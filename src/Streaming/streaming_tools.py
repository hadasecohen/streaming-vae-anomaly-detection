import torch

from src.utils import _as_iso

import math
import numpy as np
from collections import deque
from typing import Any
from scipy.stats import ks_2samp

    
class EWMAZScore:
    """
    Streaming EWMA mean/variance -> z-score.

    Use for: recon_loss, KL, ELBO, mah_score, etc.
    """
    def __init__(self, beta: float = 0.99, eps: float = 1e-8):
        assert 0 < beta < 1
        self.beta = beta
        self.eps = eps
        self.initialized = False
        self.mean = 0.0
        self.var = 0.0  # EWMA variance estimate

    def update(self, x: float) -> float:
        x = float(x)
        if not self.initialized:
            self.mean = x
            self.var = 0.0
            self.initialized = True
            return 0.0

        # EWMA mean
        self.mean = self.beta * self.mean + (1.0 - self.beta) * x

        # EWMA variance (of residuals)
        resid = x - self.mean
        self.var = self.beta * self.var + (1.0 - self.beta) * (resid * resid)

        return (x - self.mean) / math.sqrt(self.var + self.eps)

    def std(self) -> float:
        return math.sqrt(self.var + self.eps)

class PageHinkley:
    # If drifts fires too often: raise lambda (threshold). 
    # If too sluggish: lower threshold or increase sensitivity (slightly smaller alpha).
    # delta  : small tolerance for in-control variability
    # lambda_: threshold for declaring change
    # alpha  : forgetting factor for the mean (0.99..1.0)
    def __init__(self, delta=0.0, lambda_=50.0, alpha=0.999):
        self.delta = float(delta) # a small positive value (tolerance) to prevent 
                                  # overreacting to small fluctuations
        self.lambda_ = float(lambda_) 
        self.alpha = float(alpha)
        self.mean = 0.0
        self.cum = 0.0
        self.min_cum = 0.0  # track minimum to detect increases
        self.initialized = False

    def reset(self):
        self.initialized = False
        self.mean = 0.0
        self.cum = 0.0
        self.min_cum = 0.0

    # Returns (drift_detected: bool, score: float)
    # score grows as evidence accumulates.
    def update(self, x: float) -> bool:
        if not self.initialized:
            self.mean = x
            self.cum = 0.0
            self.min_cum = 0.0
            self.initialized = True
            return False, 0.0

        # discounted running mean
        self.mean = self.alpha * self.mean + (1 - self.alpha) * x
        # accumulates the deviations (how far the data point deviates from the running mean)
        self.cum += (x - self.mean - self.delta)
        self.min_cum = min(self.min_cum, self.cum)
        
        score = self.cum - self.min_cum  # increase evidence

        if score > self.lambda_:
            # reset after detection
            self.reset()
            return True, score

        return False, score

# Per-dimension Welford stats for vectors (mean + variance).
class WelfordVec:

    def __init__(self):
        self.eps = 1e-3
        self.n = 0
        self.mean = None
        self.m2 = None

    def update(self, mu_vec: np.ndarray):
        mu_vec = np.asarray(mu_vec, dtype=np.float64)
        self.n += 1
        if self.n == 1:
            self.mean = mu_vec.copy()
            self.m2 = np.zeros_like(mu_vec, dtype=np.float64)
        else:
            delta = mu_vec - self.mean
            self.mean += delta / self.n
            delta2 = mu_vec - self.mean
            self.m2 += delta * delta2

    def variance(self) -> np.ndarray:
        if self.n < 2:
            return np.ones_like(self.mean, dtype=np.float64)
        return np.maximum(self.m2 / (self.n - 1), self.eps)

class PCAMahalanobis:

    def __init__(
        self,
        latent_dim: int,
        k: int = 16,
        buffer_size: int = 1024,
        refit_every: int = 200,
        eps: float = 1e-2,
        min_to_fit: int = 200,
    ):
        self.latent_dim = latent_dim
        if not 1 <= k <= latent_dim :
            self.k = latent_dim
        else :
            self.k = k
        self.buffer = deque(maxlen=buffer_size)
        self.refit_every = int(refit_every)
        self.eps = float(eps)
        self.min_to_fit = int(min_to_fit)

        self._steps = 0
        self._fitted = False
        self.mean = np.zeros(latent_dim, dtype=np.float64)
        self.components = np.zeros((k, latent_dim), dtype=np.float64)  # rows are principal directions
        self.eigvals = np.ones(k, dtype=np.float64)

    def add_reference(self, mu_vec: np.ndarray):
        mu_vec = np.asarray(mu_vec, dtype=np.float64)
        self.buffer.append(mu_vec)
        self._steps += 1

        if len(self.buffer) >= self.min_to_fit and (self._steps % self.refit_every == 0):
            self._refit()

    def _refit(self):
        X = np.asarray(self.buffer, dtype=np.float64)  # (N, d)
        self.mean = X.mean(axis=0)
        Xc = X - self.mean

        # SVD: Xc = U S Vt
        # principal directions = Vt[:k]
        # eigenvalues of covariance = (S^2) / (N-1)
        try:
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        except np.linalg.LinAlgError:
            # if numerical issues: don't update fit
            return

        k = self.k
        Vt_k = Vt[:k, :]  # (k, d)
        eigvals = (S[:k] ** 2) / max(len(X) - 1, 1)

        self.components = Vt_k
        self.eigvals = np.maximum(eigvals, self.eps)
        self._fitted = True

    def score(self, mu_vec: np.ndarray) -> float:
        if not self._fitted:
            return float("nan")
        mu_vec = np.asarray(mu_vec, dtype=np.float64)
        diff = mu_vec - self.mean

        # project into PCA subspace: (k,)
        proj = self.components @ diff

        d2 = np.sum((proj * proj) / (self.eigvals + self.eps))
        return float(np.sqrt(d2))

class GMMLatentScorer:
    """GMM density estimator fitted on a rolling buffer of latent vectors.

    Score = -log p(z | GMM).
    Higher score = lower density = more anomalous.
    Returns float("nan") until enough samples have been collected to fit.

    Parameters
    ----------
    latent_dim   : dimensionality of the latent space
    n_components : number of Gaussian mixture components (default 3)
    buffer_size  : max number of latent vectors to keep (rolling window)
    refit_every  : refit the GMM every N calls to add_reference
    min_to_fit   : minimum buffer size before the first fit
    reg_covar    : regularisation added to diagonal of each covariance matrix
                   (prevents singular covariances with small buffers)
    """

    def __init__(
        self,
        latent_dim: int,
        n_components: int = 3,
        buffer_size: int = 1024,
        refit_every: int = 200,
        min_to_fit: int = 200,
        reg_covar: float = 1e-4,
    ):
        self.latent_dim = latent_dim
        self.n_components = n_components
        self.buffer = deque(maxlen=buffer_size)
        self.refit_every = int(refit_every)
        self.min_to_fit = int(min_to_fit)
        self.reg_covar = float(reg_covar)

        self._steps = 0
        self._fitted = False
        self._gmm = None

    def add_reference(self, mu_vec: np.ndarray):
        mu_vec = np.asarray(mu_vec, dtype=np.float64)
        self.buffer.append(mu_vec)
        self._steps += 1

        if len(self.buffer) >= self.min_to_fit and (self._steps % self.refit_every == 0):
            self._refit()

    def _refit(self):
        from sklearn.mixture import GaussianMixture

        X = np.asarray(self.buffer, dtype=np.float64)  # (N, d)
        # n_components must not exceed the number of samples
        n_comp = min(self.n_components, len(X))
        try:
            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type="full",
                reg_covar=self.reg_covar,
                max_iter=100,
                n_init=1,
            )
            gmm.fit(X)
            self._gmm = gmm
            self._fitted = True
        except Exception:
            # keep the previous model on failure (numerical issues, too few samples)
            pass

    def score(self, mu_vec: np.ndarray) -> float:
        """Return -log p(z | GMM). Higher = more anomalous. NaN when unfitted."""
        if not self._fitted or self._gmm is None:
            return float("nan")
        mu_vec = np.asarray(mu_vec, dtype=np.float64).reshape(1, -1)
        try:
            log_prob = float(self._gmm.score(mu_vec))  # mean log-likelihood per sample
            return -log_prob
        except Exception:
            return float("nan")


#   X: (n, d), Y: (m, d)
#   E = 2 E||X-Y|| - E||X-X'|| - E||Y-Y'||
def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    # pairwise L2 distances (O(nm))
    XY = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)
    XX = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    YY = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=-1)

    return float(2.0 * XY.mean() - XX.mean() - YY.mean())


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    sigma2 = max(float(sigma) ** 2, 1e-12)
    D2 = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
    return np.exp(-D2 / (2.0 * sigma2))

def median_heuristic_sigma(Z: np.ndarray, max_samples: int = 500) -> float:
    Z = np.asarray(Z, dtype=np.float64)
    if Z.shape[0] > max_samples:
        idx = np.random.choice(Z.shape[0], size=max_samples, replace=False)
        Z = Z[idx]

    D = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)
    # median of upper triangle excluding diagonal
    tri = D[np.triu_indices(D.shape[0], k=1)]
    med = np.median(tri) if tri.size else 1.0
    return float(max(med, 1e-6))

def mmd_rbf(X: np.ndarray, Y: np.ndarray, sigma: float | None = None) -> float:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if sigma is None:
        sigma = median_heuristic_sigma(np.vstack([X, Y]))

    Kxx = _rbf_kernel(X, X, sigma)
    Kyy = _rbf_kernel(Y, Y, sigma)
    Kxy = _rbf_kernel(X, Y, sigma)

    return float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())



# MMD^2
# Theory: https://jmlr.org/papers/volume13/gretton12a/gretton12a.pdf
# Practical hands on:
# https://apxml.com/courses/monitoring-managing-ml-models-production/chapter-2-advanced-drift-detection/practice-multivariate-drift


# my implementation (AIML425)

# Define the Gaussian RBF kernel function
def rbf_kernel(x, y, r=1.0):
    # Computes the squared Euclidean distance between each pair of points
    dist_squared = torch.sum((x[:, None, :] - y[None, :, :])**2, dim=2)
    # Applies the RBF (Gaussian) kernel formula
    return torch.exp(-dist_squared / r)

# Define the MMD function
def mmd(real_features, generated_features, r=.1):
    m = real_features.size(0)
    n = generated_features.size(0)
    K_XX = rbf_kernel(real_features, real_features, r)
    K_YY = rbf_kernel(generated_features, generated_features, r)
    K_XY = rbf_kernel(real_features, generated_features, r)    
    mmd_squared = (K_XX.sum() / (m * (m-1)) +
                   K_YY.sum() / (n * (n-1)) -
                   2 * K_XY.sum() / (m * n))
    
    return mmd_squared


