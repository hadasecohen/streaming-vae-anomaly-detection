import numpy as np
import scipy as sp
from collections import deque


class OnlineSigmaPrior:
    def __init__(self, alpha=0.01, clip_k=5.0, init_sigma=1e-3):
        self.alpha = alpha
        self.clip_k = clip_k
        self.mu = None
        self.var = init_sigma**2

    def update(self, x):
        if not np.isfinite(x):
            return  # skip NaN/Inf — would permanently corrupt the EWMA variance
        if self.mu is None:
            self.mu = float(x)
            return

        sigma = (self.var ** 0.5)
        lo = self.mu - self.clip_k * sigma
        hi = self.mu + self.clip_k * sigma
        x = float(np.clip(x, lo, hi))  # winsorize

        a = self.alpha
        delta = x - self.mu
        self.mu += a * delta
        # EWMA variance update
        self.var = (1 - a) * self.var + a * (delta * delta)

    @property
    def sigma(self):
        return float(max(self.var, 1e-24) ** 0.5)

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


# Approximate Mahalanobis using top-k PCA components from a rolling buffer.
#    Score(mu) = sqrt( sum_i (proj_i^2 / (eigval_i + eps)) )
#    where proj = U_k^T (mu - mean)
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
            return 0.0
        mu_vec = np.asarray(mu_vec, dtype=np.float64)
        diff = mu_vec - self.mean

        # project into PCA subspace: (k,)
        proj = self.components @ diff

        d2 = np.sum((proj * proj) / (self.eigvals + self.eps))
        return float(np.sqrt(d2))

