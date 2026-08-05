
# streaming/engine
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

import torch
from  torch import Tensor
import math
import copy

from collections import deque
from src.VAE.base_VAE import VAE
from src.logging_utils import Logger, Verbosity
from src.utils import MIN_TH_BOOTSTRAP, TANH_S_BY_METRIC, EPS_BY_METRIC

from src.Streaming.trainer_utils import ThresholdState, Metric, PredStats
from src.Streaming.streaming_tools import PCAMahalanobis, WelfordVec, GMMLatentScorer, OnlineSigmaPrior
from src.Streaming.evaluation import calc_sigma_mad, calc_sigma_iqr, extract_stats, pick_best_threshold, find_best_threshold_f1

# from distribution_analysis import *

# Maintain a buffer of recent reconstruction errors (or combined scores).
# Each step, estimate the q-quantile (e.g. 0.99) of the buffer and then
# EMA-smooth it into value. This gives a stable, slowly-adapting 
# threshold that follows the data.

# ---------- CONFIGURABLE DEFAULTS ----------

class StreamTrainer:
    def __init__(
        self,
        model: VAE,
        online_optimizer: torch.optim.Optimizer,
        offline_optimizer: torch.optim.Optimizer,
        offline_batch_size: int,
        device: str,
        #loss_fn: str,                     # returns (loss, recon_err, z_stats)
        clip_grads_norm: float,
        threshold: ThresholdState,
        hard_threshold_cut: bool,
        train_policy: str,            # "all" or "normal_only"
        stream_mode: str,                  # "point" or "window"
        elbo_beta: float,
        # lambda_variance: float,
        metric_weights: dict,
        anom_if_score_above: float,
        training_loss: str,       # "elbo" | "var_reg" | "elbo_var"
        lambda_variance: float,   # weight on variance regularisation term
        logger: Logger,
        update_if_score_below: float = 0.5, # conf threshold below which online update is allowed
        threshold_overrides: Optional[Dict[str, ThresholdState]] = None,
    ):
        self.model = model.to(device)
        self.online_optimizer = online_optimizer
        self.offline_optimizer =  offline_optimizer
        self.offline_batch_size = offline_batch_size
        self.base_lr = self.online_optimizer.param_groups[0]["lr"]
        self.device = device
        #self.loss_fn = loss_fn
        self.train_policy = train_policy
        self.stream_mode = stream_mode
        self.clip_grads_norm = clip_grads_norm

        self.elbo_beta = elbo_beta
        self.lambda_variance = lambda_variance
        self.training_loss = training_loss
        self.anom_if_score_above = anom_if_score_above
        self.update_if_score_below = update_if_score_below

        self.logger = logger
        self._step_idx = 0
        self.last_binary_pred: int = 0  # updated each step; 0 = normal, 1 = anomaly
        self._current_T = 1  # updated in step() from x_np.shape[1] before scoring

        # Online mean/variance of latent (diagonal covariance)
        
        global _MIN_TO_SCORE_BY_MAH 
        _MIN_TO_SCORE_BY_MAH = self.model.latent_dim + 2

        self.kl_anneal_steps = 0 # {0, 500, 2000, 6000}
        self.kl_anneal_warmup_length = 0
        self.kl_anneal_offline_epochs = 0  # set via YAML training.kl_anneal_epochs

        # --------------------------------------------------------------------------
        # Scoring
        # --------------------------------------------------------------------------
        self.pca_mah = PCAMahalanobis(latent_dim=self.model.latent_dim)
        self.mah_diag = WelfordVec()
        self.gmm_latent = GMMLatentScorer(latent_dim=self.model.latent_dim)
        
        # Define how each metric’s score is computed from (recon, kl, mu_vec)
        def _score_recon(recon, kl, mu_vec, logvar_vec):
            return recon
        
        def _score_kl(recon, kl, mu_vec, logvar_vec):
            # return kl
            # 1) scale-invariant: per-latent KL
            d = max(int(self.model.latent_dim), 1)
            kl_per_dim = float(kl) / d

            # guard (KL should be >=0, but numerical issues happen)
            if not np.isfinite(kl_per_dim):
                return float("nan")
            kl_per_dim = max(0.0, kl_per_dim)

            # 2) heavy-tail fix
            return float(np.log1p(kl_per_dim))   # or: float(np.sqrt(kl_per_dim))
            
        def _score_elbo(recon, kl, mu_vec, logvar_vec):
            return self.elbo_func(recon, kl, T=self._current_T)

        def _score_elbo_w_variance(recon, kl, mu_vec, logvar_vec):

#            variance_reg_loss = torch.mean(torch.abs(torch.exp(logvar_vec) - self.model.prior_variance))
            variance_reg = ((logvar_vec - math.log(self.model.prior_variance)) ** 2).mean()

            elbo = self.elbo_func(recon, kl, T=self._current_T)
            # print("elbo", elbo, "var_reg", variance_reg, "added", lambda_variance * variance_reg)
            return elbo + self.lambda_variance * variance_reg
        
        ########################################################
        #       mu-based metrics (latent mean location)        #
        ########################################################

        # These all answer the question:
        # Where did the encoder place this sample in latent space?


        # Distance of the latent mean from the origin.
        # return ||mu||
        # Normal data -> mu near 0
        # OOD / anomaly -> mu pushed far from 0
        # Pros: Simple, Often surprisingly effective
        # Cons: Scales with latent dimension. No notion of "normal direction vs abnormal direction"
        def _score_latent_z(recon, kl, mu_vec, logvar_vec):
            mu = np.asarray(mu_vec, dtype=np.float64)
            return float(np.linalg.norm(mu))
        
        # Dimension-normalized latent magnitude
        # ||mu|| / sqrt(d)
        # Same idea as _score_latent_z, but scale-stable when:
        # latent_dim changes and comparing across models
        # Pros: Better than raw norm when d is large
        # Comparable across experiments
        # Cons: Still ignores covariance / shape of latent cloud
        def _score_mu_norm(recon, kl, mu_vec, logvar_vec):
            mu = np.asarray(mu_vec, dtype=np.float64)
            d = max(mu.size, 1)
            x = float(np.linalg.norm(mu) / np.sqrt(d))
            # Transform the raw metric before thresholding/scoring
            # due to possible heavy-tailed
            return float(np.log1p(x))   # <- key change
        
        # sum(mu^2)
        # Squared latent energy (same as ||mu||^2)
        # Directly corresponds to the mu^2 term inside KL
        # Pros: Strong theoretical grounding, Amplifies large deviations
        # Cons: Explodes faster than norm, More sensitive to rare spikes
        def _score_mu2_sum(recon, kl, mu_vec, logvar_vec):
            mu_vec = np.asarray(mu_vec, dtype=np.float64)
            return float(np.sum(mu_vec * mu_vec))

        # -------------------------------
        # Measure distance from learned normal latent distribution:
        # How many sigmas away is this point from normal latent behavior?
        
        # Compute Mahalanobis distance of current mu_vec to the
        # online-estimated latent distribution (diagonal covariance).
        # According to: 
        # D^2 = sum ((x-mu)^2 / sigma^2) 
        def _score_mah_diag(recon, kl, mu_vec, logvar_vec):
            # Wait until Welford has enough samples
            if self.mah_diag.n < _MIN_TO_SCORE_BY_MAH:
                return float("nan")

            mu_vec = np.asarray(mu_vec, dtype=np.float64)
            diff = mu_vec - self.mah_diag.mean

            variance = self.mah_diag.variance()  # already eps-clipped
            d2 = float(np.sum((diff * diff) / variance))
            return float(np.log1p(d2))  # log1p of squared distance works great
            

        def _score_mah_full(recon, kl, mu_vec, logvar_vec):

            ## O(d^3)

            return self.pca_mah.score(mu_vec)

        def _score_gmm_latent(recon, kl, mu_vec, logvar_vec):
            return self.gmm_latent.score(mu_vec)

        ########################################################
        #       logvar-based metrics (uncertainty behavior)    #
        ########################################################
        
        # These all answer the question:
        # How uncertain is the encoder about this sample?
        
        # Average posterior log-variance
        # High: model unsure
        # Low: overconfident (possible posterior collapse)
        # Pros: Simple diagnostic, Good drift signal
        # Cons: Sign is unintuitive, Depends on clamp range
        def _score_logvar_mean(recon, kl, mu_vec, logvar_vec):
            return float(np.mean(logvar_vec))

        # mean(exp(logvar)) = mean(variance)
        # measures Average posterior variance, direct uncertainty magnitude
        # Pros: Interpretable, Scale-correct
        # Cons: Sensitive to one large dimension
        def _score_logvar_sumexp(recon, kl, mu_vec, logvar_vec):
            d = max(len(logvar_vec), 1)
            return float(np.sum(np.exp(logvar_vec)) / d)
        
        # Variance part of KL(q||N(0,1)) without mu
        # Zero when variance = 1
        # Grows when variance deviates from prior
        # Pros: Strong theoretical grounding, Direct KL component
        # Cons: Redundant with KL if used for training, Can dominate numerically
        def _score_variance_mismatch(recon, kl, mu_vec, logvar_vec):
            # sum(exp(logvar) - logvar - 1) >= 0
            lv = np.asarray(logvar_vec, dtype=np.float64)
            return float(np.sum(np.exp(lv) - lv - 1.0))
        
        # How far is the encoder’s log-variance from the log prior variance?
        # It monitors whether the posterior variance tracks prior variance.
        # Does the posterior collapse / variance explosion?
        # It signals uncertainty drift, encoder stress, distribution shift
        # Pros: Stable, Interpretable, Dimension-normalized, Excellent anomaly metric
        # Cons: Not a training signal (by design)
        def _score_logvar_target_mse(recon, kl, mu_vec, logvar_vec):
            lv = np.asarray(logvar_vec, dtype=np.float64)
            target = math.log(self.model.prior_variance)
            x = float(np.mean((lv - target) ** 2))
            return float(np.log1p(x))   # <- key change
            #return float(np.mean((lv - target)**2))
        
        #------------------


        self.confusion_stats = {
            "labels": [],
            "metrics" : {
                "final": {"scores_0_1": [], "tp": 0, "fp": 0, "tn": 0, "fn": 0},
            },
        }

        # optional: also allow channels for "final"
        # self.confusion_stats["final"]["channels"] = _empty_confidence_bucket()

        metric_definitions: dict[str, Callable[[float,float,np.ndarray], float]] = {
            "recon":     _score_recon,
            "kl":        _score_kl,
            "elbo":      _score_elbo,
            "elbo_w_var": _score_elbo_w_variance,
            "latent_z":  _score_latent_z,
            "mu_norm":   _score_mu_norm,
            "mu2_sum":   _score_mu2_sum,
            "mah_diag":       _score_mah_diag,
            "mah_full_cov":  _score_mah_full,
            "gmm_latent":    _score_gmm_latent,
            "logvar_mean": _score_logvar_mean,
            "logvar_sumexp": _score_logvar_sumexp,
            "variance_mismatch": _score_variance_mismatch,
            "logvar_target_mse": _score_logvar_target_mse,
        }

        self.metrics: Dict[str, Metric] = {}

        for name, w in metric_weights.items():
            # if w <= 0.0:
            #     continue  # disabled in YAML
            if name not in metric_definitions:
                self.logger.info_low(lambda: f"Unknown metric name in YAML:{name}")
                continue  # unknown metric name

            base = (threshold_overrides or {}).get(name, threshold)
            th = copy.deepcopy(base)

            self.metrics[name] = Metric(weight   = float(w),
                                         compute_score_fn = metric_definitions[name],
                                         threshold = th,
                                         ref_buffer   = deque(maxlen=threshold.buffer_size),
                                         sigma_prior = OnlineSigmaPrior()
                                     )
                # init
            self.confusion_stats["metrics"][name] = {"scores_0_1": [], "tp": 0, "fp": 0, "tn": 0, "fn": 0}

        self.hard_threshold_cut = hard_threshold_cut

        
        # for 2d/3d visualisations
        self.test_latents = []
        self.test_labels = []
        self.latent_ref = []
        self.decoder_sensitivity: list[float] = []

    def elbo_func(self, recon_loss, kl_loss, beta: float | None = None, T: int | None = None):
        if beta is None:
            beta = float(self.kl_beta())
        else:
            beta = float(beta)

        # kl_auto_scale: divide beta by kl_scale_factor(T) so that elbo_beta has a
        # consistent "per (B,T,Z)-element" meaning regardless of kl_mode.
        # T is required when kl_mode="no_norm"; ignored for "norm" and "norm_T".
        if self.model.kl_auto_scale:
            if T is None:
                raise ValueError("elbo_func: T must be provided when kl_auto_scale=True")
            beta = beta / self.model.kl_scale_factor(T)

        # Flow KL correction: _kl_flow sums over Z before .mean(B,T), making it
        # O(Z) regardless of kl_mode. After kl_auto_scale, Gaussian KL is O(1).
        # Apply an additional factor so the configured elbo_beta has the same
        # per-element meaning for both Gaussian and flow posteriors:
        #
        #   beta_flow = beta × kl_scale_factor(T) / latent_dim
        #
        # For kl_mode='norm'    (scale=1):   factor = 1/Z  → divides by Z  ← main fix
        # For kl_mode='norm_T'  (scale=Z):   factor = Z/Z = 1 → no change (already matched)
        # For kl_mode='no_norm' (scale=Z×T): factor = T       → rare with flows
        if self.model.flow is not None:
            _T = T if T is not None else 1
            beta = beta * self.model.kl_scale_factor(_T) / self.model.latent_dim

        elbo = recon_loss + beta * kl_loss
        return elbo
    
    def kl_beta(self):
        if self.kl_anneal_steps <= 0:
            return self.elbo_beta
        t = max(self._step_idx - self.kl_anneal_warmup_length, 0)
        frac = min(1.0, t / float(self.kl_anneal_steps))
        return self.elbo_beta * frac

    def _latent_to_vec(self, mu_x: torch.Tensor, logvar_x: torch.Tensor, agg: str = "mean"):
        # mu_x/logvar_x: (1,T,Z) for forward_one
        mu = mu_x[0]         # (T,Z)
        lv = logvar_x[0]     # (T,Z)

        if agg == "mean":
            return mu.mean(dim=0), lv.mean(dim=0)          # (Z,), (Z,)
        if agg == "max":
            return mu.abs().max(dim=0).values, lv.max(dim=0).values
        if agg == "last":
            return mu[-1], lv[-1]
        raise ValueError(agg)

    # single-sample path (streaming eval, online threshold updates, calibration,
    # confusion stats). Keep it batch=1 only.
    def forward_one(self, x_np: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
        # Returns (recon, kl, elbo) for one sample/window without updating anything.
        # x_np must be 3-D: (1, 1, M) for point or (1, T, M) for window.

        x = torch.from_numpy(x_np).float().to(self.device)
        self.model.eval()
        with torch.inference_mode():
            recon, kl, mu_x, logvar_x = self.model.vae_step(x)

        if mu_x.dim() == 2:
            # old architecture: (B,Z)
            mu_vec_t = mu_x[0]
            logvar_vec_t = logvar_x[0]
        elif mu_x.dim() == 3:
            # new architecture: (B,T,Z)
            if mu_x.size(0) != 1:
                raise ValueError("forward_one expects single sample/window")
            mu_vec_t, logvar_vec_t = self._latent_to_vec(mu_x, logvar_x, agg="mean")
        else:
            raise ValueError("forward_one expects single sample/window")
        mu_vec = mu_vec_t.detach().cpu().numpy()
        logvar_vec = logvar_vec_t.detach().cpu().numpy()
        # # mu_x/logvar_x should be (B,z). Enforce B==1 here.
        # if mu_vec_t.dim() != 2:
        #     raise ValueError(f"Expected mu_vec_t to be (B,z). Got {mu_vec_t.shape}")
        # if mu_vec_t.size(0) != 1:
        #     raise ValueError(
        #         f"forward_one is single-sample only, but got batch size {mu_vec_t.size(0)}. "
        #         f"Use offline batch training path; do not call forward_one on batches."
        #     )
        recon_f = float(recon.detach().cpu())
        kl_f = float(kl.detach().cpu())
        # mu_vec = mu_x[0].detach().cpu().numpy()        # (z_dim,)
        # logvar_vec = logvar_x[0].detach().cpu().numpy()# (z_dim,)
        return recon_f, kl_f, mu_vec, logvar_vec

    # offline only, purely to compute a scalar ELBO for validation
    # (and training monitoring) on batches.
    @torch.no_grad()
    def eval_batch_elbo(self, Xb_np: np.ndarray, beta: float | None = None) -> float:
        x = torch.from_numpy(Xb_np).float().to(self.device)  # (B, 1, M) or (B, T, M)
        self.model.eval()
        with torch.inference_mode():
            recon, kl, _, _ = self.model.vae_step(x)
            elbo = self.elbo_func(recon, kl, beta=beta, T=x.size(1))
        return float(elbo.detach().cpu())

    def _compute_training_loss(
        self,
        x: torch.Tensor,
        beta: float | None = None,
    ) -> torch.Tensor:
        """
        Compute the scalar training loss according to self.training_loss:
          "elbo"     — standard ELBO (recon + beta*KL)
          "var_reg"  — variance calibration only: mean((logvar - log(prior_var))^2)
          "elbo_var" — ELBO + lambda_variance * var_reg
        """
        recon_t, kl_t, _, logvar_x = self.model.vae_step(x)
        T = x.size(1)

        if self.training_loss == "var_reg":
            target = math.log(self.model.prior_variance)
            return ((logvar_x - target) ** 2).mean()

        elbo_t = self.elbo_func(recon_t, kl_t, beta=beta, T=T)

        if self.training_loss == "elbo_var":
            target = math.log(self.model.prior_variance)
            var_reg = ((logvar_x - target) ** 2).mean()
            return elbo_t + self.lambda_variance * var_reg

        return elbo_t  # "elbo" (default)

    def train_one(self,
                  x_np: np.ndarray,
                  beta: float | None = None,
                  opt: torch.optim.Optimizer | None = None) -> float:

        if opt is None:
            opt = self.online_optimizer  # fallback

        x = torch.from_numpy(x_np).float().to(self.device)  # (1, 1, M) or (1, T, M)

        self.model.train()
        opt.zero_grad(set_to_none=True)

        loss = self._compute_training_loss(x, beta=beta)
        loss.backward()

        if self.clip_grads_norm and self.clip_grads_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grads_norm)

        opt.step()
        return float(loss.detach().cpu())

    def train_batch(self, Xb_np: np.ndarray, beta: float | None = None, opt=None) -> float:
        """
        Batch training step for OFFLINE warmup only.
        Xb_np: (B,T,M) or (B,M)
        """
        if opt is None:
            opt = self.offline_optimizer

        x = torch.from_numpy(Xb_np).float().to(self.device)
        self.model.train()
        opt.zero_grad(set_to_none=True)

        # var_reg has no reconstruction signal — fall back to ELBO for offline warmup
        # so the decoder is trained as a generative model before online var_reg kicks in.
        if self.training_loss == "var_reg":
            recon_t, kl_t, _, _ = self.model.vae_step(x)
            loss = self.elbo_func(recon_t, kl_t, beta=beta, T=x.size(1))
        else:
            loss = self._compute_training_loss(x, beta=beta)

        loss.backward()
        if self.clip_grads_norm and self.clip_grads_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grads_norm)

        opt.step()
        return float(loss.detach().cpu())

    def offline_fit(
        self,
        X_train: np.ndarray,          # (N, M) or (N, W, M)
        X_val: np.ndarray,            # same shape type as train
        *,
        epochs: int = 20,
        shuffle: bool = True,
        seed: int = 0,
        early_stop_patience: int = 5,
        use_fixed_beta: bool = True,  # avoids dependence on self._step_idx during offline
        kl_anneal_epochs: int = -1,   # ramp beta from 0 → elbo_beta over this many epochs; -1 = use self.kl_anneal_offline_epochs; 0 = disabled
    ):
        # Offline training on warmup train/val for epochs.
        # Does NOT use step(). Does NOT update thresholds/buffers/confusions.
        # Works for points (N,M) and windows (N,W,M) as long as VAE supports batching.
        if kl_anneal_epochs < 0:
            kl_anneal_epochs = self.kl_anneal_offline_epochs
        rng = np.random.default_rng(seed)
        
        use_val = len(X_val) > 0
        best_val = float("inf")
        best_state = None
        bad = 0

        history = {"train_elbo": [], "val_elbo": []}

        for ep in range(1, epochs + 1):

            if kl_anneal_epochs > 0:
                beta = float(self.elbo_beta) * min(1.0, ep / float(kl_anneal_epochs))
            elif use_fixed_beta:
                beta = float(self.elbo_beta)
            else:
                beta = float(self.kl_beta())

            # ---- TRAIN ----
            idx = np.arange(len(X_train))
            if shuffle:
                rng.shuffle(idx)

            train_sum, train_n = 0.0, 0
            for start in range(0, len(idx), self.offline_batch_size):
                bidx = idx[start:start + self.offline_batch_size]
                Xb = X_train[bidx]
                elbo_b = self.train_batch(Xb, beta=beta, opt=self.offline_optimizer)  # <-- TRAINING
                train_sum += elbo_b * len(Xb)
                train_n += len(Xb)

            mean_train = train_sum / max(train_n, 1)
            history["train_elbo"].append(mean_train)

            if use_val:
                # ---- VAL (no updates) ----
                val_sum, val_n = 0.0, 0
                for start in range(0, len(X_val), self.offline_batch_size):
                    Xb = X_val[start:start + self.offline_batch_size]
                    elbo_b = self.eval_batch_elbo(Xb, beta=beta)
                    val_sum += elbo_b * len(Xb)
                    val_n += len(Xb)
                mean_val = val_sum / max(val_n, 1)
                history["val_elbo"].append(mean_val)

                self.logger.info_low(lambda: f"[offline] epoch {ep}/{epochs} "
                                         f"beta={beta:.4f} train_elbo={mean_train:.4f} val_elbo={mean_val:.4f}")

                # ---- EARLY STOP (disabled during KL ramp) ----
                in_ramp = kl_anneal_epochs > 0 and ep < kl_anneal_epochs
                if not in_ramp:
                    if mean_val < best_val:
                        best_val = mean_val
                        best_state = copy.deepcopy(self.model.state_dict())
                        bad = 0
                    else:
                        bad += 1
                        if bad >= early_stop_patience:
                            self.logger.info_low(lambda: f"[offline] early stop at epoch {ep} (best val {best_val:.6f})")
                            break
            else:
                self.logger.info_low(lambda: f"[offline] epoch {ep}/{epochs} "
                                         f"beta={beta:.4f} train_elbo={mean_train:.4f} (no val)")

                # ---- EARLY STOP on train loss (disabled during KL ramp) ----
                in_ramp = kl_anneal_epochs > 0 and ep < kl_anneal_epochs
                if not in_ramp:
                    if mean_train < best_val:
                        best_val = mean_train
                        best_state = copy.deepcopy(self.model.state_dict())
                        bad = 0
                    else:
                        bad += 1
                        if bad >= early_stop_patience:
                            self.logger.info_low(lambda: f"[offline] early stop at epoch {ep} (best train {best_val:.6f})")
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history

    def offline_fit_interleaved(
        self,
        warmup_segments: list,   # list of WarmupSegment (role, X, y)
        *,
        epochs: int = 40,
        early_stop_patience: int = 5,
        use_fixed_beta: bool = True,
        kl_anneal_epochs: int = -1,   # ramp beta from 0 → elbo_beta over this many epochs; -1 = use self.kl_anneal_offline_epochs; 0 = disabled
    ) -> dict:
        """
        Offline training that walks warmup_segments in chronological order each epoch.

        - 'train' segments: mini-batch gradient updates (sequential, no shuffle).
        - 'val'   segments: no-gradient evaluation, split by label into normal / anomaly.

        At epoch end:
          - ReduceLROnPlateau steps on val_normal_elbo (lower = better reconstruction).
          - Early stopping on val_normal_elbo; best model state is restored.
          - margin = val_anom_elbo / val_normal_elbo is logged (want > 1 and growing).
        """
        if kl_anneal_epochs < 0:
            kl_anneal_epochs = self.kl_anneal_offline_epochs

        from torch.optim.lr_scheduler import ReduceLROnPlateau

        scheduler = ReduceLROnPlateau(
            self.offline_optimizer,
            mode='min', factor=0.5, patience=3, min_lr=1e-7,
        )

        best_val_normal = float("inf")
        best_state = None
        bad = 0

        history = {
            "train_elbo":      [],
            "val_normal_elbo": [],
            "val_anom_elbo":   [],
            "margin":          [],
        }

        for ep in range(1, epochs + 1):
            if kl_anneal_epochs > 0:
                beta = float(self.elbo_beta) * min(1.0, ep / float(kl_anneal_epochs))
            elif use_fixed_beta:
                beta = float(self.elbo_beta)
            else:
                beta = float(self.kl_beta())

            train_sum, train_n       = 0.0, 0
            val_norm_sum, val_norm_n = 0.0, 0
            val_anom_sum, val_anom_n = 0.0, 0

            for seg in warmup_segments:
                X_seg = seg.X   # (N_seg, W, M)
                y_seg = seg.y   # (N_seg,)

                if seg.role == 'train':
                    # Sequential mini-batch updates (chronological order within segment)
                    for start in range(0, len(X_seg), self.offline_batch_size):
                        Xb = X_seg[start:start + self.offline_batch_size]
                        elbo_b = self.train_batch(Xb, beta=beta)
                        train_sum += elbo_b * len(Xb)
                        train_n   += len(Xb)

                else:  # 'val'
                    normal_mask = (y_seg == 0)
                    anom_mask   = (y_seg == 1)

                    X_norm = X_seg[normal_mask]
                    X_anom = X_seg[anom_mask]

                    for start in range(0, len(X_norm), self.offline_batch_size):
                        Xb = X_norm[start:start + self.offline_batch_size]
                        if len(Xb):
                            elbo_b = self.eval_batch_elbo(Xb, beta=beta)
                            val_norm_sum += elbo_b * len(Xb)
                            val_norm_n   += len(Xb)

                    for start in range(0, len(X_anom), self.offline_batch_size):
                        Xb = X_anom[start:start + self.offline_batch_size]
                        if len(Xb):
                            elbo_b = self.eval_batch_elbo(Xb, beta=beta)
                            val_anom_sum += elbo_b * len(Xb)
                            val_anom_n   += len(Xb)

            mean_train      = train_sum      / max(train_n,      1)
            mean_val_normal = val_norm_sum   / max(val_norm_n,   1)
            mean_val_anom   = val_anom_sum   / max(val_anom_n,   1)
            margin = mean_val_anom / max(mean_val_normal, 1e-8)

            history["train_elbo"].append(mean_train)
            history["val_normal_elbo"].append(mean_val_normal)
            history["val_anom_elbo"].append(mean_val_anom)
            history["margin"].append(margin)

            self.logger.info_low(
                lambda ep=ep, epochs=epochs, mt=mean_train,
                       mvn=mean_val_normal, mva=mean_val_anom, m=margin:
                f"[offline_interleaved] epoch {ep}/{epochs} "
                f"train={mt:.4f}  val_normal={mvn:.4f}  val_anom={mva:.4f}  "
                f"margin={m:.2f}x"
            )

            scheduler.step(mean_val_normal)

            # Early stopping on val_normal_elbo
            if mean_val_normal < best_val_normal:
                best_val_normal = mean_val_normal
                best_state = copy.deepcopy(self.model.state_dict())
                bad = 0
            else:
                bad += 1
                if bad >= early_stop_patience:
                    self.logger.info_low(
                        lambda ep=ep, bv=best_val_normal:
                        f"[offline_interleaved] early stop at epoch {ep} "
                        f"(best val_normal {bv:.6f})"
                    )
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history

    def recalibrate_thresholds(self) -> None:
        """After a calibration pass, recompute each threshold as the true quantile of the
        full ref_buffer, overriding the noisy EMA value accumulated during the pass."""
        for name, metric in self.metrics.items():
            buf = np.asarray(metric.ref_buffer, dtype=np.float64)
            finite_buf = buf[np.isfinite(buf)]
            if len(finite_buf) > 0:
                metric.threshold.value = float(np.quantile(finite_buf, metric.threshold.q))
    
    def reset_eval_state(self) -> None:
        """Clear evaluation tracking accumulated during calibration.
        Call after a calibration pass so calibration samples don't pollute test metrics."""
        self.confusion_stats["labels"].clear()
        for cm in self.confusion_stats["metrics"].values():
            cm["scores_0_1"].clear()
            cm["tp"] = cm["fp"] = cm["tn"] = cm["fn"] = 0
        self.test_latents.clear()
        self.test_labels.clear()
        self.decoder_sensitivity.clear()

    @torch.no_grad()
    def compute_decoder_sensitivity(self, x_np: np.ndarray, n_perturb: int = 8, eps_scale: float = 1.0) -> float:
        """Measure how sensitive the decoder is to latent perturbations.

        Encodes x to get mu_x (the posterior mean, no reparameterization noise),
        then decodes n_perturb perturbed versions z_i = mu_x + eps_i (eps ~ N(0, eps_scale²)).
        Returns the mean std of reconstructions across perturbations — high means
        the decoder uses z (good), low means it ignores z (posterior collapse).
        """
        x = torch.from_numpy(x_np).float().to(self.device)  # (1, T, M)
        _, T, _ = x.shape
        self.model.eval()
        # Mirror base_VAE.forward: inject PE into encoder input when pos_enc_mode=="enc"
        x_enc = x
        if getattr(self.model, "pos_enc_mode", "none") in ("enc", "enc_dec"):
            pos = self.model.positional_encoding(T, x.device).expand(x.shape[0], -1, -1)
            x_enc = torch.cat([x, pos], dim=-1)
        h_seq = self.model.encode_sequence(x_enc)
        if self.model.pool_latent:
            mu_x = self.model.mu(self.model.pool_h(h_seq))  # (1, Z)
        else:
            mu_x = self.model.mu(h_seq)                     # (1, T, Z)

        recons = []
        for _ in range(n_perturb):
            z_perturbed = mu_x + torch.randn_like(mu_x) * eps_scale
            x_hat = self.model.decode_z(z_perturbed, T)     # (1, T, M)
            recons.append(x_hat)

        stacked = torch.stack(recons, dim=0)  # (n_perturb, 1, T, M)
        return float(stacked.std(dim=0).mean().item())

    # for 2D/3D display: Z_ref: (N_ref, Z) latent mu vectors from normal reference data.
    def compute_latent_reference_stats(self, Z_ref: np.ndarray):
        
        # Mean
        latent_mean = Z_ref.mean(axis=0)  # (Z,)

        # Covariance
        cov = np.cov(Z_ref, rowvar=False)      # (Z,Z)

        # Regularization (VERY IMPORTANT)
        eps = 1e-4
        cov_reg = cov + eps * np.eye(cov.shape[0])

        # Stable inverse
        latent_cov_inv = np.linalg.pinv(cov_reg)
        return latent_mean, latent_cov_inv

    # --------- streaming step ---------
    # learn (bool): allow turning training on/off per sample. 
    # Sometimes we want to evaluate without updating (e.g., 
    # in a pure test segment or when pausing updates during drift).
    def step(self,
             t_np: Optional[np.ndarray],
             x_np: np.ndarray,
             label: Optional[int],
             is_eval: bool,
             learn: bool = True,
             ):
        
        # -----------------------------------------------------------------------------
        # 1. Eval

        self._current_T = x_np.shape[1]  # needed by elbo scoring closures
        eval_recon, eval_kl, mu_x_vec, logvar_x_vec = self.forward_one(x_np)
        
        metric_scores: dict[str, float] = {}

        for name, metric in self.metrics.items():
            score = metric.compute_score_fn(eval_recon, eval_kl, mu_x_vec, logvar_x_vec)
            if np.isfinite(score):
                metric_scores[name] = float(score)

        
        if is_eval:
            if label is not None:
                self.confusion_stats["labels"].append(label)
            self.test_latents.append(mu_x_vec)
            self.test_labels.append(label)
            self.decoder_sensitivity.append(self.compute_decoder_sensitivity(x_np))
        else:
            # warmup sample: feed to reference distribution, not eval tracking
            self.latent_ref.append(mu_x_vec)

        # 1. Predict all metrics by Thresholds (if this is not a warmup)
        metric_pred_stats: dict[str, PredStats] = {}

        for name, metric in self.metrics.items():
            if name not in metric_scores:
                if is_eval:
                    self.confusion_stats["metrics"][name]["scores_0_1"].append(float("nan"))
                metric_pred_stats[name] = PredStats(name, 0, float("nan"), float("nan"), float("nan"), 0.0, 0.5, 0.0, 1.0)
                continue
            loss = float(metric_scores[name])

            pred = self.predict(name, loss, metric.threshold, metric.ref_buffer, metric.sigma_prior)
            metric_pred_stats[name] = pred

            if is_eval:
                self.feed_metric_confusion_stats(label, name, pred.pred, pred.conf_0_1)
        
        should_train = False

        # Ensemble decision
        # Using a confidence-weighted voting ensemble over multiple anomaly scores 
        # (reconstruction, KL, latent, Mahalanobis, etc.), where each score 
        # contributes a sign (normal/anomaly) and a confidence derived from its 
        # margin and empirical CDF.
        binary_pred, pred_score_01 = self.ensemble_decision(metric_pred_stats)

        if binary_pred == 0 and pred_score_01 <= self.update_if_score_below:
            should_train = True

        # During calibration (learn=False) always update reference stats.
        # During online streaming (learn=True) only update on predicted-normal samples.
        if binary_pred == 0 or not learn:
            for name, metric in self.metrics.items():
                if name not in metric_pred_stats:
                    continue  # score was NaN/Inf this step
                ps = metric_pred_stats[name]
                metric.sigma_prior.update(ps.loss)
                metric.ref_buffer.append(ps.loss)
                metric.update_threshold()

            self.update_latent_stats(mu_x_vec)       
            
        if is_eval:
            self.feed_metric_confusion_stats(label, "final", binary_pred, pred_score_01)
        metric_scores["final"] = float(pred_score_01)

        trained_bin=0
        # -----------------------------------------------------------------------------
        # Update when no obvious anomaly (skip when learn=False, e.g. calibration pass)
        if learn and ((self.train_policy == "all") or should_train) :
            # ---------- TRAIN PASS (with grad) ----------
            _ = self.train_one(x_np=x_np, opt=self.online_optimizer)
            trained_bin = 1
            # for name, metric in self.metrics.items():
            #     metric.learn_buffer.append(metric_scores[name])
        # -----------------------------------------------------------
        # If using windows, start/end times are for first and last timestamps
        # Otherwise, start/end time are the same 
        if isinstance(t_np, np.ndarray):
            data_ts_start = t_np[0]
            data_ts_end   = t_np[-1]
        else:
            data_ts_start = data_ts_end = t_np
        
        cur_ts = data_ts_end

        # ------------------------------------------------------------------------------------------

        self._step_idx += 1

        ##########################################
        #             RECORD and LOG             #
        ##########################################          
        verbose = Verbosity.INFO_HIGH

        confusion = None
        if label is not None:
            if binary_pred == 1 and label == 0: confusion = "FP"
            elif binary_pred == 0 and label == 1: confusion = "FN"
            elif binary_pred == 0 and label == 0: confusion = "TN"
            elif binary_pred == 1 and label == 1: confusion = "TP"
        

        if is_eval :
            
            current_lr = float(self.online_optimizer.param_groups[0]["lr"]) if self.online_optimizer.param_groups else float("nan")
            logger_summary = {
                "ts_start":   str(data_ts_start),
                "ts_end":     str(data_ts_end),
                "step":       self._step_idx,
                "confusion":  confusion or None,
                "label": (int(label) if label is not None else None),
                "final_pred": int(binary_pred),
                "trained":    int(trained_bin),
                "final_conf": float(pred_score_01),  # kept for CSV column compatibility
                "lr":         current_lr,
            }
            self.log_prediction(logger_summary, metric_scores, verbose)

        self.last_binary_pred = binary_pred
        return self._step_idx

    def predict(self, metric_name: str, loss: float, th: ThresholdState, buffer: deque, sigma_prior : OnlineSigmaPrior) :
        
        buf = np.asarray(buffer, dtype=np.float64)  # buffer is deque
        
        if th.value is None or not np.isfinite(th.value) or not np.isfinite(loss):
            # print(f"{self._step_idx}: 'th.value is None or not np.isfinite(th.value)'")
                           # metric_name, pred,        loss, raw_margin, threshold value, signed_conf, conf_0_1,   z, sigma
            return PredStats(metric_name,    0, float(loss),        0.0,    float("nan"),         0.0,      0.5, 0.0, 1.0)

        if buf.size < MIN_TH_BOOTSTRAP:
            # not enough reference yet -> neutral
            return PredStats(metric_name, 0, float(loss), float(loss - th.value), float(th.value), 0.0, 0.5, 0.0, 1.0)

        loss_margin = float(loss - th.value)
        # margins = buf - float(th.value)

        eps = EPS_BY_METRIC.get(metric_name, 1e-3)
        sigma_mad = calc_sigma_mad(buf, eps=eps)
        sigma_iqr = calc_sigma_iqr(buf, eps=eps)
        sigma_hat = max(sigma_mad, sigma_iqr)  # from ref_buffer
        sigma0 = sigma_prior.sigma
        sigma_eff = np.sqrt(sigma_hat**2 + sigma0**2)  # or shrinkage blend
        sigma_eff = max(sigma_eff, eps)

        # Normalized distance from threshold
        # same idea as a z-score, but centered at the threshold, not the mean
        # z = +1 : score is 1 std above threshold
        # z = −2 : score is 2 std below threshold
        # z ≈ 0 : score is right at the boundary
        z = float(np.clip(loss_margin / sigma_eff, -6.0, 6.0))

        # ---- confidence via sigmoid on z-score ----
        # conf01 = 0.5 when score == threshold, → 0 when very normal, → 1 when very anomalous
        conf01 = float(1.0 / (1.0 + np.exp(-z)))

        # map to [-1,1] for ensemble compatibility
        signed = float(2.0 * conf01 - 1.0)

        bin_pred = int(conf01 >= self.anom_if_score_above)  # equivalent to loss >= th.value

        return PredStats(
            metric_name=metric_name,
            pred=bin_pred,
            loss=float(loss),
            raw_margin=loss_margin,
            threshold=float(th.value),
            signed_conf=signed,
            conf_0_1=conf01,
            z=z,
            sigma=float(sigma_eff),
        )

    def confusion_counter(self, label, pred, confusion):
        if pred ==1 and label == 1 :
            confusion["tp"] += 1
        elif pred == 1 and label == 0:
            confusion["fp"] += 1
        elif pred == 0 and label == 0:
            confusion["tn"] += 1
        else:
            confusion["fn"] += 1
    
    def feed_metric_confusion_stats(self, label: int | None, name: str, final_pred, pred_scores_0_1):
        cm = self.confusion_stats["metrics"][name]
        cm["scores_0_1"].append(float(pred_scores_0_1))
        if label is not None :
            self.confusion_counter(int(label), int(final_pred), cm)


    def ensemble_decision(self, pred_stats_by_metric: Dict[str, PredStats]) -> tuple[int, float]:

        num = 0.0
        den = 0.0

        for name, metric in self.metrics.items():
            w = float(metric.weight)
            if w <= 0:
                continue
            c = float(pred_stats_by_metric[name].signed_conf)  # in [-1, 1]
            num += w * c
            den += w

        if den <= 1e-12:
            # fallback: "uncertain"
            signed = 0.0
        else:
            signed = num / den  # guaranteed in [-1, 1]


        pred_score_01 = 0.5 * (signed + 1.0) # 0=confident normal, 0.5=uncertain, 1=confident anomaly
        high_conf_anom = int(pred_score_01 >= self.anom_if_score_above)

        # return (pred, confidence_0_1, signed_margin)
        return high_conf_anom, pred_score_01
    # Welford’s algorithm


    def update_latent_stats(self, mu_vec: np.ndarray) -> None:
        # Online updates the latent mean and per-dimension variance using Welford's method.
        # mu_vec: shape (latent_dim,)
        mu_vec = mu_vec.astype(np.float64)
        self.pca_mah.add_reference(mu_vec) # update buffer and refit occasionally
        self.mah_diag.update(mu_vec)
        self.gmm_latent.add_reference(mu_vec)


    def log_prediction(self, logger_summary, metric_scores, verbose) :
        # Add all stats blocks in a loop
        # logger_summary.update(metric_scores)
        
        # Build structured metrics payload
        logger_summary["metrics"] = {
            name: {
                "score": float(metric_scores[name]),
                "threshold": float(self.metrics[name].threshold.value)
                             if name in self.metrics else None,
            }
            for name in metric_scores
        }

        # Record logger_summary. should obey Vervosity
        self.logger.event(verbose, "pred_outcome", logger_summary)


    # --------- finalize & evaluate ---------
    def finalize_metrics_stats(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {"metrics": {}}

        y = np.asarray(self.confusion_stats["labels"], dtype=int)
        
        for name, metric in self.confusion_stats["metrics"].items() :
            if name != "final" :
                results["metrics"][name] = extract_stats(y, metric, f"metric:{name}")

        # final bucket
        results["metrics"]["final"] = extract_stats(y, self.confusion_stats["metrics"]["final"], "final")

        # After final stats computed:
        y = np.asarray(self.confusion_stats["labels"], dtype=int)
        final_scores = np.asarray(self.confusion_stats["metrics"]["final"]["scores_0_1"], dtype=float)

        best_thr, best_stats, _ = pick_best_threshold(y, final_scores, objective="f1")
        results["final_threshold_tuning"] = {
            "best_thr_f1": best_thr,
            "best_stats": best_stats,
        }

        best_thr_f2, best_stats_f2, _ = pick_best_threshold(y, final_scores, objective="fbeta", beta=2.0)
        results["final_threshold_tuning_f2"] = {
            "best_thr_f1": best_thr_f2,
            "best_stats": best_stats_f2,
        }

        best_thr_j, best_stats_j, _ = pick_best_threshold(y, final_scores, objective="youden")
        results["final_threshold_tuning_youden"] = {
            "best_thr_f1": best_thr_j,
            "best_stats": best_stats_j,
        }

        # Only do this if you actually have labels
        if len(y) == len(final_scores) and len(y) > 0:
            best = find_best_threshold_f1(y, final_scores)
            results["final_threshold_search"] = best


        return results
