import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Any, Optional

from src.VAE_model.VAE.Attn_pool import AttnPool
from src.VAE_model.utils import Activations



@dataclass
class LossFunc:
    loss_fn_type : str
    recon_loss_kind : str
    feature_weights: Optional[torch.Tensor] = None  # for wmse only

# -------- Base class --------
class VAE(nn.Module):
    def __init__(self,
                 input_dim : int,
                 latent_dim : int,
                 loss_func: LossFunc,
                 pooling : str,
                 prior_variance : float,
                 clamp_logvar : bool,
                 clamp_bounds: list,
                 pooling_attn_dim: Optional[int] = None,
                 pooling_attn_dropout: float = 0.0,
                 enc_out_dim: int | None = None,
                 arch_str: str = "VAE",
                 pool_latent: bool = False,
                 free_bits: float = 0.0,
                 do_kl_normalize: bool = False,   # legacy compat — ignored; use kl_mode
                 kl_mode: str | None = None,
                 kl_auto_scale: bool = False,
                 pos_enc_freqs: Optional[List] = None,
                 pos_enc_mode: str = "dec",
                 flow_type: Optional[str] = None,
                 flow_num_steps: int = 0,
                 flow_hidden_dim: Optional[int] = None,
                 divergence_kind: str = "kl",
                 ):
        super().__init__()
        self.free_bits = free_bits
        self.kl_auto_scale = kl_auto_scale

        # kl_mode controls how KL is aggregated over the (B, T, Z) tensor:
        #   "norm"    — mean over B, T, Z  → O(1); beta is dim- and length-independent
        #   "norm_T"  — sum over Z, mean over B, T  → O(Z); legacy do_kl_normalize=False
        #   "no_norm" — sum over Z and T, mean over B only  → O(Z*T); true no-normalisation
        #
        if divergence_kind not in ("kl", "mmd"):
            raise ValueError(f"divergence_kind must be 'kl' or 'mmd'. Got '{divergence_kind}'.")
        self.divergence_kind = divergence_kind

        if kl_mode is not None:
            if kl_mode not in ("norm", "norm_T", "no_norm"):
                raise ValueError(f"kl_mode must be 'norm', 'norm_T', or 'no_norm'. Got '{kl_mode}'.")
            self.kl_mode = kl_mode
        else:
            self.kl_mode = "norm_T"

        self.arch = arch_str
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        # Cyclic-feature handling — set by the environment after construction:
        #   n_cyclic_features > 0  : how many trailing input cols are cyclic (sin/cos) time features
        #   cyclic_recon = False   : exclude them from the reconstruction loss (default)
        #   cyclic_recon = True    : include them in the reconstruction loss (produces a roll in
        #                            the latent PCA plot because reconstructing sin/cos of time
        #                            creates gradient pressure to encode time in the latent)
        self.n_cyclic_features: int = 0
        self.cyclic_recon: bool = False
        self.pooling = pooling
        self.prior_variance = prior_variance
        self._handle_logvar = (self._clamp_logvar if clamp_logvar else self._no_clamp_logvar)
        self._clamp_lo, self._clamp_hi = (clamp_bounds[0], clamp_bounds[1] )
        self.pool_latent = pool_latent
        self.enc_out_dim = enc_out_dim
        self.pooling_attn_dim = pooling_attn_dim
        self.pooling_attn_dropout = pooling_attn_dropout

        # Positional encoding
        # pos_enc_mode: "enc"     — inject PE into encoder input (x)
        #               "dec"     — inject PE into decoder input (z_rep, legacy behaviour)
        #               "enc_dec" — inject PE into both encoder and decoder inputs
        #               "none"    — no PE
        self.pos_enc_freqs = pos_enc_freqs if pos_enc_freqs is not None else []
        self.pos_enc_mode  = pos_enc_mode if pos_enc_freqs is not None else "none"
        self.use_pos_enc   = pos_enc_freqs is not None  # kept for compat
        self._pos_cache: dict = {}  # key: (T, device.type, device.index)

        # Normalizing flow (optional posterior enrichment)
        if flow_type is not None and flow_num_steps > 0:
            if divergence_kind == "mmd":
                raise ValueError("divergence_kind='mmd' is incompatible with normalizing flows.")
            if flow_type == "temporal_iaf" and pool_latent:
                raise ValueError(
                    "flow_type='temporal_iaf' requires pool_latent=False. "
                    "The temporal LSTM needs a per-timestep latent sequence (B,T,Z); "
                    "pooled mode collapses T to 1, leaving no causal context."
                )
            from src.VAE_model.VAE.normalizing_flows import NormalizingFlowChain
            self.flow: nn.Module | None = NormalizingFlowChain(
                flow_type, latent_dim, flow_num_steps, hidden_dim=flow_hidden_dim
            )
        else:
            self.flow = None
        self._nf_cache: tuple | None = None  # (z_0, z_K, log_det) — set by forward(), consumed by vae_step()

        # Declare shared attributes (to be set in subclasses)
        self.encoder: nn.Module | None = None
        self.mu: nn.Module | None = None
        self.logvar: nn.Module | None = None
        self.decoder: nn.Module | None = None
        self.attn_pool: nn.Module | None = None
        
        # Hybrid model
        # When labels are available, add a supervised term - a simple binary classifier head on latent (BCE)
        self.bce_clf = nn.Linear(latent_dim, 1)  # logits for binary anomaly (0/1)

        self.set_recon_loss_func(loss_func)

        self.set_pooling_func(pooling)
        # attention pooling (optional)
        if self.pooling == "attention":
            self.attn_pool = AttnPool(hidden_dim=self.enc_out_dim,
                                      attn_dim=self.pooling_attn_dim if self.pooling_attn_dim is not None else self.enc_out_dim // 2, 
                                      dropout=self.pooling_attn_dropout) 
 
    # Reparameterization trick for making sampling differentiable w.r.t. mu and logvar.
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) # sigma =sqrt(sigma^2) = exp((1/2)​log(sigma^2))
        eps = torch.randn_like(std) # epsilon ~ N(0, I)
        return mu + eps * std # z ~ N(mu, diag(sigma^2))
    
    # x: (B, T, M) — always 3-D.
    # Returns: recon_loss, kl_loss, mu_x, logvar_x
    def vae_step(self, x: torch.Tensor):
        if x.dim() != 3:
            raise ValueError(
                f"vae_step expects a 3-D tensor (B, T, M). Got shape {x.shape}. "
                "Ensure data adapters and trainers always produce 3-D inputs."
            )
        x_in = x

        recon_x, mu_x, logvar_x = self(x_in)  # *_VAE forward; also sets self._nf_cache

        if self.n_cyclic_features > 0 and not self.cyclic_recon:
            n = self.n_cyclic_features
            recon_loss = self._recon_loss_fn(recon_x[:, :, :-n], x_in[:, :, :-n])
        else:
            recon_loss = self._recon_loss_fn(recon_x, x_in)

        if self.divergence_kind == "mmd":
            self._nf_cache = None
            div_loss = self._mmd_gaussian(mu_x, logvar_x)
        elif self._nf_cache is not None:
            z_0, z_K, log_det = self._nf_cache
            self._nf_cache = None
            div_loss = self._kl_flow(mu_x, logvar_x, z_0, z_K, log_det)
        else:
            div_loss = self._kl_gaussian(mu_x, logvar_x)

        return recon_loss, div_loss, mu_x, logvar_x

    def _kl_gaussian(self, mu_x: torch.Tensor, logvar_x: torch.Tensor) -> torch.Tensor:
        """Closed-form KL: KL( N(mu, sigma²) || N(0, pv·I) ).

        Aggregation controlled by self.kl_mode:
          "norm"    — mean over B, T, Z            → O(1)
          "norm_T"  — sum over Z, mean over B, T   → O(Z)
          "no_norm" — sum over Z and T, mean over B → O(Z*T)

        When kl_auto_scale=True the trainer divides elbo_beta by kl_scale_factor(T)
        so that the configured beta has a consistent per-element meaning.
        """
        pv = self.prior_variance
        kl = 0.5 * (
            (mu_x.pow(2) + logvar_x.exp()) / pv
            - 1 - logvar_x + math.log(pv)
        )  # (B, T, Z)

        # Free bits: clamp each dimension's KL from below.
        if self.free_bits > 0.0:
            kl = kl.clamp(min=self.free_bits)

        if self.kl_mode == "norm":
            return kl.mean()
        elif self.kl_mode == "norm_T":
            return kl.sum(dim=-1).mean()
        elif self.kl_mode == "no_norm":
            return kl.sum(dim=(-1, -2)).mean()
        raise ValueError(f"Unknown kl_mode '{self.kl_mode}'")

    def _kl_flow(
        self,
        mu_x: torch.Tensor,
        logvar_x: torch.Tensor,
        z_0: torch.Tensor,
        z_K: torch.Tensor,
        log_det: torch.Tensor,
    ) -> torch.Tensor:
        """MC-ELBO KL for a flow-enriched posterior.

        KL = E[ log q_0(z_0|x) - sum_k log|det J_k| - log p(z_K) ]

        z_0/z_K: (B, Z) for pool_latent=True, (B, T, Z) for pool_latent=False.
        log_det: (B,) or (B, T) — sum of log|det J| across all K flow steps.

        Note: free_bits is not applicable here (KL is a scalar MC estimate, not
        per-dimension). kl_mode is also bypassed; always mean over batch (and T).
        """
        pv = self.prior_variance
        Z  = z_0.size(-1)

        # log q_0(z_0 | x) = log N(z_0; mu, sigma^2), summed over Z
        log_q0 = -0.5 * (
            Z * math.log(2 * math.pi)
            + logvar_x.sum(-1)
            + ((z_0 - mu_x).pow(2) * (-logvar_x).exp()).sum(-1)
        )  # (B,) or (B, T)

        # log p(z_K) = log N(z_K; 0, pv·I), summed over Z
        log_p_zK = -0.5 * (
            Z * math.log(2 * math.pi * pv)
            + z_K.pow(2).sum(-1) / pv
        )  # (B,) or (B, T)

        return (log_q0 - log_det - log_p_zK).mean()

    # Maximum number of samples used for the O(n²) MMD kernel matrix.
    # Streaming batches are tiny (B=1, T≤168 → N≤168); offline batches with large
    # T are subsampled to this cap so GPU memory stays bounded.
    _MMD_MAX_SAMPLES: int = 256

    def _mmd_gaussian(self, mu_x: torch.Tensor, logvar_x: torch.Tensor) -> torch.Tensor:
        """Unbiased MMD² between q(z|x) samples and the prior N(0, pv·I).

        Uses an RBF kernel with bandwidth σ² = latent_dim × prior_variance.
        This matches the expected squared distance between two prior samples:
          E[||z1 - z2||²] = 2 × Z × pv  →  σ² = Z × pv keeps k(z,z') ≈ e^{-1} ≈ 0.37.

        Scale: O(1), bounded in [0, 2].  Use with the same elbo_beta as kl_mode='norm'.
        kl_mode and free_bits are ignored — MMD is a single unbiased scalar.

        Ref: Tolstikhin et al. 2018 (Wasserstein Auto-Encoders);
             Gretton et al. 2012 (A Kernel Two-Sample Test).
        """
        # Sample z_q ~ q(z|x) via reparameterization — gradient flows through mu/logvar
        z_q = self.reparameterize(mu_x, logvar_x)   # (B,Z) or (B,T,Z)
        if z_q.dim() == 3:
            B, T, Z = z_q.shape
            z_q = z_q.reshape(B * T, Z)
        # z_q: (N, Z)
        N, Z = z_q.shape

        # Subsample to keep the quadratic kernel matrix manageable
        if N > self._MMD_MAX_SAMPLES:
            idx = torch.randperm(N, device=z_q.device)[: self._MMD_MAX_SAMPLES]
            z_q = z_q[idx]
            N = self._MMD_MAX_SAMPLES

        pv = self.prior_variance
        z_p = torch.randn(N, Z, device=z_q.device, dtype=z_q.dtype) * math.sqrt(pv)

        sigma2 = float(Z * pv)

        def _rbf(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            diff2 = ((a.unsqueeze(1) - b.unsqueeze(0)) ** 2).sum(-1)   # (N, M)
            return torch.exp(-diff2 / (2.0 * sigma2))

        K_qq = _rbf(z_q, z_q)
        K_pp = _rbf(z_p, z_p)
        K_qp = _rbf(z_q, z_p)

        if N > 1:
            off_diag = 1.0 - torch.eye(N, device=z_q.device)
            mmd2 = (
                (K_qq * off_diag).sum() / (N * (N - 1))
                + (K_pp * off_diag).sum() / (N * (N - 1))
                - 2.0 * K_qp.mean()
            )
        else:
            mmd2 = -2.0 * K_qp.mean()

        return mmd2.clamp(min=0.0)

    def kl_scale_factor(self, T: int) -> float:
        """
        Returns the factor by which kl_loss is larger relative to a fully-normalised
        (mean over B,T,Z) baseline, given the current kl_mode and sequence length T.

        Used by the trainer when kl_auto_scale=True to normalise elbo_beta so that
        the configured value has a consistent 'per element' meaning across all modes:

            effective_beta = elbo_beta / kl_scale_factor(T)

        Mode       kl_loss scale    factor
        --------   -------------    ------------------
        norm       O(1)             1
        norm_T     O(Z)             latent_dim
        no_norm    O(Z * T)         latent_dim * T
        """
        if self.kl_mode == "norm":
            return 1.0
        elif self.kl_mode == "norm_T":
            return float(self.latent_dim)
        elif self.kl_mode == "no_norm":
            return float(self.latent_dim * T)
        raise ValueError(self.kl_mode)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode_sequence(self, decoder_input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def positional_encoding(self, T: int, device: torch.device) -> torch.Tensor:
        """Returns (1, T, 2*len(freqs)) sinusoidal PE.
        t ∈ [0, 1] normalised within window; freqs are powers-of-2 up to sqrt(seq_len).
        Cached by (T, device)."""
        key = (T, device.type, device.index)
        pe = self._pos_cache.get(key)
        if pe is not None:
            return pe
        t = torch.linspace(0, 1, T, device=device).view(1, T, 1)
        parts = []
        for k in self.pos_enc_freqs:
            parts.append(torch.sin(2 * math.pi * k * t))
            parts.append(torch.cos(2 * math.pi * k * t))
        pe = torch.cat(parts, dim=-1)   # (1, T, 2*len(freqs))
        self._pos_cache[key] = pe
        return pe

    def decode_z(self, z: torch.Tensor, T: int) -> torch.Tensor:
        """Decode from a latent tensor, handling pooled vs per-step.
        z: (B, Z) for pool_latent=True, (B, T, Z) for pool_latent=False.
        Returns x_hat: (B, T, M).
        """
        B = z.size(0)
        if self.pool_latent:
            z_rep = z.unsqueeze(1).expand(B, T, z.size(-1))
        else:
            z_rep = z
        if self.pos_enc_mode in ("dec", "enc_dec"):
            pos = self.positional_encoding(T, z_rep.device).expand(B, -1, -1)
            z_rep = torch.cat([z_rep, pos], dim=-1)
        return self.decode_sequence(z_rep)
    
    def forward(self, x: torch.Tensor):
        if x.dim() != 3:
            raise ValueError(f"Expected (B,T,M). Got {x.shape}")

        B, T, _ = x.shape
        if self.pos_enc_mode in ("enc", "enc_dec"):
            pos = self.positional_encoding(T, x.device).expand(B, -1, -1)
            x = torch.cat([x, pos], dim=-1)
        h_seq = self.encode_sequence(x)   # (B,T,H)

        if self.pool_latent:
            # --- pooled latent: one z per sequence ---
            # h_pool: (B,H)  →  mu/logvar: (B,Z)  →  z: (B,Z)
            h_pool = self.pool_h(h_seq)
            mu_x    = self.mu(h_pool)
            logvar_x = self.logvar(h_pool)
            logvar_x = self._handle_logvar(logvar_x)
            z_0 = self.reparameterize(mu_x, logvar_x)   # (B,Z)
            if self.flow is not None:
                z_K, log_det = self.flow(z_0)            # z_K: (B,Z), log_det: (B,)
                self._nf_cache = (z_0, z_K, log_det)
            else:
                z_K = z_0
                self._nf_cache = None
            # broadcast z_K across time for the decoder
            z_rep = z_K.unsqueeze(1).expand(B, T, z_K.size(-1))  # (B,T,Z)
        else:
            # --- sequence latent: one z per timestep ---
            # mu/logvar: (B,T,Z)  →  z: (B,T,Z)
            mu_x    = self.mu(h_seq)
            logvar_x = self.logvar(h_seq)
            logvar_x = self._handle_logvar(logvar_x)
            z_0 = self.reparameterize(mu_x, logvar_x)   # (B,T,Z)
            if self.flow is not None:
                z_K, log_det = self.flow(z_0)            # z_K: (B,T,Z), log_det: (B,T)
                self._nf_cache = (z_0, z_K, log_det)
            else:
                z_K = z_0
                self._nf_cache = None
            z_rep = z_K

        if self.pos_enc_mode in ("dec", "enc_dec"):
            pos = self.positional_encoding(T, z_rep.device).expand(B, -1, -1)
            z_rep = torch.cat([z_rep, pos], dim=-1)

        x_hat = self.decode_sequence(z_rep)   # (B,T,M)
        return x_hat, mu_x, logvar_x
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Return the number of parameters in the model.
        If trainable_only=True, count only those with requires_grad=True.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    # KL divergence: KL( N(mu, exp(logvar)) || N(0, pv·I) )
    #   per-dimension closed form:
    #     kl_d = 0.5 * [ (mu_d^2 + exp(logvar_d)) / pv  −  1  −  logvar_d  +  log(pv) ]
    #
    # Anomaly intuition:
    #   normal data  → mu ≈ 0, exp(logvar) ≈ pv  → kl_d ≈ 0
    #   anomalous    → mu shifts away from 0       → kl_d increases
    #   over-certain → exp(logvar) → 0             → kl_d increases
    #   uncertain    → exp(logvar) → ∞             → kl_d increases


    # Ensure all lists have the same length.
    # lists: a list of lists to compare
    # names: for logging      
    @staticmethod
    def arg_val_assert_equal_lengths(lists: List[List[Any]], names: List[str]) -> None:

        lengths = [len(lst) for lst in lists]
        if len(set(lengths)) != 1:
            raise ValueError(
                "Config length mismatch: " +
                ", ".join(f"{n}={l}" for n, l in zip(names, lengths))
            )

    # ---- value checks ----
    @staticmethod
    def arg_val_hidden_dims(hidden_dims) :
        # hidden dims: positive & within some reasonable bound (e.g., <= 10k)
        for i, h in enumerate(hidden_dims):
            if not (1 <= h <= 10_000):
                raise ValueError(f"Hidden dim at index {i} is invalid: {h} (must be 1..10000)")

    @staticmethod
    def arg_val_dropouts(dropouts) :
        # dropouts: between 0 and 0.8
        for i, d in enumerate(dropouts):
            if not (0.0 <= d < 0.8):
                raise ValueError(f"Dropout at index {i} is invalid: {d} (must be in [0,0.8))")

    @staticmethod
    def arg_val_activation_funcs(activation_funcs) :
        # activations: must be in allowed dict
        for i, act in enumerate(activation_funcs):
            if act not in Activations:
                raise ValueError(
                    f"Activation at index {i} is invalid: '{act}' "
                    f"(must be one of {list(Activations.keys())})"
                )

    def _no_clamp_logvar(self, logvar):
        return logvar

    def _clamp_logvar(self, logvar):
        return logvar.clamp(self._clamp_lo, self._clamp_hi) # optional: avoid huge exp()

    # If attention is enabled, use it; otherwise use configured pooling.
    # h: (B,T,H) 
    # returns pooled: (B,H)
    def pool_h(self, h: torch.Tensor):

        if self.attn_pool is not None:
            pooled, _ = self.attn_pool(h)
            return pooled
        return self._pooling(h)
    
    def set_pooling_func(self, pooling):
        if pooling == "mean":
            self._pooling = self._pooling_mean
        elif pooling == "winsorize_mean":
            self._pooling = self._pooling_winsorize_mean
        elif pooling == "median":
            self._pooling = self._pooling_median
        elif pooling == "last":
            self._pooling = self._pooling_last
        elif pooling == "max":
            self._pooling = self._pooling_max
        elif pooling == "lme":
            self._pooling = self._pooling_lme
        elif pooling == "max_abs":
            self._pooling = self._pooling_max_abs
        else:
            if pooling != "attention":
                # "pooling='attention' requires self.attn_pool to be set in the subclass"
                raise ValueError(f"Unknown pooling: {pooling}")
        
    def _pooling_mean(self, h) :  # h: (B,T,H)
        return h.mean(dim=1)
    
    def _pooling_winsorize_mean(self, h, dim=1, p=0.1) :  # h: (B,T,H)
        q_low  = h.quantile(p, dim=dim, keepdim=True)
        q_high = h.quantile(1-p, dim=dim, keepdim=True)
        h_clamped = torch.clamp(h, q_low, q_high)
        return h_clamped.mean(dim=1)

    def _pooling_median(self, h) :  # h: (1,T,H)
        return h.median(dim=1).values    # (1,H)
    
    def _pooling_last(self, h) :
        return h[:, -1, :]
    
    def _pooling_max(self, h) :
        return h.max(dim=1).values
    
    def _pooling_lme(self, h) :
        # log(1/n sum(e^h for every h_n in h))
        return torch.logsumexp(h, dim=1) - np.log(h.size(1))

    def _pooling_max_abs(self, h) :
        # For each hidden dim, take the timestep with the largest absolute value,
        # preserving its sign.  Symmetric counterpart to max — captures both positive
        # and negative extremes (e.g. a cold-temperature anomaly encoding as a strongly
        # negative latent dim is not ignored).
        # h: (B, T, H) → (B, H)
        idx = h.abs().argmax(dim=1, keepdim=True)   # (B, 1, H)
        return h.gather(1, idx).squeeze(1)           # (B, H)
    
    def _pooling_attn(self, h) :
            # w = self.attn(h).squeeze(-1)          # (1,T)
            # a = torch.softmax(w, dim=1).unsqueeze(-1)  # (1,T,1)
            # return (a * h).sum(dim=1)             # (1,H)
        raise ValueError("attn pooling is implemented using pool_h only")

    def set_recon_loss_func(self, loss_func: LossFunc):
        if loss_func.recon_loss_kind == "mse":
            self._recon_loss_fn = self.vae_mse_loss
        elif loss_func.recon_loss_kind == "mae":
            self._recon_loss_fn = self.vae_mae_loss
        elif loss_func.recon_loss_kind == "huber":
            self._recon_loss_fn = self.vae_huber_loss
        elif loss_func.recon_loss_kind == "charbonnier":
            self._recon_loss_fn = self.vae_charbonnier_loss
        elif loss_func.recon_loss_kind == "bce":
            # Assumption: x_in values must be in [0, 1].
            # Valid for raw (unscaled) spectral features. NOT valid after z-scoring.
            # A warning is issued at first call if this is violated.
            self._bce_warned = False
            self._recon_loss_fn = self.vae_bce_loss
        elif loss_func.recon_loss_kind == "spectral_kl":
            # Assumption: features form a meaningful spectral distribution (all positive
            # values representing energy). Softmax is applied along the feature dim to
            # convert both input and reconstruction into probability vectors before KL.
            # Works on z-scored data since softmax handles negative values, but the
            # resulting distribution may be less interpretable.
            self._recon_loss_fn = self.vae_spectral_kl_loss
        elif loss_func.recon_loss_kind == "spectral_js":
            # Jensen-Shannon divergence: symmetric, bounded in [0, log(2)] ≈ [0, 0.693].
            # JS(P||Q) = 0.5*KL(P||M) + 0.5*KL(Q||M), M = 0.5*(P+Q).
            # Better than spectral_kl when the reconstruction is far from the input
            # (KL can blow up; JS stays bounded). Same softmax-normalisation assumption.
            self._recon_loss_fn = self.vae_spectral_js_loss
        elif loss_func.recon_loss_kind == "cosine":
            # No range assumptions. Measures the angle between input and reconstruction
            # vectors; blind to magnitude, sensitive to spectral shape/direction.
            self._recon_loss_fn = self.vae_cosine_loss
        elif loss_func.recon_loss_kind == "wmse":
            # Weighted MSE: per-feature weights scale the squared error.
            # Default: uniform weights (equivalent to mse).
            # Call set_feature_weights(weights) to set data-driven weights after init.
            self._feature_weights = (
                loss_func.feature_weights.float()
                if loss_func.feature_weights is not None
                else None  # deferred to first call / set_feature_weights
            )
            self._wmse_weights_ready = loss_func.feature_weights is not None
            self._recon_loss_fn = self.vae_wmse_loss
        else:
            raise ValueError(loss_func.recon_loss_kind)

    # ------------------------------------------------------------------
    # Feature-weight management (for wmse)
    # ------------------------------------------------------------------

    def set_feature_weights(self, weights: torch.Tensor) -> None:
        """Set per-feature weights for the 'wmse' loss.

        Args:
            weights: 1-D tensor of shape (M,) with non-negative values.
                     Typically set to 1/var(feature) over the warmup set so that
                     low-variance features (consistent in normal data) are penalised
                     more when they deviate.
        """
        if weights.ndim != 1 or weights.shape[0] != self.input_dim:
            raise ValueError(
                f"feature_weights must be 1-D with length input_dim={self.input_dim}, "
                f"got shape {tuple(weights.shape)}"
            )
        if (weights < 0).any():
            raise ValueError("feature_weights must all be non-negative")
        self._feature_weights = weights.float()
        self._wmse_weights_ready = True

    # ------------------------------------------------------------------
    # point-wise error losses
    # ------------------------------------------------------------------

    def vae_mse_loss(self, recon, x_in):
        return F.mse_loss(recon, x_in, reduction="mean")

    def vae_mae_loss(self, recon, x_in):
        return F.l1_loss(recon, x_in, reduction="mean")

    def vae_huber_loss(self, recon, x_in):
        return F.smooth_l1_loss(recon, x_in, reduction="mean")

    def vae_charbonnier_loss(self, recon, x_in):
        eps = 1e-6
        return torch.sqrt(((recon - x_in)**2 + eps**2)).mean()

    # ------------------------------------------------------------------
    # Spectral Losses
    # ------------------------------------------------------------------
  
    def vae_bce_loss(self, recon, x_in):
        # Binary cross-entropy with logits
        # --------------------------------  
        # Treats recon as logits (raw decoder output, no sigmoid needed).
        # Assumption: x_in must be in [0, 1]. Valid for raw spectral energy
        # features that are naturally bounded. Violated by z-scored inputs —
        # a one-time warning is issued in that case.
        out_of_range_frac = ((x_in < 0) | (x_in > 1)).float().mean().item()
        if not self._bce_warned and out_of_range_frac > 0.01:
            import warnings
            warnings.warn(
                f"[bce recon loss] {out_of_range_frac*100:.1f}% of x_in values are outside [0, 1]. "
                "BCE requires targets in [0, 1]. Your features may be z-scored — consider "
                "switching to 'spectral_kl' or 'cosine', or using raw (unscaled) features.",
                stacklevel=2,
            )
            self._bce_warned = True
        return F.binary_cross_entropy_with_logits(recon, x_in.clamp(0.0, 1.0), reduction="mean")

    def vae_spectral_kl_loss(self, recon, x_in):
        # KL divergence treating the feature vector as a probability distribution.

        # Applies softmax along the feature dim (M) to convert both x_in and recon
        # into probability vectors, then computes KL(p_input || p_recon).

        # This is sensitive to the *shape* of the spectrum (relative energy distribution
        # across bins) rather than absolute magnitudes — useful for detecting species
        # changes in wing-beat frequency data where energy shifts between bins.
        
        # (B, T, M) → normalise over M to get prob vectors
        p = F.softmax(x_in,  dim=-1)   # target distribution
        q = F.log_softmax(recon, dim=-1)   # log of predicted distribution
        # F.kl_div expects (input=log_q, target=p); 'batchmean' divides by B*T
        return F.kl_div(q, p, reduction="batchmean")

    def vae_spectral_js_loss(self, recon, x_in):
        # Jensen-Shannon divergence between softmax-normalised input and reconstruction.

        # JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M),  M = 0.5 * (P + Q)

        # Advantages over spectral_kl:
        # - Symmetric: JS(P||Q) == JS(Q||P)
        # - Bounded:   always in [0, log(2)] ≈ [0, 0.693], never diverges
        # - Smoother gradient when P and Q are very different (where KL → ∞, JS stays finite)
        p = F.softmax(x_in, dim=-1)   # (B, T, M)
        q = F.softmax(recon, dim=-1)
        m = 0.5 * (p + q)
        log_m = m.log()
        # KL(p||m) + KL(q||m) each computed via F.kl_div(log_m, x, reduction="batchmean")
        kl_pm = F.kl_div(log_m, p, reduction="batchmean")
        kl_qm = F.kl_div(log_m, q, reduction="batchmean")
        return 0.5 * (kl_pm + kl_qm)

    def vae_cosine_loss(self, recon, x_in):
        # cosine_similarity, averaged over all (B, T) positions.

        # Measures the angle between the input and reconstructed spectral vectors.
        # Insensitive to magnitude; catches shape/direction shifts in the spectrum.
        # A different insect species with energy in different frequency bins will
        # produce a large angle even if overall power is similar.
        B, T, M = x_in.shape
        x_flat = x_in.reshape(B * T, M)
        r_flat = recon.reshape(B * T, M)
        cos_sim = F.cosine_similarity(x_flat, r_flat, dim=-1)  # (B*T,)
        return (1.0 - cos_sim).mean()

    def vae_wmse_loss(self, recon, x_in):
        # Weighted MSE: per-feature weights scale the squared error.

        # Low-variance features (those that are nearly constant during normal operation)
        # get higher weight, making small deviations in those features more costly.
        # Call set_feature_weights() after init to provide data-driven weights.
        # Falls back to uniform weights (= standard MSE) if not yet set.
        if not self._wmse_weights_ready:
            # Fallback to unweighted until set_feature_weights is called
            return F.mse_loss(recon, x_in, reduction="mean")
        w = self._feature_weights.to(x_in.device)  # (M,)
        sq = (recon - x_in) ** 2                   # (B, T, M)
        return (sq * w).mean()
    
    def apply_weight_init(self, scheme: str = "xavier_uniform") -> None:
        """Apply weight initialisation to all nn.Linear layers in the model.

        LSTM internal weight matrices and biases are intentionally skipped so that
        the forget-gate bias initialisation set in LSTM_VAE.__init__ is preserved.

        Supported schemes:
          "xavier_uniform"  - nn.init.xavier_uniform_  (good for tanh / sigmoid)
          "xavier_normal"   - nn.init.xavier_normal_   (good for tanh / sigmoid)
          "kaiming_uniform" - nn.init.kaiming_uniform_ (good for relu / silu)
          "kaiming_normal"  - nn.init.kaiming_normal_  (good for relu / silu)
          "orthogonal"      - nn.init.orthogonal_
          "none"            - leave PyTorch defaults unchanged
        """
        if scheme == "none":
            return
        for module in self.modules():
            if not isinstance(module, nn.Linear):
                continue
            # always apply near-zero to logvar, regardless of scheme
            if module is self.logvar:
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)
                continue
            if scheme == "xavier_uniform":
                nn.init.xavier_uniform_(module.weight)
            elif scheme == "xavier_normal":
                nn.init.xavier_normal_(module.weight)
            elif scheme == "kaiming_uniform":
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            elif scheme == "kaiming_normal":
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif scheme == "orthogonal":
                nn.init.orthogonal_(module.weight)
            else:
                raise ValueError(
                    f"Unknown weight_init scheme: '{scheme}'. "
                    "Choose from: xavier_uniform, xavier_normal, "
                    "kaiming_uniform, kaiming_normal, orthogonal, none."
                )
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def latent_for_classification(self, h):
        # Mean over time dimension
        # h: (B, T, D) -> (B, D)
        if h.dim() == 3:
            return h.mean(dim=1)
        return h  # if already (B, D)
    
