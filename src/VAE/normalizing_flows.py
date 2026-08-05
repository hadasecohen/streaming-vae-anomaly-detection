"""
Normalizing flows for VAE posterior enrichment.

Usage in base_VAE: after reparameterize(), optionally transform z_0 → z_K
and accumulate log|det J| for the MC-ELBO KL.

Supported flow types:
    "planar"         — Rezende & Mohamed (2015), O(Z) cost per step.
    "radial"         — Rezende & Mohamed (2015), radial contractions/expansions
                       around a learned reference point; O(Z) cost per step.
    "affine_coupling"— RealNVP (Dinh et al. 2017), splits z and applies a
                       learned scale+shift; exact log-det, O(Z²) cost per step.
                       NormalizingFlowChain automatically alternates which half
                       is transformed so all dimensions interact across steps.
    "temporal_iaf"   — Temporal Inverse Autoregressive Flow. A causal LSTM
                       over the latent sequence generates per-timestep affine
                       parameters (log_s_t, m_t). Requires pool_latent=False
                       and window streaming (T > 1). Unlike pointwise flows,
                       this flow is context-aware in time: the transformation
                       at step t conditions on the latent history z_{<t},
                       making it sensitive to temporal distribution shifts.
                       log|det J| = Σ_t Σ_z log|s_t_z| (exact, triangular J).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlanarFlow(nn.Module):
    """Single planar normalizing flow: f(z) = z + u_hat * tanh(w^T z + b).

    Enforces invertibility via the u_hat reparameterisation (u^T w >= -1).
    Input z must be 2-D: (N, Z).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim) * 0.01)
        self.u = nn.Parameter(torch.randn(dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))

    def _u_hat(self) -> torch.Tensor:
        wu = self.w @ self.u
        return self.u + (-1 + F.softplus(wu) - wu) * self.w / (self.w @ self.w + 1e-8)

    def forward(self, z: torch.Tensor):
        """z: (N, Z)  →  (z_new, log_det)  where log_det: (N,)"""
        u_hat = self._u_hat()                              # (Z,)
        lin   = (z @ self.w + self.b).unsqueeze(-1)        # (N, 1)
        h     = torch.tanh(lin)                            # (N, 1)
        z_new = z + u_hat * h                              # (N, Z)  broadcasts (Z,)*(N,1)
        psi   = (1.0 - h.pow(2)) * self.w                 # (N, Z)  — tanh'(lin) * w
        log_det = torch.log((1.0 + psi @ u_hat).abs() + 1e-8)  # (N,)
        return z_new, log_det


class RadialFlow(nn.Module):
    """Single radial normalizing flow: f(z) = z + β̂·h(r)·(z − z₀)

    where r = ‖z − z₀‖₂, h(r) = 1/(α + r), β̂ = −α + softplus(β_raw).
    Invertibility is guaranteed by β̂ > −α (Rezende & Mohamed 2015, Appx. A).

    log|det J| = (Z−1)·log|1 + β̂h| + log|1 + β̂α/(α + r)²|

    Input z must be 2-D: (N, Z).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.z0       = nn.Parameter(torch.randn(dim) * 0.01)
        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.beta_raw  = nn.Parameter(torch.zeros(1))

    def forward(self, z: torch.Tensor):
        """z: (N, Z)  →  (z_new, log_det)  where log_det: (N,)"""
        alpha    = F.softplus(self.log_alpha) + 1e-8          # scalar > 0
        beta_hat = -alpha + F.softplus(self.beta_raw)         # β̂ > −α

        diff  = z - self.z0                                    # (N, Z)
        r     = diff.norm(dim=-1, keepdim=True)                # (N, 1)
        h     = 1.0 / (alpha + r)                             # (N, 1)

        z_new = z + beta_hat * h * diff                       # (N, Z)

        bh    = beta_hat * h                                   # (N, 1)
        # Using the simplified form: 1 + β̂h + β̂h'r = 1 + β̂α/(α+r)²
        term1 = (z.shape[-1] - 1) * torch.log((1.0 + bh).abs() + 1e-8)
        term2 = torch.log((1.0 + beta_hat * alpha / (alpha + r).pow(2)).abs() + 1e-8)
        log_det = (term1 + term2).squeeze(-1)                  # (N,)

        return z_new, log_det


class AffineCouplingFlow(nn.Module):
    """Single affine coupling layer (RealNVP, Dinh et al. 2017).

    Splits z into (z1, z2) at the midpoint. When flip=False, z1 conditions
    a scale+shift applied to z2; when flip=True the roles are swapped.
    NormalizingFlowChain alternates flip across steps so all dimensions are
    transformed.

    log|det J| = Σ log|s_i| (exact, O(Z) summation).
    Input z must be 2-D: (N, Z).
    """

    needs_alternating = True  # signals NormalizingFlowChain to alternate flip

    def __init__(self, dim: int, flip: bool = False):
        super().__init__()
        self.flip = flip
        self.d1   = dim // 2
        self.d2   = dim - self.d1
        d_in  = self.d2 if flip else self.d1   # conditioning half
        d_out = self.d1 if flip else self.d2   # transformed half
        hidden = max(d_in * 2, 16)
        self.net_s = nn.Sequential(
            nn.Linear(d_in, hidden), nn.Tanh(), nn.Linear(hidden, d_out)
        )
        self.net_t = nn.Sequential(
            nn.Linear(d_in, hidden), nn.Tanh(), nn.Linear(hidden, d_out)
        )

    def forward(self, z: torch.Tensor):
        """z: (N, Z)  →  (z_new, log_det)  where log_det: (N,)"""
        z1, z2 = z[:, :self.d1], z[:, self.d1:]
        z_cond, z_trans = (z2, z1) if self.flip else (z1, z2)

        log_s     = self.net_s(z_cond).clamp(-5, 5)           # (N, d_out)
        t         = self.net_t(z_cond)                         # (N, d_out)
        z_trans_new = z_trans * log_s.exp() + t               # (N, d_out)

        if self.flip:
            z_new = torch.cat([z_trans_new, z_cond], dim=-1)  # [z1_new, z2]
        else:
            z_new = torch.cat([z_cond, z_trans_new], dim=-1)  # [z1, z2_new]

        log_det = log_s.sum(dim=-1)                            # (N,)
        return z_new, log_det


class TemporalIAFFlow(nn.Module):
    """Temporal Inverse Autoregressive Flow for latent sequences.

    Applies z_t = exp(log_s_t) ⊙ z_t + m_t where (log_s_t, m_t) are produced
    by a causal LSTM conditioned on the shifted latent history z_{<t}.

    Unlike planar/radial/affine flows which treat each timestep independently,
    this flow is context-aware in time: the transformation at step t sees all
    previous latents, making it sensitive to temporal trajectory anomalies.

    The Jacobian is block lower-triangular in time → exact log-det:
        log|det J| = Σ_t Σ_z log|s_t_z|  (shape (B, T))

    Requires pool_latent=False and T > 1.  For T=1 (point mode) the LSTM
    receives only zero context and produces a near-identity transform.

    Args:
        dim:        latent dimension Z
        hidden_dim: LSTM hidden size (default: max(2*Z, 32))
    """

    needs_temporal = True  # signals NormalizingFlowChain to preserve the T dim

    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        self.dim = dim
        hidden = hidden_dim if hidden_dim is not None else max(dim * 2, 32)
        self.lstm = nn.LSTM(dim, hidden, batch_first=True)
        self.to_scale_shift = nn.Linear(hidden, dim * 2)
        # Near-identity initialisation: s≈1, m≈0 at the start of training
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, z: torch.Tensor):
        """z: (B, T, Z) → (z_new (B,T,Z), log_det (B,T))"""
        if z.dim() != 3:
            raise ValueError(
                f"TemporalIAFFlow requires 3-D input (B, T, Z), got shape {z.shape}. "
                "Use pool_latent=False and window streaming."
            )
        B, T, Z = z.shape

        # Causal context: position t sees z_{1..t-1}; position 1 sees zeros.
        z_ctx = torch.cat([
            torch.zeros(B, 1, Z, device=z.device, dtype=z.dtype),
            z[:, :-1, :],
        ], dim=1)  # (B, T, Z)

        h, _ = self.lstm(z_ctx)            # (B, T, hidden)
        ss   = self.to_scale_shift(h)      # (B, T, 2Z)
        log_s = ss[..., :Z].clamp(-5, 5)   # (B, T, Z)
        m     = ss[..., Z:]                # (B, T, Z)

        z_new   = log_s.exp() * z + m
        log_det = log_s.sum(dim=-1)        # (B, T) — sum over Z per timestep
        return z_new, log_det


_FLOW_REGISTRY: dict[str, type] = {
    "planar":          PlanarFlow,
    "radial":          RadialFlow,
    "affine_coupling": AffineCouplingFlow,
    "temporal_iaf":    TemporalIAFFlow,
}


class NormalizingFlowChain(nn.Module):
    """K normalizing flow steps chained sequentially.

    Pointwise flows (planar, radial, affine_coupling):
        Accept (B, Z) or (B, T, Z); the T dimension is flattened to (B*T, Z)
        before each step and restored afterwards.

    Temporal flows (temporal_iaf):
        Require (B, T, Z) and preserve the T dimension throughout. The T dim
        must not be flattened because the LSTM conditions on the time axis.
        These flows also require pool_latent=False in the parent VAE.

    For flow types with needs_alternating=True (AffineCouplingFlow), the chain
    automatically alternates the flip parameter across steps so that every
    dimension is transformed.

    Returns:
        z_K          — transformed latent, same shape as input
        log_det_sum  — total log|det J|; shape (B,) for pooled, (B, T) per-step
    """

    def __init__(self, flow_type: str, dim: int, num_steps: int,
                 hidden_dim: int | None = None):
        super().__init__()
        cls = _FLOW_REGISTRY.get(flow_type)
        if cls is None:
            raise ValueError(
                f"Unknown flow_type '{flow_type}'. Supported: {list(_FLOW_REGISTRY)}"
            )
        self._is_temporal = getattr(cls, "needs_temporal", False)

        if getattr(cls, "needs_alternating", False):
            self.flows = nn.ModuleList(
                [cls(dim, flip=(k % 2 == 1)) for k in range(num_steps)]
            )
        elif self._is_temporal:
            self.flows = nn.ModuleList(
                [cls(dim, hidden_dim=hidden_dim) for _ in range(num_steps)]
            )
        else:
            self.flows = nn.ModuleList([cls(dim) for _ in range(num_steps)])

    def forward(self, z: torch.Tensor):
        if self._is_temporal:
            # Temporal flows require (B, T, Z) — do NOT flatten T into the batch.
            if z.dim() != 3:
                raise ValueError(
                    f"Temporal flow chain requires 3-D input (B, T, Z), got {z.shape}."
                )
            B, T, Z = z.shape
            ld_sum = torch.zeros(B, T, device=z.device, dtype=z.dtype)
            for flow in self.flows:
                z, ld = flow(z)
                ld_sum = ld_sum + ld
            return z, ld_sum

        # Pointwise flows: flatten T into the batch dimension, restore afterwards.
        is_3d = z.dim() == 3
        if is_3d:
            B, T, Z = z.shape
            z_flat = z.reshape(B * T, Z)
        else:
            z_flat = z

        ld_sum = torch.zeros(z_flat.size(0), device=z.device, dtype=z.dtype)
        for flow in self.flows:
            z_flat, ld = flow(z_flat)
            ld_sum = ld_sum + ld

        if is_3d:
            return z_flat.reshape(B, T, Z), ld_sum.reshape(B, T)
        return z_flat, ld_sum
