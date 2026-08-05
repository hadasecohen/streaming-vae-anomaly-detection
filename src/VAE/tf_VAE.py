"""
TF_VAE: Pure Transformer Variational Autoencoder
=================================================
A clean transformer encoder/decoder VAE for ablation comparison against
MLP_VAE (no temporal structure) and LSTM_VAE (sequential recurrence).

Architecture
------------
Encoder:
  x (B, T, M)
  → input_proj: Linear(M→d_model) + LayerNorm
  → additive sinusoidal PE
  → num_enc_layers × TransformerEncoderLayer(d_model, nhead, ff_dim)
  → LayerNorm
  → pool → mu(B,Z), logvar(B,Z)  [pool_latent=True]

Decoder:
  z (B,Z) → broadcast → z_rep (B,T,Z)
  → latent_proj: Linear(Z→d_model) + LayerNorm
  → additive sinusoidal PE
  → num_dec_layers × TransformerEncoderLayer(d_model, nhead, ff_dim)
  → LayerNorm
  → output_proj: Linear(d_model→M)
  → x_hat (B, T, M)

Key design choices
------------------
- Internal additive sinusoidal PE (Vaswani 2017), applied in both encoder and decoder.
  No external pos_enc configuration needed.
- pool_latent=True by default (one z per sequence), consistent with
  LSTM_VAE comparison configs.
"""

import torch
import torch.nn as nn
from typing import Optional

from src.VAE.base_VAE import VAE, LossFunc
from src.VAE.transformer_utils import sinusoidal_pe_additive_cached


class TF_VAE(VAE):
    """
    Pure Transformer VAE.

    Parameters (new vs base VAE)
    ----------------------------
    d_model        : Transformer width throughout
    num_enc_layers : Encoder TransformerEncoderLayer blocks
    num_dec_layers : Decoder TransformerEncoderLayer blocks
    nhead          : Attention heads (must divide d_model)
    ff_dim         : FFN hidden size (default 4*d_model)
    dropout        : Dropout in all TransformerEncoderLayers
    activation     : "relu" | "gelu"
    norm_first     : Pre-LayerNorm (True = recommended Pre-LN)
    """

    def __init__(
        self,
        # ---- standard VAE args ----
        input_dim: int,
        latent_dim: int,
        loss_func: LossFunc,
        pooling: str,
        prior_variance: float,
        clamp_logvar: bool,
        clamp_bounds: list,
        pooling_attn_dim: Optional[int] = None,
        pooling_attn_dropout: float = 0.0,
        pool_latent: bool = True,
        arch_str: str = "TF_VAE",
        free_bits: float = 0.0,
        do_kl_normalize: bool = False,
        kl_mode: str | None = None,
        kl_auto_scale: bool = False,
        # ---- TF_VAE specific ----
        d_model: int = 64,
        num_enc_layers: int = 3,
        num_dec_layers: int = 3,
        nhead: int = 4,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
        # ---- optional normalizing flow ----
        flow_type: Optional[str] = None,
        flow_num_steps: int = 0,
        flow_hidden_dim: Optional[int] = None,
        divergence_kind: str = "kl",
    ):
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        if d_model // nhead < 16:
            raise ValueError(
                f"head_dim = d_model/nhead = {d_model // nhead} < 16 (too small)"
            )
        if activation not in ("relu", "gelu"):
            raise ValueError(f"activation must be 'relu' or 'gelu', got {activation!r}")
        if not (0.0 <= dropout < 0.8):
            raise ValueError(f"dropout must be in [0, 0.8), got {dropout}")

        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            loss_func=loss_func,
            pooling=pooling,
            prior_variance=prior_variance,
            clamp_logvar=clamp_logvar,
            clamp_bounds=clamp_bounds,
            pooling_attn_dim=pooling_attn_dim,
            pooling_attn_dropout=pooling_attn_dropout,
            enc_out_dim=d_model,
            pool_latent=pool_latent,
            arch_str=arch_str,
            free_bits=free_bits,
            do_kl_normalize=do_kl_normalize,
            kl_mode=kl_mode,
            kl_auto_scale=kl_auto_scale,
            flow_type=flow_type,
            flow_num_steps=flow_num_steps,
            flow_hidden_dim=flow_hidden_dim,
            divergence_kind=divergence_kind,
        )

        self.d_model = d_model
        _ff = ff_dim if ff_dim is not None else 4 * d_model
        self._pe_cache: dict = {}

        # ----------------------------------------------------------------
        # Input projection: Linear(M→d_model)
        # ----------------------------------------------------------------
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # ----------------------------------------------------------------
        # Encoder: k × TransformerEncoderLayer
        # ----------------------------------------------------------------
        self.enc_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=_ff,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                norm_first=norm_first,
            )
            for _ in range(num_enc_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)

        # ----------------------------------------------------------------
        # Latent projections
        # ----------------------------------------------------------------
        self.mu     = nn.Linear(d_model, latent_dim)
        self.logvar = nn.Linear(d_model, latent_dim)

        # ----------------------------------------------------------------
        # Decoder: latent_proj + k × TransformerEncoderLayer + output_proj
        # ----------------------------------------------------------------
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.dec_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=_ff,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                norm_first=norm_first,
            )
            for _ in range(num_dec_layers)
        ])
        self.dec_norm    = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, input_dim)

    # ------------------------------------------------------------------
    # encode_sequence / decode_sequence  (called by base_VAE.forward)
    # ------------------------------------------------------------------

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, M) → (B, T, d_model)"""
        h = self.input_proj(x)                                    # (B, T, d_model)
        pe = sinusoidal_pe_additive_cached(
            self._pe_cache, h.size(1), self.d_model, h.device
        )
        h = h + pe
        for layer in self.enc_layers:
            h = layer(h)
        return self.enc_norm(h)

    def decode_sequence(self, decoder_input: torch.Tensor) -> torch.Tensor:
        """(B, T, latent_dim) → (B, T, M)"""
        h = self.latent_proj(decoder_input)                       # (B, T, d_model)
        pe = sinusoidal_pe_additive_cached(
            self._pe_cache, h.size(1), self.d_model, h.device
        )
        h = h + pe
        for layer in self.dec_layers:
            h = layer(h)
        h = self.dec_norm(h)
        return self.output_proj(h)                                # (B, T, M)
