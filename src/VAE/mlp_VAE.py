import torch
import torch.nn as nn
from typing import List, Optional

from src.VAE.base_VAE import VAE, LossFunc
from src.utils import Activations


class _BN1d(nn.Module):
    """BatchNorm1d that accepts (B, T, H) by merging the batch and time dims."""
    def __init__(self, num_features: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        return self.bn(x.reshape(B * T, H)).reshape(B, T, H)


# -------- Subclass: MLP --------
class MLP_VAE(VAE):
    def __init__(self,
                 input_dim : int,
                 latent_dim : int,
                 loss_func: LossFunc,
                 pooling : str,
                 prior_variance : float,
                 # Encoder config
                 enc_dec_hidden_dims: List[int],
                 enc_dec_activation_funcs: List[str],
                 enc_dec_dropouts: List[float],
                 # Decoder config (if None, mirrors encoder in reverse)
                 dec_hidden_dims: Optional[List[int]] = None,
                 dec_activation_funcs: Optional[List[str]] = None,
                 dec_dropouts: Optional[List[float]] = None,
                 # Normalization: None | "layer" | "batch"
                 norm: Optional[str] = None,
                 clamp_logvar : bool = False,
                 clamp_bounds: float = 8.0,
                 pos_enc_freqs: Optional[List[float]] = None,
                 pos_enc_mode: str = "dec",
                 pooling_attn_dim: int | None = None,
                 pooling_attn_dropout: float = 0.0,
                 pool_latent: bool = False,
                 arch_str: str = "MLP_VAE",
                 free_bits: float = 0.0,
                 do_kl_normalize: bool = False,
                 kl_mode: str | None = None,
                 kl_auto_scale: bool = False,
                 # optional normalizing flow
                 flow_type: Optional[str] = None,
                 flow_num_steps: int = 0,
                 flow_hidden_dim: Optional[int] = None,
                 divergence_kind: str = "kl",
                 ):

        super().__init__(input_dim=input_dim,
                         latent_dim=latent_dim,
                         loss_func=loss_func,
                         pooling=pooling,
                         prior_variance=prior_variance,
                         clamp_logvar=clamp_logvar,
                         clamp_bounds=clamp_bounds,
                         pooling_attn_dim=pooling_attn_dim,
                         pooling_attn_dropout=pooling_attn_dropout,
                         enc_out_dim=enc_dec_hidden_dims[-1],
                         pool_latent=pool_latent,
                         arch_str=arch_str,
                         free_bits=free_bits,
                         do_kl_normalize=do_kl_normalize,
                         kl_mode=kl_mode,
                         kl_auto_scale=kl_auto_scale,
                         pos_enc_freqs=pos_enc_freqs,
                         pos_enc_mode=pos_enc_mode,
                         flow_type=flow_type,
                         flow_num_steps=flow_num_steps,
                         flow_hidden_dim=flow_hidden_dim,
                         divergence_kind=divergence_kind)

        enc_input = input_dim
        if pos_enc_freqs is not None and self.pos_enc_mode in ("enc", "enc_dec"):
            enc_input += 2 * len(pos_enc_freqs)

        # ---- Encoder arg checks ---- #
        VAE.arg_val_assert_equal_lengths(
            [ enc_dec_hidden_dims ,  enc_dec_activation_funcs ,  enc_dec_dropouts],
            ["enc_dec_hidden_dims", "enc_dec_activation_funcs", "enc_dec_dropouts"])
        VAE.arg_val_hidden_dims(enc_dec_hidden_dims)
        VAE.arg_val_dropouts(enc_dec_dropouts)
        VAE.arg_val_activation_funcs(enc_dec_activation_funcs)

        # ---- Resolve decoder config (fall back to reversed encoder if not set) ---- #
        _dec_hidden_dims      = dec_hidden_dims      if dec_hidden_dims      is not None else enc_dec_hidden_dims[::-1]
        _dec_activation_funcs = dec_activation_funcs if dec_activation_funcs is not None else enc_dec_activation_funcs[::-1]
        _dec_dropouts         = dec_dropouts         if dec_dropouts         is not None else enc_dec_dropouts[::-1]

        # ---- Decoder arg checks ---- #
        VAE.arg_val_assert_equal_lengths(
            [ _dec_hidden_dims ,  _dec_activation_funcs ,  _dec_dropouts],
            ["dec_hidden_dims", "dec_activation_funcs", "dec_dropouts"])
        VAE.arg_val_hidden_dims(_dec_hidden_dims)
        VAE.arg_val_dropouts(_dec_dropouts)
        VAE.arg_val_activation_funcs(_dec_activation_funcs)

        # ---- Encoder ---- #
        encoder_layers = self.create_mlp_layers(
            enc_input,
            enc_dec_hidden_dims,
            enc_dec_activation_funcs,
            enc_dec_dropouts,  # last dropout must be 0.0 (right before latent)
            norm=norm,
        )
        self.encoder = nn.Sequential(*encoder_layers)

        # ---- Latent ---- #
        self.mu = nn.Linear(self.enc_out_dim, latent_dim)
        self.logvar = nn.Linear(self.enc_out_dim, latent_dim)

        # decoder_input = z_rep (B,T,Z) OR concat with pos features (B,T,Z+pos_dim)
        if pos_enc_freqs is not None and self.pos_enc_mode in ("dec", "enc_dec"):
            decoder_in = latent_dim + 2 * len(pos_enc_freqs)
        else:
            decoder_in = latent_dim

        # ---- Decoder ---- #
        decoder_layers = self.create_mlp_layers(
            decoder_in,
            _dec_hidden_dims,
            _dec_activation_funcs,
            _dec_dropouts,  # first dropout must be 0.0 (right after latent+pos)
            norm=norm,
        )
        decoder_layers.append(nn.Linear(_dec_hidden_dims[-1], input_dim))  # final projection
        self.decoder = nn.Sequential(*decoder_layers)

    @staticmethod
    def create_mlp_layers(input_dim, enc_dec_hidden_dims, enc_dec_activation_funcs,
                          enc_dec_dropouts, norm: Optional[str] = None):
        if norm not in (None, "layer", "batch"):
            raise ValueError(f"norm must be None, 'layer', or 'batch'; got {norm!r}")
        layers = []
        prev = input_dim
        for h, act, drop in zip(enc_dec_hidden_dims,
                                enc_dec_activation_funcs,
                                enc_dec_dropouts):
            layers.append(nn.Linear(prev, h))
            if norm == "layer":
                layers.append(nn.LayerNorm(h))
            elif norm == "batch":
                layers.append(_BN1d(h))
            layers.append(Activations[act]())
            if drop > 0:
                layers.append(nn.Dropout(drop))
            prev = h
        return layers


    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,input_dim)  →  (B,T,enc_out_dim)
        return self.encoder(x)

    def decode_sequence(self, z_rep: torch.Tensor) -> torch.Tensor:
        # z_rep: (B, T, Z) or (B, T, Z+pos_dim) when PE is active (applied by base_VAE.forward)
        return self.decoder(z_rep)   # (B, T, input_dim)
