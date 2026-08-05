import math
from typing import List, Optional

import torch
import torch.nn as nn

from src.VAE_model.VAE.transformer_utils import make_transformer_stack, sinusoidal_pe_additive_cached
from src.VAE_model.VAE.base_VAE import LossFunc
from src.VAE_model.VAE.lstm_VAE import LSTM_VAE

# LSTM_VAE + TransformerEncoderLayer stack applied to encoder outputs h_seq (B,T,H_enc).
# Uses VAE.forward() from base; only overrides encode_sequence().
class LSTM_ATTN_VAE(LSTM_VAE):

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        loss_func: LossFunc,
        pooling: str,
        prior_variance: float,

        # Inherit all LSTM_VAE knobs:
        enc_hidden_dim: int,
        enc_num_layers: int = 1,
        enc_dropout: float = 0.0,

        dec_hidden_dim: int | None = None,
        dec_num_layers: int = 1,
        dec_dropout: float = 0.0,

        enc_bidirectional: bool = False,
        use_layer_norm: bool = False,

        clamp_logvar: bool = False,
        clamp_bounds: float = 8.0,

        # --- attention pooling (optional, used when pool_h sees self.attn_pool) ---
        pooling_attn_dim: int | None = None,
        pooling_attn_dropout: float = 0.0,

        # --- transformer stack config ---
        num_attn_layers: int = 1,
        num_attn_heads: Optional[List[int]] = None,
        attn_ff_dims: Optional[List[int]] = None,
        attn_dropouts: Optional[List[float]] = None,
        attn_activations: Optional[List[str]] = None,
        attn_norm_first: Optional[List[bool]] = None,

        use_transformer_pos_enc: bool = False, # LSTM already injects order. For fair comparisons, may need both, but it’s not strictly necessary for the LSTM variant.
        pool_latent: bool = False,
        arch_str: str = "LSTM_ATTN_VAE",
        free_bits: float = 0.0,
        # optional normalizing flow
        flow_type: Optional[str] = None,
        flow_num_steps: int = 0,
    ):
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            loss_func=loss_func,
            pooling=pooling,
            prior_variance=prior_variance,
            enc_hidden_dim=enc_hidden_dim,
            enc_num_layers=enc_num_layers,
            enc_dropout=enc_dropout,
            dec_hidden_dim=dec_hidden_dim,
            dec_num_layers=dec_num_layers,
            dec_dropout=dec_dropout,
            enc_bidirectional=enc_bidirectional,
            use_layer_norm=use_layer_norm,
            clamp_logvar=clamp_logvar,
            clamp_bounds=clamp_bounds,
            pooling_attn_dim=pooling_attn_dim,
            pooling_attn_dropout=pooling_attn_dropout,
            pool_latent=pool_latent,
            arch_str=arch_str,
            free_bits=free_bits,
            flow_type=flow_type,
            flow_num_steps=flow_num_steps,
        )

        
        self.use_transformer_pos_enc = use_transformer_pos_enc
        self._tf_pe_cache = {}  # key: (T, device.type, device.index)

        self.enc_attn_layers = make_transformer_stack(
                                d_model=self.enc_hidden_dim,
                                num_attn_layers=num_attn_layers,
                                num_attn_heads=num_attn_heads,
                                attn_ff_dims=attn_ff_dims,
                                attn_dropouts=attn_dropouts,
                                attn_activations=attn_activations,
                                attn_norm_first=attn_norm_first)

    # ---- Hook override used by base VAE.forward() ----
    #      x: (B,T,input_dim)
    #      returns: (B,T,enc_out_dim) after transformer mixing
    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h_seq, _ = self.encoder(x) # h_seq: (B,T,enc_out_dim) from LSTM encoder

        if self.enc_attn_layers is None:
            return h_seq
        
        # Transformer positional encoding is additive
        if self.use_transformer_pos_enc:
            pe = sinusoidal_pe_additive_cached(cache   = self._tf_pe_cache,
                                               T       = h_seq.size(1),
                                               d_model = h_seq.size(-1),
                                               device  = h_seq.device)
            h_seq = h_seq + pe
        
        for layer in self.enc_attn_layers:
            h_seq = layer(h_seq)

        return h_seq
