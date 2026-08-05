from typing import List, Optional
import torch

from src.VAE_model.VAE.transformer_utils import make_transformer_stack, sinusoidal_pe_additive_cached
from src.VAE_model.VAE.base_VAE import LossFunc
from src.VAE_model.VAE.mlp_VAE import MLP_VAE 



# Extends MLP_VAE by inserting a stack of TransformerEncoderLayer blocks
# on top of the per-timestep MLP encoder output.

# Pipeline:
#     x (B,T,M) -> MLP encoder per timestep -> h (B,T,H_enc)
#                -> [optional + additive PE (H_enc)] -> Transformer stack -> h' (B,T,H_enc)
#                -> pool_h -> (B,H_enc) -> mu/logvar -> z
#                -> decoder as in MLP_VAE

class MLP_ATTN_VAE(MLP_VAE):

    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 loss_func: LossFunc,
                 pooling : str,
                 prior_variance : float,
                
                 # encoder/decoder args
                 enc_dec_hidden_dims: List[int],
                 enc_dec_activation_funcs: List[str],
                 enc_dec_dropouts: List[float],
                 
                 clamp_logvar : bool,
                 clamp_bounds: float,
                 pos_enc_freqs=None,

                 # --- attention pooling (optional, used when pool_h sees self.attn_pool) ---
                 pooling_attn_dim: int | None = None,
                 pooling_attn_dropout: float = 0.0,

                 # --- transformer stack on encoder output (self-attention) ---
                 num_attn_layers: int = 1,  # num of TransformerEncoderLayer blocks
                 num_attn_heads: Optional[List[int]] = None,
                 attn_ff_dims: Optional[List[int]] = None,  # if empty, defaults to 4*H_enc per layer
                 attn_dropouts: Optional[List[float]] = None,
                 attn_activations: Optional[List[str]] = None,  # "relu" or "gelu"
                 attn_norm_first: Optional[List[bool]] = None,
                 
                 # Transformer positional encoding: additive
                 use_transformer_pos_enc: bool = True,
                 pool_latent: bool = False,
                 arch_str: str = "MLP_ATTN_VAE",
                 free_bits: float = 0.0,
                 do_kl_normalize: bool = False,
                 kl_mode: str | None = None,
                 kl_auto_scale: bool = False,
                #  use_attention: bool = False,
                #  use_transformer: bool = False,
                #  attn_dim: int | None = None,
                #  attn_dropout: float = 0.0,
                 # optional normalizing flow
                 flow_type: Optional[str] = None,
                 flow_num_steps: int = 0,
                 ):

        # Build the base MLP encoder/decoder first
        super().__init__(input_dim=input_dim,
                         latent_dim=latent_dim,
                         loss_func=loss_func,
                         pooling=pooling,
                         prior_variance=prior_variance,
                         enc_dec_hidden_dims=enc_dec_hidden_dims,
                         enc_dec_activation_funcs=enc_dec_activation_funcs,
                         enc_dec_dropouts=enc_dec_dropouts,
                         clamp_logvar=clamp_logvar,
                         clamp_bounds=clamp_bounds,
                         pos_enc_freqs=pos_enc_freqs,
                         pooling_attn_dim=pooling_attn_dim,
                         pooling_attn_dropout=pooling_attn_dropout,
                         pool_latent=pool_latent,
                         arch_str=arch_str,
                         free_bits=free_bits,
                         do_kl_normalize=do_kl_normalize,
                         kl_mode=kl_mode,
                         kl_auto_scale=kl_auto_scale,
                         flow_type=flow_type,
                         flow_num_steps=flow_num_steps)
                        
        if len(enc_dec_hidden_dims) == 0:
            raise ValueError("enc_dec_hidden_dims must contain at least one hidden size.")

        self.use_transformer_pos_enc = use_transformer_pos_enc
        self._tf_pe_cache = {}  # key: (T, device.type, device.index)

        # Encoder output dim after MLP encoder
        self.enc_out_dim = enc_dec_hidden_dims[-1]
        self.enc_attn_layers = make_transformer_stack(
                                d_model=self.enc_out_dim,
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
        h_seq = self.encoder(x) # h_seq: (B,T,enc_out_dim) from MLP encoder

        if self.enc_attn_layers is None:
            return h_seq
        
        if self.use_transformer_pos_enc:
            pe = sinusoidal_pe_additive_cached(cache   = self._tf_pe_cache,
                                               T       = h_seq.size(1),
                                               d_model = self.enc_out_dim,
                                               device  = h_seq.device)
            h_seq = h_seq + pe
        
        for layer in self.enc_attn_layers:
            h_seq = layer(h_seq)

        return h_seq
