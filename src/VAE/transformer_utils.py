# src/VAE/transformer_utils.py
import math
import torch
import torch.nn as nn
from typing import List, Optional

def make_transformer_stack(
    d_model: int,
    num_attn_layers: int,
    num_attn_heads: Optional[List[int]] = None, # A list as there is one value per layer
    attn_ff_dims: Optional[List[int]] = None,
    attn_dropouts: Optional[List[float]] = None,
    attn_activations: Optional[List[str]] = None,
    attn_norm_first: Optional[List[bool]] = None,
):
    def validate_heads(d_model: int, nhead: int, layer_idx: int, min_head_dim: int = 16):
        # head divisibility
        if d_model % nhead != 0:
            raise ValueError(f"Layer {layer_idx}: d_model={d_model} must be divisible by nhead={nhead}.")
        head_dim = d_model // nhead
        if head_dim < min_head_dim:
            raise ValueError(
                f"Layer {layer_idx}: head_dim=d_model/nhead={head_dim} is too small "
                f"(d_model={d_model}, nhead={nhead}). Recommend >= {min_head_dim}."
            )
        
    if num_attn_heads is None:
        num_attn_heads = [4] * num_attn_layers
    if attn_dropouts is None:
        attn_dropouts = [0.1] * num_attn_layers
    if attn_activations is None:
        attn_activations = ["gelu"] * num_attn_layers
    if attn_norm_first is None:
        attn_norm_first = [True] * num_attn_layers
    if attn_ff_dims is None or len(attn_ff_dims) == 0:
        attn_ff_dims = [4 * d_model] * num_attn_layers # [4*d_model, 4*d_model, .., 4*d_model] (repeat it per layer)
    
    # length checks
    if not (len(num_attn_heads) == len(attn_dropouts) == len(attn_activations) == len(attn_norm_first) == len(attn_ff_dims) == num_attn_layers):
        raise ValueError("Transformer config lists must all have length == num_layers")

    for i, nh in enumerate(num_attn_heads):
        validate_heads(d_model, nh, i)
        if attn_activations[i] not in ("relu", "gelu"):
            raise ValueError(f"nn.TransformerEncoderLayer supports \"relu\" and \"gelu\". Got {attn_activations[i]}")
        if not (0.0 <= attn_dropouts[i] < 0.8):
            raise ValueError(f"dropout bounds should be 0.0 <= droupout < 0.8. Got {attn_dropouts[i]}")
    
    layers = nn.ModuleList([
        nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_attn_heads[i],
            dim_feedforward=attn_ff_dims[i],
            dropout=attn_dropouts[i],
            activation=attn_activations[i],
            batch_first=True,
            norm_first=attn_norm_first[i],
        )
        for i in range(num_attn_layers)
    ]) if num_attn_layers > 0 else None

    return layers

# Store Positional encoding once in cache[key] and reuse it.
# Returns (1,T,d_model) sinusoidal PE, cached by (T, device.type, device.index, d_model).
def sinusoidal_pe_additive_cached(cache: dict, T: int, d_model: int, device: torch.device) -> torch.Tensor:
    
    # Positional encoding depends only on T, d_model and device.
    key = (T, device.type, device.index, d_model)
    pe = cache.get(key)
    if pe is not None:
        return pe

    position = torch.arange(T, device=device).float().unsqueeze(1)  # (T,1)
    # torch.arange(0, d_model, 2) gives even indices [0,2,4,...] up to d_model-1, length is d_model/2 
    # exp(...), gets frequencies that range from slow to fast.
    # div_term shape: (d_model/2,)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model)
    )
    pe_t = torch.zeros(T, d_model, device=device)
    pe_t[:, 0::2] = torch.sin(position * div_term) # 0::2 - even dimensions get sine
    pe_t[:, 1::2] = torch.cos(position * div_term) # 1::2 - odd dimensions get cosine
    
    pe = pe_t.unsqueeze(0)  # (1,T,d_model)

    cache[key] = pe
    return pe
