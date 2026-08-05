from typing import Optional
import torch
import torch.nn as nn

# Attention as a module that is optionally attached
#   Additive attention pooling over time.
#   Input:  h (B, T, H)
#   Output: pooled (B, H), attn_weights (B, T)

class AttnPool(nn.Module):
    def __init__(self, hidden_dim: int, attn_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        attn_dim = attn_dim or hidden_dim
        self.proj = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # input: h -> (B, T, H)
    #          B = batch size
    #          T = time steps
    #          H = hidden dimension
    #          h[b, t, :] is the feature vector at time.
    # returns (B,H) - Which timesteps matter most for this sample?
    def forward(self, h: torch.Tensor):
        # A learned transformation into an attention space:
        proj_h = self.proj(h)  # (B,T,H) → (B,T,A)
        tanh_proj_h = torch.tanh(proj_h)  # make attention additive and stabilize (B,T,A) 
        scores_v = self.v(tanh_proj_h) # Score computation - Each timestep now has one scalar score
        scores = scores_v.squeeze(-1) # (B,T,1) -> (B,T) 
        # Now: scores[b, t] = importance of timestep t 
        # How much should each timestep contribute to the final pooled representation?
        # Softmax over time (i.e., attention weights over time)
        a = torch.softmax(scores, dim=1) # such that sum(a[b,t])=1 for all t
        a = self.dropout(a)
        # Weight hidden states:
        # h is (B, T, H):
        #    h[b, t] is a vector of length H, such that h_0, h_1, h_2, ..., h_t  
        # a is (B,T):
        #    a[b, t]  is a scalar importance weight for timestep t:
        #             = a_0, a_1, a_2, ..., a_t  where a_t >= 0 and sum(a_t) = 1
        # (h * a)[b, t, k] = a[b, t] * h[b, t, k] (the unsqueeze converts (B,T) -> (B,T,1) -> (B,T,H), like h)
        h_by_a = h * a.unsqueeze(-1) 
        # Each timestep’s feature vector is scaled by its importance.
        # pooled = a_0 * h_0 + a_1 * h_1 + a_2 * h_2 + ... + a_t * h_t
        pooled = torch.sum(h_by_a, dim=1) # Sum over time
        # pooled is a weighted average of time steps (fixed-size regardless of T)
        return pooled, a

        # Attention example:
        # h = [
        #      [1.0, 0.0],   ← timestep 0
        #      [0.0, 2.0],   ← timestep 1
        #      [1.0, 1.0],   ← timestep 2
        # ] # shape is (3, 2)
        #
        # a = [0.1, 0.7, 0.2] # attention weights (learned by the model)
        #     (timestep 1 is most important, timestep 0 is mostly irrelevant)
        # a.unsqueeze(-1) = [
        #                     [0.1],
        #                     [0.7],
        #                     [0.2]
        #                   ] 
        # h * a.unsqueeze(-1) = [
        #                         [0.1, 0.0],   ← 0.1 · [1.0, 0.0]
        #                         [0.0, 1.4],   ← 0.7 · [0.0, 2.0]
        #                         [0.2, 0.2],   ← 0.2 · [1.0, 1.0]
        #                   ]
        # pooled = [0.1 + 0.0 + 0.2 , 0.0 + 1.4 + 0.2] 
        #        = [      0.3       ,       1.6      ]
        #
        # Mean pooling treats timestep 0 (irrelevant) the same as timestep 1 (important):
        # mean = [
        #          (1.0 + 0.0 + 1.0) / 3,
        #          (0.0 + 2.0 + 1.0) / 3
        #        ] = [0.67, 1.0]
