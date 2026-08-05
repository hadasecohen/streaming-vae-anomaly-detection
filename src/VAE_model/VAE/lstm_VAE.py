import torch
import torch.nn as nn
from typing import Optional
from src.VAE_model.VAE.base_VAE import VAE, LossFunc

# -------- Subclass: LSTM --------
class LSTM_VAE(VAE):
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 loss_func: LossFunc,
                 pooling: str,
                 prior_variance: float,

                 # Encoder LSTM config
                 enc_hidden_dim: int,
                 enc_num_layers: int, # = 1,
                 enc_dropout: float, # = 0.0,

                 # Decoder LSTM config
                 dec_hidden_dim: int, # | None = None,
                 dec_num_layers: int, # = 1,
                 dec_dropout: float, # = 0.0,

                 enc_bidirectional: bool, # = False,

                 clamp_logvar: bool , #= False,
                 clamp_bounds: list , #= 8.0,
                 pooling_attn_dim: int | None = None,
                 pooling_attn_dropout: float = 0.0,
                 pool_latent: bool = False,
                 arch_str: str = "LSTM_VAE",
                 free_bits: float = 0.0,
                 do_kl_normalize: bool = False,
                 kl_mode: str | None = None,
                 kl_auto_scale: bool = False,
                 use_layer_norm: bool = False,
                 pos_enc_freqs=None,
                 pos_enc_mode: str = "dec",
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
                         enc_out_dim = enc_hidden_dim * 2 if enc_bidirectional else enc_hidden_dim,
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

        lstm_input = input_dim
        if pos_enc_freqs is not None and pos_enc_mode in ("enc", "enc_dec"):
            lstm_input += 2 * len(pos_enc_freqs)

        # args value check
        if enc_hidden_dim <= 0:
            raise ValueError("enc_hidden_dim must be > 0")
        if enc_num_layers <= 0:
            raise ValueError("enc_num_layers must be > 0")
        if not (0.0 <= enc_dropout < 0.8):
            raise ValueError("enc_dropout must be in [0, 0.8)")
        if dec_hidden_dim is None:
            dec_hidden_dim = enc_hidden_dim  # common choice, but configurable
        if dec_hidden_dim <= 0:
            raise ValueError("dec_hidden_dim must be > 0")
        if dec_num_layers <= 0:
            raise ValueError("dec_num_layers must be > 0")
        if not (0.0 <= dec_dropout < 0.8):
            raise ValueError("dec_dropout must be in [0, 0.8)")

        self.enc_dir = 2 if enc_bidirectional else 1
        self.enc_bidirectional = enc_bidirectional

        # ----- Encoder -----
        self.enc_hidden_dim = enc_hidden_dim
        self.enc_num_layers = enc_num_layers

        self.encoder = nn.LSTM(
            input_size=lstm_input,
            hidden_size=enc_hidden_dim,
            num_layers=enc_num_layers, # ?
            batch_first=True,
            dropout=enc_dropout if enc_num_layers > 1 else 0.0, # ?
            bidirectional=enc_bidirectional, # ?
        )

        # Initialize forget gate biases to a large value to remember more
        # PyTorch LSTM bias structure: [input, forget, cell, output]
        for names in self.encoder._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.encoder, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                # Initialize forget gate (second quarter)
                bias.data[start:end].fill_(1.0) # Larger value = remember more

        self.enc_ln = nn.LayerNorm(self.enc_out_dim) if use_layer_norm else nn.Identity()

        # ----- Latent -----
        self.mu = nn.Linear(self.enc_out_dim, latent_dim)
        self.logvar = nn.Linear(self.enc_out_dim, latent_dim)

        # ----- Decoder -----
        self.dec_hidden_dim = dec_hidden_dim
        self.dec_num_layers = dec_num_layers

        decoder_input_size = (latent_dim + 2 * len(pos_enc_freqs)
                              if pos_enc_freqs and pos_enc_mode in ("dec", "enc_dec") else latent_dim)
        self.decoder_rnn = nn.LSTM(
            input_size=decoder_input_size,
            hidden_size=dec_hidden_dim,
            num_layers=dec_num_layers,
            batch_first=True,
            dropout=dec_dropout if dec_num_layers > 1 else 0.0, # ?
            bidirectional=False,
        )

        for names in self.decoder_rnn._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.decoder_rnn, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                # Initialize forget gate (second quarter)
                bias.data[start:end].fill_(1.0) # Larger value = remember more

        self.dec_ln = nn.LayerNorm(dec_hidden_dim) if use_layer_norm else nn.Identity()
        self.decoder_out = nn.Linear(dec_hidden_dim, input_dim)

        # ---- map latent z to initial decoder hidden/cell ----
        # h0/c0 must be shaped (num_layers, B, dec_hidden_dim)
        self.z_to_h = nn.Linear(latent_dim, dec_num_layers * dec_hidden_dim)
        self.z_to_c = nn.Linear(latent_dim, dec_num_layers * dec_hidden_dim)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h_seq, _ = self.encoder(x)
        h_seq = self.enc_ln(h_seq)
        return h_seq

    def decode_sequence(self, decoder_input):
        # decoder_input: (B, T, latent_dim) — z_rep broadcast or per-timestep z
        B = decoder_input.size(0)
        z0 = decoder_input[:, 0, :self.latent_dim]   # (B, latent_dim) — z at first step

        # # map z -> initial states
        h0 = self.z_to_h(z0).view(self.dec_num_layers, B, self.dec_hidden_dim).contiguous()
        c0 = self.z_to_c(z0).view(self.dec_num_layers, B, self.dec_hidden_dim).contiguous()

        dec_out, _ = self.decoder_rnn(decoder_input, (h0, c0))   # (B,T,dec_hidden_dim)
        dec_out = self.dec_ln(dec_out)

        x_hat = self.decoder_out(dec_out)                        # (B,T,input_dim)
        return x_hat
