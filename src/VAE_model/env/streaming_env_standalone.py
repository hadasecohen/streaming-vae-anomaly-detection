from collections import deque
from typing import Dict
import torch
from torch.optim import Adam

from src.logging_utils import create_run_files_obj
from src.Data_tools.dataset_loader import data_fn_builder
from src.VAE_model.utils import format_metrics_stats, to_plain, make_positional_freqs, make_hourly_domain_freqs
from src.VAE_model.env.streaming_env import StreamEnv
from src.VAE_model.Streaming.streaming_loop import AnomalyDetection
from src.VAE_model.VAE.mlp_VAE import MLP_VAE
from src.VAE_model.VAE.lstm_VAE import LSTM_VAE
from src.VAE_model.VAE.tf_VAE import TF_VAE
from src.VAE_model.VAE.base_VAE import LossFunc
from src.VAE_model.Streaming.trainer_utils import ThresholdState
from src.VAE_model.Streaming.stream_trainer import StreamTrainer


class StreamEnvStandalone(StreamEnv):
    
    def __init__(
        self, arch, cfg, seed, device, config_path: str = None
    ):
        super().__init__(device)


        # Build once; no other path/file handling anywhere else
        self.run_files_output = create_run_files_obj(cfg["common"]["logging"])
        self.trial_files_output = self.run_files_output.create_trial_files_obj(0, standalone=True)
        print(
            f"Logger: Verbosity={self.run_files_output.logger.level}, "
            f"logging into: '{self.run_files_output.logger.stream.name}'"
        )

        if config_path is not None:
            self.run_files_output.logger.info_low(f"CONFIG = {config_path}")
        self.run_files_output.logger.info_low(f"GLOBAL SEED = {seed}")

        # Build the dataset function closure from cfg["data"]
        get_data_adapter_fn = data_fn_builder(cfg["data"], self.trial_files_output.logger)
        adapter, metadata = get_data_adapter_fn()
        
        eval_idx  = metadata["eval_idx"] # what index is no longer wamup (anomalies expected)
        
        stream_cfg = cfg["stream"]
        training_cfg = cfg["training"]
        model_cfg = cfg["model"] 
        
        ###############################################
        #                                             #
        #                     MODEL                   #
        #                                             #
        ###############################################

        # COMMON
        stream_mode = stream_cfg["mode"] # window or point
        input_dim = metadata["input_dim"] # size of input
        latent_dim = model_cfg["latent_dim"]
        loss_fn_type = "default"
        recon_loss_kind = model_cfg["recon_loss_kind"]
        loss_fn = LossFunc(loss_fn_type, recon_loss_kind)
        pooling = model_cfg.get("pooling", "mean") 
        pool_latent = model_cfg["pool_latent"]
        prior_variance = model_cfg["prior_variance"]
        clamp_logvar = model_cfg["clamp_logvar"]
        clamp_bounds = model_cfg["clamp_bounds"]
        free_bits = model_cfg.get("free_bits", 0.0)
        # kl_mode: "norm" | "norm_T" | "no_norm"
        # Explicit kl_mode takes precedence; do_kl_normalize is the legacy fallback.
        kl_mode = model_cfg.get("kl_mode") or (
            "norm" if model_cfg.get("do_kl_normalize", False) else "norm_T"
        )
        kl_auto_scale = model_cfg.get("kl_auto_scale", False)
        
        seq_len = stream_cfg["seq_len"] if stream_mode == "window" else 1
        # pos_enc: "enc" | "dec" | "enc_dec" | "none"
        # Backward compat: use_pos_enc: true → "dec"
        pos_enc_mode = model_cfg.get("pos_enc", None)
        if pos_enc_mode is None:
            pos_enc_mode = "dec" if model_cfg.get("use_pos_enc", False) else "none"
        elif pos_enc_mode not in ("enc", "dec", "enc_dec", "none"):
            raise ValueError(f"model.pos_enc must be 'enc', 'dec', 'enc_dec', or 'none'. Got '{pos_enc_mode}'.")
        if pos_enc_mode != "none" and arch == "TF_VAE":
            raise ValueError(
                f"pos_enc='{pos_enc_mode}' is not valid for {arch}: transformer models use "
                "internal sinusoidal PE. Set pos_enc: none or remove the key."
            )
        use_hourly_freqs = model_cfg.get("use_hourly_freqs", False)
        if pos_enc_mode != "none" and stream_mode == "window":
            pos_enc_freqs = make_hourly_domain_freqs(seq_len) if use_hourly_freqs else make_positional_freqs(seq_len)
        else:
            pos_enc_freqs = None
            pos_enc_mode = "none"

        fixed_cfg = {
            "device": self.device,
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "recon_loss_kind": recon_loss_kind,
            "pooling": pooling,
            "pool_latent": pool_latent,
            "prior_variance": prior_variance,
            "clamp_logvar": clamp_logvar,
            "clamp_bounds": clamp_bounds,
            "stream_mode": stream_mode,
            "seq_len": seq_len,
            "free_bits": free_bits,
            "kl_mode": kl_mode,
            "kl_auto_scale": kl_auto_scale,
            "cfg_data_args": cfg["data"]["args"]
        }
        
        # MLP encoder
        hidden_dims = model_cfg.get("hidden_dims", None)
        acts = model_cfg.get("acts", None)
        drops = model_cfg.get("drops", None)
        if drops is not None and len(drops) > 0 and drops[-1] != 0.0:
            raise ValueError(f"The last encoder dropout must be 0.0. Got {drops}")

        # MLP decoder (optional asymmetric config; None → mirror encoder in reverse)
        dec_hidden_dims = model_cfg.get("dec_hidden_dims", None)
        dec_acts        = model_cfg.get("dec_acts", None)
        dec_drops       = model_cfg.get("dec_drops", None)
        if dec_drops is not None and len(dec_drops) > 0 and dec_drops[0] != 0.0:
            raise ValueError(f"The first decoder dropout must be 0.0. Got {dec_drops}")

        # MLP normalization: None | "layer" | "batch"
        mlp_norm = model_cfg.get("norm", None)

        #LSTM
        enc_hidden_dim = model_cfg.get("enc_hidden_dim", None)
        enc_num_layers = model_cfg.get("enc_num_layers", 1)
        enc_dropout = model_cfg.get("enc_dropout", 0.0)

        dec_hidden_dim = model_cfg.get("dec_hidden_dim", None)
        dec_num_layers = model_cfg.get("dec_num_layers", 1)
        dec_dropout = model_cfg.get("dec_dropout", 0.0)

        enc_bidirectional = model_cfg.get("enc_bidirectional", False)
        lstm_use_layer_norm = model_cfg.get("use_layer_norm", False)

        # ATTN
        num_attn_heads = model_cfg.get("num_attn_heads", None)
        dim_feedforward = model_cfg.get("dim_feedforward", None)
        attn_dropouts=model_cfg.get("attn_dropouts", None)
        attn_acts=model_cfg.get("attn_acts", None)
        attn_norm_first=model_cfg.get("attn_norm_first", None)
        use_transformer_pos_enc=model_cfg.get("use_transformer_pos_enc", None)

        flow_type       = model_cfg.get("flow_type", None)
        flow_num_steps  = model_cfg.get("flow_num_steps", 0)
        flow_hidden_dim = model_cfg.get("flow_hidden_dim", None)
        divergence_kind = model_cfg.get("divergence_kind", "kl")

        # TF_VAE / MS_TF params
        tf_d_model        = model_cfg.get("d_model", 64)
        tf_num_enc_layers = model_cfg.get("num_enc_layers", 3)
        tf_num_dec_layers = model_cfg.get("num_dec_layers", 3)
        tf_nhead          = model_cfg.get("nhead", 4)
        tf_ff_dim         = model_cfg.get("ff_dim", None)
        tf_dropout        = model_cfg.get("dropout", 0.1)
        tf_activation     = model_cfg.get("activation", "gelu")
        tf_norm_first     = model_cfg.get("norm_first", True)

        if arch == "MLP" :
            model = MLP_VAE(
                input_dim=input_dim,
                latent_dim=latent_dim,
                loss_func=loss_fn,
                pooling=pooling,
                pool_latent=pool_latent,
                prior_variance=prior_variance,
                enc_dec_hidden_dims=hidden_dims,
                enc_dec_activation_funcs=acts,
                enc_dec_dropouts=drops,
                dec_hidden_dims=dec_hidden_dims,
                dec_activation_funcs=dec_acts,
                dec_dropouts=dec_drops,
                norm=mlp_norm,
                clamp_logvar=clamp_logvar,
                clamp_bounds=clamp_bounds,
                pos_enc_freqs=pos_enc_freqs,
                pos_enc_mode=pos_enc_mode,
                free_bits=free_bits,
                kl_mode=kl_mode,
                kl_auto_scale=kl_auto_scale,
                flow_type=flow_type,
                flow_num_steps=flow_num_steps,
                flow_hidden_dim=flow_hidden_dim,
                divergence_kind=divergence_kind,
            ).to(self.device)

            fixed_cfg["pos_enc_freqs"] = pos_enc_freqs
            fixed_cfg["pos_enc_mode"]  = pos_enc_mode
            fixed_cfg["enc_dec_hidden_dims"] = hidden_dims
            fixed_cfg["enc_dec_activation_funcs"] = acts
            fixed_cfg["enc_dec_dropouts"] = drops
            fixed_cfg["dec_hidden_dims"] = dec_hidden_dims
            fixed_cfg["dec_activation_funcs"] = dec_acts
            fixed_cfg["dec_dropouts"] = dec_drops
            fixed_cfg["norm"] = mlp_norm


        elif arch == "LSTM" :

            model = LSTM_VAE(
                input_dim=input_dim,
                latent_dim=latent_dim,
                loss_func=loss_fn,
                pooling=pooling,
                pool_latent=pool_latent,
                prior_variance=prior_variance,

                # ----- encoder -----
                enc_hidden_dim=enc_hidden_dim,
                enc_num_layers=enc_num_layers,
                enc_dropout=enc_dropout,

                # ----- decoder -----
                dec_hidden_dim=dec_hidden_dim,   # SAME as encoder hidden size
                dec_num_layers=dec_num_layers,
                dec_dropout=dec_dropout,

                enc_bidirectional=enc_bidirectional,
                use_layer_norm=lstm_use_layer_norm,

                clamp_logvar=clamp_logvar,
                clamp_bounds=clamp_bounds,
                free_bits=free_bits,
                kl_mode=kl_mode,
                kl_auto_scale=kl_auto_scale,
                pos_enc_freqs=pos_enc_freqs,
                pos_enc_mode=pos_enc_mode,
                flow_type=flow_type,
                flow_num_steps=flow_num_steps,
                flow_hidden_dim=flow_hidden_dim,
                divergence_kind=divergence_kind,
            ).to(self.device)

            fixed_cfg["pos_enc_freqs"] = pos_enc_freqs
            fixed_cfg["pos_enc_mode"]  = pos_enc_mode
            fixed_cfg["enc_hidden_dim"] = enc_hidden_dim
            fixed_cfg["enc_num_layers"] = enc_num_layers
            fixed_cfg["enc_dropout"] = enc_dropout
            fixed_cfg["dec_hidden_dim"] = dec_hidden_dim
            fixed_cfg["dec_num_layers"] = dec_num_layers
            fixed_cfg["dec_dropout"] = dec_dropout
            fixed_cfg["enc_bidirectional"] = enc_bidirectional
            fixed_cfg["use_layer_norm"] = lstm_use_layer_norm

        elif arch == "TF_VAE":
            model = TF_VAE(
                input_dim=input_dim,
                latent_dim=latent_dim,
                loss_func=loss_fn,
                pooling=pooling,
                pool_latent=pool_latent,
                prior_variance=prior_variance,
                clamp_logvar=clamp_logvar,
                clamp_bounds=clamp_bounds,
                free_bits=free_bits,
                kl_mode=kl_mode,
                kl_auto_scale=kl_auto_scale,
                d_model=tf_d_model,
                num_enc_layers=tf_num_enc_layers,
                num_dec_layers=tf_num_dec_layers,
                nhead=tf_nhead,
                ff_dim=tf_ff_dim,
                dropout=tf_dropout,
                activation=tf_activation,
                norm_first=tf_norm_first,
                flow_type=flow_type,
                flow_num_steps=flow_num_steps,
                flow_hidden_dim=flow_hidden_dim,
                divergence_kind=divergence_kind,
            ).to(self.device)

            fixed_cfg["d_model"]          = tf_d_model
            fixed_cfg["num_enc_layers"]   = tf_num_enc_layers
            fixed_cfg["num_dec_layers"]   = tf_num_dec_layers
            fixed_cfg["nhead"]            = tf_nhead
            fixed_cfg["ff_dim"]           = tf_ff_dim
            fixed_cfg["dropout"]          = tf_dropout
            fixed_cfg["activation"]       = tf_activation
            fixed_cfg["norm_first"]       = tf_norm_first

        # weight_init = model_cfg.get("weight_init", "xavier_uniform")
        # model.apply_weight_init(weight_init)

        n_cyc = metadata.get("n_cyclic_features", 0)
        if n_cyc > 0:
            model.n_cyclic_features = n_cyc
            model.cyclic_recon = bool(model_cfg.get("cyclic_recon", False))

        # ---------------------------
        # Opetimizer
        # ---------------------------
        lr_offline = training_cfg["lr_offline"]
        lr_online = training_cfg["lr_online"]
        
        offline_batch_size = training_cfg["offline_batch_size"]
        wd = training_cfg["weight_decay"]

        opt_offline = torch.optim.Adam(
            model.parameters(),
            lr=float(lr_offline),
            weight_decay=float(wd),
        )

        opt_online = torch.optim.Adam(
            model.parameters(),
            lr=float(lr_online),
            weight_decay=float(wd),
        )



        # -----------------------------
        # prediction Threshold settings 
        # -----------------------------
        threshold_cfg=training_cfg["threshold"]
        threshold_q=threshold_cfg["q"]
        threshold_buffer_size = threshold_cfg["buffer_length"]
        threshold_ema_alpha = threshold_cfg["ema"]

        fixed_cfg["threshold_q"] = threshold_q
        fixed_cfg["threshold_buffer_size"] = threshold_buffer_size
        fixed_cfg["threshold_ema_alpha"] = threshold_ema_alpha

        threshold = ThresholdState(
            threshold_q,
            threshold_buffer_size,
            threshold_ema_alpha,
            float("nan")
        )

        threshold_overrides: Dict[str, ThresholdState] = {
            name: ThresholdState(
                q=cfg.get("q", threshold_q),
                buffer_size=cfg.get("buffer_length", threshold_buffer_size),
                ema_alpha=cfg.get("ema", threshold_ema_alpha),
                value=float("nan"),
            )
            for name, cfg in training_cfg.get("threshold_per_metric", {}).items()
        }
        hard_threshold_cut = training_cfg["hard_threshold_cut"]
        
        ###############################################
        #                                             #
        #                   TRAINER                   #
        #                                             #
        ###############################################

        # Trainer configs
    
        clip_grads_norm = training_cfg["clip_grads_norm"] # optional gradient clipping to reduce NaN risk
        policy = training_cfg["policy"]
        elbo_beta = training_cfg["elbo_beta"]
        kl_anneal_epochs = training_cfg.get("kl_anneal_epochs", 0)
        update_if_score_below = training_cfg.get("update_if_score_below", 0.5)
        metric_weights      = training_cfg["ensemble"]["metric_weights"]
        anom_if_score_above  = float(training_cfg.get("anom_if_score_above", 0.5))
        training_loss       = str(training_cfg.get("training_loss", "elbo"))
        _valid_training_losses = {"elbo", "var_reg", "elbo_var"}
        if training_loss not in _valid_training_losses:
            raise ValueError(
                f"Invalid training_loss: '{training_loss}'. "
                f"Must be one of {sorted(_valid_training_losses)}. "
                f"Note: 'elbo_w_var' is a scoring metric, not a training loss."
            )
        lambda_variance     = float(training_cfg.get("lambda_variance", 0.0))
        

        trainer = StreamTrainer(
            model=model,
            online_optimizer=opt_online,
            offline_optimizer=opt_offline,
            offline_batch_size=offline_batch_size,
            device = self.device,
            # loss_fn,   # loss_fn: must return (loss, recon_err, z_summary)
            clip_grads_norm=clip_grads_norm,
            threshold=threshold,
            hard_threshold_cut = hard_threshold_cut,
            train_policy=policy,
            stream_mode=stream_mode,
            elbo_beta=elbo_beta,
            # lambda_variance=lambda_variance,
            metric_weights=metric_weights,
            anom_if_score_above=anom_if_score_above,
            training_loss=training_loss,
            lambda_variance=lambda_variance,
            logger=self.trial_files_output.logger,
            update_if_score_below=update_if_score_below,
            threshold_overrides=threshold_overrides,
        )
        trainer.kl_anneal_offline_epochs = kl_anneal_epochs

        # ---------------------------
        # Run streaming configs
        # ---------------------------
        stream_stride = stream_cfg["stride"]
        stream_shuffle = stream_cfg["shuffle"]
        stream_max_steps = stream_cfg.get("max_steps", None)
        offline_enabled = training_cfg.get("offline_enabled", True)
        offline_epochs = training_cfg.get("offline_epochs", 40)
        early_stop_patience = training_cfg.get("early_stop_patience", 5)
        
        
        ###############################################
        #                                             #
        #               Print Configs                 #
        #                                             #
        ###############################################   
        
        fixed_cfg["lr_offline"] = lr_offline
        fixed_cfg["lr_online"] = lr_online
        fixed_cfg["offline_batch_size"] = offline_batch_size
        fixed_cfg["weight_decay"] = wd
        fixed_cfg["elbo_beta"] = elbo_beta
        fixed_cfg["kl_anneal_epochs"]= kl_anneal_epochs
        fixed_cfg["anom_if_score_above"] = anom_if_score_above
        fixed_cfg["update_if_score_below"] = update_if_score_below
        fixed_cfg["metric_weights"] = metric_weights
        fixed_cfg["training_loss"] = training_loss
        fixed_cfg["lambda_variance"] = lambda_variance   
        # fixed_cfg["lambda_variance"]=lambda_variance
        fixed_cfg["stride"] = stream_stride
        fixed_cfg["shuffle"] = stream_shuffle
        fixed_cfg["max_steps"] = stream_max_steps
        fixed_cfg["eval_idx"] = eval_idx
        fixed_cfg["offline_enabled"] = offline_enabled
        fixed_cfg["clip_grads_norm"] = clip_grads_norm
        fixed_cfg["policy"]               = policy
        fixed_cfg["offline_epochs"]        = offline_epochs
        fixed_cfg["early_stop_patience"]   = early_stop_patience
        fixed_cfg["flow_type"]             = flow_type
        fixed_cfg["flow_num_steps"]        = flow_num_steps
        fixed_cfg["flow_hidden_dim"]       = flow_hidden_dim
        fixed_cfg["divergence_kind"]       = divergence_kind

        evt = {
            "event": "run_config",
            "fixed": {k: to_plain(v) for k, v in fixed_cfg.items()},
        }
        # single JSON line (easy to parse to CSV)
        self.trial_files_output.logger.info_low(lambda: f"{evt}")
        self.trial_files_output.logger.info_low(lambda: f"Number of Parameters: {model.count_parameters()}")

        ###############################################
        #                                             #
        #                    RUN                      #
        #                                             #
        ###############################################

        anomaly_detection = AnomalyDetection(
            adapter,
            trainer,
            stream_mode,
            seq_len,
            stream_stride,
            stream_shuffle,
            stream_max_steps,
            self.trial_files_output,
            seed,
            offline_enabled=offline_enabled,
            offline_epochs=offline_epochs,
            early_stop_patience=early_stop_patience,
            plot_settings=cfg.get("plot", {}),
        )

        anomaly_detection.train_warmup()

        metrics_stats = anomaly_detection.run_stream()

        try:
            _n_warmup = int(getattr(anomaly_detection.adapter.train_ds, "X",
                                    __import__("numpy").array([])).shape[0])
            anomaly_detection.perf.finalize(
                trial_dir=self.trial_files_output.trial_dir,
                logger=self.trial_files_output.logger,
                n_warmup_samples=_n_warmup,
                n_stream_steps=anomaly_detection.n_stream_steps,
                n_params=model.count_parameters(),
            )
        except Exception as _perf_err:
            self.trial_files_output.logger.info_low(f"[perf] tracking failed: {_perf_err}")

        self.trial_files_output.logger.info_low(lambda: format_metrics_stats(metrics_stats, nd=3))
        self.trial_files_output.close()
        
        
        


