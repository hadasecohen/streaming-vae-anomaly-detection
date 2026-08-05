from collections import deque
from typing import Dict
import torch
from torch.optim import Adam
import optuna

import numpy as np

from src.VAE_model.env.streaming_env import StreamEnv
from src.VAE_model.Streaming.streaming_loop import AnomalyDetection
from src.logging_utils import RunFilesObj, create_run_files_obj

from src.Data_tools.dataset_loader import data_fn_builder
from src.VAE_model.env.optuna_utils import suggest_param, make_report_fn

from src.VAE_model.VAE.base_VAE import LossFunc
from src.VAE_model.Streaming.stream_trainer import StreamTrainer
from src.VAE_model.Streaming.trainer_utils import ThresholdState
from src.VAE_model.utils import format_metrics_stats, to_plain, make_positional_freqs, make_hourly_domain_freqs

class StreamEnvOptuna(StreamEnv):
    
    def __init__(
        self, cfg, seed, device
    ): 
        super().__init__((device))
        
        # Build once; no other path/file handling anywhere else
        self.run_files_output: RunFilesObj = create_run_files_obj(cfg["common"]["logging"])
        print(
            f"Logger: Verbosity={self.run_files_output.logger.level}, "
            f"logging into: {self.run_files_output.logger.stream}"
        )
        
        self.run_files_output.logger.info_low(f"GLOBAL SEED = {seed}")

        # Build the dataset function closure from cfg["data"]
        get_data_adapter_fn = data_fn_builder(cfg["data"], self.run_files_output.logger)        

        
        # Convenience wrapper to create an Optuna study and optimize the objective.
        optuna_cfg = cfg["optuna"]        
        self.optuna_n_trials = optuna_cfg.get("n_trials", 4)
        
        self.objective = self.build_objective(self.device, cfg, get_data_adapter_fn, self.run_files_output, seed)

    def run_study(self) :
        study = optuna.create_study(direction="maximize")

        study.optimize(self.objective, n_trials=self.optuna_n_trials)

        print("\nBest Trial:")
        print("  Value :", study.best_value)
        print("  Params:", study.best_trial.params)
    
    
    # Returns objective(trial) that:
    #   - samples hyperparams from cfg YAML via suggest_param
    #   - builds model/optimizer/trainer
    #   - runs streaming (warmup + test-then-train)
    #   - reports intermediate PR and final F1 to Optuna
    def build_objective(self, device, cfg, get_data_adapter_fn, run_files_output: RunFilesObj, seed: int) :
        
        model_type = cfg["model"]["type"]
        # Lazy import so this file doesn’t require model at import time
        match model_type:
            case "MLP":
                from src.VAE_model.VAE.mlp_VAE import MLP_VAE
            case "MLP_Attn" | "MLP attn":
                from src.VAE_model.VAE.mlp_attn_VAE import MLP_ATTN_VAE
            case "LSTM":
                from src.VAE_model.VAE.lstm_VAE import LSTM_VAE
            case "LSTM_Attn" | "LSTM attn":
                from src.VAE_model.VAE.lstm_attn_VAE import LSTM_ATTN_VAE
            case "TF_VAE":
                from src.VAE_model.VAE.tf_VAE import TF_VAE
            case _:
                raise ValueError(f"No such model: {model_type}")


        def objective(trial: optuna.Trial,
                      cfg=cfg,
                      device=device,
                      get_data_adapter_fn=get_data_adapter_fn,
                      run_files_output=run_files_output):

            # per-trial artifacts+logger (unique subdir)
            trial_files_output = run_files_output.create_trial_files_obj(trial.number, standalone=False)

            # ---------------------------
            # Build adapter from the data function
            # ---------------------------
            adapter, meta = get_data_adapter_fn()
            eval_idx  = meta["eval_idx"] # what index is no longer wamup (anomalies expected)

            # ---------------------------
            # Sample search-space params
            # ---------------------------
            ctx = {}

            
            training_cfg = cfg["training"]
            stream_cfg   = cfg["stream"]
            
            stream_mode = stream_cfg["mode"] # "point" or "window"

            if stream_mode == "window" :
                seq_len = suggest_param(trial, "seq_len", stream_cfg["seq_len"], ctx)
            else :
                seq_len = 1
            model_cfg_optuna = cfg["model"]
            pos_enc_mode = model_cfg_optuna.get("pos_enc", None)
            if pos_enc_mode is None:
                pos_enc_mode = "dec" if model_cfg_optuna.get("use_pos_enc", False) else "none"
            elif pos_enc_mode not in ("enc", "dec", "none"):
                raise ValueError(f"model.pos_enc must be 'enc', 'dec', or 'none'. Got '{pos_enc_mode}'.")
            use_hourly_freqs = model_cfg_optuna.get("use_hourly_freqs", False)
            if pos_enc_mode != "none" and stream_mode == "window":
                pos_enc_freqs = make_hourly_domain_freqs(seq_len) if use_hourly_freqs else make_positional_freqs(seq_len)
            else:
                pos_enc_freqs = None
                pos_enc_mode = "none"
   
            ###############################################
            #                                             #
            #                     MODEL                   #
            #                                             #
            ###############################################
            
            model_cfg     = cfg["model"]

            input_dim = meta["input_dim"] # size of input
            latent_dim = suggest_param(trial, "latent_dim", model_cfg["latent_dim"], ctx)
            loss_fn_type = "default" # TODO: other loss_func?
            recon_loss_kind = suggest_param(trial, "recon_loss_kind", model_cfg["recon_loss_kind"], ctx)
            loss_fn = LossFunc(loss_fn_type, recon_loss_kind)
            pooling = suggest_param(trial, "pooling_method", model_cfg["pooling"], ctx)
            pool_latent = suggest_param(trial, "pool_latent", model_cfg["pool_latent"], ctx)
            prior_variance  = suggest_param(trial, "prior_variance",  model_cfg["prior_variance"], ctx)
            clamp_logvar = model_cfg.get("clamp_logvar", True)
            clamp_bounds = model_cfg.get("clamp_bounds", 10.0)
            kl_mode = model_cfg.get("kl_mode") or (
                "norm" if model_cfg.get("do_kl_normalize", False) else "norm_T"
            )
            kl_auto_scale = model_cfg.get("kl_auto_scale", False)
            
            fixed_cfg = {
                "device": self.device,
                "stream_mode": stream_mode,
                "input_dim": input_dim,
                "clamp_logvar": clamp_logvar,
                "clamp_bounds": clamp_bounds,
                "stream_mode": stream_mode,
            }
            
            chosen_cfg = {
                "latent_dim": latent_dim,
                "recon_loss_kind": recon_loss_kind,
                "pooling": pooling,
                "pool_latent": pool_latent,
                "prior_variance": prior_variance,
                "seq_len": seq_len,
            }
            match model_type:
                case "MLP" :
                    enc_depth = suggest_param(trial, "enc_depth", model_cfg["enc_depth"], ctx)
                    ctx["enc_depth"] = enc_depth            
                    
                    hidden_dims = suggest_param(trial, "widths_list", model_cfg["hidden_dims"], ctx)
                    acts        = suggest_param(trial, "acts_list", model_cfg["acts"], ctx)
                    drops       = suggest_param(trial, "drops_list", model_cfg["drops"], ctx)
                    
                    # Force last dropout = 0
                    if len(drops) > 0:
                        drops[-1] = 0.0

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
                        clamp_logvar=clamp_logvar,
                        clamp_bounds=clamp_bounds,
                        pos_enc_freqs=pos_enc_freqs,
                        pos_enc_mode=pos_enc_mode,
                        kl_mode=kl_mode,
                        kl_auto_scale=kl_auto_scale,
                    ).to(device)

                    chosen_cfg["enc_dec_hidden_dims"] = hidden_dims,
                    chosen_cfg["enc_dec_activation_funcs"] = acts
                    chosen_cfg["enc_dec_dropouts"] = drops

                case "LSTM" :
                    enc_hidden_dim = suggest_param(trial, "enc_hidden_dim", model_cfg["enc_hidden_dim"], ctx)
                    enc_num_layers =  suggest_param(trial, "enc_num_layers", model_cfg["enc_num_layers"], ctx)
                    enc_dropout = suggest_param(trial, "enc_dropout", model_cfg["enc_dropout"], ctx) 

                    dec_hidden_dim_opts = model_cfg.get("dec_hidden_dim", None)
                    dec_num_layers_opts = model_cfg.get("dec_num_layers", 1)
                    dec_dropout_opts = model_cfg.get("dec_dropout", 0.0)

                    dec_hidden_dim = None if dec_hidden_dim_opts == None else suggest_param(trial, "dec_hidden_dim", dec_hidden_dim_opts, ctx)
                    dec_num_layers = suggest_param(trial, "dec_num_layers", dec_num_layers_opts, ctx)
                    dec_dropout =    suggest_param(trial, "dec_dropout", dec_dropout_opts, ctx)
                    
                    enc_bidirectional = suggest_param(trial, "enc_bidirectional", model_cfg["enc_bidirectional"], ctx)

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
                        dec_hidden_dim=dec_hidden_dim, # dec_hidden_dim,   # SAME as encoder hidden size
                        dec_num_layers=dec_num_layers, # dec_num_layers,
                        dec_dropout=dec_dropout, # dec_dropout,

                        enc_bidirectional = enc_bidirectional,

                        clamp_logvar=clamp_logvar,
                        clamp_bounds=clamp_bounds,
                        kl_mode=kl_mode,
                        kl_auto_scale=kl_auto_scale,
                        pos_enc_freqs=pos_enc_freqs,
                        pos_enc_mode=pos_enc_mode,
                    ).to(self.device)

                    chosen_cfg["enc_hidden_dim"] = enc_hidden_dim
                    chosen_cfg["enc_num_layers"] = enc_num_layers
                    chosen_cfg["enc_dropout"] = enc_dropout
                    chosen_cfg["enc_bidirectional"] = enc_bidirectional

                    chosen_cfg["dec_hidden_dim"] = dec_hidden_dim
                    chosen_cfg["dec_num_layers"] = dec_num_layers
                    chosen_cfg["dec_dropout"] = dec_dropout

                case "TF_VAE":
                    tf_d_model        = suggest_param(trial, "d_model",        model_cfg["d_model"],        ctx)
                    tf_num_enc_layers = suggest_param(trial, "num_enc_layers", model_cfg["num_enc_layers"], ctx)
                    tf_num_dec_layers = suggest_param(trial, "num_dec_layers", model_cfg["num_dec_layers"], ctx)
                    tf_nhead          = model_cfg.get("nhead", 4)
                    tf_ff_dim         = model_cfg.get("ff_dim", None)
                    tf_dropout        = suggest_param(trial, "dropout",        model_cfg.get("dropout", 0.1), ctx)
                    tf_activation     = model_cfg.get("activation", "gelu")
                    tf_norm_first     = model_cfg.get("norm_first", True)

                    model = TF_VAE(
                        input_dim=input_dim,
                        latent_dim=latent_dim,
                        loss_func=loss_fn,
                        pooling=pooling,
                        pool_latent=pool_latent,
                        prior_variance=prior_variance,
                        clamp_logvar=clamp_logvar,
                        clamp_bounds=clamp_bounds,
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
                    ).to(device)

                    chosen_cfg["d_model"]        = tf_d_model
                    chosen_cfg["num_enc_layers"] = tf_num_enc_layers
                    chosen_cfg["num_dec_layers"] = tf_num_dec_layers
                    chosen_cfg["nhead"]          = tf_nhead
                    chosen_cfg["dropout"]        = tf_dropout

            weight_init = model_cfg.get("weight_init", "xavier_uniform")
            model.apply_weight_init(weight_init)

            # ---------------------------
            # Opetimizer
            # ---------------------------
            #lr = suggest_param(trial, "lr", training_cfg["lr"], ctx)
            lr_offline = suggest_param(trial, "lr_offline", training_cfg["lr_offline"], ctx)
            lr_online = suggest_param(trial, "lr_online", training_cfg["lr_online"], ctx)
            wd = suggest_param(trial, "weight_decay", training_cfg["weight_decay"], ctx)

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

            offline_batch_size = suggest_param(trial, "offline_batch_size", training_cfg["offline_batch_size"], ctx)
            chosen_cfg["lr_online"] = lr_online
            chosen_cfg["lr_offline"] = lr_offline
            chosen_cfg["wd"] = wd
            chosen_cfg["offline_batch_size"]=offline_batch_size
            # TODO  - enable/disable Adaptive lr/wd

            # -----------------------------
            # prediction Threshold settings 
            # -----------------------------
            threshold_cfg=training_cfg["threshold"]
            
            threshold_q=suggest_param(trial, "threshold_q", threshold_cfg["q"], ctx)
            threshold_buffer_length=suggest_param(trial, "threshold_buffer_length", threshold_cfg["buffer_length"], ctx)
            threshold_ema_alpha = suggest_param(trial, "threshold_ema", threshold_cfg["ema"], ctx)
        
            threshold = ThresholdState(
                threshold_q,
                threshold_buffer_length,
                threshold_ema_alpha,
                float("nan")
            )

            hard_threshold_cut = suggest_param(trial, "hard_threshold_cut", training_cfg["hard_threshold_cut"], ctx)
            
            chosen_cfg["threshold_q"] = threshold_q
            chosen_cfg["threshold_buffer_length"] = threshold_buffer_length
            chosen_cfg["threshold_ema_alpha"] = threshold_ema_alpha
            chosen_cfg["hard_threshold_cut"]=hard_threshold_cut
        
            ###############################################
            #                                             #
            #                   TRAINER                   #
            #                                             #
            ###############################################

            
            clip_grads_norm = suggest_param(trial, "clip_grads_norm", training_cfg["clip_grads_norm"],ctx) # optional gradient clipping to reduce NaN risk            
            policy = training_cfg["policy"] # "all" or "normal_only"
            elbo_beta = suggest_param(trial, "elbo_beta", training_cfg["elbo_beta"], ctx)
            #lambda_variance = suggest_param(trial, "lambda_variance", training_cfg["lambda_variance"], ctx)

            chosen_cfg["clip_grads_norm"] = clip_grads_norm
            fixed_cfg["policy"] = policy
            chosen_cfg["elbo_beta"] = elbo_beta
            # chosen_cfg["lambda_variance"] = lambda_variance
            
            metric_weights = {}

            for name, w_cfg in training_cfg["ensemble"]["metric_weights"].items():
                metric_weights[name] = suggest_param(
                    trial,
                    f"metric_weight_{name}",
                    w_cfg,
                    ctx="ensemble.metric_weights"
                )
            
            chosen_cfg["metric_weights"]=metric_weights
            anom_if_score_above   = float(training_cfg.get("anom_if_score_above", 0.5))
            update_if_score_below = float(training_cfg.get("update_if_score_below", 0.5))
            chosen_cfg["anom_if_score_above"]   = anom_if_score_above
            chosen_cfg["update_if_score_below"] = update_if_score_below
            training_loss   = str(training_cfg.get("training_loss", "elbo"))
            _valid_training_losses = {"elbo", "var_reg", "elbo_var"}
            if training_loss not in _valid_training_losses:
                raise ValueError(
                    f"Invalid training_loss: '{training_loss}'. "
                    f"Must be one of {sorted(_valid_training_losses)}. "
                    f"Note: 'elbo_w_var' is a scoring metric, not a training loss."
                )
            lambda_variance = float(training_cfg.get("lambda_variance", 0.0))
            chosen_cfg["training_loss"]   = training_loss
            chosen_cfg["lambda_variance"] = lambda_variance
            
           
            trainer = StreamTrainer(
                model,
                opt_online,
                opt_offline,
                offline_batch_size,
                device,
                # loss_fn,   # loss_fn: must return (loss, recon_err, z_summary)
                clip_grads_norm,
                threshold,
                hard_threshold_cut,
                policy,
                stream_mode,
                elbo_beta,
                # lambda_variance,
                metric_weights,
                anom_if_score_above,
                training_loss,
                lambda_variance,
                trial_files_output.logger
            )

            
            # ---------------------------
            # Run streaming configs
            # ---------------------------

            window_size = seq_len

            stream_stride = suggest_param(trial, "stream_stride", stream_cfg["stride"], ctx)

            stream_shuffle=stream_cfg["shuffle"]
            # update_threshold = training_cfg.get("update_threshold", True)
            stream_max_steps = stream_cfg.get("max_steps", None)
            offline_enabled = training_cfg.get("offline_enabled", True)
            offline_epochs = training_cfg.get("offline_epochs", 40)
            early_stop_patience = training_cfg.get("early_stop_patience", 5)

            ###############################################
            #                                             #
            #               Print Configs                 #
            #                                             #
            ###############################################
            

            # fixed_cfg: dict of non-tuned settings used this trial (device, data meta, etc.)
            # sampled_cfg: dict of all sampled hyperparams for this trial
            evt = {
                "event": "trial_config",
                "trial_number": getattr(trial, "number", None),
                "fixed": {k: to_plain(v) for k, v in fixed_cfg.items()},
                "sampled": {k: to_plain(v) for k, v in chosen_cfg.items()},
            }
            # single JSON line (easy to parse to CSV)
            run_files_output.logger.info_low(lambda: f"{evt}")
            trial_files_output.logger.info_low(lambda: f"{evt}")
            
            ###############################################
            #                                             #
            #                    RUN                      #
            #                                             #
            ###############################################

            anomaly_detection = AnomalyDetection(
                adapter,
                trainer,
                stream_mode,
                window_size,
                stream_stride,
                stream_shuffle,
                stream_max_steps,
                trial_files_output,
                seed,
                offline_enabled=offline_enabled,
                offline_epochs=offline_epochs,
                early_stop_patience=early_stop_patience,
            )

            anomaly_detection.train_warmup()

            metrics_stats = anomaly_detection.run_stream()

            run_files_output.logger.info_low(lambda: format_metrics_stats(metrics_stats, nd=3))
            trial_files_output.logger.info_low(lambda: format_metrics_stats(metrics_stats, nd=3))

            trial_files_output.close()

            
            final = metrics_stats["metrics"]["final"]

            value = final["f1"]
            # store the whole dict for later inspection
            trial.set_user_attr("metrics_stats", metrics_stats)

            # Store common quick fields at top-level
            trial.set_user_attr("final_conf_pr", final.get("conf_pr"))
            trial.set_user_attr("final_conf_roc", final.get("conf_roc"))
            trial.set_user_attr("final_f1", final.get("f1"))
            trial.set_user_attr("final_acc", final.get("accuracy"))

            return value
            
        return objective