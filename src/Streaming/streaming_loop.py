
import time
import numpy as np

from src.Data_tools.dataset_adapter import DatasetAdapter
from src.logging_utils import Logger, TrialFilesObj
from src.Streaming.stream_trainer import StreamTrainer
from src.Streaming.perf_tracker import PerfTracker, STEP_BLOCK_SIZE


class AnomalyDetection:
    def __init__(
        self,
        adapter: DatasetAdapter,                 # DatasetAdapter
        trainer : StreamTrainer,
        mode: str,
        window_size: int,
        stride: int,
        shuffle: bool,
        max_steps: int,
        trial_files: TrialFilesObj,
        seed: int,
        offline_enabled: bool = True,
        offline_epochs: int = 40,
        early_stop_patience: int = 5,
        plot_settings: dict = None,
    ):
        self.adapter = adapter
        self.trainer = trainer
        self.mode = mode
        self.window_size = window_size
        self.stride = stride
        self.shuffle = shuffle
        self.max_steps = max_steps
        self.trial_files = trial_files
        self.seed = seed
        self.offline_enabled = offline_enabled
        self.offline_epochs = offline_epochs
        self.early_stop_patience = early_stop_patience
        self.plot_settings = plot_settings or {}

        self.perf = PerfTracker(device=str(getattr(trainer, "device", "cpu")))
        self.n_stream_steps = 0

        if window_size == 1 :
            self.adapter.preprocess_datasets_in_points()
        else :
            self.adapter.preprocess_datasets_in_windows(window_size, stride, train_on_normal_only=True)

    def train_warmup(self) :
        self.perf.mark("warmup_start")
        self.X_train = self.adapter.train_ds.X
        self.X_val   = self.adapter.val_ds.X

        print("train NaN:", np.isnan(self.X_train).any(), "val NaN:", np.isnan(self.X_val).any())

        # wmse: compute inverse-variance weights from train split (normal data only)
        if getattr(self.trainer.model, "_wmse_weights_ready", None) is not None and \
                not self.trainer.model._wmse_weights_ready:
            import torch
            X_flat = self.X_train.reshape(-1, self.X_train.shape[-1])  # (N, M)
            var = X_flat.var(axis=0).clip(1e-6)                         # (M,)
            weights = torch.tensor(1.0 / var, dtype=torch.float32)
            weights = weights / weights.mean()                           # normalise so mean weight = 1
            self.trainer.model.set_feature_weights(weights)
            print(f"[wmse] feature weights set: min={weights.min():.3f}  max={weights.max():.3f}  mean={weights.mean():.3f}")

        if not self.offline_enabled:
            # All warmup samples will flow through online step() in run_stream().
            return

        # Interleaved path: used by adapters that expose warmup_segments (e.g. FraudAdapter).
        # Walks train/val segments in chronological order; calibrates on train-only (normal).
        # Standard path: used by SWaT/iTrust adapters — both train and val are normal,
        # so calibration on their concatenation is valid.
        warmup_segs = getattr(self.adapter, 'warmup_segments', None)
        if warmup_segs is not None:
            history = self.trainer.offline_fit_interleaved(
                warmup_segs,
                epochs=self.offline_epochs,
                early_stop_patience=self.early_stop_patience,
            )
            X_calib = self.X_train   # train-role windows only (anomaly-free by construction)
        else:
            history = self.trainer.offline_fit(
                self.X_train,
                self.X_val,
                epochs=self.offline_epochs,
                shuffle=self.shuffle,
                seed=self.seed,
                early_stop_patience=self.early_stop_patience,
            )
            if len(self.X_val) > 0:
                X_calib = np.concatenate([self.X_train, self.X_val], axis=0)
            else:
                X_calib = self.X_train

        # Calibrate thresholds via the same online scoring path (learn=False = no grad).
        # This ensures ref_buffer, update_threshold, sigma_prior, and latent stats
        # are populated through the exact same code path as online step().
        for x in X_calib:
            self.trainer.step(t_np=None, x_np=x[np.newaxis], label=None, is_eval=False, learn=False)
        self.trainer.recalibrate_thresholds()

        # Move calibration latents to latent_ref for downstream visualization,
        # then clear evaluation state so calibration samples don't pollute test metrics.
        self.trainer.latent_ref.extend(self.trainer.test_latents)
        self.trainer.reset_eval_state()

        # var_reg trains offline with ELBO then switches to var_reg online, so the
        # calibrated thresholds are based on an ELBO-era model.  Store the tail of
        # the calibration window (definitely-normal data, chronologically closest
        # to the stream) for a deferred re-calibration after the model has had a
        # chance to adapt its var_reg updates.
        if self.trainer.training_loss == "var_reg":
            buf_size = next(iter(self.trainer.metrics.values())).ref_buffer.maxlen
            self._recal_tail = X_calib[-buf_size:]
        else:
            self._recal_tail = None

        self.perf.mark("warmup_end")


    def run_stream(self) :
        # online test then train:
        # If offline is disabled, warmup samples were never trained — stream everything.
        self.perf.mark("stream_start")
        stream = self.adapter.stream_all() if not self.offline_enabled else self.adapter.stream(shuffle=False)
        n_eval = 0
        eval_timestamps = []   # ts_start per eval step, for post-hoc month analysis

        # var_reg deferred re-calibration: fire once after buffer_size eval steps.
        # By that point the model has had buffer_size var_reg online updates; we
        # replay the warmup tail (definitely-normal, chronologically closest to the
        # stream) through the updated model to flush ref_buffer with stream-era
        # normal scores, snap the threshold to the exact new quantile, and discard
        # the stale early predictions made under the ELBO-calibrated threshold.
        # _recal_tail is only set (non-None) when training_loss == "var_reg".
        _recal_tail = getattr(self, "_recal_tail", None)
        _recal_at   = (next(iter(self.trainer.metrics.values())).ref_buffer.maxlen
                       if _recal_tail is not None else 0)

        _t_block = time.perf_counter()
        _n_block = 0

        for sample in stream:
            if sample.is_eval:
                if self.max_steps is not None and n_eval >= self.max_steps:
                    break
                n_eval += 1
                # record the leading timestamp of the window/point
                t = sample.t
                if isinstance(t, np.ndarray) and t.ndim >= 1:
                    eval_timestamps.append(t[0])
                else:
                    eval_timestamps.append(t)
            self.trainer.step(sample.t, sample.x, sample.y, sample.is_eval)

            if _recal_at and n_eval == _recal_at:
                for x in _recal_tail:
                    self.trainer.step(t_np=None, x_np=x[np.newaxis], label=None, is_eval=False, learn=False)
                self.trainer.recalibrate_thresholds()
                self.trainer.reset_eval_state()
                eval_timestamps.clear()
                n_eval = 0
                _recal_at = 0   # fire once only

            if sample.x_raw is not None and self.trainer.last_binary_pred == 0:
                self.adapter.update_scaler(sample.x_raw)

            _n_block += 1
            if _n_block >= STEP_BLOCK_SIZE:
                self.perf.record_block(time.perf_counter() - _t_block, STEP_BLOCK_SIZE)
                _t_block = time.perf_counter()
                _n_block = 0

        if _n_block > 0:
            self.perf.record_block(time.perf_counter() - _t_block, _n_block)

        self.perf.mark("stream_end")
        self.n_stream_steps = n_eval

        metrics_stats = self.trainer.finalize_metrics_stats()
        metrics_weights = {name: metric.weight for name, metric in self.trainer.metrics.items()}

        payload=self.trainer.confusion_stats
        payload["metric_weights"]=metrics_weights
        payload["anom_if_score_above"]=self.trainer.anom_if_score_above
        payload["decoder_sensitivity"]=list(self.trainer.decoder_sensitivity)

        Z_test = np.stack(self.trainer.test_latents)
        labels = np.array(self.trainer.test_labels)
        Z_ref = np.stack(self.trainer.latent_ref, axis=0)   # shape (N_ref, Z)
        latent_mean, latent_cov_inv = self.trainer.compute_latent_reference_stats(Z_ref)

        payload["latent_mu"] = Z_test
        payload["labels"] = labels
        payload["latent_mean"] = latent_mean   # computed during calibration
        payload["latent_cov_inv"] = latent_cov_inv
        payload["ensemble_threshold"] = self.trainer.anom_if_score_above
        payload["timestamps"] = eval_timestamps
        payload["plot_settings"] = self.plot_settings

        # Save latent_mu for post-hoc visualisation (timestamps/labels/scores
        # are recoverable from trial_predictions.csv — see scripts/plot_latent_*.py)
        np.save(self.trial_files.paths.latent_mu_npy, Z_test)


        # after the trial: save trial-specific plots to trial folder
        self.trial_files.logger.visual_event(
            kind="confusions", 
            payload=payload
        )
        self.trial_files.logger.visual_event(
            kind="scores_over_time", 
            payload=payload
        )
        self.trial_files.logger.visual_event(
            kind="2d_3d_display", 
            payload=payload
        )

        
        return metrics_stats
        

        

    