"""
Lightweight runtime and GPU-memory tracker for one streaming run.

Writes two outputs when finalize() is called:
  _perf_summary.json  — machine-readable, parsed by cross_compare.py
  run.log block       — human-readable PERFORMANCE section
"""

import os
import json
import time
import numpy as np

STEP_BLOCK_SIZE = 200  # steps per timing block (reduces perf_counter overhead)


class PerfTracker:

    def __init__(self, device: str):
        self.device = str(device)
        self._is_cuda = self.device.startswith("cuda")
        self._marks: dict[str, float] = {}
        self._step_block_times: list[float] = []

        if self._is_cuda:
            try:
                import torch
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception:
                pass

    def mark(self, name: str) -> None:
        if self._is_cuda:
            try:
                import torch
                torch.cuda.synchronize()
            except Exception:
                pass
        self._marks[name] = time.perf_counter()

    def elapsed(self, a: str, b: str) -> float:
        return max(0.0, self._marks.get(b, 0.0) - self._marks.get(a, 0.0))

    def record_block(self, elapsed_s: float, n_steps: int) -> None:
        """Record wall time for a contiguous block of n_steps."""
        if n_steps > 0:
            self._step_block_times.append(elapsed_s / n_steps)

    def _gpu_peak_mb(self) -> float:
        if not self._is_cuda:
            return 0.0
        try:
            import torch
            return torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
        except Exception:
            return 0.0

    def _gpu_reserved_mb(self) -> float:
        if not self._is_cuda:
            return 0.0
        try:
            import torch
            return torch.cuda.max_memory_reserved(self.device) / 1024 ** 2
        except Exception:
            return 0.0

    def finalize(
        self,
        trial_dir: str,
        logger,
        n_warmup_samples: int,
        n_stream_steps: int,
        n_params: int,
    ) -> dict:
        warmup_s = self.elapsed("warmup_start", "warmup_end")
        stream_s = self.elapsed("stream_start", "stream_end")
        total_s  = warmup_s + stream_s

        warmup_sps = n_warmup_samples / warmup_s if warmup_s > 0 else 0.0
        stream_sps = n_stream_steps   / stream_s if stream_s > 0 else 0.0

        gpu_peak_mb     = self._gpu_peak_mb()
        gpu_reserved_mb = self._gpu_reserved_mb()

        summary: dict = {
            "total_s":                round(total_s,   2),
            "warmup_s":               round(warmup_s,  2),
            "stream_s":               round(stream_s,  2),
            "n_warmup_samples":       n_warmup_samples,
            "n_stream_steps":         n_stream_steps,
            "warmup_samples_per_sec": round(warmup_sps, 1),
            "stream_steps_per_sec":   round(stream_sps, 1),
            "n_params":               n_params,
            "gpu_peak_mb":            round(gpu_peak_mb,     1),
            "gpu_reserved_mb":        round(gpu_reserved_mb, 1),
        }

        if self._step_block_times:
            arr = np.array(self._step_block_times) * 1_000  # s → ms
            summary["step_ms_mean"] = round(float(arr.mean()),             3)
            summary["step_ms_p50"]  = round(float(np.percentile(arr, 50)), 3)
            summary["step_ms_p99"]  = round(float(np.percentile(arr, 99)), 3)
            summary["step_ms_max"]  = round(float(arr.max()),              3)

        try:
            with open(os.path.join(trial_dir, "_perf_summary.json"), "w") as fh:
                json.dump(summary, fh, indent=2)
        except Exception:
            pass

        try:
            if logger is not None:
                n_blk = len(self._step_block_times)
                lines = [
                    "─" * 60,
                    "PERFORMANCE",
                    f"  total          {total_s:>8.1f} s",
                    f"  warmup         {warmup_s:>8.1f} s   ({n_warmup_samples:,} samples,  {warmup_sps:,.0f} samp/s)",
                    f"  streaming      {stream_s:>8.1f} s   ({n_stream_steps:,} steps,   {stream_sps:,.0f} steps/s)",
                    f"  n_params       {n_params:>13,}",
                    f"  gpu_peak_mem   {gpu_peak_mb:>8.1f} MB",
                    f"  gpu_reserved   {gpu_reserved_mb:>8.1f} MB",
                ]
                if self._step_block_times:
                    arr = np.array(self._step_block_times) * 1_000
                    lines.append(
                        f"  step_ms ({n_blk:>4d} blks)  "
                        f"mean={arr.mean():.3f}  p50={np.percentile(arr,50):.3f}"
                        f"  p99={np.percentile(arr,99):.3f}  max={arr.max():.3f}"
                    )
                lines.append("─" * 60)
                logger.info_low("\n".join(lines))
        except Exception:
            pass

        return summary
