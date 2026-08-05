import numpy as np
from sklearn.metrics import average_precision_score
import optuna

# ---- Reporting helper (with grace period, throttling, and class guards) ----
def make_report_fn(trainer,
                   min_eval_samples,
                   report_every,
                   grace_steps,
                   trial=None,           # can be None for non-optuna runs
                   logger=None):    # optional: log AP somewhere
    started = False
    last_step = 0

    def report_fn(step, out):
        nonlocal started, last_step

        # throttle reporting
        if (step - last_step) < report_every:
            return
        last_step = step

        # grace window before reporting/pruning
        if step < grace_steps:
            return

        labels = np.array(trainer.stats["labels"])
        scores = np.array(trainer.stats["scores"])

        # not enough eval yet
        if labels.size < min_eval_samples:
            return

        # must have both classes for AP
        if labels.max() == 0 or labels.min() == 1:
            return

        # first valid moment: arm but don’t report yet (stabilize once)
        if not started:
            started = True
            return

        pr = float(average_precision_score(labels, scores))

        # ---------- branch: Optuna vs non-Optuna ----------
        if trial is not None:
            # classic Optuna reporting + pruning
            trial.report(pr, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        else:
            # non-Optuna run: maybe just log it
            if logger is not None:
                logger.info_low(lambda: f"[report] step={step} AP={pr:.4f}")

    return report_fn


def suggest_param(trial, name, spec, ctx):
    if not isinstance(spec, dict):
        return spec  # plain scalar — treat as fixed, no sampling
    t = spec["type"]

    step = spec.get("step", None)

    if t == "int":
        if step is None:
            return trial.suggest_int(name, spec["low"], spec["high"])
        else:
            return trial.suggest_int(name, spec["low"], spec["high"], step=step)
    if t == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    if t == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if t.endswith("_list"):
        # Determine list length
        count = spec.get("count")
        if count is None:
            cp = spec.get("count_param")
            if cp is None:
                raise KeyError(
                    f"{name}: list spec must provide either 'count' or 'count_param'. "
                    f"Spec was: {spec}"
                )
            if cp not in ctx:
                raise KeyError(
                    f"{name}: count_param '{cp}' not found in ctx yet. "
                    f"Ensure '{cp}' is defined in 'common' and suggested BEFORE any *_list "
                    f"that depends on it."
                )
            count = int(ctx[cp])
        # Build inner items using key_fmt
        key_fmt = spec["key_fmt"]  # e.g., "enc_w{idx}"
        inner_type = t.replace("_list", "")
        vals = []
        for idx in range(count):
            inner = dict(spec)
            inner["type"] = inner_type
            inner.pop("count", None)
            inner.pop("count_param", None)
            inner_name = key_fmt.format(idx=idx)
            vals.append(suggest_param(trial, inner_name, inner, ctx))
        return vals

    raise ValueError(f"Unknown spec type: {t}")

def suggest_from_space(trial, space, model_key):
    ctx = {}
    def add_block(block):
        for name, spec in block.items():
            # 👇 NEW: skip suggest_param for constants / metadata
            if not (isinstance(spec, dict) and "type" in spec):
                ctx[name] = spec  # constant like epochs: 10
                continue
            ctx[name] = suggest_param(trial, name, spec, ctx)

    # 1) common first (so count_param exists)
    add_block(space.get("common", {}))
    
    # 2) then model-specific
    if model_key:
        if model_key:
            add_block(space.get("models", {}).get(model_key, {}))
    
    # 3) train first (so count_param exists)
    add_block(space.get("train", {}))
    return ctx
