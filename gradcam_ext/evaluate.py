"""Out-of-sample evaluation of the GPU-trained ensembles with the metrics of ../study.py.

Writes results/baseline_summary.json. If no_volume models exist as well, the volume
ablation of ../study.py is re-estimated with the stronger models (same paired block
bootstrap), and results/volume_summary.json is written in the original format.
"""
import json

import numpy as np
import torch

from common import HERE, panel
from study import monthly_scores, summary


def main():
    images, dates, returns, caps, monthly, _ = panel()
    test = np.load(HERE / "predictions" / "test_index.npy")
    y = np.isfinite(returns) & (returns > 0)
    probs, seeds = {}, {}
    for features in ("full", "no_volume"):
        files = sorted((HERE / "predictions").glob(f"{features}_seed*.npy"))
        if files:
            seeds[features] = {f.stem: np.load(f) for f in files}
            probs[features] = np.mean(list(seeds[features].values()), axis=0)
    histories = {p.stem: dict(best_epoch=s["best_epoch"], epochs=len(s["history"]),
                              best_val_loss=min(h["val_loss"] for h in s["history"]),
                              best_val_accuracy=s["history"][s["best_epoch"] - 1]["val_accuracy"])
                 for p in sorted((HERE / "models").glob("*.pt"))
                 for s in [torch.load(p, weights_only=False)]}
    out = {"histories": histories}
    ref = probs["full"]
    pseudo = {"full": ref, "no_volume": probs.get("no_volume", ref),
              "constant": np.full(len(test), y[test].mean())}
    rows = monthly_scores(dates[test], y[test], caps[test], monthly[test], pseudo)
    stats = summary(rows, 2000)
    keep = ("period", "group", "months", "count", "full_auc", "full_accuracy", "full_logloss",
            "full_spread", "full_sharpe", "constant_accuracy", "constant_logloss")
    out["ensemble_full"] = [{k: r[k] for k in keep if k in r} for r in stats if r["group"] != "small_minus_large"]
    out["pooled_accuracy_full"] = float(np.mean((ref > .5) == y[test]))
    out["per_seed"] = {}
    for features, d in seeds.items():
        for tag, p in d.items():
            r = monthly_scores(dates[test], y[test], caps[test], monthly[test],
                               {"full": p, "no_volume": p, "constant": pseudo["constant"]})
            s = summary(r, 2000)[0]
            out["per_seed"][tag] = {k: s[f"full_{k}"] for k in ("auc", "accuracy", "sharpe", "spread")}
    if "no_volume" in probs:
        out["volume_ablation"] = stats
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "baseline_summary.json").write_text(json.dumps(out, indent=2, allow_nan=False))
    print(json.dumps({k: out[k] for k in ("pooled_accuracy_full", "per_seed")}, indent=2))
    print(json.dumps(out["ensemble_full"][0], indent=2))
    if "no_volume" in probs:
        print(json.dumps(stats[0], indent=2))


if __name__ == "__main__":
    main()
