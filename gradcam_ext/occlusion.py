"""Inference-time region occlusion for the trained I20/R20 ensemble (no retraining).

Each condition blacks out one region of the input image (sets its pixels to the background
value 0) and re-scores the same test images with the same trained models. Comparing with the
unoccluded predictions shows which regions the fitted model actually relies on, and so tests
what the Grad-CAM maps suggest. Occluded images lie outside the training distribution, so the
changes measure the fitted model's sensitivity, not the value of retraining without a region
(the retrained no-volume ensemble in train_baseline.py answers that question).

Sample: a fixed stratified draw of up to --per-month test images in every month 2000-2019.
Outputs: results/occlusion_summary.json, results/occlusion_monthly.csv, results/occlusion.png
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from common import (HERE, HEIGHT, VOLUME_START, load_model, normalise, paired_split, panel,  # noqa: E402
                    take_images)
from study import auc, block_interval  # noqa: E402

PRICE_END = 51


def day_columns(days):
    return np.concatenate([np.arange(3 * d, 3 * d + 3) for d in days])


def occluders():
    """Name -> function(raw uint8 batch) returning an occluded copy."""
    def rows(r0, r1):
        def f(x):
            x = x.copy()
            x[:, r0:r1] = 0
            return x
        return f

    def cols(c, r0=0, r1=HEIGHT):
        def f(x):
            x = x.copy()
            x[:, r0:r1, c] = 0
            return x
        return f

    def by_range(kind, k=5):
        """Occlude the k days with the widest (or narrowest, or random) high-low bar, all rows."""
        rng = np.random.default_rng(11)

        def f(x):
            hl = (x[:, :PRICE_END, 1::3] > 0).sum(1).astype(float)          # (n, 20)
            hl += 1e-3 * rng.random(hl.shape)                                  # break ties
            if kind == "random":
                score = rng.random(hl.shape)
            else:
                score = hl if kind == "widest" else -hl
            days = np.argsort(-score, axis=1)[:, :k]
            mask = np.zeros((len(x), 60), bool)
            for j in range(3):
                np.put_along_axis(mask, 3 * days + j, True, axis=1)
            x = x.copy()
            x[np.broadcast_to(mask[:, None, :], x.shape)] = 0
            return x
        return f

    half = PRICE_END // 2 + 1
    return {
        "volume band": rows(VOLUME_START, HEIGHT),
        "price area (OHLC + MA)": rows(0, PRICE_END),
        "upper half of price area": rows(0, half),
        "lower half of price area": rows(half, PRICE_END),
        "days 1-5 (oldest)": cols(day_columns(range(0, 5))),
        "days 6-10": cols(day_columns(range(5, 10))),
        "days 11-15": cols(day_columns(range(10, 15))),
        "days 16-20 (most recent)": cols(day_columns(range(15, 20))),
        "open-tick columns": cols(np.arange(0, 60, 3), 0, PRICE_END),
        "high-low bar columns": cols(np.arange(1, 60, 3), 0, PRICE_END),
        "close-tick columns": cols(np.arange(2, 60, 3), 0, PRICE_END),
        "5 widest-range days": by_range("widest"),
        "5 narrowest-range days": by_range("narrowest"),
        "5 random days": by_range("random"),
    }


@torch.no_grad()
def ensemble_prob(models, raw, device, batch=2048):
    out = np.zeros(len(raw), dtype=np.float64)
    for model, state in models:
        for start in range(0, len(raw), batch):
            x = normalise(raw[start:start + batch], state["center"], state["scale"], device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                z = model(x).float()
            out[start:start + batch] += z.softmax(1)[:, 1].cpu().numpy()
    return out / len(models)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-month", type=int, default=500)
    parser.add_argument("--models", type=Path, default=HERE / "models")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    args.out.mkdir(exist_ok=True)
    paths = sorted(args.models.glob("full_seed*.pt"))
    models = [load_model(p, device) for p in paths]
    print(f"models: {[p.stem for p in paths]}", flush=True)

    images, dates, returns, caps, monthly, _ = panel()
    # Same split as train_baseline.py (seed 390); recomputed so no saved predictions are needed.
    _, _, test = paired_split(dates, returns, 2000, 390, 0)
    months = dates[test].astype("datetime64[M]")
    rng = np.random.default_rng(2024)
    sample = np.sort(np.concatenate([
        rng.choice(rows, min(len(rows), args.per_month), replace=False)
        for m in np.unique(months) for rows in [np.flatnonzero(months == m)]]))
    rows_ = test[sample]
    raw = take_images(images, rows_)
    y = returns[rows_] > 0
    month = months[sample]
    print(f"sample: {len(raw)} images in {len(np.unique(month))} months", flush=True)

    base = ensemble_prob(models, raw, device)
    results, monthly_rows = {}, []
    ink = (raw > 0).sum((1, 2))
    for name, occlude in [("none", None)] + list(occluders().items()):
        x = raw if occlude is None else occlude(raw)
        p = base if occlude is None else ensemble_prob(models, x, device)
        removed = 1 - (x > 0).sum((1, 2)) / np.maximum(ink, 1)
        per_month = []
        for m in np.unique(month):
            i = month == m
            if y[i].any() and not y[i].all():
                per_month.append((str(m), auc(y[i], p[i]), auc(y[i], base[i]),
                                  spearmanr(p[i], base[i])[0]))
        auc_occ = np.array([r[1] for r in per_month])
        auc_base = np.array([r[2] for r in per_month])
        delta = auc_occ - auc_base
        results[name] = dict(
            ink_removed=float(removed.mean()),
            mean_prob_up=float(p.mean()),
            mean_shift_prob_up=float((p - base).mean()),
            mean_abs_shift=float(np.abs(p - base).mean()),
            class_flip_rate=float(np.mean((p > .5) != (base > .5))),
            rank_corr_with_unoccluded=float(np.mean([r[3] for r in per_month])),
            auc=float(auc_occ.mean()),
            delta_auc=float(delta.mean()),
            delta_auc_95=block_interval(delta) if name != "none" else [0.0, 0.0],
        )
        for r in per_month:
            monthly_rows.append(dict(condition=name, month=r[0], auc=r[1], base_auc=r[2], rank_corr=r[3]))
        print(f"{name:28s} dAUC={100 * delta.mean():+.3f}pp  shift={results[name]['mean_shift_prob_up']:+.4f} "
              f"flip={results[name]['class_flip_rate']:.3f}  rho={results[name]['rank_corr_with_unoccluded']:.3f}",
              flush=True)

    (args.out / "occlusion_summary.json").write_text(json.dumps(dict(
        models=[p.stem for p in paths], images=int(len(raw)), per_month=args.per_month,
        base_auc=results["none"]["auc"], conditions=results), indent=2))
    with (args.out / "occlusion_monthly.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(monthly_rows[0]))
        w.writeheader()
        w.writerows(monthly_rows)

    names = [n for n in results if n != "none"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True, layout="constrained")
    pos = np.arange(len(names))
    d = np.array([100 * results[n]["delta_auc"] for n in names])
    ci = np.array([[100 * v for v in results[n]["delta_auc_95"]] for n in names])
    axes[0].hlines(pos, ci[:, 0], ci[:, 1], color="#1f5f99", lw=2)
    axes[0].scatter(d, pos, color="#1f5f99", zorder=3)
    axes[0].axvline(0, color="#666", ls="--", lw=1)
    axes[0].set_xlabel("Change in monthly AUC (pp), 95% block bootstrap")
    shift = [100 * results[n]["mean_shift_prob_up"] for n in names]
    axes[1].barh(pos, shift, color=["#b2182b" if s > 0 else "#2166ac" for s in shift])
    axes[1].axvline(0, color="#666", lw=1)
    axes[1].set_xlabel("Mean shift in P(up) (pp)")
    rho = [results[n]["rank_corr_with_unoccluded"] for n in names]
    axes[2].barh(pos, rho, color="#4d4d4d")
    axes[2].set_xlim(min(rho) - .05, 1)
    axes[2].set_xlabel("Rank corr. with unoccluded forecast")
    axes[0].set_yticks(pos, names)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis="x", color="#ddd")
        ax.set_axisbelow(True)
    fig.suptitle(f"Region occlusion of the I20/R20 ensemble ({len(paths)} models, {len(raw):,} test images, "
                 f"base AUC {results['none']['auc']:.4f})")
    fig.savefig(args.out / "occlusion.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
