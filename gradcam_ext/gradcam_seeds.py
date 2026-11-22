"""Robustness of the Grad-CAM conclusions across the five independently trained CNNs.

Per-image Grad-CAM maps of different seeds turned out to be nearly uncorrelated, so this
script asks whether the *class-level* regularities survive: for every seed it explains that
seed's own predicted class on the same 2019 sample and recomputes the summary statistics
and the deletion (faithfulness) test.

Outputs: results/gradcam_seeds.json and results/gradcam_seeds.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import HERE, load_model, panel, take_images  # noqa: E402
from gradcam import deletion_test, gradcam, image_stats  # noqa: E402

KEYS = ("share_upper_half", "centroid_minus_ink", "share_volume", "ink_lift", "open_lift", "bar_lift",
        "close_lift", "last5_share", "corr_range_cam", "corr_volume_price_cam")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=6000)
    parser.add_argument("--deletion-n", type=int, default=2000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--plot-only", action="store_true", help="redraw from results/gradcam_seeds.json")
    args = parser.parse_args()
    if args.plot_only:
        return plot(json.loads((HERE / "results" / "gradcam_seeds.json").read_text()))
    device = torch.device(args.device)
    images, dates, returns, *_ = panel()
    year = dates.astype("datetime64[Y]").astype(int) + 1970
    pool = np.flatnonzero((year == 2019) & np.isfinite(returns))
    rows = np.sort(np.random.default_rng(77).choice(pool, args.n, replace=False))
    raw = take_images(images, rows)

    report = {}
    for path in sorted((HERE / "models").glob("full_seed*.pt")):
        model, state = load_model(path, device)
        p, maps = gradcam(model, raw, state["center"], state["scale"], device)
        up = p > .5
        stats = image_stats(raw, maps)
        entry = dict(share_classified_up=float(up.mean()))
        for layer in range(3):
            s = dict(stats[layer])
            s["centroid_minus_ink"] = s["cam_centroid_row"] - s["ink_centroid_row"]
            for key in KEYS:
                v = s[key]
                entry[f"layer{layer + 1}_{key}"] = {
                    "up": float(np.nanmean(v[up])), "down": float(np.nanmean(v[~up])),
                    "up_minus_down": float(np.nanmean(v[up]) - np.nanmean(v[~up])),
                    "up_minus_down_se": float(np.sqrt(np.nanvar(v[up]) / up.sum() + np.nanvar(v[~up]) / (~up).sum()))}
        sub = slice(0, args.deletion_n)
        entry["deletion"] = deletion_test(model, state, raw[sub], maps[:, sub], up[sub].astype(int), device,
                                          fractions=(.05, .2))
        report[path.stem] = entry
        print(path.stem, f"up={up.mean():.3f}",
              {f"L{l}": round(entry[f"layer{l}_share_upper_half"]["up_minus_down"], 4) for l in (1, 2, 3)},
              flush=True)
    (HERE / "results" / "gradcam_seeds.json").write_text(json.dumps(report, indent=2))
    plot(report)


def plot(report):
    seeds = list(report)
    panels = [("share_upper_half", "Upper-half share of price-area CAM,\nup minus down"),
              ("centroid_minus_ink", "CAM centroid minus ink centroid (rows),\nup minus down (negative = higher)"),
              ("close_lift", "Close-tick lift,\nup minus down"),
              ("corr_range_cam", "Within-image corr(high-low range, CAM),\nup minus down")]
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(17, 3.8), layout="constrained")
    x = np.arange(3)
    width = .8 / len(seeds)
    colors = plt.cm.viridis(np.linspace(.1, .85, len(seeds)))
    for ax, (key, title) in zip(axes, panels):
        for i, seed in enumerate(seeds):
            vals = [report[seed][f"layer{l}_{key}"]["up_minus_down"] for l in (1, 2, 3)]
            errs = [1.96 * report[seed][f"layer{l}_{key}"]["up_minus_down_se"] for l in (1, 2, 3)]
            ax.bar(x + (i - (len(seeds) - 1) / 2) * width, vals, width, yerr=errs, color=colors[i],
                   label=seed.replace("full_", ""), capsize=1.5, error_kw=dict(lw=.6))
        ax.axhline(0, color="#444", lw=.8)
        ax.set_xticks(x, ["Layer 1", "Layer 2", "Layer 3"])
        ax.set_title(title, fontsize=9)
    ax = axes[-1]
    for i, seed in enumerate(seeds):
        d = report[seed]["deletion"]["0.20"]
        vals = [d[f"layer{l}"][0] - d["random"][0] for l in (1, 2, 3)]
        ax.bar(x + (i - (len(seeds) - 1) / 2) * width, vals, width, color=colors[i])
    ax.axhline(0, color="#444", lw=.8)
    ax.set_xticks(x, ["Layer 1", "Layer 2", "Layer 3"])
    ax.set_title("Deletion of 20% of lit pixels:\nCAM-ranked minus random margin drop", fontsize=9)
    axes[0].legend(fontsize=7, ncol=1)
    fig.savefig(HERE / "results" / "gradcam_seeds.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
