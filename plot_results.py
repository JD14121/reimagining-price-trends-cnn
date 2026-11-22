"""Draw the paired masking example and the AUC sensitivity estimates."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from study import HEIGHT, WIDTH, VOLUME_START, without_volume


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    report = json.loads((args.results / "summary.json").read_text(encoding="utf-8"))
    data = Path(report["settings"]["data"])
    sample = np.memmap(data / "20d_month_has_vb_[20]_ma_2017_images.dat",
                       dtype=np.uint8, mode="r").reshape(-1, HEIGHT, WIDTH)
    full = sample[:1]
    masked = without_volume(full)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.6), layout="constrained")
    for ax, image, title in zip(axes, (full[0], masked[0]), ("Full input", "Volume masked")):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.axhline(VOLUME_START - .5, color="#46a8d8", linestyle="--", linewidth=1.1)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.savefig(args.results / "volume_mask.png", dpi=180)
    plt.close(fig)

    rows = report["results"]
    overall, early, late = dict.fromkeys(r["period"] for r in rows)
    specs = [(overall, "all", f"All stocks, {overall}"),
             (early, "all", f"All stocks, {early}"),
             (late, "all", f"All stocks, {late}"),
             (overall, "small", "Smallest market-cap quartile"),
             (overall, "large", "Largest market-cap quartile"),
             (overall, "small_minus_large", "Small minus large effect")]
    fig, ax = plt.subplots(figsize=(9.2, 4.6), layout="constrained")
    ax.axvline(0, linestyle="--", linewidth=1, color="#556276")
    for pos, (period, group, label) in enumerate(specs):
        row = next(r for r in rows if r["period"] == period and r["group"] == group)
        value = 100 * row["full_minus_no_volume_auc"]
        lo, hi = np.array(row["bootstrap_95"]) * 100
        color = "#004e83" if group in ("all", "small_minus_large") else "#147c72"
        ax.hlines(pos, lo, hi, color=color, linewidth=2)
        ax.scatter(value, pos, color=color, s=38, zorder=3)
    ax.set_yticks(range(len(specs)), [label for _, _, label in specs])
    ax.invert_yaxis()
    ax.set_xlabel("AUC difference, full input minus masked input (percentage points)")
    ax.set_title("Incremental value of the volume band")
    ax.grid(axis="x", color="#dce1e7", alpha=.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.savefig(args.results / "volume_auc.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

