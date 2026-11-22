"""Grad-CAM for the I20/R20 CNN (Selvaraju et al., 2017), following Section 6.3 of the
working-paper version of Jiang, Kelly and Xiu, plus quantitative checks of its claims.

For class c with logit z_c and block-l activation A (after LeakyReLU, before pooling):
    alpha_k = mean_{h,w} dz_c / dA_k,      CAM_l = ReLU(sum_k alpha_k A_k)
Each CAM_l is mapped onto the 64x60 input grid using the exact receptive-field centres of
the feature-map rows (common.ROW_CENTRES) rather than a plain image resize.

Run: python gradcam.py            (needs models/full_seed*.pt from train_baseline.py)
Outputs go to results/: figures, gradcam_stats.json and the per-image table gradcam_images.csv.
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from common import (HERE, HEIGHT, I20CNN, VOLUME_START, WIDTH, load_model, normalise,  # noqa: E402
                    panel, take_images, to_input_grid)

CMAP = LinearSegmentedColormap.from_list("cam", ["#000000", "#08306b", "#2171b5", "#9ecae1", "#ffffff"])
LAYERS = 3
PRICE_END = 51


def gradcam(model, raw, center, scale, device, target=None, batch=256):
    """Return (prob_up, maps) with maps of shape (LAYERS, N, 64, 60) on the input grid.
    target: per-image class to explain (default: the model's predicted class)."""
    probs, maps = [], [[] for _ in range(LAYERS)]
    model.eval()
    for start in range(0, len(raw), batch):
        x = normalise(raw[start:start + batch], center, scale, device)
        logits, acts = model(x, keep=True)
        for a in acts:
            a.retain_grad()
        p = logits.detach().softmax(1)[:, 1]
        cls = (p > .5).long() if target is None else torch.as_tensor(
            target[start:start + batch], device=device).long()
        model.zero_grad(set_to_none=True)
        logits.gather(1, cls[:, None]).sum().backward()
        for layer, a in enumerate(acts):
            alpha = a.grad.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((alpha * a).sum(1)).detach().cpu().numpy()
            maps[layer].append(to_input_grid(cam, layer))
        probs.append(p.cpu().numpy())
    return np.concatenate(probs), np.stack([np.concatenate(m) for m in maps]).astype(np.float32)


@torch.no_grad()
def margin(model, raw, center, scale, device, cls, batch=1024):
    """Logit margin of class cls over the other class."""
    out = []
    for start in range(0, len(raw), batch):
        z = model(normalise(raw[start:start + batch], center, scale, device)).float()
        c = torch.as_tensor(cls[start:start + batch], device=device).long()
        out.append((z.gather(1, c[:, None])[:, 0] - z.gather(1, 1 - c[:, None])[:, 0]).cpu().numpy())
    return np.concatenate(out)


def unit(maps):
    """Scale each map to [0, 1] for display."""
    hi = maps.max(axis=(-2, -1), keepdims=True)
    return np.divide(maps, hi, out=np.zeros_like(maps), where=hi > 0)


def mass(maps):
    """Normalise each map to a probability distribution over pixels."""
    total = maps.sum(axis=(-2, -1), keepdims=True)
    return np.divide(maps, total, out=np.zeros_like(maps), where=total > 0)


# --------------------------------------------------------------------------- figures
def example_figure(raw, maps, title, path):
    n = len(raw)
    fig, axes = plt.subplots(1 + LAYERS, n, figsize=(1.25 * n, 1.32 * (1 + LAYERS)))
    shown = unit(maps)
    for j in range(n):
        axes[0, j].imshow(raw[j], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        for layer in range(LAYERS):
            axes[1 + layer, j].imshow(shown[layer, j], cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    for i, label in enumerate(["Image", "Layer 1", "Layer 2", "Layer 3"]):
        axes[i, 0].set_ylabel(label, fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(pad=.25, h_pad=.15, w_pad=.15, rect=(0, 0, 1, .94))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def average_figure(raw, maps, up, path):
    """Mean ink and mean CAM mass for 'up' vs 'down' classified images, and their difference."""
    ink = (raw > 0).astype(np.float32)
    rows = [("Ink density", ink)] + [(f"Layer {l + 1} CAM", mass(maps[l]) * HEIGHT * WIDTH)
                                     for l in range(LAYERS)]
    fig, axes = plt.subplots(len(rows), 3, figsize=(6.6, 2.0 * len(rows)), layout="constrained")
    for i, (name, value) in enumerate(rows):
        a, b = value[up].mean(0), value[~up].mean(0)
        hi = max(a.max(), b.max())
        for j, (img, label) in enumerate(((a, "'Up' images"), (b, "'Down' images"))):
            axes[i, j].imshow(img, cmap=CMAP, vmin=0, vmax=hi, interpolation="nearest")
            axes[i, j].set_title(f"{name}\n{label}", fontsize=8)
        d = a - b
        lim = np.abs(d).max()
        im = axes[i, 2].imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        axes[i, 2].set_title(f"{name}\nUp minus down", fontsize=8)
        fig.colorbar(im, ax=axes[i, 2], shrink=.8)
        for ax in axes[i]:
            ax.axhline(VOLUME_START - 1, color="#e6a700", lw=.6, ls="--")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.savefig(path, dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------- statistics
def image_stats(raw, maps):
    """Per-image, per-layer attribution statistics (all shares are of total CAM mass)."""
    lit = raw > 0
    n = len(raw)
    rows = np.arange(HEIGHT, dtype=float)[None, :, None]
    price_lit = lit[:, :PRICE_END]
    ink_rows = (price_lit * rows[:, :PRICE_END]).sum((1, 2)) / np.maximum(price_lit.sum((1, 2)), 1)
    # Per-day quantities: OHLC column roles inside the price area (day d uses columns 3d..3d+2).
    hl_range = lit[:, :PRICE_END, 1::3].sum(1)                  # (n, 20) high-low bar length
    volume = lit[:, VOLUME_START:, 1::3].sum(1)                 # (n, 20) volume bar height
    out = {}
    for layer in range(LAYERS):
        p = mass(maps[layer].astype(np.float64))
        pp = p[:, :PRICE_END]
        price_total = np.maximum(pp.sum((1, 2)), 1e-12)
        day_price = pp.reshape(n, PRICE_END, 20, 3).sum((1, 3))  # CAM mass per day, price area
        day_volume = p[:, VOLUME_START:].reshape(n, HEIGHT - VOLUME_START, 20, 3).sum((1, 3))

        def lift(col):
            """Mean CAM mass per lit pixel of one OHLC column role (0 open, 1 high-low bar,
            2 close) in the price area, relative to a uniform map (1 = no emphasis)."""
            m = lit[:, :PRICE_END, col::3]
            return HEIGHT * WIDTH * (pp[:, :, col::3] * m).sum((1, 2)) / np.maximum(m.sum((1, 2)), 1)

        def within_corr(a, b):
            res = np.full(n, np.nan)
            for i in range(n):
                if np.ptp(a[i]) > 0 and np.ptp(b[i]) > 0:
                    res[i] = spearmanr(a[i], b[i])[0]
            return res

        out[layer] = dict(
            share_price=p[:, :PRICE_END].sum((1, 2)),
            share_volume=p[:, VOLUME_START:].sum((1, 2)),
            share_on_ink=(p * lit).sum((1, 2)),
            ink_fraction=lit.mean((1, 2)),
            share_upper_half=pp[:, :PRICE_END // 2 + 1].sum((1, 2)) / price_total,
            cam_centroid_row=(pp * rows[:, :PRICE_END]).sum((1, 2)) / price_total,
            ink_centroid_row=ink_rows,
            open_lift=lift(0),
            bar_lift=lift(1),
            close_lift=lift(2),
            ink_lift=HEIGHT * WIDTH * (p * lit).sum((1, 2)) / np.maximum(lit.sum((1, 2)), 1),
            last5_share=day_price[:, -5:].sum(1) / price_total,
            corr_range_cam=within_corr(hl_range, day_price),
            corr_volume_price_cam=within_corr(volume, day_price),
            corr_volume_volume_cam=within_corr(volume, day_volume),
            day_profile=day_price / price_total[:, None],
        )
    return out


def summarise(stats, up):
    def ci(x):
        x = x[np.isfinite(x)]
        return [float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))]

    report = {}
    for layer, s in stats.items():
        entry = {}
        for key, value in s.items():
            if key == "day_profile":
                entry[key] = {"up": value[up].mean(0).tolist(), "down": value[~up].mean(0).tolist()}
                continue
            entry[key] = {"all": ci(value), "up": ci(value[up]), "down": ci(value[~up])}
        offset = s["cam_centroid_row"] - s["ink_centroid_row"]
        entry["centroid_minus_ink"] = {"all": ci(offset), "up": ci(offset[up]), "down": ci(offset[~up])}
        report[f"layer{layer + 1}"] = entry
    return report


def deletion_test(model, state, raw, maps, cls, device, fractions=(.02, .05, .1, .2), draws=3):
    """Erase the lit pixels with the highest CAM (or random lit pixels) and measure the drop in
    the explained class's logit margin. Faithful maps should beat random deletion."""
    rng = np.random.default_rng(7)
    c, s = state["center"], state["scale"]
    base = margin(model, raw, c, s, device, cls)
    lit = raw > 0
    noise = rng.random(raw.shape)  # tie-breaker so equal CAM values are not ordered by position
    result = {"base_margin": float(base.mean())}
    rankings = {f"layer{l + 1}": maps[l] + 1e-9 * noise for l in range(LAYERS)}
    for frac in fractions:
        row = {}
        for name, score in list(rankings.items()) + [("random", None)]:
            drops = []
            for d in range(1 if score is not None else draws):
                order = rng.random(raw.shape) if score is None else score
                key = np.where(lit, order, -np.inf).reshape(len(raw), -1)
                k = np.maximum(1, np.round(frac * lit.reshape(len(raw), -1).sum(1))).astype(int)
                thresh = -np.sort(-key, axis=1)[np.arange(len(raw)), k - 1]
                erased = raw.copy()
                erased[(key >= thresh[:, None]).reshape(raw.shape) & lit] = 0
                drops.append(base - margin(model, erased, c, s, device, cls))
            drops = np.mean(drops, axis=0)
            row[name] = [float(drops.mean()), float(1.96 * drops.std(ddof=1) / np.sqrt(len(drops)))]
        result[f"{frac:.2f}"] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--stats-n", type=int, default=20000, help="images for the statistics")
    parser.add_argument("--deletion-n", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=1, help="model whose maps are drawn")
    parser.add_argument("--models", type=Path, default=HERE / "models")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    out = args.out
    out.mkdir(exist_ok=True)
    model_paths = sorted(args.models.glob("full_seed*.pt"))
    model, state = load_model(args.models / f"full_seed{args.seed}.pt", device)

    images, dates, returns, caps, monthly, _ = panel()
    year = dates.astype("datetime64[Y]").astype(int) + 1970
    pool = np.flatnonzero((year == args.year) & np.isfinite(returns))
    rng = np.random.default_rng(2019)
    raw_all = take_images(images, pool)
    with torch.no_grad():
        prob = np.concatenate([model(normalise(raw_all[i:i + 1024], state["center"], state["scale"],
                                                device)).softmax(1)[:, 1].cpu().numpy()
                               for i in range(0, len(raw_all), 1024)])
    up_rows, down_rows = np.flatnonzero(prob > .5), np.flatnonzero(prob <= .5)
    print(f"{args.year}: {len(pool)} images, classified up={len(up_rows)} down={len(down_rows)}", flush=True)

    # 1. Replica of Figure 13: ten random 'up' and ten random 'down' images.
    for name, rows in (("up", up_rows), ("down", down_rows)):
        pick = rng.choice(rows, 10, replace=False)
        _, maps = gradcam(model, raw_all[pick], state["center"], state["scale"], device)
        example_figure(raw_all[pick], maps, f"I20/R20 Grad-CAM, {args.year}: ten random images classified "
                       f"'{name}'", out / f"gradcam_examples_{name}.png")
    # Most confident examples (top/bottom of the predicted-probability distribution).
    for name, pick in (("most_up", np.argsort(-prob)[:10]), ("most_down", np.argsort(prob)[:10])):
        _, maps = gradcam(model, raw_all[pick], state["center"], state["scale"], device)
        example_figure(raw_all[pick], maps, f"I20/R20 Grad-CAM, {args.year}: ten most confident "
                       f"'{name.split('_')[1]}' predictions", out / f"gradcam_examples_{name}.png")

    # 2. Statistics on a large random sample of the year.
    sample = np.sort(rng.choice(len(pool), min(args.stats_n, len(pool)), replace=False))
    raw = raw_all[sample]
    p, maps = gradcam(model, raw, state["center"], state["scale"], device)
    up = p > .5
    average_figure(raw, maps, up, out / "gradcam_average_maps.png")
    stats = image_stats(raw, maps)
    report = dict(year=args.year, model=f"full_seed{args.seed}", images=int(len(raw)),
                  classified_up=int(up.sum()), layers=summarise(stats, up))
    # Confident tails only (top and bottom deciles of predicted probability).
    lo, hi = np.quantile(p, [.1, .9])
    tail = (p <= lo) | (p >= hi)
    report["confident_tails"] = summarise({k: {n: v[tail] for n, v in s.items()} for k, s in stats.items()},
                                          up[tail])
    with (out / "gradcam_images.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        keys = [k for k in stats[0] if k != "day_profile"]
        writer.writerow(["pool_row", "prob_up"] + [f"l{l + 1}_{k}" for l in range(LAYERS) for k in keys])
        for i in range(len(raw)):
            writer.writerow([int(pool[sample[i]]), float(p[i])] +
                            [float(stats[l][k][i]) for l in range(LAYERS) for k in keys])

    # 3. Faithfulness: deletion test on a sub-sample, explaining the predicted class.
    sub = slice(0, args.deletion_n)
    report["deletion"] = deletion_test(model, state, raw[sub], maps[:, sub], up[sub].astype(int), device)

    # 4. Sanity and stability: maps from other seeds and from an untrained network.
    check = slice(0, 2000)
    def corr(a, b):
        a = a.reshape(len(a), -1)
        b = b.reshape(len(b), -1)
        a = a - a.mean(1, keepdims=True)
        b = b - b.mean(1, keepdims=True)
        den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
        return float(np.nanmean(np.divide((a * b).sum(1), den, out=np.full(len(a), np.nan), where=den > 0)))

    target = up[check].astype(int)
    stability = {}
    for path in model_paths:
        if path.stem == f"full_seed{args.seed}":
            continue
        other, st = load_model(path, device)
        _, m = gradcam(other, raw[check], st["center"], st["scale"], device, target=target)
        stability[path.stem] = [corr(maps[l, check], m[l]) for l in range(LAYERS)]
    torch.manual_seed(0)
    untrained = I20CNN().to(device).eval()
    _, m = gradcam(untrained, raw[check], state["center"], state["scale"], device, target=target)
    ink = (raw[check] > 0).astype(np.float32)
    report["stability"] = dict(
        other_seeds=stability,
        untrained_network=[corr(maps[l, check], m[l]) for l in range(LAYERS)],
        trained_vs_ink=[corr(maps[l, check], ink) for l in range(LAYERS)],
        untrained_vs_ink=[corr(m[l], ink) for l in range(LAYERS)],
    )
    (out / "gradcam_stats.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("deletion", "stability")}, indent=2))


if __name__ == "__main__":
    main()
