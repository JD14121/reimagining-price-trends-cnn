import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import torch
from scipy.stats import rankdata
from torch import nn

HEIGHT, WIDTH = 64, 60
VOLUME_START = 52
FEATURES = ("full", "no_volume")
YEARS = range(1993, 2020)


def without_volume(images):
    masked = images.copy()
    masked[:, VOLUME_START:] = 0
    return masked


def paired_split(dates, returns, test_year, seed, per_month):
    rng = np.random.default_rng(seed)
    first_test = np.datetime64(f"{test_year}-01", "M")
    months = dates.astype("datetime64[M]")
    finite = np.isfinite(returns)
    development = np.flatnonzero(finite & (months < first_test - 1))
    if per_month:
        development = np.concatenate([
            rng.choice(rows, min(len(rows), per_month), replace=False)
            for month in np.unique(months[development])
            for rows in [development[months[development] == month]]
        ])
    development = rng.permutation(development)
    n_train = int(len(development) * .70)
    return (development[:n_train], development[n_train:],
            np.flatnonzero(finite & (months >= first_test)))


def load_panel(path):
    dates, returns, caps, monthly, images, audit = [], [], [], [], [], []
    for year in YEARS:
        prefix = path / f"20d_month_has_vb_[20]_ma_{year}"
        table = feather.read_table(str(prefix) + "_labels_w_delay.feather")
        day = table["Date"].to_numpy().astype("datetime64[D]")
        ret = table["Ret_20d"].to_numpy()
        cap = table["MarketCap"].to_numpy()
        chart = np.memmap(str(prefix) + "_images.dat", mode="r", dtype=np.uint8)
        if chart.size != len(ret) * HEIGHT * WIDTH:
            raise ValueError(f"{year}: image count disagrees with label count")
        chart = chart.reshape(-1, HEIGHT, WIDTH)
        # Validate the separator and daily bar columns across the full release.
        if (chart[:, 51].any() or chart[:, VOLUME_START:, ::3].any()
                or chart[:, VOLUME_START:, 2::3].any()):
            raise ValueError(f"{year}: unexpected pixels in the volume band")
        if np.any(day.astype("datetime64[Y]") != np.datetime64(str(year), "Y")):
            raise ValueError(f"{year}: unexpected dates")
        finite = np.isfinite(ret)
        record = dict(year=year, rows=len(ret), valid=int(finite.sum()),
                      positive=int((finite & (ret > 0)).sum()),
                      negative_market_cap=int((cap < 0).sum()))
        if "Retx_20d_label" in table.column_names:
            raw_label = table["Retx_20d_label"].to_numpy()
            overlap = finite & np.isin(raw_label, [0, 1])
            record.update(retx_label_overlap=int(overlap.sum()),
                          retx_label_disagreement=int(np.sum(overlap & (raw_label != (ret > 0)))))
        audit.append(record)
        dates.append(day)
        returns.append(ret)
        caps.append(np.abs(cap))
        monthly.append(table["Ret_month"].to_numpy())
        images.append(chart)
    return images, *(np.concatenate(x) for x in (dates, returns, caps, monthly)), audit


def take_images(annual_images, indices):
    edges = np.cumsum([0, *(len(image) for image in annual_images)])
    batch = np.empty((len(indices), HEIGHT, WIDTH), dtype=np.uint8)
    for image, start, end in zip(annual_images, edges[:-1], edges[1:]):
        positions = np.flatnonzero((indices >= start) & (indices < end))
        if len(positions):
            batch[positions] = image[indices[positions] - start]
    return batch


def cnn(width):
    blocks = []
    previous = 1
    for depth, channels in enumerate((width, 2 * width, 4 * width)):
        stride = (3, 1) if depth == 0 else (1, 1)
        dilation = (2, 1) if depth == 0 else (1, 1)
        conv = nn.Conv2d(previous, channels, (5, 3), stride=stride,
                         dilation=dilation, padding=(2 * dilation[0], 1))
        blocks.extend((conv, nn.BatchNorm2d(channels),
                       nn.LeakyReLU(.01), nn.MaxPool2d((2, 1))))
        previous = channels
    model = nn.Sequential(*blocks, nn.Flatten(), nn.Dropout(.5),
                          nn.Linear(2 * WIDTH * previous, 2))
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return model


def float_images(raw, device, center, scale):
    images = torch.from_numpy(raw[:, None]).to(device, dtype=torch.float32)
    return images.sub_(center).div_(scale).contiguous(memory_format=torch.channels_last)


@torch.inference_mode()
def predict(model, raw, device, center, scale, batch_size):
    model.eval()
    result = []
    for start in range(0, len(raw), batch_size):
        inputs = float_images(raw[start:start + batch_size], device, center, scale)
        result.append(model(inputs).softmax(dim=1)[:, 1].cpu().numpy())
    return np.concatenate(result)


def logloss(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return -np.mean(np.where(y, np.log(p), np.log1p(-p)))


def train_model(x, y, xv, yv, seed, args, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = cnn(args.width).to(device, memory_format=torch.channels_last)
    # Release images store only black (0) and white (255) pixels.
    white_share = np.count_nonzero(x) / x.size
    center = 255 * white_share
    scale = 255 * math.sqrt(white_share * (1 - white_share))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    best_loss, stale, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(x))
        for start in range(0, len(order), args.batch):
            index = order[start:start + args.batch]
            inputs = float_images(x[index], device, center, scale)
            labels = torch.from_numpy(y[index]).to(device, dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            criterion(model(inputs), labels).backward()
            opt.step()
        value = logloss(yv, predict(model, xv, device, center, scale, args.batch))
        history.append(value)
        print(f"seed={seed} epoch={epoch} validation_loss={value:.6f}", flush=True)
        if value < best_loss:
            best_loss, stale, best_epoch = value, 0, epoch
            best = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        else:
            stale += 1
            if stale == 2:
                break
    model.load_state_dict(best)
    return model, center, scale, dict(seed=seed, best_epoch=best_epoch, validation_loss=history)


def auc(y, p):
    positives = int(y.sum())
    negatives = len(y) - positives
    ranks = rankdata(p, method="average")
    return (ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives)


def monthly_scores(dates, target, caps, monthly_returns, probabilities):
    months = dates.astype("datetime64[M]")
    rows = []
    for month in np.unique(months):
        eligible = np.flatnonzero(months == month)
        cap_valid = eligible[np.isfinite(caps[eligible]) & (caps[eligible] > 0)]
        cap_rank = cap_valid[np.argsort(caps[cap_valid], kind="stable")]
        quartile = len(cap_rank) // 4
        groups = {"all": eligible, "small": cap_rank[:quartile], "large": cap_rank[-quartile:]}
        for group, index in groups.items():
            y = target[index]
            scores = dict(month=str(month), group=group, count=len(y), positive_rate=float(y.mean()))
            tradable = index[np.isfinite(monthly_returns[index])]
            for method, probability in probabilities.items():
                p = probability[index]
                rank = tradable[np.argsort(probability[tradable], kind="stable")]
                decile = len(rank) // 10
                spread = (np.mean(monthly_returns[rank[-decile:]]) -
                          np.mean(monthly_returns[rank[:decile]])) if decile else math.nan
                scores.update({f"{method}_auc": auc(y, p) if y.any() and not y.all() else math.nan,
                               f"{method}_accuracy": float(np.mean((p > .5) == y)),
                               f"{method}_logloss": logloss(y, p),
                               f"{method}_spread": float(spread)})
            rows.append(scores)
    return rows


def block_interval(values, block=6):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(390)
    starts = rng.integers(len(values), size=(5000, math.ceil(len(values) / block), 1))
    positions = ((starts + np.arange(block)) % len(values)).reshape(5000, -1)[:, :len(values)]
    return [float(x) for x in np.quantile(np.nanmean(values[positions], axis=1), [.025, .975])]


def summary(rows, first_year):
    summaries = []
    midpoint = (first_year + 2020) // 2
    periods = ((first_year, 2019), (first_year, midpoint - 1), (midpoint, 2019))
    for start, end in periods:
        for group in ("all", "small", "large"):
            selected = [r for r in rows if r["group"] == group and start <= int(r["month"][:4]) <= end]
            delta = [r["full_auc"] - r["no_volume_auc"] for r in selected]
            record = dict(period=f"{start}-{end}", group=group, months=len(selected),
                          count=sum(r["count"] for r in selected),
                          full_minus_no_volume_auc=float(np.nanmean(delta)),
                          bootstrap_95=block_interval(delta))
            for method in (*FEATURES, "constant"):
                for metric in ("auc", "accuracy", "logloss"):
                    record[f"{method}_{metric}"] = float(np.nanmean([r[f"{method}_{metric}"] for r in selected]))
                if method != "constant":
                    spread = np.array([r[f"{method}_spread"] for r in selected])
                    record[f"{method}_spread"] = float(np.nanmean(spread))
                    record[f"{method}_sharpe"] = float(np.sqrt(12) * np.nanmean(spread) / np.nanstd(spread, ddof=1))
            summaries.append(record)
        period_rows = [r for r in rows if start <= int(r["month"][:4]) <= end]
        deltas = {
            group: np.array([r["full_auc"] - r["no_volume_auc"]
                             for r in period_rows if r["group"] == group])
            for group in ("small", "large")
        }
        interaction = deltas["small"] - deltas["large"]
        summaries.append(dict(period=f"{start}-{end}", group="small_minus_large",
                              months=len(interaction), full_minus_no_volume_auc=float(np.nanmean(interaction)),
                              bootstrap_95=block_interval(interaction)))
    return summaries


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/course/monthly_20d"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--test-year", type=int, choices=(2000, 2001), default=2000)
    parser.add_argument("--per-month", type=int, default=400, help="0 uses all development rows")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(args.threads)
    print(f"device={device} torch={torch.__version__}", flush=True)
    images, dates, returns, caps, monthly, audit = load_panel(args.data)
    write_csv(args.out / "data_audit.csv", audit)
    train, validation, test = paired_split(dates, returns, args.test_year, 390, args.per_month)
    y = np.isfinite(returns) & (returns > 0)
    print(f"train={len(train)} validation={len(validation)} test={len(test)}", flush=True)
    raw_train, raw_val = take_images(images, train), take_images(images, validation)
    probabilities, histories, per_seed = {}, {}, {}
    for method in FEATURES:
        print(method, flush=True)
        transform = without_volume if method == "no_volume" else np.asarray
        x, xv = transform(raw_train), transform(raw_val)
        models, histories[method] = [], []
        for seed in args.seeds:
            model, mean, std, history = train_model(x, y[train], xv, y[validation], seed, args, device)
            models.append((model, mean, std))
            histories[method].append(history)
        result = np.empty((len(models), len(test)), dtype=np.float32)
        for start in range(0, len(test), 8192):
            index = test[start:start + 8192]
            raw = transform(take_images(images, index))
            for row, (model, mean, std) in enumerate(models):
                result[row, start:start + len(index)] = predict(
                    model, raw, device, mean, std, args.batch * 2)
        probabilities[method] = result.mean(axis=0)
        per_seed[method] = result
        print(f"{method} test inference complete", flush=True)
    probabilities["constant"] = np.full(len(test), y[train].mean())
    rows = monthly_scores(dates[test], y[test], caps[test], monthly[test], probabilities)
    write_csv(args.out / "monthly_metrics.csv", rows)
    seed_auc = []
    for i, seed in enumerate(args.seeds):
        scores = monthly_scores(dates[test], y[test], caps[test], monthly[test],
                                {method: per_seed[method][i] for method in FEATURES})
        seed_auc.append(dict(seed=seed, **{
            method: float(np.nanmean([r[f"{method}_auc"] for r in scores if r["group"] == "all"]))
            for method in FEATURES
        }))
    info = dict(
        settings={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        torch=torch.__version__, device=str(device), train=len(train), validation=len(validation),
        test=len(test), train_positive_rate=float(y[train].mean()), histories=histories,
        seed_auc=seed_auc, results=summary(rows, args.test_year),
    )
    (args.out / "summary.json").write_text(json.dumps(info, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(info["results"][0], indent=2), flush=True)


if __name__ == "__main__":
    main()

