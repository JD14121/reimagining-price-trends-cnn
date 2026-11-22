"""Train the paper-faithful I20/R20 CNN on the full 1993-1999 development sample (GPU).

Follows the paper's estimation recipe: Xavier init, Adam (lr 1e-5), batch 128, 50% dropout
before the FC layer, cross-entropy loss, early stopping after two epochs without validation
improvement, and an ensemble of independently trained seeds. The development rows, labels
and December-1999 embargo are exactly those of ../study.py (split seed 390, 70/30).

Outputs per seed: models/<features>_seed<k>.pt (weights, normalisation, history) and
predictions/<features>_seed<k>.npy (test-set up-probabilities, aligned with test_index.npy).
"""
import argparse
import json
import math
import time

import numpy as np
import torch
from torch import nn

from common import HERE, I20CNN, paired_split, panel, take_images, without_volume


def batches(n, size, rng=None):
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for start in range(0, n, size):
        yield order[start:start + size]


@torch.no_grad()
def predict(model, images_gpu, center, scale, size=1024):
    """images_gpu may live on the CPU (pinned); batches are moved to the model's device."""
    device = next(model.parameters()).device
    model.eval()
    out = []
    for index in batches(len(images_gpu), size):
        x = (images_gpu[index].to(device, non_blocking=True).float().unsqueeze(1) - center) / scale
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x.contiguous(memory_format=torch.channels_last))
        out.append(logits.float().softmax(1)[:, 1])
    out = torch.cat(out)
    torch.cuda.empty_cache()  # return cached activation buffers to the shared laptop GPU
    return out


def logloss(y, p):
    p = p.clamp(1e-7, 1 - 1e-7)
    return float(-(y * p.log() + (1 - y) * (1 - p).log()).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", choices=("full", "no_volume"), default="full")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-epochs", type=int, default=5,
                        help="early-stopping counter starts after this epoch (noisy early validation)")
    parser.add_argument("--limit", type=int, default=0, help="debug: cap training rows")
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    (HERE / "models").mkdir(exist_ok=True)
    (HERE / "predictions").mkdir(exist_ok=True)

    images, dates, returns, caps, monthly, _ = panel()
    train, val, test = paired_split(dates, returns, 2000, 390, 0)
    if args.limit:
        train, val = train[:args.limit], val[:args.limit // 2]
    y = (np.isfinite(returns) & (returns > 0)).astype(np.float32)
    np.save(HERE / "predictions" / "test_index.npy", test)
    transform = without_volume if args.features == "no_volume" else np.asarray
    t0 = time.time()
    # Images stay in pinned host memory: the 8 GB laptop GPU is shared with the desktop, and
    # keeping ~2.6 GB of images on it pushed the process into slow shared-memory paging.
    xtr = torch.from_numpy(transform(take_images(images, train))).pin_memory()
    xva = torch.from_numpy(transform(take_images(images, val))).pin_memory()
    ytr = torch.from_numpy(y[train]).to(device)
    yva = torch.from_numpy(y[val]).to(device)
    print(f"{args.features}: train={len(train)} val={len(val)} test={len(test)} "
          f"loaded in {time.time() - t0:.0f}s", flush=True)
    white = float(np.count_nonzero(xtr.numpy()) / xtr.numel())
    center, scale = 255 * white, 255 * math.sqrt(white * (1 - white))

    for seed in args.seeds:
        tag = f"{args.features}_seed{seed}"
        if (HERE / "models" / f"{tag}.pt").exists():
            print(f"{tag}: exists, skipped", flush=True)
            continue
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = I20CNN().to(device, memory_format=torch.channels_last)
        assert sum(p.numel() for p in model.parameters()) == 708_866
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()
        best, stale, history = math.inf, 0, []
        for epoch in range(1, args.max_epochs + 1):
            model.train()
            t0, total = time.time(), torch.zeros((), device=device)
            for index in batches(len(xtr), args.batch, rng):
                x = xtr[torch.from_numpy(index)].to(device, non_blocking=True)
                index = torch.from_numpy(index).to(device)
                x = ((x.float().unsqueeze(1) - center) / scale).contiguous(
                    memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = loss_fn(model(x).float(), ytr[index].long())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.detach() * len(index)
            pv = predict(model, xva, center, scale)
            val_loss = logloss(yva, pv)
            val_acc = float(((pv > .5).float() == yva).float().mean())
            total = float(total)
            history.append(dict(epoch=epoch, train_loss=total / len(xtr), val_loss=val_loss,
                                val_accuracy=val_acc, seconds=time.time() - t0))
            print(f"{tag} epoch={epoch} train={total / len(xtr):.5f} val={val_loss:.5f} "
                  f"acc={val_acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if val_loss < best:
                best, stale = val_loss, 0
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += epoch > args.min_epochs  # only count after the warm-up epochs
                if stale >= args.patience:
                    break
        model.load_state_dict(state)
        best_epoch = min(history, key=lambda h: h["val_loss"])["epoch"]
        torch.save(dict(model=state, width=64, center=center, scale=scale, features=args.features,
                        seed=seed, history=history, best_epoch=best_epoch, lr=args.lr,
                        batch=args.batch, train_rows=len(train), val_rows=len(val)),
                   HERE / "models" / f"{tag}.pt")
        # Out-of-sample prediction on every 2000-2019 test image.
        prob = np.empty(len(test), dtype=np.float32)
        for start in range(0, len(test), 65536):
            chunk = test[start:start + 65536]
            raw = torch.from_numpy(transform(take_images(images, chunk)))
            prob[start:start + len(chunk)] = predict(model, raw, center, scale).cpu().numpy()
        np.save(HERE / "predictions" / f"{tag}.npy", prob)
        print(f"{tag}: best_epoch={best_epoch} best_val={best:.5f} test predictions saved",
              flush=True)
    (HERE / "models" / f"split_{args.features}.json").write_text(json.dumps(dict(
        train=len(train), validation=len(val), test=len(test), split_seed=390,
        development="1993-01..1999-11 (Dec-1999 embargoed)", test_period="2000-01..2019-12")))


if __name__ == "__main__":
    main()
