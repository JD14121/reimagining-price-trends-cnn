"""Shared pieces for the Grad-CAM extension: the paper-faithful I20 CNN and data access.

Data access re-uses the audited loaders in ../study.py (row alignment, label rule,
development/test split with the December-1999 embargo).
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from study import (HEIGHT, WIDTH, VOLUME_START, load_panel, paired_split,  # noqa: E402
                   take_images, without_volume)

DATA = ROOT / "data" / "course" / "monthly_20d"
PRICE_ROWS = slice(0, 51)      # OHLC bars and 20-day moving average
GAP_ROW = 51                   # always black separator
VOLUME_ROWS = slice(VOLUME_START, HEIGHT)


class I20CNN(nn.Module):
    """Baseline 20-day CNN of Jiang, Kelly and Xiu (Figure 7 of the working paper).

    Three blocks of conv(5x3) -> BN -> LeakyReLU -> maxpool(2x1) with 64/128/256 filters.
    The first convolution uses vertical stride 3, dilation 2 and padding (12, 1), which gives
    feature maps of height 27 -> 13 -> 6 -> 3 and a 3*60*256 = 46,080-dimensional FC input.
    The total parameter count is 708,866, the number reported in the paper.
    """

    def __init__(self, width=64, dropout=0.5):
        super().__init__()
        channels = (width, 2 * width, 4 * width)
        layers, previous = [], 1
        for depth, out in enumerate(channels):
            first = depth == 0
            conv = nn.Conv2d(previous, out, (5, 3), stride=(3, 1) if first else (1, 1),
                             dilation=(2, 1) if first else (1, 1),
                             padding=(12, 1) if first else (2, 1))
            layers.append(nn.Sequential(conv, nn.BatchNorm2d(out), nn.LeakyReLU(0.01)))
            previous = out
        self.blocks = nn.ModuleList(layers)
        self.pool = nn.MaxPool2d((2, 1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(3 * WIDTH * previous, 2))
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def features(self, x):
        """Block activations (after LeakyReLU, before pooling) and the pooled final map."""
        activations = []
        for block in self.blocks:
            x = block(x)
            activations.append(x)
            x = self.pool(x)
        return activations, x

    def forward(self, x, keep=False):
        activations, x = self.features(x)
        logits = self.head(x)
        return (logits, activations) if keep else logits


# Vertical position (in input-pixel rows) of the receptive-field centre of every row of the
# activation map that Grad-CAM reads at each block (after LeakyReLU, before pooling).
#   block 1: conv rows i tap input rows 3i-12 + {0,2,4,6,8}  -> centre 3i-8   (27 rows)
#   block 2: row j = pooled block-1 rows 2j, 2j+1            -> centre 6j-6.5 (13 rows)
#   block 3: row m = pooled block-2 rows 2m, 2m+1            -> centre 12m-3.5 (6 rows)
ROW_CENTRES = (3.0 * np.arange(27) - 8.0, 6.0 * np.arange(13) - 6.5, 12.0 * np.arange(6) - 3.5)


def to_input_grid(cam, layer):
    """Map an (N, h, 60) layer map onto the (N, 64, 60) input grid by linear interpolation
    between receptive-field centres (columns are already aligned: stride 1, padding 1)."""
    centres = ROW_CENTRES[layer]
    rows = np.arange(HEIGHT, dtype=float)
    pos = np.clip(np.interp(rows, centres, np.arange(len(centres))), 0, len(centres) - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(centres) - 1)
    w = (pos - lo)[None, :, None]
    return cam[:, lo, :] * (1 - w) + cam[:, hi, :] * w


def normalise(raw, center, scale, device):
    x = torch.as_tensor(raw[:, None], device=device, dtype=torch.float32)
    return (x - center) / scale


def load_model(path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    model = I20CNN(state.get("width", 64)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


def panel(validate=False):
    """Same arrays as study.load_panel. By default skips its full scan of the volume band of
    all 2.2M images (already validated by study.py) and only checks image/label row counts."""
    if validate:
        return load_panel(DATA)
    import pyarrow.feather as feather
    from study import YEARS
    images, dates, returns, caps, monthly = [], [], [], [], []
    for year in YEARS:
        prefix = DATA / f"20d_month_has_vb_[20]_ma_{year}"
        table = feather.read_table(str(prefix) + "_labels_w_delay.feather",
                                   columns=["Date", "Ret_20d", "MarketCap", "Ret_month"])
        chart = np.memmap(str(prefix) + "_images.dat", mode="r", dtype=np.uint8)
        if chart.size != table.num_rows * HEIGHT * WIDTH:
            raise ValueError(f"{year}: image count disagrees with label count")
        images.append(chart.reshape(-1, HEIGHT, WIDTH))
        dates.append(table["Date"].to_numpy().astype("datetime64[D]"))
        returns.append(table["Ret_20d"].to_numpy())
        caps.append(np.abs(table["MarketCap"].to_numpy()))
        monthly.append(table["Ret_month"].to_numpy())
    return images, *(np.concatenate(x) for x in (dates, returns, caps, monthly)), []


__all__ = ["I20CNN", "ROW_CENTRES", "to_input_grid", "normalise", "load_model", "panel",
           "paired_split", "take_images", "without_volume", "HEIGHT", "WIDTH", "VOLUME_START",
           "PRICE_ROWS", "GAP_ROW", "VOLUME_ROWS", "DATA", "HERE", "ROOT"]
