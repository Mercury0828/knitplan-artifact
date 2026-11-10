"""Device table of the paper (distinct outcomes per shot, median over subcircuits, by subcircuit width) from
derisk/results/M1/qpu_support.json (written by bench/qpu_support.py). Usage: python bench/device_table.py"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rows = json.loads((ROOT / "derisk/results/M1/qpu_support.json").read_text())["rows"]
for name, sel in (("width <= 24", lambda w: w <= 24), ("width 25-33", lambda w: 25 <= w <= 33)):
    g = [r for r in rows if sel(r["width"])]
    vals = {k: float(np.median([r[k] for r in g])) for k in ("qpu_4096", "noisy_1000", "noisy_10000", "ideal_10000")}
    print(f"{name} ({len(g)} subcircuits): device {vals['qpu_4096']:.2f}  noisy 1e3 {vals['noisy_1000']:.2f}  "
          f"noisy 1e4 {vals['noisy_10000']:.2f}  ideal 1e4 {vals['ideal_10000']:.3f}")
