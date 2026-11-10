"""Load the archived ibm_cleveland noise model (derisk/data/noise_model_ibm_cleveland_2026-11-07.pkl, a
NoiseModel.to_dict(serializable=True) dump: complex matrices stored as [..., (re, im)] pairs, which
NoiseModel.from_dict does not undo for Kraus / unitary parameters)."""
import copy, pickle
from pathlib import Path

import numpy as np

PKL = Path(__file__).resolve().parents[1] / "derisk/data/noise_model_ibm_cleveland_2026-11-07.pkl"


def _cplx(m):
    a = np.asarray(m, dtype=float)
    return a[..., 0] + 1j * a[..., 1] if a.ndim >= 3 and a.shape[-1] == 2 else np.asarray(m)


def load():
    from qiskit_aer.noise import NoiseModel
    d = copy.deepcopy(pickle.load(open(PKL, "rb")))
    if not isinstance(d, dict):
        return d
    for e in d["errors"]:
        for seq in e.get("instructions", []):
            for op in seq:
                if op["name"] == "kraus":
                    op["params"] = [_cplx(m) for m in op["params"]]
                elif op["name"] == "unitary":
                    op["params"] = [_cplx(m) for m in op["params"]]
    return NoiseModel.from_dict(d)
