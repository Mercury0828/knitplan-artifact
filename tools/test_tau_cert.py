"""Tests for the certified tau_eq. Run: python tools/test_tau_cert.py"""
import sys, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knitplan import engine as E


def adversarial_case():
    # c = 0, r = 1, 17 outputs, S = 2^55: one outcome S - 2^16, 2^16 outcomes with count 1; query = total mass.
    S = 2 ** 55
    m = E.SubModel(0, [], {}, {}, list(range(17)), list(range(17)), [], [])
    idx = np.arange(2 ** 16 + 1, dtype=np.int64)
    cnt = np.ones(idx.size, dtype=np.int64); cnt[0] = S - 2 ** 16
    tabs = {0: {((), ()): (idx, cnt)}}
    w = cnt.astype(float) / S
    big_first = w[0]
    for x in w[1:]:
        big_first += x
    small_first = sum(w[1:].tolist()) + w[0]
    diff = abs(big_first - small_first)
    tau, B = E.tau_eq([m], tabs, S, set(), None, 0)
    tau_l, W = E.tau_eq_legacy([m], tabs, S, set(), None, 0)
    print(f"order difference {diff:.3e}; legacy tau {tau_l:.3e} (fails: {diff > tau_l}); certified tau {tau:.3e} "
          f"(holds: {diff <= tau}); B = {B}")
    assert diff > tau_l and diff <= tau


def r2_overflow_case():
    # int64 count sum overflows (S = 2^63, 2048 counts of 2^52): B and tau must stay positive.
    S = 2 ** 63
    m = E.SubModel(0, [], {}, {}, list(range(11)), list(range(11)), [], [])
    idx = np.arange(2048, dtype=np.int64); cnt = np.full(2048, 2 ** 52, dtype=np.int64)
    tau, B = E.tau_eq([m], {0: {((), ()): (idx, cnt)}}, S, set(), None, 0)
    print(f"overflow case: B = {B}, tau = {tau:.3e}"); assert B > 0 and tau > 0


def r2_unnormalised_case():
    # c = 0, counts [100, 100], S = 100 -> expanded absolute mass 2; B must be >= 2.
    m = E.SubModel(0, [], {}, {}, [0], [0], [], [])
    tau, B = E.tau_eq([m], {0: {((), ()): (np.array([0, 1], np.int64), np.array([100, 100], np.int64))}}, 100, set(), None, 0)
    print(f"unnormalised case: B = {B}"); assert B >= 2


if __name__ == "__main__":
    adversarial_case()
    r2_overflow_case()
    r2_unnormalised_case()
    print("ok")
