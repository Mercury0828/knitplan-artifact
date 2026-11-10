"""M1-N recheck (the pre-registered protocol note, 2026-11-09): the Entry 7a check (derisk/m1_noise_check.py) took the gate errors
from the live backend object's properties() on 2026-11-08, while the production GD path (knitplan.shots, K = "gd")
takes them from the archived 2026-11-07 properties snapshot that the archived noise model was built from. This rerun is
the same check, same configurations, with eps from the archived snapshot (knitplan.shots.gd_eps), so that the adoption
rule is evaluated on exactly the production method. Output derisk/results/M1/N_archived.json.
Usage: python derisk/m1_noise_recheck.py   (light: one thread)
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "derisk"))
import m1_noise_check as M
from knitplan.shots import archived_gate_errors, gd_eps

ERRS = archived_gate_errors()
M.eps_of = lambda tc, props: gd_eps(tc, ERRS)
M.OUT = ROOT / "derisk/results/M1"
_orig_write = Path.write_text


def _redirect(self, data, *a, **k):
    if self.name == "N.json" and self.parent == M.OUT:
        self = self.with_name("N_archived.json")
    return _orig_write(self, data, *a, **k)


Path.write_text = _redirect

if __name__ == "__main__":
    M.main()
