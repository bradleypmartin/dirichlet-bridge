#!/usr/bin/env python3
"""One-command reproduction of every figure in the discrete <-> continuous bridge.

Runs each of the arc drivers in `bridge/` as a fresh subprocess. Each driver
self-validates (asserts its identities to full precision) and writes its figure(s)
to `bridge/figures/`, so a green run of this script reproduces the whole arc from
scratch and confirms the numerics on the way.

Usage
-----
    python repro.py                # run all drivers, print a timing summary
    python repro.py -v             # also stream each driver's validation output
    python repro.py cont_eta rate_law   # run only the named driver(s)

Exit code is non-zero if any driver errors or fails to (re)write its figure(s).
Total runtime is roughly 21-23 minutes (see the per-driver estimates below); the
migration drivers `harmonic_bridge`, `warp_bridge` and `warp_eta` dominate.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent
BRIDGE_DIR = REPO_ROOT / "bridge"
FIGURES_DIR = BRIDGE_DIR / "figures"

# (driver stem, [figure file(s) it writes], approx runtime) -- in arc order.
# Runtimes are wall-clock on a laptop; the two migration drivers dominate (~10 min total).
DRIVERS: List[Tuple[str, List[str], str]] = [
    ("cont_eta",          ["cont_eta.png"],                              "~10 s"),
    ("comb_vs_warp",      ["comb_vs_warp.png"],                          "~1 min"),
    ("harmonic_bridge",   ["harmonic_bridge.png"],                       "~4 min"),
    ("warp_bridge",       ["warp_bridge.png"],                           "~5 min"),
    ("warp_coordinate",   ["warp_coordinate.png"],                       "~5 s"),
    ("warp_phase_compare", ["warp_phase_compare.png"],                   "~5 s"),
    ("warp_alpha",        ["warp_alpha.png"],                            "~20 s"),
    ("warp_eta",          ["warp_eta.png"],                              "~3 min"),
    ("rate_law",          ["rate_law.png", "rate_law_loose_ends.png"],   "~3 min"),
    ("stability",         ["stability.png"],                             "~2 min"),
    ("trivial_zeros",     ["trivial_zeros.png"],                         "~2 min"),
    ("eta_two_component", ["eta_two_component.png"],                     "~2 s"),
    ("jonquiere_zeros",   ["jonquiere_zeros.png"],                       "~1 min"),
    ("lfunction_bridge",  ["lfunction_bridge.png"],                      "~3 min"),
]


def _run_driver(stem: str, figures: List[str], verbose: bool) -> Tuple[bool, float, str]:
    """Run one driver as a subprocess; return (ok, elapsed_seconds, detail)."""
    script = BRIDGE_DIR / (stem + ".py")
    if not script.exists():
        return False, 0.0, "driver script not found: {}".format(script)

    # Force a non-interactive matplotlib backend so the run never blocks on a GUI.
    env = dict(os.environ, MPLBACKEND="Agg")
    started = time.time()
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=(None if verbose else subprocess.PIPE),
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.time() - started

    if proc.returncode != 0:
        tail = "" if verbose else (proc.stdout or "")[-2000:]
        return False, elapsed, "exited with code {}\n{}".format(proc.returncode, tail)

    # Confirm every expected figure exists and was (re)written by *this* run.
    missing = []
    for fig in figures:
        path = FIGURES_DIR / fig
        if not path.exists() or path.stat().st_mtime < started - 1:
            missing.append(fig)
    if missing:
        return False, elapsed, "did not (re)write: {}".format(", ".join(missing))

    return True, elapsed, ", ".join(figures)


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate every bridge figure.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="stream each driver's validation output")
    parser.add_argument("drivers", nargs="*",
                        help="optional subset of driver stems to run (default: all)")
    args = parser.parse_args()

    selected = DRIVERS
    if args.drivers:
        wanted = set(args.drivers)
        unknown = wanted - {stem for stem, _f, _n in DRIVERS}
        if unknown:
            parser.error("unknown driver(s): {}".format(", ".join(sorted(unknown))))
        selected = [d for d in DRIVERS if d[0] in wanted]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Reproducing {} driver(s) with {}\n".format(len(selected), sys.executable))

    results = []
    for stem, figures, note in selected:
        print("=> {:<20} ({}) ...".format(stem, note), end="", flush=True)
        ok, elapsed, detail = _run_driver(stem, figures, args.verbose)
        results.append((stem, ok, elapsed, detail))
        print("\r{} {:<18} {:>7.1f}s  {}".format(
            "[OK]  " if ok else "[FAIL]", stem, elapsed, detail))

    total = sum(elapsed for _s, _ok, elapsed, _d in results)
    n_ok = sum(1 for _s, ok, _e, _d in results if ok)
    print("\n{}/{} drivers OK in {:.1f}s total.".format(n_ok, len(results), total))
    print("Figures written to {}".format(FIGURES_DIR))
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
