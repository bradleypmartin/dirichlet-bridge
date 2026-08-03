"""lambda_census.py -- the lambda-census at scale (issue #49, sub-issue of #37).

EXPERIMENTAL. Lives in `explorations/`, disjoint from the frozen manuscript arc;
tests run with `pytest explorations/tests`; the driver is local/manual only
(the heavy steps are hours -- never in CI). Experimental mathematics;
**no RH/GRH claims**.

The question (split out of #44, PR #46)
---------------------------------------
#44's lambda-side verdict at t <= 36.2 (N = 11 flows -> 6 seeds) was
FATE-SELECTION: the seed-fated cohort is already 2-adically phased at lam = 1,
and ensemble order develops only as the 5-zero surplus is expelled. But with 6
seeds every lam = 0 instrument reads deterministic geometry; the statistical
questions need a t <= ~120 box (N ~ 60 flows). Two of them are quantitative:

1.  THE SEED-FATED FRACTION has a parameter-free prediction. Seeds have density
    ln q / 2 pi (main comb ln(q-1)/2pi + conductor string ln(q/(q-1))/2pi, the
    #43 mean-motion bookkeeping) while the raw K = 1 census has RvM density
    ln(q t/2 pi)/2 pi -- so the surviving fraction should fall as

        f(t) = ln q / ln(q t / 2 pi),

    the expelled surplus making up the rest. #44's one data point: predicted
    5-6 of 11, measured 6. A t <= 120 box turns the point into a curve.
2.  PHASE AT BIRTH: does the seed-fated cohort's early phasing (<cos(gamma ln 2)>
    flat from lam = 1) persist at scale -- i.e. is a flow's fate readable at
    birth from its 2-adic phase alone?

The machinery (the two #44 blockers, plus three more this census forced)
------------------------------------------------------------------------
1.  `warp_alpha.find_zeros` polishes each grid seed with a SINGLE-SEED secant
    findroot, whose default second iterate x0 + 1/4 stalls between packed zeros
    (the #43 gotcha); above t ~ 36 it starts losing zeros -- one narrow-basin
    zero at 0.18 + 30.88i needed a winding-guided rescue even below that.
    Here: `find_zeros_muller` (Muller local-triple polish + escape clamp) run
    per t-window and CERTIFIED by the argument principle (`certify_window`:
    the window's winding count must equal the scan count; a deficit triggers
    locally refined rescans, then the every-node `rescue_deficits` hunt).
    The census of record is certified, not scanned.
2.  The lam-evaluator's moment grids keyed on the exact working precision,
    which changes with every height above t ~ 40 -- grid rebuilds dominated the
    #44 flow cost. Fixed in `zero_birth._bucket_dps` (dps bucketed to steps of
    5, evaluated at the bucket ceiling, moment headroom): grids are reused
    across ~11-unit t windows. With the Muller flow steps this collapsed the
    projected overnight run to ~5 minutes of flow wall time at --jobs 10.
3.  THE AMBIENT-18 ERROR LOBES (found by this census, root-caused to the
    warp_bridge guard ramp having been tuned at ambient dps 30): see
    `band_dps`. All census work above t = 38 runs at ambient 24.
4.  TRAJECTORY BASIN HOPS: flow_lambda's default max_jump = 2.0 exceeds the
    local zero spacing above t ~ 90, letting a solve silently converge onto a
    neighbor's zero (measured as double landings: 12 of 21 seeds multiply
    claimed). The #44 walk policy (jump gate at 0.6x local spacing) plus the
    `repair_flows` set-integrity pass (argument principle: one flow per zero
    of G_lam, one landing per simple seed; mid-flow merge losers re-solved by
    local scans, double landings adjudicated by the quadratic lam -> 0
    extrapolant) reduce residual collisions to zero.
5.  THE SCHEDULE-FLOOR AMBIGUITY: two distinct zeros of G_0.012 can share one
    findroot basin at lam = 0. `resolve_tails` extends such flows down
    LAM_TAIL (needs +4 digits: G = F/lam amplifies the evaluator floor by
    1/lam) so the capture and the fleeing neighbor visibly separate; a flow
    still unresolved stays counted NOT-seed-fated (the conservative side).

Honest framing
--------------
Experimental mathematics; no RH/GRH claims. The Riemann-von Mangoldt density,
Bohr almost-periodicity / mean-motion zero counts for exponential polynomials,
and Hurwitz's zero-convergence theorem are classical. What has no published
analog we know of (per the #37 adversarial lit pass) is the finite-K family's
amplitude-flow census itself: the certified raw-census -> seed-set flow at
scale, the measured seed-fated fraction against the ln q / ln(q t/2 pi) law,
and the phase-at-birth fate statistics.

Run directly to replot from the committed caches (the set-integrity repair
re-verifies in a few seconds). The heavy steps, at --jobs 10 on 14 cores:
`--recompute-scan` (the certified census + seed set, ~4 min wall) and
`--recompute-flows` (~6 min wall) -- the full pipeline is ~12 minutes. The
issue budgeted this as an overnight run; the grid bucketing, the Muller
steps, and the band_dps policy (which prevents wasted refinement rounds and
findroot flailing against noise floors) are where the two orders of
magnitude came from.
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_BRIDGE = _HERE.parent / "bridge"
for _p in (str(_HERE), str(_BRIDGE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

import arithmetic_clock as ac  # noqa: E402  (flow_fate, lam_instruments)
import character_bridge as cb  # noqa: E402
import zero_birth as zb  # noqa: E402

PI = math.pi
LN2 = math.log(2)
CHI = cb.CHI3
Q = CHI.q

# the census box: off-round edges dodge contour zeros (the #43 convention);
# sigma reaches left so conductor-string landings stay in view (#44's box)
BOX_SIG = (-3.19, 2.01)
T_LO = 2.03
T_MAX_DEFAULT = 120.29
WINDOW_H = 8.2                # certification window height (off-round edges)
# The seed scan reaches this far ABOVE the flow box: a flow born just under
# t_max can land on the next seed up (the #44 flow at t = 35.91 lands on the
# m = 4 main tooth at t ~ 36.4), so the landing targets must outrange the box.
SEED_MARGIN = 6.0

RAW_CSV = _HERE / "data" / "lambda_census_raw.csv"
WIN_CSV = _HERE / "data" / "lambda_census_windows.csv"
SEED_CSV = _HERE / "data" / "lambda_census_seeds.csv"
FLOW_CSV = _HERE / "data" / "lambda_census_flow.csv"
FIG_PATH = _HERE / "figures" / "lambda_census.png"


# --------------------------------------------------------------------------
# 1. the Muller-triple scan and the per-window certification
# --------------------------------------------------------------------------
def band_dps(t_lo, t_hi):
    """Ambient working dps for census work touching (t_lo, t_hi).

    The fast evaluator's |Im s|-guard ramp (warp_bridge._GUARD_ONSET = 40,
    slope 0.45/unit) was tuned at the manuscript arc's ambient dps 30, whose
    ~15 spare digits absorb any ramp shortfall. At the census's lean ambient
    18 there is no spare, and measured ERROR LOBES open wherever the
    ceil-quantized guard dips under the true need: |F| errors reach 2e-4
    (t = 52), 2e-2 (t = 65), and a second lobe 5e-4..9e-3 over t ~ (84, 89)
    -- far above the 1e-6 zero acceptance -- while other heights sit at
    1e-14. The lobe pattern is a warp_bridge internality we do NOT try to
    model: everything above the guard-onset region runs at ambient 24, which
    measures <= 1e-8 at every probed height. Below t = 38 ambient 18
    measures ~3e-8 (the validated #43/#44 regime)."""
    return 24 if t_hi > 38.0 else 18


def flow_dps(gamma):
    """Ambient dps for a flow born at height gamma: the #44 policy (15 below
    t = 30 -- the dps-15 noise floor crosses the 1e-6 residual acceptance
    there -- and 18 to t = 36), then the band_dps uniform 24 above."""
    if gamma >= 36.0:
        return 24
    return 18 if gamma > 30 else 15


def _dedupe(zeros, tol=1e-3):
    """Merge a zero list within `tol`, keeping first occurrences, (Im, Re) order."""
    out = []  # type: List[mp.mpc]
    for z in sorted(zeros, key=lambda w: (float(w.imag), float(w.real))):
        if all(abs(z - w) > tol for w in out):
            out.append(z)
    return out


def find_zeros_muller(fn, sig_range, t_range, dsig=0.28, dt=0.5, workdps=18,
                      tol=1e-6, dedupe=1e-3, pad=0.35, max_jump=1.5):
    """warp_alpha.find_zeros with the single-seed secant polish replaced by a
    Muller local-triple polish: the #49 scan.

    Same shape: sample |fn| once per grid node, run findroot only from local
    minima (a blind seed sends iterates to expensive far-field s), keep roots
    inside the padded box with |fn| < tol, dedupe. The polish differences:
    a Muller solve from a tight local triple (immune to the secant's x0 + 1/4
    second-iterate stall between packed zeros), and an escape clamp that aborts
    any iterate leaving the 3*max_jump neighborhood of its seed (runaway
    iterates at large |Im s| are expensive -- the zero_birth clamp).

    The acceptance tol matches the flow's zb._RESID_OK, NOT warp_alpha's 1e-8:
    the fast evaluator's noise floor at workdps 18 measures ~3e-8 near t ~ 36
    (a fixed 1e-8 gate silently rejects perfectly polished zeros there), the
    root location is still good to ~floor/|fn'| ~ 1e-7, and the census count
    authority is the winding certification, not the scan residual.
    """
    (smin, smax), (tmin, tmax) = sig_range, t_range
    h = mp.mpc("1e-3", "1e-3")
    zeros = []  # type: List[mp.mpc]
    with mp.workdps(workdps):
        ftol = mp.mpf(10) ** (-(workdps - 4))
        sig_vals = []  # type: List[float]
        s = smin
        while s <= smax + 1e-9:
            sig_vals.append(s)
            s += dsig
        t_vals = []  # type: List[float]
        t = tmin
        while t <= tmax + 1e-9:
            t_vals.append(t)
            t += dt
        grid = [[abs(fn(mp.mpc(sv, tv))) for tv in t_vals] for sv in sig_vals]
        ni, nj = len(sig_vals), len(t_vals)
        for i in range(ni):
            for j in range(nj):
                nbrs = [grid[i2][j2] for i2 in (i - 1, i, i + 1)
                        for j2 in (j - 1, j, j + 1)
                        if 0 <= i2 < ni and 0 <= j2 < nj and (i2, j2) != (i, j)]
                if grid[i][j] > min(nbrs):               # not a local minimum
                    continue
                seed = mp.mpc(sig_vals[i], t_vals[j])

                def g(s, seed=seed):
                    if abs(s - seed) > 3 * max_jump:
                        raise ValueError("iterate escaped")
                    return fn(s)

                try:
                    z = mp.findroot(g, (seed, seed + h, seed - mp.conj(h)),
                                    solver="muller", tol=ftol, maxsteps=20)
                except Exception:
                    continue
                if abs(fn(z)) >= tol:
                    # findroot's own success test is |f|^2 <= tol, so a solve
                    # that limps to maxsteps can pass it at |f| ~ sqrt(ftol)
                    # yet fail the census acceptance; a fresh tight triple at
                    # the near-root re-polishes in a couple of steps. Keep the
                    # re-polish only if it actually improves (near the noise
                    # floor a fresh Muller parabola can bounce).
                    try:
                        z2 = mp.findroot(g, (z, z + h / 1000,
                                             z - mp.conj(h) / 1000),
                                         solver="muller", tol=ftol,
                                         maxsteps=10)
                        if abs(fn(z2)) < abs(fn(z)):
                            z = z2
                    except Exception:
                        pass
                if (smin - pad <= z.real <= smax + pad
                        and tmin - pad <= z.imag <= tmax + pad
                        and abs(fn(z)) < tol
                        and all(abs(z - w) > dedupe for w in zeros)):
                    zeros.append(mp.mpc(z))
    zeros.sort(key=lambda z: (float(z.imag), float(z.real)))
    return zeros


def certify_window(fn, sig_range, t_window, dsig=0.28, dt=0.5, workdps=18,
                   max_refine=2, wind_dps=12):
    """One t-window of the certified census: Muller-scan the window, then
    require the argument-principle count to match.

    The winding count (adaptive phase walk, zb.winding_count) is the census
    authority; a scan deficit triggers refined rescans (grid steps halved per
    round, results unioned) up to max_refine times. Returns
    (zeros, n_wind, refines, status) with status OK / DEFICIT / SURPLUS --
    a DEFICIT is an honest miss (reported, never papered over); SURPLUS would
    mean winding undersampling and warrants a by-hand look.
    """
    lo, hi = t_window
    try:
        n_wind = zb.winding_count(fn, sig_range, (lo, hi), dsig=0.35, dt=0.5,
                                  workdps=wind_dps)
    except RuntimeError:
        # a phase step too coarse (or a zero grazing the contour): one harder
        # try; if that fails too, the window is honestly uncertifiable --
        # never let one bad contour kill a multi-hour run
        try:
            n_wind = zb.winding_count(fn, sig_range, (lo, hi), dsig=0.2,
                                      dt=0.3, workdps=wind_dps + 2, refine=0.5)
        except RuntimeError:
            zs = [z for z in find_zeros_muller(fn, sig_range, (lo, hi), dsig,
                                               dt, workdps=workdps)
                  if lo < float(z.imag) <= hi]
            return zs, -1, 0, "WINDFAIL"
    zs = [z for z in find_zeros_muller(fn, sig_range, (lo, hi), dsig, dt,
                                       workdps=workdps)
          if lo < float(z.imag) <= hi]
    refines = 0
    while len(zs) < n_wind and refines < max_refine:
        refines += 1
        finer = find_zeros_muller(fn, sig_range, (lo, hi),
                                  dsig / (2 ** refines), dt / (2 ** refines),
                                  workdps=workdps)
        zs = _dedupe(zs + [z for z in finer if lo < float(z.imag) <= hi])
    status = ("OK" if len(zs) == n_wind
              else ("DEFICIT" if len(zs) < n_wind else "SURPLUS"))
    return zs, n_wind, refines, status


def windows_for(t_lo, t_hi, height=WINDOW_H):
    """Contiguous certification windows covering (t_lo, t_hi]; the off-round
    t_lo and the irrational-ish height keep edges off zeros."""
    edges = [t_lo]
    while edges[-1] < t_hi - 1e-9:
        edges.append(min(edges[-1] + height, t_hi))
    return list(zip(edges, edges[1:]))


def _census_fn(kind):
    """The two census targets, by picklable name: 'raw' = the K = 1, lam = 1
    warp family (the flow's starting object); 'seed' = the lam = 0 seed D."""
    if kind == "raw":
        return lambda s: zb.F_lam(s, 1, 1, CHI)
    if kind == "seed":
        return lambda s: zb.seed_D(s, CHI)
    raise ValueError(kind)


def _scan_worker(job):
    """Multiprocessing worker: one window's certified scan -> flat rows.
    The winding contour gets a dps bump alongside the scan's band_dps bump
    (in the bubble the ambient-12 phase error is ~10x the ambient-18 one)."""
    kind, lo, hi, dsig, dt, workdps, max_refine = job
    fn = _census_fn(kind)
    zs, n_wind, refines, status = certify_window(
        fn, BOX_SIG, (lo, hi), dsig, dt, workdps=workdps,
        max_refine=max_refine, wind_dps=(14 if workdps >= 24 else 12))
    return (lo, hi, n_wind, refines, status,
            [(float(z.real), float(z.imag)) for z in zs])


def run_scan(t_max=T_MAX_DEFAULT, jobs=1, verbose=True):
    """The certified raw census: scan + certify every window of the K = 1
    warp family to t_max, cache zeros (RAW_CSV) and the window report
    (WIN_CSV). The first heavy step (--recompute-scan)."""
    t0 = time.time()
    wins = windows_for(T_LO, t_max)
    jobsq = [("raw", lo, hi, 0.28, 0.5, band_dps(lo, hi), 2)
             for lo, hi in wins]
    results = []
    if jobs > 1:
        import multiprocessing as mp_proc
        with mp_proc.get_context("spawn").Pool(jobs) as pool:
            for res in pool.imap_unordered(_scan_worker, jobsq):
                results.append(res)
                if verbose:
                    lo, hi, nw, rf, st, zs = res
                    print(f"   window ({lo:7.2f}, {hi:7.2f}): winding {nw:>2}, "
                          f"found {len(zs):>2}, refines {rf}  {st}  "
                          f"({time.time() - t0:.0f} s)", flush=True)
    else:
        for job in jobsq:
            res = _scan_worker(job)
            results.append(res)
            if verbose:
                lo, hi, nw, rf, st, zs = res
                print(f"   window ({lo:7.2f}, {hi:7.2f}): winding {nw:>2}, "
                      f"found {len(zs):>2}, refines {rf}  {st}  "
                      f"({time.time() - t0:.0f} s)", flush=True)
    results.sort(key=lambda r: r[0])
    zeros = [z for _lo, _hi, _nw, _rf, _st, zs in results for z in zs]
    zeros.sort(key=lambda z: z[1])
    RAW_CSV.parent.mkdir(exist_ok=True)
    with RAW_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "sigma", "gamma"])
        for i, (sig, gam) in enumerate(zeros):
            w.writerow([i, f"{sig:.9f}", f"{gam:.9f}"])
    with WIN_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_lo", "t_hi", "winding", "found", "refines", "status"])
        for lo, hi, nw, rf, st, zs in results:
            w.writerow([f"{lo:.2f}", f"{hi:.2f}", nw, len(zs), rf, st])
    if verbose:
        n_bad = sum(st != "OK" for _l, _h, _n, _r, st, _z in results)
        print(f"   certified census: {len(zeros)} zeros, "
              f"{n_bad} uncertified window(s), {time.time() - t0:.0f} s")
    return read_raw(), read_windows()


def _rescue_window(fn, lo, hi, have, workdps=26, n_slice=6, verbose=True):
    """The deficit-rescue pass for one window: winding-localize the missing
    zeros into ~1.4-unit slices, then polish from EVERY node of a fine grid
    (not just |fn| local minima -- a tight pair merges into one minimum and
    the coarse seeding can only ever find one of them) at raised precision
    (the workdps-18 noise floor is ~3e-8; dps 22 buys ~1e-10, so the narrow
    basins the base scan cannot see resolve). Returns the newly found zeros.
    """
    edges = [lo + (hi - lo) * i / n_slice for i in range(n_slice + 1)]
    h = mp.mpc("1e-3", "1e-3")
    news = []  # type: List[mp.mpc]

    def _hunt(slo, shi, n_w):
        """Every-node polish over one short slice; returns when its winding
        count is met (or the grid is exhausted)."""
        with mp.workdps(workdps):
            ftol = mp.mpf(10) ** (-(workdps - 6))
            sv = BOX_SIG[0]
            while sv <= BOX_SIG[1]:
                tv = slo
                while tv <= shi:
                    seed = mp.mpc(sv, tv)
                    try:
                        z = mp.findroot(fn, (seed, seed + h,
                                             seed - mp.conj(h)),
                                        solver="muller", tol=ftol,
                                        maxsteps=15)
                    except Exception:
                        z = None
                    if (z is not None and slo < z.imag <= shi
                            and BOX_SIG[0] - 0.35 <= z.real
                            <= BOX_SIG[1] + 0.35
                            and abs(fn(z)) < 1e-6
                            and all(abs(z - w) > 1e-3 for w in have + news)):
                        news.append(mp.mpc(z))
                        if verbose:
                            print(f"      RESCUED {complex(z):.6f}",
                                  flush=True)
                        n_here = len([w for w in have + news
                                      if slo < float(w.imag) <= shi])
                        if n_w >= 0 and n_here == n_w:
                            return
                    tv += 0.25
                sv += 0.14

    for slo, shi in zip(edges, edges[1:]):
        try:
            n_w = zb.winding_count(fn, BOX_SIG, (slo, shi), dsig=0.35, dt=0.4,
                                   workdps=12)
        except RuntimeError:
            n_w = -1                       # slice edge grazes a zero: hunt anyway
        got = [z for z in have + news if slo < float(z.imag) <= shi]
        if n_w == len(got):
            continue
        if verbose:
            print(f"      rescue slice ({slo:.2f}, {shi:.2f}): winding {n_w}, "
                  f"have {len(got)} -> every-node polish", flush=True)
        _hunt(slo, shi, n_w)
    return news


def rescan_windows(statuses=("DEFICIT", "SURPLUS", "WINDFAIL"), jobs=1,
                   verbose=True):
    """Re-run the certified scan on flagged windows and REBUILD the raw cache.

    The repair path for a pre-flows census (it rewrites RAW_CSV with fresh
    ascending indices, so it must run BEFORE any flow cache exists -- the
    post-flows repair is rescue_deficits, which is append-only). Used to
    re-certify the error-bubble windows after the band_dps fix: same code
    path as run_scan, now at the corrected precision."""
    if FLOW_CSV.exists():
        raise RuntimeError("flow cache exists -- use rescue_deficits instead "
                           "(rescan_windows rewrites raw indices)")
    raw, wins = read_raw(), read_windows()
    todo = [(lo, hi) for lo, hi, _nw, _nf, _rf, st in wins if st in statuses]
    if not todo:
        if verbose:
            print("   no flagged windows -- nothing to rescan")
        return raw, wins
    jobsq = [("raw", lo, hi, 0.28, 0.5, band_dps(lo, hi), 2)
             for lo, hi in todo]
    t0 = time.time()
    results = []
    if jobs > 1 and len(jobsq) > 1:
        import multiprocessing as mp_proc
        with mp_proc.get_context("spawn").Pool(min(jobs, len(jobsq))) as pool:
            for res in pool.imap_unordered(_scan_worker, jobsq):
                results.append(res)
                if verbose:
                    lo, hi, nw, rf, st, zs = res
                    print(f"   rescan ({lo:7.2f}, {hi:7.2f}): winding {nw:>2}, "
                          f"found {len(zs):>2}, refines {rf}  {st}  "
                          f"({time.time() - t0:.0f} s)", flush=True)
    else:
        for job in jobsq:
            res = _scan_worker(job)
            results.append(res)
            if verbose:
                lo, hi, nw, rf, st, zs = res
                print(f"   rescan ({lo:7.2f}, {hi:7.2f}): winding {nw:>2}, "
                      f"found {len(zs):>2}, refines {rf}  {st}  "
                      f"({time.time() - t0:.0f} s)", flush=True)
    by_win = {(lo, hi): (nw, rf, st, zs) for lo, hi, nw, rf, st, zs in results}
    keep = [(s, g) for _i, s, g in raw
            if not any(lo < g <= hi for lo, hi in todo)]
    merged = keep + [z for _w, (_n, _r, _s, zs) in by_win.items() for z in zs]
    merged.sort(key=lambda p: p[1])
    with RAW_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "sigma", "gamma"])
        for i, (sig, gam) in enumerate(merged):
            w.writerow([i, f"{sig:.9f}", f"{gam:.9f}"])
    with WIN_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_lo", "t_hi", "winding", "found", "refines", "status"])
        for lo, hi, nw, nf, rf, st in wins:
            if (lo, hi) in by_win:
                nw, rf, st, zs = by_win[(lo, hi)]
                nf = len(zs)
            w.writerow([f"{lo:.2f}", f"{hi:.2f}", nw, nf, rf, st])
    if verbose:
        n_bad = sum(1 for _w, (_n, _r, st, _z) in by_win.items()
                    if st != "OK")
        print(f"   rescan: {len(merged)} zeros total, "
              f"{n_bad} window(s) still flagged")
    return read_raw(), read_windows()


def _rescue_worker(job):
    """Multiprocessing worker: one DEFICIT window's rescue -> plain tuples."""
    lo, hi, zs = job
    have = [mp.mpc(s, g) for s, g in zs]
    news = _rescue_window(_census_fn("raw"), lo, hi, have)
    return lo, hi, [(float(z.real), float(z.imag)) for z in news]


def rescue_deficits(jobs=1, verbose=True):
    """Hunt every DEFICIT window's missing zeros and repair the caches:
    rescued zeros are APPENDED to RAW_CSV with fresh indices (append-only so
    existing flow rows keep their keys), and the window report is updated
    (DEFICIT -> RESCUED when the winding count is reached). Run before the
    missing flows (`run_flows(..., only_missing=True)`)."""
    raw, wins = read_raw(), read_windows()
    jobsq = []
    for lo, hi, n_w, _nf, _rf, st in wins:
        if st != "DEFICIT":
            continue
        zs = [(s, g) for _i, s, g in raw if lo < g <= hi]
        jobsq.append((lo, hi, zs))
    if not jobsq:
        if verbose:
            print("   no DEFICIT windows -- nothing to rescue")
        return raw, wins
    results = []
    if jobs > 1 and len(jobsq) > 1:
        import multiprocessing as mp_proc
        with mp_proc.get_context("spawn").Pool(min(jobs, len(jobsq))) as pool:
            results = list(pool.imap_unordered(_rescue_worker, jobsq))
    else:
        for job in jobsq:
            if verbose:
                print(f"   rescuing window ({job[0]:.2f}, {job[1]:.2f})",
                      flush=True)
            results.append(_rescue_worker(job))
    idx_next = max(i for i, _s, _g in raw) + 1 if raw else 0
    news_by_win = {(lo, hi): zs for lo, hi, zs in results}
    with RAW_CSV.open("a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        for (_lo, _hi), zs in sorted(news_by_win.items()):
            for sig, gam in sorted(zs, key=lambda p: p[1]):
                w.writerow([idx_next, f"{sig:.9f}", f"{gam:.9f}"])
                idx_next += 1
    with WIN_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_lo", "t_hi", "winding", "found", "refines", "status"])
        for lo, hi, n_w, nf, rf, st in wins:
            n_new = len(news_by_win.get((lo, hi), []))
            if st == "DEFICIT" and n_new:
                nf += n_new
                st = "RESCUED" if nf == n_w else "DEFICIT"
            w.writerow([f"{lo:.2f}", f"{hi:.2f}", n_w, nf, rf, st])
    if verbose:
        n_total = sum(len(zs) for zs in news_by_win.values())
        print(f"   rescue: {n_total} zero(s) recovered across "
              f"{len(jobsq)} window(s)")
    return read_raw(), read_windows()


def run_seed_scan(t_max=T_MAX_DEFAULT, verbose=True):
    """The certified seed set: zeros of D = B + q^{-s} L(0, chi) in the box
    (cheap -- D is a 3-term exponential polynomial). Cached at SEED_CSV with
    each seed classified main (the wobbled sigma = 0 comb) vs conductor (the
    far-left ln(q/(q-1)) string)."""
    fn = _census_fn("seed")
    rows = []
    for lo, hi in windows_for(T_LO, t_max):
        zs, n_wind, refines, status = certify_window(fn, BOX_SIG, (lo, hi),
                                                     dsig=0.25, dt=0.5)
        if verbose and status != "OK":
            print(f"   seed window ({lo:.2f}, {hi:.2f}): {status} "
                  f"(winding {n_wind}, found {len(zs)})")
        rows.extend(zs)
    rows.sort(key=lambda z: float(z.imag))
    with SEED_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "sigma", "gamma", "kind"])
        for i, z in enumerate(rows):
            kind = "main" if float(z.real) > -1.5 else "conductor"
            w.writerow([i, f"{float(z.real):.9f}", f"{float(z.imag):.9f}",
                        kind])
    if verbose:
        n_main = sum(1 for z in rows if float(z.real) > -1.5)
        print(f"   seed set: {len(rows)} seeds ({n_main} main, "
              f"{len(rows) - n_main} conductor) to t = {t_max:.0f}")
    return read_seeds()


# --------------------------------------------------------------------------
# 2. the flow runner
# --------------------------------------------------------------------------
def _flow_worker(job):
    """Multiprocessing worker: one raw zero's lambda flow -> CSV-ready rows.
    Ambient dps from flow_dps (the #44 noise-floor fix + the bubble bump);
    Muller local-triple steps throughout."""
    idx, sig, gam = job
    wd = flow_dps(gam)
    # jump gate at 0.6x the local zero spacing (the #44 walk_zero policy):
    # flow_lambda's default 2.0 exceeds the spacing above t ~ 90, letting a
    # solve silently converge onto a neighbor's zero -- the double-landing
    # artifact. A too-far solve now fails instead, engaging the lam
    # substepping (real tracking) or an honest loss.
    mj = 0.6 * ac.local_spacing(gam, Q)
    try:
        traj = zb.flow_lambda(mp.mpc(sig, gam), 1, CHI, workdps=wd,
                              muller=True, max_jump=mj)
    except Exception as exc:               # never let one flow kill the pool
        print(f"   flow {idx}: UNEXPECTED {type(exc).__name__}: {exc}",
              flush=True)
        traj = [(1.0, None)]
    rows = []
    for lam, z in traj:
        if z is None:
            rows.append((idx, lam, float("nan"), float("nan")))
        else:
            rows.append((idx, lam, float(z.real), float(z.imag)))
    return rows


def run_flows(raw, jobs=1, verbose=True, only_missing=False):
    """Flow every raw census zero lam: 1 -> 0 and cache the trajectories.
    The second heavy step (--recompute-flows). only_missing=True flows just
    the raw indices absent from the cache (the post-rescue top-up) and
    appends, preserving the committed trajectories."""
    t0 = time.time()
    have = {}  # type: Dict[int, List[Tuple[float, float, float]]]
    if only_missing and FLOW_CSV.exists():
        have = read_flows()
    jobsq = [(i, sig, gam) for i, sig, gam in raw if i not in have]
    all_rows = []  # type: List[Tuple[int, float, float, float]]
    if jobs > 1:
        import multiprocessing as mp_proc
        with mp_proc.get_context("spawn").Pool(jobs) as pool:
            for rows in pool.imap_unordered(_flow_worker, jobsq):
                all_rows.extend(rows)
                if verbose:
                    idx = rows[0][0]
                    end = rows[-1]
                    tag = (f"seed {end[2]:+.3f}{end[3]:+.3f}i"
                           if end[1] == 0.0 and not math.isnan(end[2])
                           else "exits/lost")
                    print(f"   flow {idx:>2}: {tag}  "
                          f"({time.time() - t0:.0f} s)", flush=True)
    else:
        for job in jobsq:
            rows = _flow_worker(job)
            all_rows.extend(rows)
            if verbose:
                end = rows[-1]
                tag = (f"seed {end[2]:+.3f}{end[3]:+.3f}i"
                       if end[1] == 0.0 and not math.isnan(end[2])
                       else "exits/lost")
                print(f"   flow {job[0]:>2}: {tag}  "
                      f"({time.time() - t0:.0f} s)", flush=True)
    all_rows.sort(key=lambda r: (r[0], -r[1]))
    with FLOW_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "lam", "sigma", "gamma"])
        for idx, lam, sig, gam in all_rows:
            w.writerow([idx, lam,
                        "nan" if math.isnan(sig) else f"{sig:.9f}",
                        "nan" if math.isnan(gam) else f"{gam:.9f}"])
    if verbose:
        print(f"   flows: {len(jobsq)} trajectories in "
              f"{time.time() - t0:.0f} s -> {FLOW_CSV.name}")
    return read_flows()


# --------------------------------------------------------------------------
# 3. cache readers
# --------------------------------------------------------------------------
def read_raw(path=RAW_CSV):
    """[(idx, sigma, gamma)] ascending in gamma."""
    out = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.append((int(r["idx"]), float(r["sigma"]), float(r["gamma"])))
    return out


def read_windows(path=WIN_CSV):
    """[(t_lo, t_hi, winding, found, refines, status)]."""
    out = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.append((float(r["t_lo"]), float(r["t_hi"]), int(r["winding"]),
                        int(r["found"]), int(r["refines"]), r["status"]))
    return out


def read_seeds(path=SEED_CSV):
    """[(idx, sigma, gamma, kind)]."""
    out = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.append((int(r["idx"]), float(r["sigma"]), float(r["gamma"]),
                        r["kind"]))
    return out


def read_flows(path=FLOW_CSV):
    """{idx: [(lam, sigma, gamma)]} descending in lam (nan = honest loss);
    the arithmetic_clock.lam_instruments / flow_fate format."""
    out = {}  # type: Dict[int, List[Tuple[float, float, float]]]
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(int(r["idx"]), []).append(
                (float(r["lam"]), float(r["sigma"]), float(r["gamma"])))
    return out


# --------------------------------------------------------------------------
# 4. the instruments
# --------------------------------------------------------------------------
def rvm_count(t0, t1, q=Q):
    """The smooth entire-L zero count in (t0, t1): integral of the RvM density
    ln(q t/2 pi)/2 pi -- (1/2pi)[t ln(q t/2 pi e)] between the endpoints."""
    def F(t):
        return t / (2 * PI) * (math.log(q * t / (2 * PI)) - 1)
    return F(t1) - F(t0)


def seed_count(t0, t1, q=Q):
    """The mean-motion seed count in (t0, t1): density ln q/2 pi (main comb
    ln(q-1) + conductor string ln(q/(q-1)), the #43 bookkeeping)."""
    return (t1 - t0) * math.log(q) / (2 * PI)


def predicted_fraction(t, q=Q):
    """The parameter-free seed-fated fraction law f(t) = ln q / ln(q t/2 pi)
    (seed density over RvM density), clamped to 1 below the crossover."""
    x = math.log(q * float(t) / (2 * PI))
    if x <= math.log(q):
        return 1.0
    return math.log(q) / x


def census_fates(flows, seeds):
    """Per-flow bookkeeping: {idx: (fate, t_birth, seed_idx or None, dist)}.
    A seed-fated flow is matched to the nearest certified seed; the match
    distance should be ~0 (the lam = 0 leg IS resolved on D), so a large one
    flags a mismatch rather than silently claiming a landing."""
    out = {}
    for idx, tr in flows.items():
        fate = ac.flow_fate(tr)
        t_birth = tr[0][2]
        match = None  # type: Optional[int]
        dist = float("nan")
        if fate == "seed":
            lam0, s0, g0 = tr[-1]
            ds = [(math.hypot(s0 - sig, g0 - gam), si)
                  for si, sig, gam, _k in seeds]
            dist, match = min(ds)
        out[idx] = (fate, t_birth, match, dist)
    return out


def fraction_windows(fates, windows):
    """The measured seed-fated fraction per certification window:
    [(t_lo, t_hi, n_flows, n_seed, f_meas, f_pred)] with f_pred the
    window-integrated law seed_count/rvm_count."""
    out = []
    for lo, hi, _nw, _nf, _rf, _st in windows:
        sel = [(fate, tb) for fate, tb, _m, _d in fates.values()
               if lo < tb <= hi]
        if not sel:
            continue
        n = len(sel)
        k = sum(1 for fate, _tb in sel if fate == "seed")
        f_pred = min(1.0, seed_count(lo, hi) / max(rvm_count(lo, hi), 1e-9))
        out.append((lo, hi, n, k, k / n, f_pred))
    return out


def fraction_cumulative(fates, t_grid=None):
    """The cumulative curve: N_seed(<= T)/N(<= T) at each flow birth height
    (measured), against the integrated law. Returns (T, f_meas, f_pred)."""
    births = sorted((tb, fate) for fate, tb, _m, _d in fates.values())
    Ts, fm, fp = [], [], []
    n = k = 0
    for tb, fate in births:
        n += 1
        k += fate == "seed"
        Ts.append(tb)
        fm.append(k / n)
        fp.append(min(1.0, seed_count(T_LO, tb) / max(rvm_count(T_LO, tb),
                                                      1e-9)))
    return np.array(Ts), np.array(fm), np.array(fp)


def collision_report(flows, tol=1e-4):
    """Set-integrity check along the flow: at every lam level the tracked
    points must be DISTINCT zeros of G_lam (argument principle: a simple seed
    zero attracts exactly one), so any two flows within `tol` of each other
    have merged -- a tracking artifact, not mathematics. Returns
    {lam: [(idx_a, idx_b), ...]}; the honest census target is {} everywhere,
    and at lam = 0 it enforces <= one landing per seed."""
    by_lam = {}  # type: Dict[float, List[Tuple[int, float, float]]]
    for idx, tr in flows.items():
        for lam, sig, gam in tr:
            if not math.isnan(gam):
                by_lam.setdefault(lam, []).append((idx, sig, gam))
    out = {}  # type: Dict[float, List[Tuple[int, int]]]
    for lam, pts in by_lam.items():
        pts = sorted(pts, key=lambda p: p[2])
        hits = []
        for (ia, sa, ga), (ib, sb, gb) in zip(pts, pts[1:]):
            if math.hypot(sa - sb, ga - gb) < tol:
                hits.append((ia, ib))
        if hits:
            out[lam] = hits
    return out


def _local_zero_scan(fn, center, taken, radius=0.9, step=0.15, workdps=18,
                     exclude=1e-3):
    """|fn| grid minima in a box around `center`, Muller-polished, excluding
    roots within `exclude` of any `taken` point. The flow-side analog of
    arithmetic_clock._local_rescue. Returns the nearest accepted root or None."""
    h = mp.mpc("1e-3", "1e-3")
    hits = []  # type: List[mp.mpc]
    with mp.workdps(workdps):
        ftol = mp.mpf(10) ** (-(workdps - 6))
        n = int(2 * radius / step) + 1
        vals = [[abs(fn(center + mp.mpc(-radius + i * step,
                                        -radius + j * step)))
                 for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(n):
                nb = [vals[i2][j2] for i2 in (i - 1, i, i + 1)
                      for j2 in (j - 1, j, j + 1)
                      if 0 <= i2 < n and 0 <= j2 < n and (i2, j2) != (i, j)]
                if vals[i][j] > min(nb):
                    continue
                seed = center + mp.mpc(-radius + i * step, -radius + j * step)
                try:
                    z = mp.findroot(fn, (seed, seed + h, seed - mp.conj(h)),
                                    solver="muller", tol=ftol, maxsteps=20)
                except Exception:
                    continue
                if (abs(z - center) <= radius * 2
                        and abs(fn(z)) < zb._RESID_OK
                        and all(abs(z - w) > exclude for w in taken)
                        and all(abs(z - w) > 1e-6 for w in hits)):
                    hits.append(mp.mpc(z))
    if not hits:
        return None
    return min(hits, key=lambda z: abs(z - center))


def _extrapolate_to_zero(tr):
    """Quadratic-in-lam extrapolation of a trajectory's last three tracked
    schedule points to lam = 0 (the flow is analytic in lam near a simple
    seed, so this is accurate over the 0.03 -> 0 tail). None if < 3 points."""
    pts = [(lam, s, g) for lam, s, g in tr
           if lam > 0 and not math.isnan(g)][-3:]
    if len(pts) < 3:
        return None
    lams = np.array([p[0] for p in pts])
    cs = np.polyfit(lams, np.array([p[1] for p in pts]), 2)
    ct = np.polyfit(lams, np.array([p[2] for p in pts]), 2)
    return complex(np.polyval(cs, 0.0), np.polyval(ct, 0.0))


def repair_flows(flows, seeds, verbose=True, max_pass=3):
    """Set-integrity repair of the flow census -- dedup_patch's philosophy
    along lam. Two failure classes, both objective violations of the argument
    principle (one flow per zero of G_lam; one landing per simple seed):

    1. MID-FLOW MERGES: two flows on one point at some lam > 0. The loser
       (farther from the shared point at the previous lam) is re-solved by a
       local scan around its pre-merge position, excluding taken zeros, and
       its remaining schedule is re-flowed from the rescued zero. No rescue
       -> the trajectory is truncated at the merge (honest loss).
    2. DOUBLE LANDINGS at lam = 0 (distinct at lam = 0.012, same seed after
       the final resolve: both 0.012-zeros sat in one findroot basin). The
       quadratic lam -> 0 extrapolant adjudicates: the closer flow keeps the
       seed; the loser is re-resolved on D near its own extrapolant (tight
       jump gate) if that lands a DIFFERENT seed, else truncated at 0.012.

    Operates on a copy; writes the repaired FLOW_CSV; returns the new dict.
    Idempotent once collision_report is clean."""
    flows = {i: list(tr) for i, tr in flows.items()}
    sched = sorted(zb.LAM_SCHEDULE, reverse=True)
    n_fix = n_trunc = 0
    for _pass in range(max_pass):
        coll = collision_report(flows)
        if not coll:
            break
        mid = sorted(((lam, pr) for lam, prs in coll.items() if lam > 0
                      for pr in prs), key=lambda x: -x[0])
        handled = set()
        changed = False
        for lam_m, (ia, ib) in mid:
            if (ia, ib) in handled:
                continue
            handled.add((ia, ib))
            tra, trb = flows[ia], flows[ib]
            pa = {lam: (s, g) for lam, s, g in tra}
            pb = {lam: (s, g) for lam, s, g in trb}
            lam_prev = min((l for l in pa if l > lam_m), default=None)
            if lam_prev is None or lam_prev not in pb:
                continue
            merged = complex(*pa[lam_m])
            da = abs(complex(*pa[lam_prev]) - merged)
            db = abs(complex(*pb[lam_prev]) - merged)
            lose, tr_l = (ia, tra) if da > db else (ib, trb)
            g_birth = tr_l[0][2]
            prev_pt = mp.mpc(*({ia: pa, ib: pb}[lose][lam_prev]))
            taken = [mp.mpc(s, g) for i2, tr2 in flows.items() if i2 != lose
                     for lam2, s, g in tr2
                     if lam2 == lam_m and not math.isnan(g)]
            fn = lambda s, l=lam_m: zb.G_lam(s, 1, l, CHI)
            wd = flow_dps(g_birth)
            zz = _local_zero_scan(fn, prev_pt, taken, workdps=wd)
            kept = [(lam, s, g) for lam, s, g in tr_l if lam > lam_m]
            if zz is None:
                flows[lose] = kept        # truncate: honest loss at the merge
                n_trunc += 1
                if verbose:
                    print(f"   flow {lose}: no rescue at lam={lam_m} -- "
                          f"truncated")
            else:
                rest = [l for l in sched if l < lam_m]
                mj = 0.6 * ac.local_spacing(g_birth, Q)
                cont = zb.flow_lambda(zz, 1, CHI, schedule=rest, workdps=wd,
                                      muller=True, max_jump=mj)
                new_tr = kept + [(lam_m, float(zz.real), float(zz.imag))]
                for lam, z in cont:
                    if z is None:
                        break
                    new_tr.append((lam, float(z.real), float(z.imag)))
                flows[lose] = new_tr
                n_fix += 1
                changed = True
                if verbose:
                    end = new_tr[-1]
                    print(f"   flow {lose}: rescued at lam={lam_m} -> "
                          f"({end[1]:+.3f}, {end[2]:.3f}) at lam={end[0]}")
        # lam = 0 double landings (recomputed each pass, after mid repairs)
        landings = {}  # type: Dict[int, List[int]]
        fates = census_fates(flows, seeds)
        for idx, (fate, _tb, m, _d) in fates.items():
            if fate == "seed" and m is not None:
                landings.setdefault(m, []).append(idx)
        for m, idxs in sorted(landings.items()):
            if len(idxs) < 2:
                continue
            s_seed = complex(seeds[m][1], seeds[m][2])
            exts = {}
            for idx in idxs:
                e = _extrapolate_to_zero(flows[idx])
                exts[idx] = abs(e - s_seed) if e is not None else float("inf")
            keep = min(exts, key=lambda i: exts[i])
            for idx in idxs:
                if idx == keep:
                    continue
                e = _extrapolate_to_zero(flows[idx])
                tr = [(lam, s, g) for lam, s, g in flows[idx] if lam > 0]
                relanded = False
                if e is not None:
                    others = [(abs(complex(sg, gm) - e), si)
                              for si, sg, gm, _k in seeds if si != m]
                    d_o, si_o = min(others)
                    if d_o < 0.6 * ac.local_spacing(seeds[si_o][2], Q):
                        zz = _local_zero_scan(
                            lambda s: zb.seed_D(s, CHI), mp.mpc(e),
                            [mp.mpc(s_seed)], radius=0.6,
                            workdps=flow_dps(flows[idx][0][2]))
                        if zz is not None:
                            tr = tr + [(0.0, float(zz.real), float(zz.imag))]
                            relanded = True
                flows[idx] = tr
                changed = True
                if relanded:
                    n_fix += 1
                    if verbose:
                        print(f"   flow {idx}: double landing on seed {m} -> "
                              f"re-landed at ({tr[-1][1]:+.3f}, "
                              f"{tr[-1][2]:.3f})")
                else:
                    n_trunc += 1
                    if verbose:
                        print(f"   flow {idx}: double landing on seed {m} -> "
                              f"truncated at lam=0.012 (unresolved)")
        if not changed:
            break
    left = collision_report(flows)
    if verbose:
        print(f"   repair: {n_fix} rescued/re-landed, {n_trunc} truncated; "
              f"residual collisions: {sum(map(len, left.values())) if left else 0}")
    rows = [(idx, lam, s, g) for idx, tr in sorted(flows.items())
            for lam, s, g in tr]
    with FLOW_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "lam", "sigma", "gamma"])
        for idx, lam, s, g in rows:
            w.writerow([idx, lam,
                        "nan" if math.isnan(s) else f"{s:.9f}",
                        "nan" if math.isnan(g) else f"{g:.9f}"])
    return flows


# the tail extension below the #43 schedule floor: G = F/lam amplifies the
# evaluator's absolute floor by 1/lam, so these need ~3-4 extra digits
LAM_TAIL = [0.008, 0.005, 0.003, 0.002]


def resolve_tails(flows, seeds, verbose=True):
    """Adjudicate flows left tracked-but-unresolved at the schedule floor
    (a trajectory ending at lam = 0.012 with no lam = 0 entry: the truncated
    double-landing losers, and rescues whose final resolve failed). The flow
    is extended down LAM_TAIL at +4 digits and the lam = 0 resolve re-tried
    with the tight jump gate: near lam = 0 the argument principle separates
    the true capture from the fleeing neighbor. A tail that dies -> the flow
    stays unresolved (counted not-seed-fated, conservatively). Writes
    FLOW_CSV; returns the dict."""
    n_land = n_flee = 0
    for idx in sorted(flows):
        tr = flows[idx]
        good = [(l, s, g) for l, s, g in tr if not math.isnan(g)]
        if not good or good[-1][0] != 0.012 \
                or any(l == 0.0 for l, _s, _g in good):
            continue
        _l0, s0, g0 = good[-1]
        g_birth = tr[0][2]
        wd = flow_dps(g_birth) + 4
        mj = 0.6 * ac.local_spacing(g_birth, Q)
        cont = zb.flow_lambda(mp.mpc(s0, g0), 1, CHI, schedule=LAM_TAIL,
                              workdps=wd, muller=True, max_jump=mj)
        new_rows = []
        for lam, z in cont:
            if z is None:
                break
            new_rows.append((lam, float(z.real), float(z.imag)))
        flows[idx] = good + new_rows
        landed = new_rows and new_rows[-1][0] == 0.0
        n_land += bool(landed)
        n_flee += not landed
        if verbose:
            if landed:
                print(f"   flow {idx}: tail-resolved -> "
                      f"({new_rows[-1][1]:+.3f}, {new_rows[-1][2]:.3f})")
            else:
                last = (new_rows or good)[-1]
                print(f"   flow {idx}: tail dies at lam={last[0]} "
                      f"({last[1]:+.3f}, {last[2]:.3f}) -- stays unresolved")
    if verbose:
        print(f"   tails: {n_land} resolved to a seed, {n_flee} unresolved")
    rows = [(idx, lam, s, g) for idx, tr in sorted(flows.items())
            for lam, s, g in tr]
    with FLOW_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "lam", "sigma", "gamma"])
        for idx, lam, s, g in rows:
            w.writerow([idx, lam,
                        "nan" if math.isnan(s) else f"{s:.9f}",
                        "nan" if math.isnan(g) else f"{g:.9f}"])
    return flows


def birth_phase_split(flows):
    """The phase-at-birth question: cos(gamma ln 2) at lam = 1, split by fate.

    Returns {fate: [(idx, t_birth, cos_at_birth)]} plus the best single-
    threshold classification accuracy over the pooled sample -- the honest
    'is fate readable at birth from the phase alone?' number."""
    split = {"seed": [], "exit": []}  # type: Dict[str, List[Tuple]]
    for idx, tr in flows.items():
        lam1, _s1, g1 = tr[0]
        if math.isnan(g1):
            continue
        split[ac.flow_fate(tr)].append((idx, g1, math.cos(g1 * LN2)))
    pooled = ([(c, 1) for _i, _t, c in split["seed"]]
              + [(c, 0) for _i, _t, c in split["exit"]])
    best = 0.0
    for thr, _lab in pooled:
        acc = max(
            sum((c >= thr) == bool(lab) for c, lab in pooled),
            sum((c < thr) == bool(lab) for c, lab in pooled)) / len(pooled)
        best = max(best, acc)
    return split, best


# --------------------------------------------------------------------------
# 5. driver
# --------------------------------------------------------------------------
def _print_census(raw, windows, seeds):
    print("== the certified raw census (K = 1 warp family, lam = 1) ==")
    print("   window            winding  found  refines  status")
    for lo, hi, nw, nf, rf, st in windows:
        print(f"   ({lo:7.2f},{hi:7.2f})   {nw:>4}   {nf:>4}     {rf}     {st}")
    n_bad = sum(st != "OK" for *_r, st in windows)
    t_hi = windows[-1][1]
    pred = rvm_count(T_LO, t_hi)
    print(f"   total: {len(raw)} zeros to t = {t_hi:.1f} "
          f"({n_bad} uncertified window(s)); RvM smooth count = {pred:.1f}")
    n_main = sum(1 for _i, _s, _g, k in seeds if k == "main")
    n_cond = len(seeds) - n_main
    print(f"   seed set: {len(seeds)} certified ({n_main} main + {n_cond} "
          f"conductor; mean-motion ln q/2pi x T = "
          f"{seed_count(T_LO, t_hi):.1f})")


def _print_fractions(fates, fr_win, seeds):
    print("\n== the seed-fated fraction vs ln q / ln(q t/2 pi) ==")
    print("   window            N   seed   f_meas   f_pred")
    for lo, hi, n, k, fm, fp in fr_win:
        print(f"   ({lo:7.2f},{hi:7.2f})  {n:>3}   {k:>3}    {fm:.3f}    "
              f"{fp:.3f}")
    n = len(fates)
    k = sum(1 for f, *_x in fates.values() if f == "seed")
    print(f"   whole box: {k}/{n} seed-fated = {k / n:.3f} "
          f"(integrated law {min(1.0, seed_count(T_LO, max(tb for _f, tb, _m, _d in fates.values())) / rvm_count(T_LO, max(tb for _f, tb, _m, _d in fates.values()))):.3f})")
    # seed coverage: how much of the certified seed set is actually reached
    hit = [m for f, _tb, m, _d in fates.values() if f == "seed"]
    multi = {m: hit.count(m) for m in set(hit) if hit.count(m) > 1}
    worst = max((d for f, _tb, _m, d in fates.values() if f == "seed"),
                default=float("nan"))
    print(f"   seed coverage: {len(set(hit))}/{len(seeds)} distinct seeds hit; "
          f"double-landings: {multi if multi else 'none'}; "
          f"worst match dist {worst:.2e}")


def _print_integrity(flows, fates):
    print("\n== flow set-integrity (argument principle: one flow per zero) ==")
    coll = collision_report(flows)
    if not coll:
        print("   no collisions at any lam level")
    else:
        for lam in sorted(coll, reverse=True):
            pairs = ", ".join(f"{a}~{b}" for a, b in coll[lam])
            print(f"   lam={lam:5.3f}: merged {pairs}")
    ends = []
    for idx, tr in flows.items():
        if fates[idx][0] == "exit":
            good = [(l, s, g) for l, s, g in tr if not math.isnan(g)]
            if good:
                ends.append((idx, good[-1]))
    if ends:
        far_left = sum(1 for _i, (_l, s, _g) in ends if s < -0.5)
        print(f"   expelled flows: {len(ends)} tracked-then-lost; {far_left} "
              f"last seen at sigma < -0.5 (fleeing left)")


def _print_phase(split, best_acc, lam_seed, lam_exit):
    print("\n== phase at birth: is fate readable at lam = 1? ==")
    for fate in ("seed", "exit"):
        cs = [c for _i, _t, c in split[fate]]
        if cs:
            m = float(np.mean(cs))
            sd = float(np.std(cs)) / math.sqrt(len(cs))
            print(f"   {fate:>5}-fated (N = {len(cs):>2}): "
                  f"<cos(gamma ln 2)> at lam = 1 = {m:+.3f} +- {sd:.3f}")
    print(f"   best single-threshold fate classifier: {best_acc:.2f} accuracy")
    print("   cohort order along the flow (lam: 1 -> 0):")
    print("   lam     seed-fated <cos>  rho_1   |  exiting <cos>  rho_1")
    ex = {r[0]: r for r in lam_exit}
    for lam, n, ms, _c3 in lam_seed:
        e = ex.get(lam)
        etxt = (f"  {e[2][1][0]:+.3f}    {e[2][1][1]:.3f} (N={e[1]})"
                if e else "        --")
        print(f"   {lam:5.3f}   {ms[1][0]:+.3f}    {ms[1][1]:.3f} (N={n})   |"
              + etxt)


def _figure(raw, windows, seeds, flows, fates, fr_win, cum, split, best_acc,
            lam_all, lam_seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figstyle
    figstyle.enlarge()
    plt.rcParams.update({"axes.titlesize": 12, "legend.fontsize": 9.5,
                         "figure.titlesize": 15})
    fig, axes = plt.subplots(1, 4, figsize=(21, 10),
                             gridspec_kw={"width_ratios": [1.15, 1, 1, 1]})
    axA, axB, axC, axD = axes

    # (A) the flow map: every trajectory, colored by fate; seeds starred
    for idx, tr in flows.items():
        pts = [(s, g) for _l, s, g in tr if not math.isnan(g)]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        col = "C0" if fates[idx][0] == "seed" else "C3"
        axA.plot(xs, ys, "-", lw=0.9, color=col, alpha=0.7)
        axA.plot(xs[0], ys[0], "k.", ms=5)
    axA.plot([s for _i, s, _g, k in seeds if k == "main"],
             [g for _i, _s, g, k in seeds if k == "main"], "*", color="C1",
             ms=11, mec="k", mew=0.4, ls="none", label="main seeds", zorder=5)
    axA.plot([s for _i, s, _g, k in seeds if k == "conductor"],
             [g for _i, _s, g, k in seeds if k == "conductor"], "D",
             color="C1", ms=7, mfc="none", ls="none", label="conductor seeds",
             zorder=5)
    axA.plot([], [], "-", color="C0", label="seed-fated flow")
    axA.plot([], [], "-", color="C3", label="expelled flow")
    axA.axvline(0, ls=":", lw=1, color="0.5")
    for lo, _hi, *_r in windows[1:]:
        axA.axhline(lo, lw=0.4, color="0.85", zorder=0)
    axA.set_xlabel(r"$\sigma$")
    axA.set_ylabel(r"$t$")
    axA.set_xlim(BOX_SIG[0] - 0.15, BOX_SIG[1] + 0.15)
    axA.set_title("the flow map: raw census $\\to$ seeds,\n"
                  "surplus expelled leftward")
    axA.legend(loc="lower left")

    # (B) the fraction curve
    for lo, hi, n, k, fm, fp in fr_win:
        tc = 0.5 * (lo + hi)
        err = math.sqrt(max(fm * (1 - fm), 0.25 / n) / n)
        axB.errorbar(tc, fm, yerr=err, fmt="o", color="C0", ms=6,
                     capsize=3)
        axB.plot([lo, hi], [fp, fp], "-", lw=2, color="C1", alpha=0.8)
    ts = np.linspace(6, windows[-1][1], 300)
    axB.plot(ts, [predicted_fraction(t) for t in ts], "--", lw=1.4,
             color="C1", label=r"$\ln q\,/\,\ln(qt/2\pi)$")
    T, fm_c, fp_c = cum
    axB.plot(T, fm_c, "-", lw=1.6, color="C0",
             label=r"cumulative $N_{\rm seed}/N$ (measured)")
    axB.plot(T, fp_c, ":", lw=1.6, color="C1",
             label="cumulative (law)")
    axB.plot([], [], "o", color="C0", label="per-window measured")
    axB.set_xlabel(r"$t$")
    axB.set_ylabel("seed-fated fraction")
    axB.set_ylim(0, 1.05)
    axB.set_title("the seed-fated fraction:\nparameter-free law vs census")
    axB.legend(loc="upper right")

    # (C) the handoff at scale (the #44 panel F, N ~ 6x larger)
    lams = [r[0] for r in lam_seed if r[0] > 0]
    axC.plot(lams, [r[2][1][0] for r in lam_seed if r[0] > 0], "o-",
             color="C0", label=r"$\langle\cos(\gamma\ln 2)\rangle$ seed-fated")
    axC.plot(lams, [r[2][1][1] for r in lam_seed if r[0] > 0], "s-",
             color="tab:cyan", label=r"$\rho_1$ seed-fated")
    lams_a = [r[0] for r in lam_all if r[0] > 0]
    axC.plot(lams_a, [r[2][1][0] for r in lam_all if r[0] > 0], "o--",
             color="C0", alpha=0.35, label="all tracked")
    n_typ = max(r[1] for r in lam_seed)
    axC.axhspan(-1 / math.sqrt(2 * n_typ), 1 / math.sqrt(2 * n_typ),
                color="0.9", zorder=0, label=r"$1/\sqrt{2N}$ noise")
    seed_row = next((r for r in lam_seed if r[0] == 0.0), None)
    if seed_row is not None:
        axC.plot([0.011], [seed_row[2][1][0]], "*", ms=15, color="C0")
    axC.set_xscale("log")
    axC.invert_xaxis()
    axC.set_xlabel(r"$\lambda$ (1 $\to$ 0; star = seed set)")
    axC.set_title("the handoff at scale:\nselection, not migration")
    axC.legend(loc="center left")

    # (D) phase at birth, by fate
    for fate, col, mk in (("seed", "C0", "o"), ("exit", "C3", "^")):
        pts = split[fate]
        axD.plot([t for _i, t, _c in pts], [c for _i, _t, c in pts], mk,
                 color=col, ms=6, ls="none", alpha=0.8,
                 label=f"{fate}-fated (N={len(pts)})")
        if pts:
            m = float(np.mean([c for _i, _t, c in pts]))
            axD.axhline(m, color=col, lw=1.2, ls="--", alpha=0.7)
    axD.axhline(0, color="0.6", lw=0.8)
    axD.set_xlabel(r"birth height $t$ (at $\lambda = 1$)")
    axD.set_ylabel(r"$\cos(\gamma\ln 2)$ at $\lambda = 1$")
    axD.set_title(f"phase at birth: cohort means split;\n"
                  f"best threshold reads fate at {best_acc:.0%}")
    axD.legend(loc="lower left")

    fig.suptitle(r"The $\lambda$-census at scale: certified K = 1 census "
                 r"$\to$ seed set, the ln q / ln(qt/2$\pi$) fraction law, "
                 "and phase-fated birth (issue #49)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    print(f"\nfigure -> {FIG_PATH}")


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recompute-scan", action="store_true",
                    help="rebuild the certified raw census (+seed set; hours "
                         "serial, see --jobs)")
    ap.add_argument("--recompute-flows", action="store_true",
                    help="rebuild the lambda-flow cache (hours; see --jobs)")
    ap.add_argument("--rescue", action="store_true",
                    help="hunt DEFICIT windows' missing zeros (append-only), "
                         "then flow the newly found ones")
    ap.add_argument("--jobs", type=int, default=1,
                    help="worker processes for the heavy steps (default 1)")
    ap.add_argument("--t-max", type=float, default=T_MAX_DEFAULT,
                    help=f"census height ceiling (default {T_MAX_DEFAULT})")
    args = ap.parse_args(argv)

    if args.recompute_scan or not RAW_CSV.exists():
        raw, windows = run_scan(t_max=args.t_max, jobs=args.jobs)
        seeds = run_seed_scan(t_max=args.t_max + SEED_MARGIN)
    else:
        raw, windows = read_raw(), read_windows()
        seeds = read_seeds()
    if args.rescue:
        raw, windows = rescue_deficits(jobs=args.jobs)
    if args.recompute_flows or not FLOW_CSV.exists():
        flows = run_flows(raw, jobs=args.jobs)
    elif args.rescue:
        flows = run_flows(raw, jobs=args.jobs, only_missing=True)
    else:
        flows = read_flows()
    print("== flow set-integrity repair (one flow per zero, one landing "
          "per seed) ==")
    flows = repair_flows(flows, seeds)
    flows = resolve_tails(flows, seeds)
    flows = repair_flows(flows, seeds)      # re-adjudicate any new landings

    fates = census_fates(flows, seeds)
    _print_census(raw, windows, seeds)
    _print_integrity(flows, fates)
    fr_win = fraction_windows(fates, windows)
    _print_fractions(fates, fr_win, seeds)
    split, best_acc = birth_phase_split(flows)
    # keep the cohort tables on the shared schedule levels: the LAM_TAIL
    # extension rows exist only for the handful of adjudicated flows
    on_sched = set(zb.LAM_SCHEDULE) | {0.0}
    lam_all = [r for r in ac.lam_instruments(flows) if r[0] in on_sched]
    lam_seed = [r for r in ac.lam_instruments(flows, fate="seed")
                if r[0] in on_sched]
    lam_exit = [r for r in ac.lam_instruments(flows, fate="exit")
                if r[0] in on_sched]
    _print_phase(split, best_acc, lam_seed, lam_exit)
    cum = fraction_cumulative(fates)
    _figure(raw, windows, seeds, flows, fates, fr_win, cum, split, best_acc,
            lam_all, lam_seed)


if __name__ == "__main__":
    _main()
