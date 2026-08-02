"""arithmetic_clock.py -- the arithmetic birth clock (issue #44, sub-issue of #37).

The question, sharpened by #43's instant-census finding: the box census of the
finite-K family L_comb_K(s, chi_3) is already complete at K = 1 (geometric birth
is a lambda-phenomenon), so the K-knob from there is pure O(1/K) displacement.
How, then, does the ARITHMETIC -- the chi-twisted explicit-formula prime
resonance that #38 measured on the finished zero set -- converge along that
displacement field?

Three instruments, run on the finite-K zero census (2-D findroot walks seeded
down the K schedule from the cached chi_3 critical zeros; no Hardy-Z at finite
K -- the finite-K family has no functional equation):

1.  The resonance timeline. R_omega(K) = <cos(gamma_K omega)> over the walked
    census, at the 2-adic tower omega = m ln 2, the dark conductor omega = ln 3,
    ln 9, and the control primes ln 5, ln 7. First-order prediction: the
    displacement field s_K - rho ~ a_1(rho)/K (a_1 = disp_coeff_chi, verified
    3% at #43) perturbs each cosine linearly, so

        R_omega(K) - R_omega(inf) ~ C_omega / K,
        C_omega = -omega < sin(gamma omega) Im a_1(rho) >,

    a COMPUTABLE constant: the resonance converges at O(1/K) exactly where the
    displacement field does. Because a_1 = (s q/2 pi^2) B(rho+1)/L'(rho) carries
    the one-period section B(s) = 1 - 2^{-s} (q = 3), Im a_1 itself oscillates
    at frequency ln 2 -- the <sin * Im a_1> average RECTIFIES at the 2-adic test
    frequencies and dies at incommensurate ones: the displacement field inherits
    exactly the section's frequency content, which is the explicit formula's
    surviving-prime set (#40's dark-conductor dropout, read through the knob).

2.  The resolution clock. The walk measures the complex amplification
    A = K(s_K - rho)/a_1 per zero per K. The asymptotic (A -> 1) regime turns
    on only past the stationary-phase band edge of the a = 1 Hurwitz moment
    (e^{2 pi i k y} y^{-s-1} has interior stationary phase y* = t/2 pi k >= 1/q
    up to k ~ q t/2 pi), so the honest clock variable is

        kappa = 2 pi K / (q t)

    -- #40's "harmonics per residue class" q/K clock with height in the role of
    the conductor (its fixed-s measurement at t ~ 2 pi makes the two coincide).
    Instrumented as a data collapse: |A - 1| against kappa across the whole
    (t, K) grid, and as the physical front t_front(K) = 2 pi K/q sweeping up
    through the sample -- arithmetic locks in behind it, stays unborn ahead.

3.  The lambda-side (the birth flow re-instrumented). The same resonance on the
    K = 1 warp family as the knob amplitude lambda sweeps 1 -> 0 (zero_birth's
    F_lam), asking where the GUE-flavored zero field hands off to the crystal:
    the seed comb near t = 2 pi m/ln 2 is phase-LOCKED (resonance -> +1 at
    every m), so the 2-adic order parameter rho_m = |<e^{i m ln 2 gamma}>|
    must climb from GUE-noise to ~1 along the flow, while the conductor
    frequency ln 3 has no seed-side structure to lock to.

Cost note (why #44 was deferred): the K-walk is a few hundred Muller solves of
L_comb_K per K value; the closed-form comb costs ~2 ms per harmonic per call,
so the full 400-zero walk runs ~1-2 h serial (--jobs parallelizes it; the
lambda flow adds ~40 min, precision-inflated by |Im s|). All heavy output is
cached in data/*.csv; the default driver replots from the caches in seconds.

Honest framing: experimental mathematics; no RH/GRH claims. The explicit-formula
resonance, stationary phase, and first-order zero perturbation are classical
individually. What has no published analog we know of (per the #37 adversarial
lit pass) is the finite-K measurement itself: the resonance timeline along the
discreteness knob, the q t/2 pi K resolution clock unifying #40's conductor
sweep with the height axis, and the lambda-flow handoff.
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

from mpmath import mp  # noqa: E402
import numpy as np  # noqa: E402

import character_bridge as cb  # noqa: E402
import chi6_two_component as c6  # noqa: E402
import zero_birth as zb  # noqa: E402

_J = mp.j
PI = mp.pi
CHI = cb.CHI3
Q = CHI.q
LN2 = float(mp.log(2))

A1_CSV = _HERE / "data" / "arithmetic_clock_a1.csv"
KWALK_CSV = _HERE / "data" / "arithmetic_clock_kwalk.csv"
LAMRAW_CSV = _HERE / "data" / "arithmetic_clock_lamraw.csv"
LAMFLOW_CSV = _HERE / "data" / "arithmetic_clock_lamflow.csv"
FIG_PATH = _HERE / "figures" / "arithmetic_clock.png"

# Layer A: the fixed-K timeline, every cached zero (geometric, ratio ~sqrt(2)).
K_SCHEDULE = [16, 24, 34, 48, 68, 96, 136, 192]
# Layer B: the clock schedule -- every KAPPA_STRIDE-th zero also walks the K
# values that realize these kappa = 2 pi K/(q t) targets, so the collapse covers
# kappa in [0.5, 4] at every height (K up to ~4 q t/2 pi ~ 1050 at t ~ 550).
KAPPA_SCHEDULE = [0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0]
KAPPA_STRIDE = 8
# the locked sub-census for the C_omega/K fit: all fit K values keep the whole
# sub-census past kappa ~ 1.9 (t <= 100 at K = 96), comfortably asymptotic.
FIT_T_MAX = 100.0
FIT_KS = [96, 136, 192]

# resonance test frequencies: (label, omega, n) with n feeding the chi-twisted
# explicit-formula amplitude (None marks the comb-commensurate 2-adic tower).
TEST_FREQS = [("ln2", LN2, 2), ("ln4", 2 * LN2, 4), ("ln8", 3 * LN2, 8),
              ("ln16", 4 * LN2, 16), ("ln3", math.log(3), 3),
              ("ln9", math.log(9), 9), ("ln5", math.log(5), 5),
              ("ln7", math.log(7), 7)]

_RESID_OK = 1e-8      # acceptance residual for the walk's Muller solves


# --------------------------------------------------------------------------
# 1. the clock variable and the displacement field
# --------------------------------------------------------------------------
def kappa_of(K, t):
    """The resolution-clock variable kappa = 2 pi K/(q t): harmonics per residue
    class per unit height. kappa ~ 1 is the a = 1/q moment's stationary-phase
    band edge; the O(1/K) displacement law is the kappa >> 1 asymptote."""
    return 2.0 * math.pi * float(K) / (Q * float(t))


def K_of_kappa(kap, t):
    """The smallest integer K whose clock reads at least kappa at height t."""
    return max(1, int(math.ceil(kap * Q * float(t) / (2.0 * math.pi))))


def local_spacing(t):
    """The mean zero spacing 2 pi/ln(q t/2 pi) at height t (RvM density)."""
    return 2.0 * math.pi / math.log(max(Q * float(t) / (2.0 * math.pi), 1.5))


def a1_of(gamma, workdps=18):
    """disp_coeff_chi at rho = 1/2 + i gamma (the comb-route first-order
    displacement coefficient), at walk-grade precision."""
    with mp.workdps(workdps):
        return complex(zb.disp_coeff_chi(mp.mpc(0.5, gamma), CHI))


def a1_table(gammas, verbose=False):
    """[(gamma, a_1)] for the whole zero sample, cached in A1_CSV."""
    if A1_CSV.exists():
        out = []
        with A1_CSV.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                out.append((float(r["gamma"]),
                            complex(float(r["a1_re"]), float(r["a1_im"]))))
        if len(out) == len(gammas):
            return out
    out = [(float(g), a1_of(g)) for g in gammas]
    A1_CSV.parent.mkdir(exist_ok=True)
    with A1_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "gamma", "a1_re", "a1_im"])
        for i, (g, a1) in enumerate(out, start=1):
            w.writerow([i, f"{g:.12f}", f"{a1.real:.8e}", f"{a1.imag:.8e}"])
    if verbose:
        print(f"   a_1 table: {len(out)} zeros -> {A1_CSV.name}")
    return out


# --------------------------------------------------------------------------
# 2. the K-walk: descend the schedule, predictor-seeded Muller solves
# --------------------------------------------------------------------------
def _muller(fn, seed, ftol, max_jump):
    """One Muller solve from a tight local triple around `seed`; None on escape
    (jump beyond max_jump -- a neighbor's basin) or a bad residual. The plain-
    secant default second iterate x0 + 1/4 stalls near packed zeros (the #43
    zero_birth gotcha), hence Muller throughout."""
    h = mp.mpc("1e-3", "1e-3")
    try:
        zz = mp.findroot(fn, (seed, seed + h, seed - mp.conj(h)),
                         solver="muller", tol=ftol, maxsteps=30)
    except (ValueError, ZeroDivisionError):
        return None
    if abs(zz - seed) > max_jump or abs(fn(zz)) > _RESID_OK:
        return None
    return mp.mpc(zz)


def _walk_step(z, K_from, K_to, a1, ftol, max_jump, depth=3):
    """One K step of the walk, first-order predictor seed + adaptive substepping:
    on failure insert the geometric-mean K and recurse (the flow_lambda pattern).
    In the pre-asymptotic zone (kappa < 1) the true motion outruns the a_1/K
    predictor by the amplification A(kappa); substeps keep each hop local."""
    seed = z + mp.mpc(a1) * (mp.mpf(1) / K_to - mp.mpf(1) / K_from)
    zz = _muller(lambda s: cb.L_comb_K(s, K_to, CHI), seed, ftol, max_jump)
    if zz is not None or depth == 0:
        return zz
    K_mid = int(round(math.sqrt(K_from * K_to)))
    if K_mid in (K_from, K_to):
        return None
    zm = _walk_step(z, K_from, K_mid, a1, ftol, max_jump, depth - 1)
    if zm is None:
        return None
    return _walk_step(zm, K_mid, K_to, a1, ftol, max_jump, depth - 1)


def walk_zero(gamma, a1, Ks, workdps=15):
    """The descendant of rho = 1/2 + i gamma through the K schedule (descending).

    Seeds the top-K solve at rho + a_1/K (the asymptotic predictor -- the top of
    every schedule is chosen at or past the clock edge, where it is accurate),
    then walks down with per-step predictors and substepping. A failed step
    falls back to a fresh asymptotic anchor before recording an honest None;
    the walk continues from the last good root either way. Returns
    [(K, s_K or None)] descending in K.
    """
    rho = mp.mpc(0.5, gamma)
    max_jump = 0.6 * local_spacing(gamma)
    out = []  # type: List[Tuple[int, Optional[mp.mpc]]]
    z = None
    K_prev = None
    with mp.workdps(workdps):
        ftol = mp.mpf(10) ** (-(workdps - 4))
        for K in sorted(Ks, reverse=True):
            fn = lambda s, K=K: cb.L_comb_K(s, K, CHI)
            if z is None:
                zz = _muller(fn, rho + mp.mpc(a1) / K, ftol, max_jump)
            else:
                zz = _walk_step(z, K_prev, K, a1, ftol, max_jump)
                if zz is None:
                    zz = _muller(fn, rho + mp.mpc(a1) / K, ftol, max_jump)
            out.append((K, zz))
            if zz is not None:
                z, K_prev = zz, K
    return out


def walk_schedule(idx, gamma):
    """The per-zero K schedule: Layer A for everyone; every KAPPA_STRIDE-th zero
    adds the K values that realize KAPPA_SCHEDULE at its height (Layer B)."""
    Ks = set(K_SCHEDULE)
    if idx % KAPPA_STRIDE == 0:
        Ks.update(K_of_kappa(kap, gamma) for kap in KAPPA_SCHEDULE)
    return sorted(Ks)


def _walk_worker(job):
    """Multiprocessing worker: one zero's full walk -> flat CSV-ready rows."""
    idx, gamma, a1_re, a1_im = job
    a1 = complex(a1_re, a1_im)
    rows = []
    for K, s_K in walk_zero(gamma, a1, walk_schedule(idx, gamma)):
        if s_K is None:
            rows.append((idx, gamma, K, float("nan"), float("nan"), "lost"))
        else:
            rows.append((idx, gamma, K, float(s_K.real), float(s_K.imag), "ok"))
    return rows


# --------------------------------------------------------------------------
# 3. census repair: per-K dedup of collided walks + local rescue scans
# --------------------------------------------------------------------------
def _local_rescue(K, sig_c, t_lo, t_hi, taken, workdps=15):
    """Recover a walk's lost root: |L_comb_K| minima on a local grid over
    (sig_c -+ 1.1) x (t_lo, t_hi), Muller-polished, excluding roots already
    `taken`. Returns the recovered root or None."""
    with mp.workdps(workdps):
        ftol = mp.mpf(10) ** (-(workdps - 4))
        fn = lambda s: cb.L_comb_K(s, K, CHI)
        sig_grid = [sig_c - 1.1 + 0.2 * i for i in range(12)]
        t_grid = [t_lo + 0.08 + 0.12 * j
                  for j in range(int((t_hi - t_lo - 0.16) / 0.12) + 1)]
        if len(t_grid) < 3:
            return None
        vals = [[abs(fn(mp.mpc(sg, tg))) for tg in t_grid] for sg in sig_grid]
        best = []
        for i, sg in enumerate(sig_grid):
            for j, tg in enumerate(t_grid):
                nb = [vals[i2][j2] for i2 in (i - 1, i, i + 1)
                      for j2 in (j - 1, j, j + 1)
                      if 0 <= i2 < len(sig_grid) and 0 <= j2 < len(t_grid)
                      and (i2, j2) != (i, j)]
                if vals[i][j] < min(nb):
                    best.append((vals[i][j], sg, tg))
        for _v, sg, tg in sorted(best):
            zz = _muller(fn, mp.mpc(sg, tg), ftol, max_jump=1.0)
            if zz is not None and t_lo < zz.imag < t_hi \
                    and all(abs(zz - w) > 1e-5 for w in taken):
                return zz
    return None


def dedup_patch(rows, verbose=True):
    """Set-integrity pass per K: walks that collided onto one root (possible in
    the deep pre-asymptotic zone, where displacements reach the mean spacing)
    are split -- the farther source zero keeps 'dup' status and gets a local
    rescue scan between its neighbors' roots ('patched' on success). Operates
    on the flat row list in place; returns the per-K repair counts."""
    byK = {}  # type: Dict[int, List[int]]
    for i, r in enumerate(rows):
        byK.setdefault(r[2], []).append(i)
    repairs = {}
    for K, idxs in sorted(byK.items()):
        ok = [i for i in idxs if rows[i][5] == "ok"]
        ok.sort(key=lambda i: rows[i][4])
        n_dup = n_fix = 0
        for a, b in zip(ok, ok[1:]):
            za = complex(rows[a][3], rows[a][4])
            zbb = complex(rows[b][3], rows[b][4])
            if abs(za - zbb) > 1e-6:
                continue
            # collision: the source gamma farther from the shared root loses it
            lose = a if abs(rows[a][1] - za.imag) > abs(rows[b][1] - za.imag) \
                else b
            n_dup += 1
            others = [complex(rows[i][3], rows[i][4]) for i in ok
                      if i != lose and rows[i][5] == "ok"]
            g = rows[lose][1]
            below = max([w.imag for w in others if w.imag < za.imag - 1e-6],
                        default=g - 1.2 * local_spacing(g))
            above = min([w.imag for w in others if w.imag > za.imag + 1e-6],
                        default=g + 1.2 * local_spacing(g))
            zz = _local_rescue(K, 0.5, below, above, [complex(w) for w in others])
            if zz is None:
                rows[lose] = rows[lose][:5] + ("dup",)
            else:
                rows[lose] = (rows[lose][0], g, K, float(zz.real),
                              float(zz.imag), "patched")
                n_fix += 1
        if n_dup:
            repairs[K] = (n_dup, n_fix)
            if verbose:
                print(f"   K={K:>4}: {n_dup} collided walk(s), {n_fix} rescued")
    return repairs


# --------------------------------------------------------------------------
# 4. the K-walk runner and its cache
# --------------------------------------------------------------------------
def run_kwalk(jobs=1, verbose=True):
    """Walk every cached chi_3 zero down its K schedule, repair the census, and
    cache the rows. The heavy step (--recompute-kwalk); ~1-2 h serial at
    jobs=1, ~/jobs with multiprocessing."""
    gammas = c6.load_chi3_zeros()
    a1s = a1_table(gammas, verbose=verbose)
    jobsq = [(i, g, a1.real, a1.imag) for i, (g, a1) in enumerate(a1s)]
    t0 = time.time()
    rows = []  # type: List[Tuple]
    if jobs > 1:
        import multiprocessing as mp_proc
        with mp_proc.get_context("spawn").Pool(jobs) as pool:
            for i, rr in enumerate(pool.imap_unordered(_walk_worker, jobsq)):
                rows.extend(rr)
                if verbose and (i + 1) % 25 == 0:
                    print(f"   walked {i + 1}/{len(jobsq)} zeros "
                          f"({time.time() - t0:.0f} s)")
    else:
        for i, job in enumerate(jobsq):
            rows.extend(_walk_worker(job))
            if verbose and (i + 1) % 25 == 0:
                print(f"   walked {i + 1}/{len(jobsq)} zeros "
                      f"({time.time() - t0:.0f} s)")
    rows.sort(key=lambda r: (r[0], -r[2]))
    dedup_patch(rows, verbose=verbose)
    KWALK_CSV.parent.mkdir(exist_ok=True)
    with KWALK_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "gamma", "K", "sigma_K", "gamma_K", "status"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.12f}", r[2],
                        "nan" if r[5] in ("lost", "dup") else f"{r[3]:.9f}",
                        "nan" if r[5] in ("lost", "dup") else f"{r[4]:.9f}",
                        r[5]])
    if verbose:
        n_bad = sum(r[5] in ("lost", "dup") for r in rows)
        print(f"   K-walk: {len(rows)} rows ({n_bad} unrecovered) in "
              f"{time.time() - t0:.0f} s -> {KWALK_CSV.name}")
    return rows


def read_kwalk():
    """The cached walk rows: [(n, gamma, K, sigma_K, gamma_K, status)]."""
    rows = []
    with KWALK_CSV.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append((int(r["n"]), float(r["gamma"]), int(r["K"]),
                         float(r["sigma_K"]), float(r["gamma_K"]), r["status"]))
    return rows


# --------------------------------------------------------------------------
# 5. instruments -- the resonance timeline and the C_omega law
# --------------------------------------------------------------------------
def resonance_constant(gammas, a1s, omega):
    """C_omega = -omega <sin(gamma omega) Im a_1>: the computable first-order
    convergence constant of the resonance at test frequency omega."""
    g = np.asarray(gammas, dtype=float)
    ia = np.asarray([a.imag for a in a1s], dtype=float)
    return float(-omega * np.mean(np.sin(g * omega) * ia))


def field_prediction(gammas, a1s, omega, K):
    """The full first-order-field prediction <cos((gamma + Im a_1/K) omega)>:
    the exact cosine of the predicted displacement (no linearization of cos),
    isolating O(1/K^2) field errors from linearization error."""
    g = np.asarray(gammas, dtype=float)
    ia = np.asarray([a.imag for a in a1s], dtype=float)
    return float(np.mean(np.cos((g + ia / K) * omega)))


def timeline(rows, a1s_by_n, Ks=None, t_max=None):
    """The resonance table: {K: {label: (measured, field_pred, linear_pred)}}
    plus the K = inf reference {label: R_inf} and the census size per K.

    `measured` runs over the walked census (ok + patched rows) at each K;
    the two predictions run over the SAME source-zero subset, so a handful of
    unrecovered walks cannot masquerade as resonance drift."""
    Ks = sorted(set(r[2] for r in rows)) if Ks is None else Ks
    src = sorted({(r[0], r[1]) for r in rows
                  if t_max is None or r[1] <= t_max})
    g_src = np.array([g for _n, g in src])
    ref = {label: c6.prime_resonance(g_src, omega)
           for label, omega, _n in TEST_FREQS}
    out = {}
    counts = {}
    for K in Ks:
        sel = [r for r in rows if r[2] == K and r[5] in ("ok", "patched")
               and (t_max is None or r[1] <= t_max)]
        if not sel:
            continue
        gK = np.array([r[4] for r in sel])
        g0 = np.array([r[1] for r in sel])
        a1 = [a1s_by_n[r[0]] for r in sel]
        out[K] = {}
        counts[K] = len(sel)
        for label, omega, _n in TEST_FREQS:
            r_inf = c6.prime_resonance(g0, omega)
            out[K][label] = (c6.prime_resonance(gK, omega),
                             field_prediction(g0, a1, omega, K),
                             r_inf + resonance_constant(g0, a1, omega) / K)
    return out, ref, counts


def locked_fit(rows, a1s_by_n):
    """The Q1 verdict: on the locked sub-census (t <= FIT_T_MAX, all FIT_KS past
    kappa ~ 1.9), fit K Delta R = C + c'/K and compare the intercept to the
    computable C_omega. Returns {label: (C_fit, C_pred, slope)}."""
    byK = {K: {r[0]: r for r in rows if r[2] == K and r[1] <= FIT_T_MAX
               and r[5] in ("ok", "patched")} for K in FIT_KS}
    common = sorted(set.intersection(*(set(d) for d in byK.values())))
    g0 = np.array([next(r[1] for r in rows if r[0] == n) for n in common])
    a1 = [a1s_by_n[n] for n in common]
    out = {}
    for label, omega, _n in TEST_FREQS:
        r_inf = c6.prime_resonance(g0, omega)
        xs, ys = [], []
        for K in FIT_KS:
            gK = np.array([byK[K][n][4] for n in common])
            xs.append(1.0 / K)
            ys.append(K * (c6.prime_resonance(gK, omega) - r_inf))
        if len(xs) >= 2:
            slope, C_fit = np.polyfit(xs, ys, 1)
            out[label] = (float(C_fit), resonance_constant(g0, a1, omega),
                          float(slope), xs, ys)
    return out


# --------------------------------------------------------------------------
# 5b. the C(omega) spectrum: the convergence constant as an arithmetic object
# --------------------------------------------------------------------------
# The frequency-swept constant C(omega) is a Landau-formula-flavored object:
# sum_gamma sin(gamma omega) weighted by Im a_1. Landau's classical formula
# (sum_{gamma <= T} x^rho concentrating at prime powers x, chi-twisted here)
# says the UNWEIGHTED sum has lines at omega = ln p^k with amplitude
# ~ Lambda(p^k)/sqrt(p^k); the a_1 weight keeps that line set -- measured below
# -- with a structured 2-adic anomaly (the B(rho+1) = 1 - 2^{-rho-1} factor in
# a_1 modulates the weight at frequency ln 2, putting +-ln2 sidebands on every
# line: ln2 halved, ln4 on prediction, ln8 nearly extinguished) and the
# conductor tower ln 3, ln 9 DARK at the control floor: chi_3(3^k) = 0 kills
# the channel exactly as in the explicit formula. Lines are window-limited
# (lobe width ~ pi/T ~ 0.006 at T ~ 550), so band maxima, not point values.
C_BAND_NS = [("ln2", 2), ("ln4", 4), ("ln8", 8), ("ln3", 3), ("ln9", 9),
             ("ln5", 5), ("ln7", 7), ("ln11", 11), ("ln13", 13)]
C_CONTROLS = [0.90, 1.25, 1.50, 1.87, 2.01, 2.25]   # off ln(n) for all n <= 16


def _von_mangoldt(n):
    """Lambda(n): ln p if n = p^k, else 0 (tiny n only)."""
    for p in range(2, n + 1):
        if n % p == 0:
            while n % p == 0:
                n //= p
            return math.log(p) if n == 1 else 0.0
    return 0.0


def c_spectrum(gammas, a1s, w_lo=0.4, w_hi=2.7, dw=1e-3):
    """(omegas, C(omega)) on a fine grid -- the figure's spectrum panel."""
    ws = np.arange(w_lo, w_hi, dw)
    return ws, np.array([resonance_constant(gammas, a1s, w) for w in ws])


def c_band_table(gammas, a1s, half=0.02, dw=4e-4):
    """Band maxima of |C| around the named frequencies and the controls:
    rows (label, center, max|C|, argmax omega, landau_ref) with landau_ref the
    chi-weighted Landau amplitude |Lambda(n)|/sqrt(n) under one common scale
    fitted on the odd-prime lines (0 for controls)."""
    raw = []
    for label, n in C_BAND_NS:
        w0 = math.log(n)
        ws = np.arange(w0 - half, w0 + half, dw)
        Cs = np.array([resonance_constant(gammas, a1s, w) for w in ws])
        i = int(np.argmax(np.abs(Cs)))
        raw.append([label, w0, float(abs(Cs[i])), float(ws[i]),
                    _von_mangoldt(n) / math.sqrt(n)])
    scale_pts = [(r[2], r[4]) for r in raw
                 if r[0] in ("ln5", "ln7", "ln11", "ln13")]
    scale = (sum(m for m, _l in scale_pts)
             / max(sum(l for _m, l in scale_pts), 1e-12))
    out = [(r[0], r[1], r[2], r[3], r[4] * scale) for r in raw]
    for w0 in C_CONTROLS:
        ws = np.arange(w0 - half, w0 + half, dw)
        Cs = np.array([resonance_constant(gammas, a1s, w) for w in ws])
        i = int(np.argmax(np.abs(Cs)))
        out.append(("ctl", w0, float(abs(Cs[i])), float(ws[i]), 0.0))
    return out


# --------------------------------------------------------------------------
# 6. instruments -- the resolution-clock collapse
# --------------------------------------------------------------------------
def collapse_points(rows, a1s_by_n):
    """[(t, K, kappa, A)] with A = K(s_K - rho)/a_1 the complex amplification,
    over every ok walk row (patched rows are census repairs, not trajectories --
    their per-zero identity is exactly what was in doubt, so they stay out)."""
    pts = []
    for n, g, K, sig, gK, status in rows:
        if status != "ok":
            continue
        a1 = a1s_by_n[n]
        if a1 == 0:
            continue
        A = K * complex(sig - 0.5, gK - g) / a1
        pts.append((g, K, kappa_of(K, g), A))
    return pts


def collapse_fit(pts, kappa_min=2.0):
    """Least-squares c_1/kappa + c_2/kappa^2 through |A - 1| on the asymptotic
    side (kappa >= kappa_min; the knee at kappa ~ 1-2 is outside the expansion
    and biases c_1 negative if included). Returns (c_1, c_2)."""
    xs = np.array([p[2] for p in pts if p[2] >= kappa_min])
    ys = np.array([abs(p[3] - 1) for p in pts if p[2] >= kappa_min])
    M = np.column_stack([1.0 / xs, 1.0 / xs ** 2])
    (c1, c2), *_ = np.linalg.lstsq(M, ys, rcond=None)
    return float(c1), float(c2)


def collapse_binned(pts, n_bins=14):
    """Median |A - 1| in log-spaced kappa bins: (bin centers, medians)."""
    ka = np.array([p[2] for p in pts])
    ab = np.array([abs(p[3] - 1) for p in pts])
    edges = np.geomspace(ka.min() * 0.999, ka.max() * 1.001, n_bins + 1)
    cent, med = [], []
    for lo, hi in zip(edges, edges[1:]):
        m = (ka >= lo) & (ka < hi)
        if m.sum() >= 4:
            cent.append(math.sqrt(lo * hi))
            med.append(float(np.median(ab[m])))
    return np.array(cent), np.array(med)


# --------------------------------------------------------------------------
# 7. the lambda-side: the birth flow re-instrumented for resonance
# --------------------------------------------------------------------------
def run_lamflow(t_max=60.0, verbose=True):
    """Raw K = 1 warp-family zeros (blind 2-D scan, the #43 box widened to
    t <= t_max) flowed to lambda -> 0; trajectories cached. The second heavy
    step (--recompute-lamflow, ~40 min: the warp evaluator's precision knobs
    inflate with |Im s|)."""
    import warp_alpha as wa
    t0 = time.time()
    raw = wa.find_zeros(lambda s: zb.F_lam(s, 1, 1, CHI), (-3.2, 2.0),
                        (2.0, t_max), dsig=0.3, dt=0.6)
    if verbose:
        print(f"   raw K=1 family: {len(raw)} zeros to t = {t_max:.0f} "
              f"({time.time() - t0:.0f} s)")
    with LAMRAW_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "sigma", "gamma"])
        for i, z in enumerate(raw):
            w.writerow([i, f"{float(z.real):.9f}", f"{float(z.imag):.9f}"])
    with LAMFLOW_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "lam", "sigma", "gamma"])
        for i, z0 in enumerate(raw):
            traj = zb.flow_lambda(z0, 1, CHI)
            for lam, z in traj:
                w.writerow([i, lam] + (["nan", "nan"] if z is None else
                                       [f"{float(z.real):.9f}",
                                        f"{float(z.imag):.9f}"]))
            if verbose:
                end = traj[-1]
                tag = (f"seed {complex(end[1]):.3f}" if end[0] == 0.0
                       and end[1] is not None else "exits/lost")
                print(f"      flow {i:>2} from {complex(z0):.3f}: {tag} "
                      f"({time.time() - t0:.0f} s)")
    return read_lamflow()


def read_lamflow():
    """{idx: [(lam, sigma, gamma)]} descending in lam (nan = honest loss)."""
    out = {}  # type: Dict[int, List[Tuple[float, float, float]]]
    with LAMFLOW_CSV.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(int(r["idx"]), []).append(
                (float(r["lam"]), float(r["sigma"]), float(r["gamma"])))
    return out


def flow_fate(traj):
    """'seed' if the trajectory lands on the lam = 0 seed set, 'exit' if it is
    lost mid-flow (the #43 surplus fleeing every compact leftward)."""
    lam_end, _s, g_end = traj[-1]
    return "seed" if lam_end == 0.0 and not math.isnan(g_end) else "exit"


def lam_instruments(flows, fate=None):
    """Per-lambda 2-adic order over the flowed set: rows
    (lam, n_alive, {m: (cos_m, rho_m)}, cos_ln3) with cos_m = <cos(m ln2 gamma)>
    and rho_m = |<e^{i m ln2 gamma}>| (the phase-locking order parameter).
    fate='seed' restricts to trajectories that land on the seed set (the
    lambda -> 0 census); fate='exit' to the fleeing surplus; None keeps all."""
    keep = {i: tr for i, tr in flows.items()
            if fate is None or flow_fate(tr) == fate}
    lams = sorted({lam for tr in keep.values() for lam, _s, _g in tr},
                  reverse=True)
    out = []
    for lam in lams:
        gs = [g for tr in keep.values() for lm, _s, g in tr
              if lm == lam and not math.isnan(g)]
        if len(gs) < 3:
            continue
        g = np.array(gs)
        ms = {}
        for m in (1, 2):
            ph = m * LN2 * g
            ms[m] = (float(np.cos(ph).mean()),
                     float(abs(np.exp(1j * ph).mean())))
        out.append((lam, len(gs), ms, float(np.cos(math.log(3) * g).mean())))
    return out


# --------------------------------------------------------------------------
# 8. driver
# --------------------------------------------------------------------------
def _print_timeline(tl, ref, counts):
    print("== the resonance timeline: R_omega(K) over the walked census ==")
    labels = ["ln2", "ln4", "ln8", "ln3", "ln5"]
    print("   K      N " + "".join(f"{lb + ' meas/field':>19}" for lb in labels))
    for K in sorted(tl):
        row = f"   {K:>4} {counts[K]:>4}"
        for lb in labels:
            m, fp, _lin = tl[K][lb]
            row += f"    {m:+.4f}/{fp:+.4f}"
        print(row)
    print("    inf     " + "".join(f"    {ref[lb]:+.4f} (#38 ref)"[:19].ljust(19)
                                   for lb in labels))


def _print_locked(lock):
    print("\n== the locked sub-census (t <= %.0f, K in %s): K Delta R -> "
          "C_omega ==" % (FIT_T_MAX, FIT_KS))
    print("   omega     C_fit      C_pred     |resid|   fit slope c'")
    for lb, (C_fit, C_pred, slope, _xs, _ys) in lock.items():
        print(f"   {lb:>5}   {C_fit:+.5f}   {C_pred:+.5f}   "
              f"{abs(C_fit - C_pred):.5f}   {slope:+.3f}")


def _print_collapse(pts, c1, c2):
    print("\n== the resolution clock: |A - 1| vs kappa = 2 pi K/(q t) ==")
    cent, med = collapse_binned(pts)
    for c, m in zip(cent, med):
        bar = "#" * int(min(m, 2.0) * 30)
        print(f"   kappa {c:6.2f}: median |A-1| = {m:6.3f}  {bar}")
    print(f"   asymptotic fit (kappa >= 2): |A-1| ~ {c1:.3f}/kappa + "
          f"{c2:.3f}/kappa^2")


def _print_cbands(bands):
    print("\n== the C(omega) spectrum: band maxima vs the chi-twisted Landau "
          "line set ==")
    print("   band   center    max|C|   at omega   Landau ref   meas/ref")
    for lb, w0, mx, wm, ref in bands:
        ratio = f"{mx / ref:8.2f}" if ref > 0 else "       -"
        print(f"   {lb:>5}  {w0:.4f}   {mx:6.3f}   {wm:.4f}    {ref:8.3f} "
              f"  {ratio}")
    ctl = [mx for lb, _w, mx, _wm, _r in bands if lb == "ctl"]
    print(f"   control floor: {min(ctl):.3f} .. {max(ctl):.3f} "
          "(the conductor bands ln3, ln9 sit inside it)")


def _print_dark(tl, ref, counts):
    print("\n== the dark conductor along the knob ==")
    print("   K       ln3       ln9    | ln2 (for scale)   noise ~ 1/sqrt(2N)")
    for K in sorted(tl):
        noise = 1.0 / math.sqrt(2 * counts[K])
        print(f"   {K:>4}  {tl[K]['ln3'][0]:+.4f}   {tl[K]['ln9'][0]:+.4f}"
              f"   | {tl[K]['ln2'][0]:+.4f}            {noise:.3f}")
    print(f"    inf  {ref['ln3']:+.4f}   {ref['ln9']:+.4f}   | "
          f"{ref['ln2']:+.4f}")


def _print_lambda(lam_all, lam_seed):
    print("\n== the lambda-side handoff (K = 1 family, flowed census) ==")
    for tag, lam_rows in (("all tracked flows", lam_all),
                          ("seed-fated flows only", lam_seed)):
        print(f"   -- {tag} --")
        print("   lam     N    <cos(g ln2)>  rho_1   <cos(2 g ln2)>  rho_2   "
              "<cos(g ln3)>")
        for lam, n, ms, c3 in lam_rows:
            print(f"   {lam:5.3f}  {n:>3}     {ms[1][0]:+.3f}     {ms[1][1]:.3f}"
                  f"       {ms[2][0]:+.3f}     {ms[2][1]:.3f}      {c3:+.3f}")


def _figure(tl, ref, counts, lock, pts, c1, c2, lam_rows, rows, a1s_by_n,
            spec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figstyle
    figstyle.enlarge()
    plt.rcParams["axes.titlesize"] = 11.5
    plt.rcParams["figure.titlesize"] = 15

    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))
    axA, axB, axC = axes[0]
    axD, axE, axF = axes[1]

    # (A) the resonance timeline at the first two 2-adic modes
    Ks = sorted(tl)
    xs = [1.0 / K for K in Ks]
    for lb, color in (("ln2", "tab:blue"), ("ln4", "tab:orange")):
        axA.plot(xs, [tl[K][lb][0] for K in Ks], "o-", color=color,
                 label=f"measured, omega = {lb}")
        axA.plot(xs, [tl[K][lb][1] for K in Ks], "--", color=color,
                 alpha=0.65, label=f"a_1-field prediction ({lb})")
        axA.axhline(ref[lb], color=color, ls=":", lw=1.2)
    axA.set_xlabel("1/K")
    axA.set_ylabel(r"$\langle\cos(\gamma_K\,\omega)\rangle$")
    axA.set_title("The resonance timeline (full census; dotted = K = inf)")
    axA.legend(fontsize=8.5)

    # (B) the locked sub-census: K Delta R vs 1/K with the C_omega intercepts
    for lb, color in (("ln2", "tab:blue"), ("ln4", "tab:orange"),
                      ("ln8", "tab:green")):
        if lb not in lock:
            continue
        C_fit, C_pred, slope, xk, yk = lock[lb]
        xk = np.array(xk)
        axB.plot(xk, yk, "o", color=color)
        xx = np.linspace(0, xk.max() * 1.05, 40)
        axB.plot(xx, C_fit + slope * xx, "-", color=color, lw=1.2,
                 label=f"{lb}: fit -> {C_fit:+.4f}")
        axB.plot(0, C_pred, "*", ms=14, color=color,
                 label=f"{lb}: C_pred = {C_pred:+.4f}")
    axB.set_xlabel("1/K")
    axB.set_ylabel(r"$K\,\Delta R_\omega(K)$")
    axB.set_title(f"O(1/K) with the computable constant (t <= {FIT_T_MAX:.0f})")
    axB.legend(fontsize=8.5)

    # (C) the clock collapse
    ts = np.array([p[0] for p in pts])
    ka = np.array([p[2] for p in pts])
    ab = np.array([abs(p[3] - 1) for p in pts])
    sc = axC.scatter(ka, np.maximum(ab, 1e-4), s=7, c=ts, cmap="viridis",
                     alpha=0.5)
    cent, med = collapse_binned(pts)
    axC.plot(cent, med, "r-", lw=2, label="binned median")
    kk = np.geomspace(1.0, ka.max(), 50)
    axC.plot(kk, c1 / kk + c2 / kk ** 2, "k--", lw=1.5,
             label=f"{c1:.2f}/kappa + {c2:.2f}/kappa^2")
    axC.axvline(1.0, color="gray", ls=":")
    axC.set_xscale("log")
    axC.set_yscale("log")
    axC.set_xlabel(r"$\kappa = 2\pi K / (q\,t)$")
    axC.set_ylabel(r"$|A - 1|,\ A = K(s_K-\rho)/a_1$")
    axC.set_title("The resolution clock: one collapse across (t, K)")
    axC.legend(fontsize=8.5)
    fig.colorbar(sc, ax=axC, label="t")

    # (D) the C(omega) spectrum: the convergence constant is arithmetic
    ws, Cs = spec
    axD.plot(ws, np.abs(Cs), "-", color="0.4", lw=0.7)
    for lb, n in C_BAND_NS:
        w0 = math.log(n)
        dark = lb in ("ln3", "ln9")
        axD.axvline(w0, color="tab:red" if dark else "tab:blue",
                    ls=":" if dark else "--", lw=1.1, alpha=0.8)
        axD.annotate(lb, (w0, axD.get_ylim()[1] * 0.9), fontsize=8,
                     ha="center",
                     color="tab:red" if dark else "tab:blue")
    axD.set_xlabel(r"test frequency $\omega$")
    axD.set_ylabel(r"$|C(\omega)|$")
    axD.set_title("The convergence constant is arithmetic: Landau lines at "
                  "ln p^k, conductor dark (red)")

    # (E) the dark conductor timeline
    for lb, color, mark in (("ln2", "tab:blue", "o"), ("ln5", "tab:green", "s"),
                            ("ln3", "tab:red", "^"), ("ln9", "tab:pink", "v")):
        axE.plot([1.0 / K for K in Ks], [abs(tl[K][lb][0]) for K in Ks],
                 mark + "-", color=color, label=f"|R| at {lb}")
    noise = [1.0 / math.sqrt(2 * counts[K]) for K in Ks]
    axE.plot([1.0 / K for K in Ks], noise, "k:", label="1/sqrt(2N) noise")
    axE.set_xlabel("1/K")
    axE.set_ylabel(r"$|R_\omega(K)|$")
    axE.set_title("The conductor prime stays dark at every K")
    axE.legend(fontsize=8.5)

    # (F) the lambda handoff: seed-fated flows solid, all-tracked faint
    lam_all, lam_seed = lam_rows
    lams = [r[0] for r in lam_seed if r[0] > 0]
    axF.plot(lams, [r[2][1][0] for r in lam_seed if r[0] > 0], "o-",
             color="tab:blue", label=r"$\langle\cos(\gamma\ln 2)\rangle$ (seed-fated)")
    axF.plot(lams, [r[2][1][1] for r in lam_seed if r[0] > 0], "s-",
             color="tab:cyan", label=r"$\rho_1$ (order parameter)")
    axF.plot(lams, [r[3] for r in lam_seed if r[0] > 0], "^-",
             color="tab:red", label=r"$\langle\cos(\gamma\ln 3)\rangle$")
    lams_a = [r[0] for r in lam_all if r[0] > 0]
    axF.plot(lams_a, [r[2][1][0] for r in lam_all if r[0] > 0], "o--",
             color="tab:blue", alpha=0.35, label="all tracked flows")
    seed_row = next((r for r in lam_seed if r[0] == 0.0), None)
    if seed_row is not None:
        axF.plot([0.011], [seed_row[2][1][0]], "*", ms=15, color="tab:blue")
        axF.plot([0.011], [seed_row[3]], "*", ms=15, color="tab:red")
    axF.set_xscale("log")
    axF.invert_xaxis()
    axF.set_xlabel("lambda (1 -> 0: the birth flow, stars = the seed set)")
    axF.set_title("The lambda handoff: 2-adic lock climbs, conductor stays flat")
    axF.legend(fontsize=8.5)

    fig.suptitle("The arithmetic birth clock: chi_3 prime resonance along the "
                 "K-family (issue #44)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIG_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    print(f"\nfigure -> {FIG_PATH}")


def _spot_certify(rows, verbose=True):
    """Census spot-certification: winding counts of L_comb_K over two bulk
    windows must equal the walked-census count there (set completeness, the
    #43 instrument)."""
    print("\n== census spot-certification (argument principle) ==")
    ok = True
    for K, (t_lo, t_hi) in ((48, (200.31, 210.83)), (192, (540.17, 549.61))):
        walked = [r for r in rows if r[2] == K and r[5] in ("ok", "patched")
                  and t_lo < r[4] < t_hi]
        n_wind = zb.winding_count(lambda s: cb.L_comb_K(s, K, CHI),
                                  (-0.97, 2.03), (t_lo, t_hi),
                                  dsig=0.3, dt=0.35)
        tag = "OK" if n_wind == len(walked) else "MISMATCH"
        ok &= n_wind == len(walked)
        print(f"   K={K:>4}, t in ({t_lo}, {t_hi}): winding {n_wind}, "
              f"walked {len(walked)}   {tag}")
    return ok


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recompute-kwalk", action="store_true",
                    help="rebuild the K-walk cache (~1-2 h serial; see --jobs)")
    ap.add_argument("--recompute-lamflow", action="store_true",
                    help="rebuild the lambda-flow cache (~40 min)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="worker processes for the K-walk (default 1)")
    ap.add_argument("--t-max-lam", type=float, default=60.0,
                    help="height ceiling for the lambda-side box (default 60)")
    ap.add_argument("--skip-certify", action="store_true",
                    help="skip the winding spot-checks (~2 min)")
    args = ap.parse_args(argv)

    gammas = c6.load_chi3_zeros()
    a1s = a1_table(gammas, verbose=True)
    a1s_by_n = {i: a1 for i, (_g, a1) in enumerate(a1s)}

    if args.recompute_kwalk or not KWALK_CSV.exists():
        rows = run_kwalk(jobs=args.jobs)
    else:
        rows = read_kwalk()
    if args.recompute_lamflow or not LAMFLOW_CSV.exists():
        flows = run_lamflow(t_max=args.t_max_lam)
    else:
        flows = read_lamflow()

    tl, ref, counts = timeline(rows, a1s_by_n, Ks=K_SCHEDULE)
    _print_timeline(tl, ref, counts)
    lock = locked_fit(rows, a1s_by_n)
    _print_locked(lock)
    pts = collapse_points(rows, a1s_by_n)
    c1, c2 = collapse_fit(pts)
    _print_collapse(pts, c1, c2)
    _print_dark(tl, ref, counts)
    g_src = np.array([g for g, _a in a1s])
    a1_src = [a for _g, a in a1s]
    bands = c_band_table(g_src, a1_src)
    _print_cbands(bands)
    lam_all = lam_instruments(flows)
    lam_seed = lam_instruments(flows, fate="seed")
    _print_lambda(lam_all, lam_seed)
    if not args.skip_certify:
        _spot_certify(rows)
    spec = c_spectrum(g_src, a1_src)
    _figure(tl, ref, counts, lock, pts, c1, c2, (lam_all, lam_seed), rows,
            a1s_by_n, spec)


if __name__ == "__main__":
    _main()
