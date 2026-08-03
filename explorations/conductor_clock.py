"""conductor_clock.py -- the resolution clock across conductors (issue #48).

#44 measured, at q = 3 only, that the displacement law s_K - rho ~ a_1/K turns
on past the resolution clock kappa = 2 pi K/(q t) ~ 1 and that the per-zero
amplification A = K(s_K - rho)/a_1 data-collapses in kappa across the whole
(t, K) grid. #47 read the convergence-constant spectrum C(omega) as a MOEBIUS
line spectrum under a one-sided section dressing. This module repeats the bulk
K-walk and the spectrum at two more conductors and tests the #48 predictions:

1.  UNIVERSALITY (prediction 1). The |A - 1|(kappa) collapse overlaid at
    q = 3 (the committed #44 cache), q = 4, and q = 7: same knee at kappa ~ 1,
    same asymptotic decay (0.13/kappa + 0.29/kappa^2 at q = 3), each conductor
    walked at conductor-scaled K schedules (Layer A = the #44 schedule x q/3,
    so the kappa coverage is identical by construction -- the #40 lesson that
    matched resolution is K proportional to q).

2.  THE SWAP TEST (prediction 2, the strong one). At q = 4, chi_4(2) = 0 makes
    2 the conductor prime: the ENTIRE 2-divisible column goes dark in one
    stroke (ln 2, ln 4, ln 8 -- and the even squarefree mu-lines ln 10, ln 14,
    ln 22 with them, the #47 sharpening), while ln 3, ln 5, ln 7 are live:
    #44's dark conductor with the roles of 2 and 3 exchanged. The section
    B(s) = 1 - 3^{-s} dresses the live lines one-sidedly at +ln 3 with
    beta = 3^{-3/2}: 3-divisible squarefree lines at EXACTLY 2/3 of Moebius,
    and the ln 9 vacancy satellite at 2/3 of the Landau reference -- measurably
    non-Landau, the discriminator q = 3 structurally could not supply (its
    ln 4 satellite is Landau-exact by the p = 2 identity 2/p = 1).

3.  THE CONDUCTOR DROPOUT (prediction 3). At q = 7 the C-spectrum deletes
    exactly the p | q columns (ln 7, ln 14, ln 21, ln 35, ln 49) while every
    n coprime to 7 -- INCLUDING the composite squarefree ln 6, ln 10 that both
    smaller conductors kill -- carries its Moebius line.

4.  THE DRESSING COMB, AND A CORRECTION TO #47's SIGN-FLIP PREDICTION. At
    general q the section is a full comb (a sideband at every ln a, a < q a
    unit), and the dressed line amplitude closes rationally (mobius_dressing.
    dressed_ref): scale ln n |chi(n)| |sum_{a | n, a < q, (a,q)=1} mu(n/a)/a|
    / sqrt n. The chi factor CANCELS between each satellite's section
    coefficient chi(a) and the direct line's chi(n) = chi(a) chi(n/a), so
    #47's logged sign-flip test (chi(2) = +1 conductors show 2-divisible
    lines ENHANCED x 3/2) is refuted by the algebra: the model says the ln 2
    line at chi_7(2) = +1 is HALVED, exactly as at q = 3 where chi_3(2) = -1
    -- the suppression is chi-independent. The q = 7 measurement adjudicates
    the two closed forms (1/2 vs 3/2, a factor 3: far outside band noise).

Instruments transfer from arithmetic_clock (walk, repair, certification,
timeline, collapse) with chi threaded through, and from mobius_dressing (the
spectrum, band maxima, and the three reference models, chi-general). Bulk
critical-line zeros come from the #40 Hardy-Z walk machinery
(conductor_sweep.walk_zeros: eps = +1 for real primitive chi), deepened to
t <= 550 at both conductors -- the C(omega) window needs the T ~ 550 lobe
width to resolve the dense q = 7 line set (ln 34/ln 35, ln 47/ln 49/ln 51).
The q = 7 K-WALK stops at t <= 250: the collapse is per-zero, and the walk
cost scales as q x units(q), so depth buys nothing there.

Cost note: the two K-walks are the heavy steps (--recompute-kwalk, each
~2-2.5 h serial, /jobs with --jobs). All heavy output is cached in
data/conductor_clock_*.csv; the default driver replots from the caches.

Honest framing: experimental mathematics; no RH/GRH claims. Landau's formula
(chi-twisted by Banks/Fujii), the M_chi explicit formula and 1/L'-weighted
discrete moments (Titchmarsh 14.27; Gonek; Ng), and first-order zero
perturbation are classical. What has no published analog we know of (per the
#37/#47 adversarial lit passes) is the finite-K measurement program itself:
the conductor-swept kappa collapse, and the Moebius-under-section-dressing
spectrum read across conductors with its exact rational dressing laws.
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Dict

_HERE = Path(__file__).resolve().parent
_BRIDGE = _HERE.parent / "bridge"
for _p in (str(_HERE), str(_BRIDGE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mpmath import mp  # noqa: E402
import numpy as np  # noqa: E402

import arithmetic_clock as ac  # noqa: E402
import character_bridge as cb  # noqa: E402
import chi6_two_component as c6  # noqa: E402
import conductor_sweep as cs  # noqa: E402
import mobius_dressing as md  # noqa: E402
import zero_birth as zb  # noqa: E402

DATA = _HERE / "data"
FIG_PATH = _HERE / "figures" / "conductor_clock.png"

LN = math.log


def _scaled_schedule(q):
    """Layer A: the #44 K schedule scaled by q/3 -- identical kappa coverage
    at every conductor (matched resolution K ~ q, the #40 lesson)."""
    return [int(round(K * q / 3.0)) for K in ac.K_SCHEDULE]


# Per-conductor experiment design. fit_Ks are the top Layer A rungs, all past
# kappa = 2 pi K/(q t) ~ 2 on the locked sub-census t <= fit_t_max. The census
# depth t_census serves the C(omega) window (the q = 7 line set is dense --
# ln 34/ln 35/ln 36 and ln 47/ln 49/ln 51 need the T ~ 550 lobe width to
# resolve); the K-walk stops at t_walk (the collapse is per-zero and the walk
# cost scales as q x units(q), so depth buys nothing there).
CONDUCTORS = {
    4: {"t_census": 550.0, "t_walk": 550.0, "layerA": _scaled_schedule(4),
        "kappa_stride": 8, "fit_t_max": 100.0,
        "cert": [((200.31, 210.83), 64), ((540.17, 549.61), 256)]},
    7: {"t_census": 550.0, "t_walk": 250.0, "layerA": _scaled_schedule(7),
        "kappa_stride": 8, "fit_t_max": 100.0,
        "cert": [((150.31, 160.83), 112), ((244.17, 249.61), 448)]},
}
KAPPA_SCHEDULE = ac.KAPPA_SCHEDULE          # [0.5 .. 4.0], reused verbatim

# resonance-timeline test frequencies per conductor (label, omega, n)
FREQS = {
    4: [("ln2", LN(2), 2), ("ln4", LN(4), 4), ("ln8", LN(8), 8),
        ("ln3", LN(3), 3), ("ln9", LN(9), 9), ("ln5", LN(5), 5),
        ("ln7", LN(7), 7)],
    7: [("ln2", LN(2), 2), ("ln3", LN(3), 3), ("ln5", LN(5), 5),
        ("ln6", LN(6), 6), ("ln7", LN(7), 7), ("ln10", LN(10), 10)],
}

# C(omega) band classes per conductor: (class, [n ...])
BANDS = {
    4: [("conductor", [2, 4, 8, 10, 14, 22]),      # the swapped dark column
        ("section", [3, 15, 21, 33]),              # 3-divisible sqf: 2/3 law
        ("vacancy", [9, 27]),                      # ln9 satellite; ln27 dark
        ("untouched", [5, 7, 11, 13, 35])],
    7: [("conductor", [7, 14, 21, 35, 49]),        # the p | q dropout
        ("dressed", [2, 3, 5, 6, 10, 15]),         # full-comb suppressions
        ("vacancy", [4, 8, 9]),                    # multi-satellite vacancies
        ("untouched", [11, 13])],
}
ANCHORS = {4: [3, 5, 7, 11, 13], 7: [3, 5, 11, 13]}   # scale fit (undressed)
CONTROLS = ac.C_CONTROLS                    # off ln(n) for all n <= 16
LOWER_SB = {4: [LN(5) - LN(3), LN(7) - LN(3), LN(11) - LN(3)],
            7: [LN(11) - LN(2), LN(11) - LN(3)]}

_CHIS = {}  # type: Dict[int, cb.Character]


def chi_q(q):
    """THE real primitive character mod q of the walk family (#40's pick)."""
    if q not in _CHIS:
        _CHIS[q] = cs.real_primitive(q)
    return _CHIS[q]


def zeros_csv(q):
    return DATA / f"conductor_clock_zeros_q{q}.csv"


def a1_csv(q):
    return DATA / f"conductor_clock_a1_q{q}.csv"


def kwalk_csv(q):
    return DATA / f"conductor_clock_kwalk_q{q}.csv"


# --------------------------------------------------------------------------
# 1. the bulk zero census per conductor (the #40 Hardy-Z walk, deepened)
# --------------------------------------------------------------------------
def run_zeros(q, verbose=True):
    """Walk the critical-line zeros of L(s, chi_q) to t_census and cache them.
    Census integrity: N - theta_chi/pi must sit at S(T) size (no +1: entire L)."""
    cfg = CONDUCTORS[q]
    chi = chi_q(q)
    t0 = time.time()
    zeros = cs.walk_zeros(chi, t_max=cfg["t_census"], verbose=False)
    resid = len(zeros) - float(zb.theta_chi(cfg["t_census"], chi) / mp.pi)
    if verbose:
        print(f"   q={q}: {len(zeros)} zeros to t={cfg['t_census']:.0f}, "
              f"census - theta/pi = {resid:+.2f}  [{time.time() - t0:.0f} s]")
    assert abs(resid) < 1.0, (q, resid)
    zeros_csv(q).parent.mkdir(exist_ok=True)
    with zeros_csv(q).open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "gamma"])
        for i, g in enumerate(zeros, start=1):
            w.writerow([i, f"{g:.12f}"])
    return np.array(zeros, dtype=float)


def load_zeros(q):
    """The cached zero heights for chi_q, sorted ascending."""
    out = []
    with zeros_csv(q).open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.append(float(r["gamma"]))
    return np.array(sorted(out), dtype=float)


def a1_table_q(q, gammas, verbose=True):
    """[(gamma, a_1)] for chi_q over the census (arithmetic_clock.a1_table
    with chi and the per-conductor cache path threaded through)."""
    return ac.a1_table(gammas, verbose=verbose, chi=chi_q(q), path=a1_csv(q))


# --------------------------------------------------------------------------
# 2. the K-walk per conductor
# --------------------------------------------------------------------------
def walk_schedule_q(idx, gamma, q):
    """Layer A (the q-scaled #44 schedule) for everyone; every kappa_stride-th
    zero adds the K values realizing KAPPA_SCHEDULE at its height (Layer B)."""
    cfg = CONDUCTORS[q]
    Ks = set(cfg["layerA"])
    if idx % cfg["kappa_stride"] == 0:
        Ks.update(ac.K_of_kappa(kap, gamma, q) for kap in KAPPA_SCHEDULE)
    return sorted(Ks)


def run_kwalk_q(q, jobs=1, verbose=True):
    """The heavy step at conductor q: walk every census zero down its K
    schedule, repair, cache (arithmetic_clock.run_kwalk_jobs, chi threaded)."""
    chi = chi_q(q)
    gammas = load_zeros(q)
    a1s = a1_table_q(q, gammas, verbose=verbose)
    t_walk = CONDUCTORS[q]["t_walk"]
    jobsq = [(i, g, a1.real, a1.imag, walk_schedule_q(i, g, q), chi)
             for i, (g, a1) in enumerate(a1s) if g <= t_walk]
    if verbose:
        n_solve = sum(len(j[4]) for j in jobsq)
        print(f"   q={q}: {len(jobsq)} zeros, {n_solve} (K, zero) solves")
    return ac.run_kwalk_jobs(jobsq, kwalk_csv(q), jobs=jobs, chi=chi,
                             verbose=verbose)


# --------------------------------------------------------------------------
# 3. the collapse overlay
# --------------------------------------------------------------------------
def collapse_for(q):
    """(pts, (c1, c2)) for conductor q: q = 3 reads the committed #44 cache,
    q = 4/7 read this module's."""
    if q == 3:
        rows = ac.read_kwalk()
        a1s = ac.a1_table(c6.load_chi3_zeros())
    else:
        rows = ac.read_kwalk(kwalk_csv(q))
        a1s = a1_table_q(q, load_zeros(q), verbose=False)
    a1s_by_n = {i: a1 for i, (_g, a1) in enumerate(a1s)}
    pts = ac.collapse_points(rows, a1s_by_n, q=q)
    return pts, ac.collapse_fit(pts)


def overlay_table(collapses):
    """Binned medians per conductor on one kappa grid: {q: (centers, medians)}."""
    return {q: ac.collapse_binned(pts) for q, (pts, _fit) in collapses.items()}


def universality_checks(collapses):
    """The prediction-1 gates: per conductor, pre-knee saturation (|A-1| ~ 1),
    post-knee asymptotic decay, and the q = 3 fit curve tracking every other
    conductor's binned medians within a factor band on the asymptotic side."""
    c1_ref, c2_ref = collapses[3][1]
    checks = []
    for q, (pts, (c1, c2)) in sorted(collapses.items()):
        cent, med = ac.collapse_binned(pts)
        pre = [m for c, m in zip(cent, med) if c < 0.7]
        post = [m for c, m in zip(cent, med) if c > 3.0]
        checks.append((f"q={q}: pre-knee median |A-1| ~ 1 "
                       f"({min(pre):.2f}..{max(pre):.2f})",
                       bool(pre) and min(pre) > 0.4))
        checks.append((f"q={q}: asymptotic median |A-1| < 0.15 past kappa = 3 "
                       f"(worst {max(post):.3f})", bool(post) and max(post) < 0.15))
        if q != 3:
            rel = [m / (c1_ref / c + c2_ref / c ** 2)
                   for c, m in zip(cent, med) if c > 1.5]
            checks.append((f"q={q}: q=3 fit curve tracks the medians past "
                           f"kappa = 1.5 (x{min(rel):.2f}..x{max(rel):.2f})",
                           bool(rel) and 0.35 < min(rel) and max(rel) < 2.8))
        checks.append((f"q={q}: fit c1 = {c1:.3f}, c2 = {c2:.3f} "
                       f"(q=3: {c1_ref:.2f}, {c2_ref:.2f})", True))
    return checks


# --------------------------------------------------------------------------
# 4. the C(omega) spectrum instruments per conductor
# --------------------------------------------------------------------------
def spectrum_data(q):
    """(g, a1, a1p, scale) for conductor q from the committed a_1 cache."""
    chi = chi_q(q)
    a1s = a1_table_q(q, load_zeros(q), verbose=False)
    g = np.array([gg for gg, _a in a1s])
    a1 = np.array([a for _g, a in a1s])
    a1p = md.undress(g, a1, chi)
    scale = md.fit_scale(g, a1p, anchors=ANCHORS[q])
    return g, a1, a1p, scale


def band_table_q(q, g, a1, a1p, scale):
    """Rows (class, n, omega, undressed, dressed, mobius_ref, landau_ref,
    dressed_ref) over the conductor's named bands (chi-general models)."""
    chi = chi_q(q)
    rows = []
    for cls, ns in BANDS[q]:
        for n in ns:
            w0 = math.log(n)
            mu_m, _w = md.band_max(g, a1p, w0)
            dd_m, _w = md.band_max(g, a1, w0)
            rows.append((cls, n, w0, mu_m, dd_m, md.mobius_ref(n, scale, chi),
                         md.landau_ref(n, scale), md.dressed_ref(n, scale, chi)))
    return rows


def control_table_q(q, g, a1, a1p):
    """The floor: the #44 controls plus conductor-specific lower sidebands
    (one-sidedness at q = 4; two empty combinations at q = 7)."""
    rows = []
    for w0 in CONTROLS:
        rows.append(("ctl", w0, md.band_max(g, a1p, w0)[0],
                     md.band_max(g, a1, w0)[0]))
    for w0 in LOWER_SB[q]:
        rows.append(("low-sb", w0, md.band_max(g, a1p, w0)[0],
                     md.band_max(g, a1, w0)[0]))
    return rows


def _hyp_line(n, scale):
    """The amplitude a unit-weight squarefree line at ln n would carry:
    the honest 'dark' reference. High-omega dark bands sit between strong
    lines and pick up window-sidelobe leakage well above the low-omega
    control floor, so darkness is gated as a fraction of the would-be line."""
    return scale * math.log(n) / math.sqrt(n)


def validate_q4(bands, ctls, scale):
    """The swap-test gates at q = 4."""
    by_n = {r[1]: r for r in bands}
    floor = max(m for tag, _w, mu_m, dd_m in ctls if tag == "ctl"
                for m in (mu_m, dd_m))
    checks = []
    # S1: the whole 2-divisible column dark, undressed AND dressed
    for n in (2, 4, 8, 10, 14, 22):
        _c, _n, _w, mu_m, dd_m, _mr, _lr, _dr = by_n[n]
        frac = max(mu_m, dd_m) / _hyp_line(n, scale)
        checks.append((f"S1 q4 ln{n} dark ({frac:.2f} of a would-be line; "
                       f"und {mu_m:.2f}/drs {dd_m:.2f})", frac <= 0.25))
    # S2: the ex-conductor 3 is LIVE: undressed ln3 on the Moebius ref
    for n in (3, 5, 7, 11):
        _c, _n, _w, mu_m, _dd, mref, _lr, _dr = by_n[n]
        checks.append((f"S2 q4 undressed ln{n} on Moebius ref "
                       f"({mu_m / mref:.2f})", 0.82 <= mu_m / mref <= 1.22))
    # S3: the 2/3 law on 3-divisible squarefree lines
    for n in (3, 15, 21):
        _c, _n, _w, mu_m, dd_m, _mr, _lr, _dr = by_n[n]
        checks.append((f"S3 q4 dressed/undressed ln{n} = 2/3 "
                       f"({dd_m / mu_m:.2f})", 0.55 <= dd_m / mu_m <= 0.80))
    # S4: the ln9 vacancy satellite at 2/3 of Landau -- measurably non-Landau
    _c, _n, _w, _mu, dd9, _mr, lr9, dr9 = by_n[9]
    checks.append((f"S4 q4 dressed ln9 = 2/3 of Landau ({dd9 / lr9:.3f}), "
                   f"on its closed form ({dd9 / dr9:.3f})",
                   abs(dd9 / lr9 - 2.0 / 3.0) < 0.10 and abs(dd9 / dr9 - 1) < 0.15))
    checks.append((f"S4 q4 ln27 dark ({by_n[27][4] / _hyp_line(27, scale):.2f} "
                   "of a would-be line: the satellite cascade stops)",
                   by_n[27][4] <= 0.25 * _hyp_line(27, scale)))
    # S5: untouched odd lines (3-free): dressing acts only through the factor 3
    for n in (5, 7, 35):
        _c, _n, _w, mu_m, dd_m, _mr, _lr, _dr = by_n[n]
        checks.append((f"S5 q4 dressed/undressed ln{n} = 1 "
                       f"({dd_m / mu_m:.2f})", 0.85 <= dd_m / mu_m <= 1.15))
    # S6: one-sidedness -- nothing at ln p - ln 3
    for tag, w0, mu_m, dd_m in ctls:
        if tag == "low-sb":
            checks.append((f"S6 q4 no lower sideband at {w0:.3f} "
                           f"(und {mu_m:.2f}/drs {dd_m:.2f})",
                           max(mu_m, dd_m) <= 1.35 * floor))
    return checks


def validate_q7(bands, ctls, scale, g, a1, a1p):
    """The dropout + dressing-comb gates at q = 7, and the sign-flip
    adjudication on the ln 2 line."""
    by_n = {r[1]: r for r in bands}
    checks = []
    # D1: the p | q dropout, exactly the 7-columns, gated as fractions of a
    # would-be line. The q = 7 line set is dense (every squarefree n coprime
    # to 7), so the standard +-0.02 band around a dark column can reach the
    # FIRST sidelobe of a genuine neighbor (ln 35 is 0.029 from the real line
    # ln 34: the band edge sits 1.6 lobe-widths from its peak at T ~ 550).
    # Dark columns have no line-center drift to chase, so they are re-measured
    # on a +-0.012 band that keeps >= 2 lobes of separation.
    for n in (7, 14, 21, 35, 49):
        w0 = math.log(n)
        mu_m = md.band_max(g, a1p, w0, half=0.012)[0]
        dd_m = md.band_max(g, a1, w0, half=0.012)[0]
        frac = max(mu_m, dd_m) / _hyp_line(n, scale)
        checks.append((f"D1 q7 ln{n} dark ({frac:.2f} of a would-be line; "
                       f"und {mu_m:.2f}/drs {dd_m:.2f}, narrow band)",
                       frac <= 0.30))
    # D2: everything coprime to 7 lives, including the composites 6, 10 that
    # q = 3 and q = 4 both kill
    for n in (2, 3, 5, 6, 10):
        _c, _n, _w, mu_m, _dd, mref, _lr, _dr = by_n[n]
        checks.append((f"D2 q7 undressed ln{n} on Moebius ref "
                       f"({mu_m / mref:.2f})", 0.75 <= mu_m / mref <= 1.30))
    # D3: THE ADJUDICATION. Convolution model: dressed ln2 = 1/2 of undressed
    # (chi cancels). #47's sign-flip prediction: chi_7(2) = +1 makes it 3/2.
    _c, _n, _w, mu2, dd2, _mr, _lr, _dr = by_n[2]
    r2 = dd2 / mu2
    checks.append((f"D3 q7 dressed/undressed ln2 = {r2:.2f}: convolution says "
                   f"1/2, #47 sign-flip said 3/2", 0.32 <= r2 <= 0.72))
    checks.append((f"D3 q7 ln2 NOT enhanced (sign-flip refuted: {r2:.2f} < 1)",
                   r2 < 1.0))
    # D4: the full-comb rational suppressions
    for n, target in ((3, 2.0 / 3.0), (5, 4.0 / 5.0), (6, 1.0 / 3.0),
                      (10, 0.30), (15, 7.0 / 15.0)):
        _c, _n, _w, mu_m, dd_m, _mr, _lr, dref = by_n[n]
        checks.append((f"D4 q7 dressed/undressed ln{n} = {target:.2f} "
                       f"({dd_m / mu_m:.2f})", abs(dd_m / mu_m - target) < 0.16))
    # D5: multi-satellite vacancies on their closed forms (vs Landau: 1/2 at
    # ln4, 3/4 at ln8, 2/3 at ln9 -- all non-Landau, unlike q = 3's ln4)
    for n in (4, 8, 9):
        _c, _n, _w, _mu, dd_m, _mr, lref, dref = by_n[n]
        checks.append((f"D5 q7 dressed ln{n} on its closed form "
                       f"({dd_m / dref:.2f}; vs Landau {dd_m / lref:.2f})",
                       abs(dd_m / dref - 1) < 0.25))
    return checks


# --------------------------------------------------------------------------
# 5. the q = 4 resonance timeline (the dark tower swapped, along the knob)
# --------------------------------------------------------------------------
def timeline_q(q):
    """(timeline, ref, counts, locked) for conductor q's walked census."""
    rows = ac.read_kwalk(kwalk_csv(q))
    a1s = a1_table_q(q, load_zeros(q), verbose=False)
    a1s_by_n = {i: a1 for i, (_g, a1) in enumerate(a1s)}
    cfg = CONDUCTORS[q]
    tl, ref, counts = ac.timeline(rows, a1s_by_n, Ks=cfg["layerA"],
                                  freqs=FREQS[q])
    locked = ac.locked_fit(rows, a1s_by_n, fit_Ks=cfg["layerA"][-3:],
                           fit_t_max=cfg["fit_t_max"], freqs=FREQS[q])
    return tl, ref, counts, locked


def timeline_checks(q, tl, ref, counts, locked):
    """Two separate gates, two separate physics statements.

    DRIFT FIDELITY: the walked census drifts onto the computable constant
    C_omega at EVERY test frequency -- dark ones included: the finite-sample
    constant over the locked sub-census is noise-sized but nonzero there, and
    the census tracks it (the #44 Q1 verdict, conductor-general). The SWAP
    lives in the resonance VALUES: the conductor-tower R_omega(K) sits inside
    the 1/sqrt(2N) noise band at every K (the census never has that
    arithmetic, at any resolution), while the live explicit-formula lines
    (Lambda(n) chi(n) != 0) are resolved above it."""
    dark = {4: ("ln2", "ln4", "ln8"), 7: ("ln7",)}[q]
    live = {4: ("ln3", "ln9", "ln5"), 7: ("ln2", "ln3", "ln5")}[q]
    checks = []
    for lb, (C_fit, C_pred, _s, _x, _y) in locked.items():
        checks.append((f"T q{q} drift {lb}: C_fit {C_fit:+.3f} on C_pred "
                       f"{C_pred:+.3f}",
                       abs(C_fit - C_pred) < max(0.025, 0.25 * abs(C_pred))))
    noise = max(1.0 / math.sqrt(2 * n) for n in counts.values())
    for lb in dark:
        worst = max(abs(tl[K][lb][0]) for K in tl)
        checks.append((f"T q{q} dark {lb}: |R(K)| <= {worst:.3f} at every K "
                       f"(noise {noise:.3f})", worst < 2.0 * noise))
    for lb in live:
        checks.append((f"T q{q} live {lb}: |R(inf)| = {abs(ref[lb]):.3f} "
                       f"above the noise band", abs(ref[lb]) > 2.0 * noise))
    return checks


# --------------------------------------------------------------------------
# 6. census spot-certification (argument principle)
# --------------------------------------------------------------------------
def spot_certify(q, verbose=True):
    """Winding counts of L_comb_K over per-conductor bulk windows vs the
    walked census (the #43/#44 set-completeness instrument)."""
    chi = chi_q(q)
    rows = ac.read_kwalk(kwalk_csv(q))
    ok = True
    for (t_lo, t_hi), K in CONDUCTORS[q]["cert"]:
        walked = [r for r in rows if r[2] == K and r[5] in ("ok", "patched")
                  and t_lo < r[4] < t_hi]
        n_wind = zb.winding_count(lambda s: cb.L_comb_K(s, K, chi),
                                  (-0.97, 2.03), (t_lo, t_hi),
                                  dsig=0.3, dt=0.35)
        tag = "OK" if n_wind == len(walked) else "MISMATCH"
        ok &= n_wind == len(walked)
        if verbose:
            print(f"   q={q}, K={K:>4}, t in ({t_lo}, {t_hi}): winding "
                  f"{n_wind}, walked {len(walked)}   {tag}")
    return ok


# --------------------------------------------------------------------------
# 7. driver printing
# --------------------------------------------------------------------------
def _print_checks(title, checks):
    print(f"\n== {title} ==")
    n_fail = 0
    for label, okc in checks:
        print(f"   [{'PASS' if okc else 'FAIL'}] {label}")
        n_fail += not okc
    return n_fail


def _print_bands(q, bands, scale):
    print(f"\n== q = {q} band table (scale {scale:.3f}) ==")
    print("   class      n    omega   undress   dress   Moebius   Landau"
          "   dressed-model")
    for cls, n, w0, mu_m, dd_m, mref, lref, dref in bands:
        print(f"   {cls:>9} {n:>4}   {w0:.4f}   {mu_m:7.3f}  {dd_m:6.3f}"
              f"   {mref:7.3f}  {lref:7.3f}   {dref:7.3f}")


def _print_timeline(q, tl, ref, counts):
    labels = [lb for lb, _w, _n in FREQS[q]][:5]
    print(f"\n== q = {q} resonance timeline (walked census) ==")
    print("   K      N " + "".join(f"{lb + ' meas/field':>19}" for lb in labels))
    for K in sorted(tl):
        row = f"   {K:>4} {counts[K]:>4}"
        for lb in labels:
            m, fp, _lin = tl[K][lb]
            row += f"    {m:+.4f}/{fp:+.4f}"
        print(row)
    print("    inf     " + "".join(f"    {ref[lb]:+.4f} (target)"[:19].ljust(19)
                                   for lb in labels))


def _print_collapse_overlay(collapses):
    print("\n== the collapse overlay: median |A - 1| vs kappa per conductor ==")
    for q, (pts, (c1, c2)) in sorted(collapses.items()):
        cent, med = ac.collapse_binned(pts)
        print(f"   q = {q} ({len(pts)} points; fit {c1:.3f}/kappa "
              f"+ {c2:.3f}/kappa^2):")
        for c, m in zip(cent, med):
            bar = "#" * int(min(m, 2.0) * 30)
            print(f"      kappa {c:6.2f}: {m:6.3f}  {bar}")


# --------------------------------------------------------------------------
# 8. the figure
# --------------------------------------------------------------------------
def _figure(collapses, spectra, bands_by_q, tl4):
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
    QCOL = {3: "0.45", 4: "tab:blue", 7: "tab:red"}

    # (A) the collapse overlay
    for q, (pts, (c1, c2)) in sorted(collapses.items()):
        ka = np.array([p[2] for p in pts])
        ab = np.array([abs(p[3] - 1) for p in pts])
        axA.scatter(ka, np.maximum(ab, 1e-4), s=5, color=QCOL[q], alpha=0.18)
        cent, med = ac.collapse_binned(pts)
        axA.plot(cent, med, "-o", color=QCOL[q], lw=2, ms=4,
                 label=f"q = {q} median (c1 = {c1:.2f})")
    c1r, c2r = collapses[3][1]
    kk = np.geomspace(0.9, 60, 60)
    axA.plot(kk, c1r / kk + c2r / kk ** 2, "k--", lw=1.4,
             label=f"q = 3 fit: {c1r:.2f}/k + {c2r:.2f}/k$^2$")
    axA.axvline(1.0, color="gray", ls=":")
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlabel(r"$\kappa = 2\pi K/(q\,t)$")
    axA.set_ylabel(r"$|A - 1|$")
    axA.set_title("Prediction 1: one collapse across conductors "
                  "(knee at kappa = 1)")
    axA.legend(fontsize=8.5)

    # (B) q = 4 undressed spectrum: the swap
    g4, a14, a1p4, sc4 = spectra[4]
    ws = np.arange(0.4, 3.65, 1e-3)
    axB.plot(ws, np.abs(md.spectrum(g4, a1p4, ws)), "-", color="tab:blue",
             lw=0.7)
    first = True
    for n in range(2, 37):
        r = md.mobius_ref(n, sc4, chi_q(4))
        if r > 0:
            axB.plot([LN(n)] * 2, [0, r], color="0.25", lw=2.2, alpha=0.65,
                     solid_capstyle="butt",
                     label="Moebius model (odd sqf n)" if first else None)
            first = False
    for n in (2, 4, 8, 10, 14):
        axB.plot(LN(n), _hyp_line(n, sc4), "x", ms=9, color="tab:red",
                 label="dark: the 2-divisible column" if n == 2 else None)
    axB.set_xlabel(r"$\omega$")
    axB.set_ylabel(r"$|C'(\omega)|$")
    axB.set_title("Prediction 2a at q = 4: the dark tower SWAPS\n"
                  "(2 = conductor, 3 = section)")
    axB.legend(fontsize=8.5, loc="upper left")

    # (C) q = 4 dressed spectrum vs the dressed model
    axC.plot(ws, np.abs(md.spectrum(g4, a14, ws)), "-", color="tab:orange",
             lw=0.7)
    first = True
    for n in range(2, 37):
        r = md.dressed_ref(n, sc4, chi_q(4))
        if r > 0:
            axC.plot([LN(n)] * 2, [0, r], color="0.25", lw=2.2, alpha=0.65,
                     solid_capstyle="butt",
                     label="dressed model" if first else None)
            first = False
    axC.annotate("ln9: satellite in the\nmu(9)=0 vacancy at\n2/3 of Landau",
                 xy=(LN(9), md.dressed_ref(9, sc4, chi_q(4))),
                 xytext=(0.55, 2.0), fontsize=9,
                 arrowprops=dict(arrowstyle="->", lw=0.8))
    axC.annotate("3-divisible lines\nat exactly 2/3", xy=(LN(15), 1.4),
                 xytext=(2.5, 2.4), fontsize=9,
                 arrowprops=dict(arrowstyle="->", lw=0.8))
    axC.set_xlabel(r"$\omega$")
    axC.set_ylabel(r"$|C(\omega)|$")
    axC.set_title(r"Prediction 2b at q = 4: one-sided $+\ln 3$ dressing, "
                  r"$\beta = 3^{-3/2}$")
    axC.legend(fontsize=8.5, loc="upper left")

    # (D) q = 7 undressed spectrum: the p | q dropout only
    g7, a17, a1p7, sc7 = spectra[7]
    axD.plot(ws, np.abs(md.spectrum(g7, a1p7, ws)), "-", color="tab:green",
             lw=0.7)
    first = True
    for n in range(2, 37):
        r = md.mobius_ref(n, sc7, chi_q(7))
        if r > 0:
            axD.plot([LN(n)] * 2, [0, r], color="0.25", lw=2.2, alpha=0.65,
                     solid_capstyle="butt",
                     label="Moebius model (sqf n, 7 !| n)" if first else None)
            first = False
    for n in (7, 14, 21, 35):
        axD.plot(LN(n), sc7 * LN(n) / math.sqrt(n), "x", ms=9, color="tab:red",
                 label="dark: exactly the 7-columns" if n == 7 else None)
    axD.set_xlabel(r"$\omega$")
    axD.set_ylabel(r"$|C'(\omega)|$")
    axD.set_title("Prediction 3 at q = 7: only the p | q columns\n"
                  "delete (ln6, ln10 now live)")
    axD.legend(fontsize=8.5, loc="upper left")

    # (E) q = 7 dressing ratios: the adjudication panel
    ns = [2, 3, 5, 6, 10, 15]
    meas, model = [], []
    by_n = {r[1]: r for r in bands_by_q[7]}
    for n in ns:
        _c, _n, _w, mu_m, dd_m, _mr, _lr, _dr = by_n[n]
        meas.append(dd_m / mu_m)
        model.append(md.dressed_ref(n, 1.0, chi_q(7))
                     / md.mobius_ref(n, 1.0, chi_q(7)))
    x = np.arange(len(ns))
    axE.bar(x - 0.18, meas, 0.34, color="tab:green", label="measured")
    axE.bar(x + 0.18, model, 0.34, color="0.35",
            label="convolution model (chi cancels)")
    axE.plot([-0.5], [1.5], "X", ms=13, color="tab:red")
    axE.annotate("#47 sign-flip\nprediction (3/2)\nat ln2: refuted",
                 xy=(-0.35, 1.5), xytext=(0.6, 1.35), fontsize=9,
                 arrowprops=dict(arrowstyle="->", lw=0.8))
    axE.axhline(1.0, color="0.6", lw=0.8, ls=":")
    axE.set_xticks(x)
    axE.set_xticklabels([f"ln{n}" for n in ns])
    axE.set_ylabel("dressed / undressed band max")
    axE.set_title("The q = 7 dressing comb is rational and\n"
                  "chi-INDEPENDENT (ln2 halved, not x3/2)")
    axE.legend(fontsize=8.5)

    # (F) the q = 4 dark-tower timeline
    tl, ref, counts, _locked = tl4
    Ks = sorted(tl)
    xs = [1.0 / K for K in Ks]
    for lb, color, mark in (("ln3", "tab:red", "^"), ("ln5", "tab:green", "s"),
                            ("ln2", "tab:blue", "o"), ("ln4", "tab:cyan", "v"),
                            ("ln8", "tab:purple", "d")):
        axF.plot(xs, [abs(tl[K][lb][0] - ref[lb]) for K in Ks], mark + "-",
                 color=color, label=f"|R(K) - R(inf)| at {lb}")
    noise = [1.0 / math.sqrt(2 * counts[K]) for K in Ks]
    axF.plot(xs, noise, "k:", label="1/sqrt(2N) noise")
    axF.set_xlabel("1/K")
    axF.set_ylabel(r"$|R_\omega(K) - R_\omega(\infty)|$")
    axF.set_title("q = 4 timeline: the 2-adic tower never drifts\n"
                  "(dark); ln3/ln5 drift at O(1/K)")
    axF.legend(fontsize=8)

    fig.suptitle("The resolution clock across conductors: universality, the "
                 "dark-tower swap, and the chi-cancelling dressing (issue #48)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIG_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    print(f"\nfigure -> {FIG_PATH}")


# --------------------------------------------------------------------------
# 9. main
# --------------------------------------------------------------------------
def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recompute-zeros", action="store_true",
                    help="rebuild the per-conductor zero censuses (~2 min)")
    ap.add_argument("--recompute-kwalk", action="store_true",
                    help="rebuild the K-walk caches (HEAVY: hours; see --jobs)")
    ap.add_argument("--qs", type=int, nargs="+", default=[4, 7],
                    help="conductors to (re)compute (default: 4 7)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="worker processes for the K-walks (default 1)")
    ap.add_argument("--skip-certify", action="store_true",
                    help="skip the winding spot-checks (~minutes)")
    args = ap.parse_args(argv)
    n_fail = 0

    for q in args.qs:
        if args.recompute_zeros or not zeros_csv(q).exists():
            run_zeros(q)
        a1_table_q(q, load_zeros(q))
        if args.recompute_kwalk or not kwalk_csv(q).exists():
            run_kwalk_q(q, jobs=args.jobs)

    # prediction 1: the collapse overlay
    collapses = {q: collapse_for(q) for q in [3] + args.qs}
    _print_collapse_overlay(collapses)
    n_fail += _print_checks("prediction 1: universality gates",
                            universality_checks(collapses))

    # predictions 2 + 3: the spectra
    spectra = {q: spectrum_data(q) for q in args.qs}
    bands_by_q = {}
    for q in args.qs:
        g, a1, a1p, scale = spectra[q]
        bands = band_table_q(q, g, a1, a1p, scale)
        bands_by_q[q] = bands
        _print_bands(q, bands, scale)
        ctls = control_table_q(q, g, a1, a1p)
        if q == 4:
            n_fail += _print_checks("prediction 2: the q = 4 swap test",
                                    validate_q4(bands, ctls, scale))
        if q == 7:
            n_fail += _print_checks("prediction 3 + the dressing comb at q = 7",
                                    validate_q7(bands, ctls, scale, g, a1, a1p))

    # the timeline along the knob
    tls = {}
    for q in args.qs:
        tl, ref, counts, locked = timeline_q(q)
        tls[q] = (tl, ref, counts, locked)
        _print_timeline(q, tl, ref, counts)
        n_fail += _print_checks(f"timeline gates at q = {q}",
                                timeline_checks(q, tl, ref, counts, locked))

    if not args.skip_certify:
        print("\n== census spot-certification (argument principle) ==")
        for q in args.qs:
            if not spot_certify(q):
                n_fail += 1

    _figure(collapses, spectra, bands_by_q,
            tls[4] if 4 in tls else list(tls.values())[0])
    print(f"\n{'ALL GATES PASS' if n_fail == 0 else f'{n_fail} GATE(S) FAILED'}")
    return n_fail


if __name__ == "__main__":
    sys.exit(_main())
