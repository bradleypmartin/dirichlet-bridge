r"""Rate vs conductor: the residue-class comb constant swept across q -- issue #40.

EXPERIMENTAL (sub-issue of #37; same disjoint-from-the-manuscript rules as
character_bridge.py). Run tests with `pytest explorations/tests`; the driver is
local/manual only (minutes fresh, seconds from the CSV caches; never in CI).
Experimental-mathematics framing throughout; **no RH/GRH claims**.

The question this module owns (issue #40's three bullets)
---------------------------------------------------------
character_bridge.py measured the first-order answer at q = 3..6: the residue-class
comb converges as

    L(s, chi) - L_comb_K(s, chi)  ~  rate_comb_chi(s, chi) / K
                                  =  (s q / 2 pi^2) B(s+1, chi) / K,

with B(s) = sum_a chi(a) a^{-s} the one-period section -- q times a section of
L(s+1, chi). This module is the systematic half: (1) the constant MEASURED across
every primitive chi mod q <= 50 (the `characters_any` CRT constructor is the build
item that unlocks non-cyclic moduli 8, 12, 15, 16, ...); (2) the translation to
ZERO displacement, s_K - rho ~ a_1/K with a_1 = rate_comb_chi(rho)/L'(rho), as
conductor-scaling STATISTICS over bulk zero samples per q (the #38 Hardy-Z walk,
generalized to every real primitive chi); (3) the explicit-formula face: what the
q-dependence of the constant says next to the weil-positivity-lab conductor thread.

Finding 1 -- the knob's natural clock is K/q (the measurement design)
---------------------------------------------------------------------
The naive fixed-K measurement degrades linearly in q: at K = 160 the raw K*err
matches the constant to ~0.3% at q <= 6 (the #37 numbers) but only ~4.5% at
q ~ 48. The sub-leading structure, measured here across the sweep:

    K (L - L_comb_K)  =  Delta_1  +  c_1 / K  +  c_2 / K^2 + ...,

with c_1/Delta_1 = O(1) roughly q-independent and c_2/Delta_1 ~ 0.4..0.6 q^2:
the expansion parameter is q/K, NOT 1/K. Interpretation: the k-th sawtooth
harmonic only resolves the a/q residue lattice once k ~ q, so "convergence
progress" is harmonics PER RESIDUE CLASS. The conductor enters the rate question
twice, and the two faces should not be conflated:

    amplitude:   the constant grows linearly, (sq/2 pi^2) B(s+1)   (finding 2);
    resolution:  the error series runs in q/K -- matched accuracy needs K ~ q.

The sweep therefore measures at conductor-scaled checkpoints K = (5q, 7q, 10q)
and eliminates {1/K, 1/K^2} by 3-point Richardson; that lands q-UNIFORM residuals
(~1e-3 for every chi in the sweep) where fixed-K Richardson fails (the 1/K^2 term
is the dominant contaminant at large q -- 2-point extrapolation in 1/K makes the
estimate WORSE than the raw value beyond q ~ 12).

Finding 2 -- the amplitude law, confirmed conductor by conductor
----------------------------------------------------------------
Measured lim K(L - L_comb_K) vs (sq/2 pi^2) B(s+1, chi) at s0 = 1.3 + 7i for every
primitive chi mod q, 3 <= q <= 50 -- 470 characters across 36 moduli, every
parity, real and complex, cyclic and non-cyclic: median residual 6.4e-4, WORST
8.5e-4 (raw top checkpoint K = 10q: median 3.9e-3). The measured second-order
coefficient sits at |c_2/(Delta_1 q^2)| = 0.50 median, full range 0.36..0.63,
across the whole sweep -- the resolution clock quantified. Two structural
readings:

  * the MEAN over the full character group mod q is exact and section-free:
    orthogonality keeps only the a = 1 column, (1/phi(q)) sum_chi rate_comb_chi
    = q s/2 pi^2 -- the "q times zeta's constant" spine (verified to working
    precision in the driver; the primitive-only mean scatters O(1) around it).
  * the section factor B(s0+1) is an O(1) fluctuation around 1 whose size is set
    by the SMALLEST PRIME NOT DIVIDING q: |B - 1| ~ p*^{-Re s0-1}, so the sweep's
    section factors split into measured tiers -- p* = 2 (odd q, 421 chi): mean
    |B-1| = 0.215 vs 2^{-2.3} = 0.203; p* = 3 (q = 0 mod 4, 3 !| q, 38 chi):
    0.081 vs 0.080; p* = 5 (q = 0 mod 12, 11 chi): 0.026 vs 0.025. The ramified
    primes p | q are DELETED from the section exactly as they are deleted from
    the explicit formula's prime side (#38 measured that null on the zero
    statistics) -- the comb constant carries the same surviving-prime
    fingerprint. This is the concrete bridge to the weil-positivity-lab
    conductor thread (its #22/#31): approximation-rate q-scaling and
    explicit-formula q-scaling read the same arithmetic, one through B(s+1),
    one through the prime sums.

Finding 3 -- zero displacement: linear-in-q amplitude, log-q tempering
----------------------------------------------------------------------
At a simple zero rho, s_K - rho ~ a_1/K with a_1 = rate_comb_chi(rho)/L'(rho)
(zero_birth.disp_coeff_chi, verified there at q = 3, 5 and here spot-certified
at the non-cyclic q = 8, 15 via measured K = 45 comb roots, rel 0.02-0.03). The
conductor statistics, over bulk critical-line samples (the Hardy-Z sign walk of
#38 generalized: every real primitive chi has eps = +1 by Gauss, so
Z_chi = e^{i theta_chi} L is real -- walked here for 12 conductors q = 3..43,
332 zeros to t = 50, every census within |S(T)| noise of theta_chi/pi, no +1):

    |a_1|  =  (q |rho| / 2 pi^2) |B(rho+1)| / |L'(rho)|,

numerator linear in q with an O(1) section factor; denominator's statistics grow
with the local zero DENSITY ln(q t/2 pi)/2 pi -- median |L'| climbs 2.21 -> 3.75
from q = 3 to 43 (x1.7, tracking the density's x1.6) while the density-rescaled
median |L'|/rho_dens stays q-flat at 5.4 +- 0.5 with NO trend (GUE-flavored
conductor-aspect universality, observed not assumed). Net: the per-q median
displacement grows x6.9 from q = 3 to 43, where pure-linear predicts x14.3 and
the density-tempered q/ln(q t) curve x8.9 -- the measured medians ride the
tempered curve and cleanly exclude the pure-linear one, with a heavy right tail
from near-degenerate zero pairs (small |L'| -- the close-pair migrations are the
slow ones, as in the chi_6 walk of #39). Zero condensation onto sigma = 1/2 is
therefore SLOWER at larger conductor in absolute terms, but only by the linear
amplitude: in resolution units (K/q fixed) the displacement per zero is
asymptotically conductor-free up to the section factor and the L' fluctuations.

Ride-alongs (the #43 carryover list, all closed-form, conductor-indexed)
------------------------------------------------------------------------
For the odd real primitive members of the walk family the driver tabulates the
#39 seed/ground scalars across q: L(0, chi) (rational -- the walked family lands
on the classical class-number values 1/3, 1/2, 1, 1, 2, 3, 3, 1 for q = 3, 4, 7,
11, 15, 23, 31, 43), the seed wobble radius L0/ln(q-1), the conductor-string
location ln L0 / ln(q/(q-1)) -- which parks at sigma* = 0 exactly when L0 = 1
and jumps far right for L0 >= 2 -- and its spacing 2 pi/ln(q/(q-1)) ~ 2 pi q
(the string THINS out ~ 1/q: the conductor's own seed signature fades at large
q), plus the ground/section density ln(q-1)/2 pi. The instant-census question
(#39: geometric birth completes at K = 1) is re-asked at new moduli: the
argument-principle box census of L_comb_K vs the L target for q = 5, 8, 12, 15
reads N_K = target = 11, 13, 15, 16 at EVERY K in {1, 2, 3}, both parities,
non-cyclic moduli included -- and without even the one transient extra zero
that zeta's census showed (#39): the instant census is the generic behavior.

Honest framing
--------------
Experimental mathematics; no RH/GRH claims. The CRT character construction,
Gauss's eps = +1 for real primitive chi, and the smooth counting function are
classical; the q-aspect growth of L' moments is conjectural territory
(Hughes-Keating-O'Connell-flavored) that we MEASURE, not derive. What has no
published analog we know of (per the #37 adversarial lit pass) is the finite-K
rate object itself: the conductor-swept measurement of the comb constant, the
q/K resolution clock, and the conductor statistics of the displacement
coefficient a_1.

Run directly to validate + plot: writes explorations/figures/conductor_sweep.png
and caches data/conductor_rate_sweep.csv + data/conductor_walk_a1.csv (rebuild
with --recompute-sweep / --recompute-zeros). Local/manual only -- NOT in
repro.py or CI (issue #37's CI-economy note).
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mpmath as mp

# Path shim: explorations/ module cross-importing the flat bridge/ sources plus its
# siblings (explorations/conftest.py does the same for pytest).
_HERE = Path(__file__).resolve().parent
_BRIDGE = _HERE.parent / "bridge"
for _p in (str(_HERE), str(_BRIDGE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import character_bridge as cb  # noqa: E402  (characters_any, L_chi, comb, rate)
import rate_law as rl          # noqa: E402  (rate_comb = s/2 pi^2, the spine)
import zero_birth as zb        # noqa: E402  (theta_chi, winding_count, disp, L0)

mp.mp.dps = 30

_J = mp.j
PI = mp.pi

S0 = mp.mpc("1.3", "7")            # the repo's standard off-line reference point
Q_MAX = 50
Q_WALK = [3, 4, 5, 7, 8, 11, 12, 15, 17, 23, 31, 43]   # real-primitive family
T_WALK = 50.0                      # zero-sample height window per conductor
CENSUS_QS = [5, 8, 12, 15]         # instant-census re-ask (non-cyclic included)
RATE_CSV = _HERE / "data" / "conductor_rate_sweep.csv"
WALK_CSV = _HERE / "data" / "conductor_walk_a1.csv"


# --------------------------------------------------------------------------
# 1. the measurement: conductor-scaled checkpoints + 3-point Richardson
# --------------------------------------------------------------------------
def k_triple(q):
    # type: (int) -> Tuple[int, int, int]
    """Checkpoints K = (5q, 7q, 10q), floored at (30, 40, 50) for tiny q.

    The error series of the chi comb runs in q/K (finding 1): fixed multipliers
    of q give q-uniform accuracy, fixed K degrades linearly in q.
    """
    return (max(30, 5 * q), max(40, 7 * q), max(50, 10 * q))


def comb_checkpoints(s, alpha, Ks):
    """zeta_comb_K(s, K, alpha) at several K from ONE harmonic accumulation.

    The per-q cost saver: components depend on the phase a/q only, so one run to
    max(Ks) serves every checkpoint (and every chi mod q -- the weighting into
    L_comb_K is a free finite sum per character).
    """
    s = mp.mpc(s)
    alpha = mp.mpf(alpha)
    want = set(int(K) for K in Ks)
    tot = alpha ** (1 - s) / (s - 1) + alpha ** (-s) / 2
    out = {}  # type: Dict[int, mp.mpc]
    for k in range(1, max(want) + 1):
        tot += cb.hurwitz_term(s, k, alpha)
        if k in want:
            out[k] = +tot
    return out


def rich3(Xs, Ks):
    """Fit X_K = a + b/K + c/K^2 through three checkpoints; return (a, b, c).

    a estimates Delta_1 = lim K(L - L_comb_K) with the 1/K and 1/K^2 layers
    eliminated; b and c are the measured sub-leading coefficients (c/a ~ 0.5 q^2
    is the resolution-clock evidence). NB 2-point Richardson in 1/K is WORSE
    than the raw value here -- the 1/K^2 term dominates the contamination at
    large q and enters the 1/K extrapolation with a doubled sign.
    """
    A = mp.matrix([[1, mp.mpf(1) / K, mp.mpf(1) / K ** 2] for K in Ks])
    v = mp.lu_solve(A, mp.matrix(list(Xs)))
    return v[0], v[1], v[2]


def primitive_characters(q):
    # type: (int) -> List[cb.Character]
    """The primitive characters mod q (conductor == q; empty for q = 2 mod 4)."""
    return [chi for chi in cb.characters_any(q)
            if chi.conductor == q and not chi.principal]


def measure_rate_q(q, s=S0, workdps=20):
    """Measured vs predicted comb constant for every primitive chi mod q.

    One component pass (checkpointed comb + mp.zeta target per residue class),
    then per-chi weighted sums. Returns rows
    (chi, pred, meas, rel_resid, raw_rel, c2_norm) with pred = rate_comb_chi,
    meas = the Richardson-extrapolated lim K(L - L_comb_K), raw_rel the residual
    of the raw top checkpoint (K = 10q), and c2_norm = |c/(a q^2)| the measured
    second-order coefficient in resolution units.
    """
    prims = primitive_characters(q)
    if not prims:
        return []
    Ks = k_triple(q)
    out = []
    with mp.workdps(workdps):
        us = cb.units(q)
        zs = {a: mp.zeta(s, mp.mpf(a) / q) for a in us}
        recs = {a: comb_checkpoints(s, mp.mpf(a) / q, Ks) for a in us}
        qs = mp.power(q, -s)
        for chi in prims:
            L = qs * mp.fsum(chi(a) * zs[a] for a in us)
            X = [Ks[i] * (L - qs * mp.fsum(chi(a) * recs[a][Ks[i]] for a in us))
                 for i in range(3)]
            a, b, c = rich3(X, Ks)
            pred = cb.rate_comb_chi(s, chi)
            out.append((chi, pred, a,
                        float(abs(a - pred) / abs(pred)),
                        float(abs(X[2] - pred) / abs(pred)),
                        float(abs(c / a) / q ** 2)))
    return out


def section_factor(chi, s=S0):
    """B(s+1, chi) = rate_comb_chi/(q rate_comb): the O(1) per-chi section factor."""
    return zb.section_B(mp.mpc(s) + 1, chi)


def p_star(q):
    # type: (int) -> int
    """The smallest prime NOT dividing q -- sets the section factor's tier:
    |B(s0+1) - 1| ~ p*^{-Re s0 - 1} (the ramified p | q columns are deleted)."""
    p = 2
    while q % p == 0:
        p += 1
        while any(p % d == 0 for d in range(2, int(p ** 0.5) + 1)):
            p += 1
    return p


def mean_rate_identity_residual(q, s=S0):
    """|mean over the FULL character group of rate_comb_chi - q s/2 pi^2|.

    Orthogonality keeps only the a = 1 column of the section: the group mean of
    the comb constant is exactly q times zeta's constant, section-free. Exact at
    working precision -- the spine line of the sweep panel.
    """
    chars = cb.characters_any(q)
    mean = mp.fsum(cb.rate_comb_chi(s, chi) for chi in chars) / len(chars)
    return float(abs(mean - q * rl.rate_comb(s)))


# --------------------------------------------------------------------------
# 2. bulk zeros per conductor: the Hardy-Z walk, generalized (real primitive chi)
# --------------------------------------------------------------------------
def real_primitive(q):
    # type: (int) -> cb.Character
    """THE real primitive character mod q used by the walk family (even preferred
    when both parities exist, e.g. chi_8 over chi_-8; unique at the other Q_WALK
    moduli). Real => eps(chi) = +1 (Gauss), so the Hardy-Z walk applies verbatim."""
    reals = [chi for chi in primitive_characters(q)
             if all(2 * e % chi.m == 0 for e in chi.exps.values())]
    if not reals:
        raise ValueError(f"no real primitive character mod {q}")
    return sorted(reals, key=lambda c: c.parity)[0]


def hardy_z(t, chi, dps=15):
    """Z_chi(t) = e^{i theta_chi(t)} L(1/2 + it, chi): real for real primitive chi.

    eps = +1 makes Lambda(1/2 + it) real (Gauss); phase and L evaluated at matched
    precision (the #38 pattern, conductor-general via zero_birth.theta_chi)."""
    with mp.workdps(dps):
        tt = mp.mpf(t)
        th = zb.theta_chi(tt, chi)
        return float(mp.re(mp.expj(th) * cb.L_chi(mp.mpc("0.5", tt), chi)))


def _mean_density(t, q):
    """The smooth zero density ln(q t/2 pi)/2 pi, floored for the low-t grid."""
    return max(math.log(max(q * t / (2.0 * math.pi), 1.05)) / (2.0 * math.pi), 0.02)


def _bracket_zeros(chi, t_lo, t_hi, step_frac, dps):
    """Sign-change brackets of Z_chi on an adaptive grid (the #38 walk, q-general)."""
    grid = [t_lo]
    while grid[-1] < t_hi:
        step = min(0.45, step_frac / _mean_density(grid[-1], chi.q))
        grid.append(min(grid[-1] + step, t_hi))
    vals = [hardy_z(t, chi, dps) for t in grid]
    return [(a, b) for a, b, va, vb in zip(grid, grid[1:], vals, vals[1:])
            if va * vb < 0]


def _polish_bracket(chi, a, b, dps):
    with mp.workdps(dps):
        t0 = mp.findroot(lambda t: hardy_z(float(t), chi, dps),
                         (mp.mpf(a), mp.mpf(b)), solver="anderson", maxsteps=60)
    return float(t0)


def walk_zeros(chi, t_max=T_WALK, t_min=0.3, dps=15, verbose=False):
    """The bulk critical-line sample for one real primitive chi: coarse walk +
    deficit rescans (a missed close pair is invisible to sign changes but visible
    as a theta_chi/pi counting deficit -- the #38 machinery, conductor-general).
    """
    zeros = [_polish_bracket(chi, a, b, dps)
             for a, b in _bracket_zeros(chi, t_min, t_max, 0.35, dps)]
    for _ in (2, 3):
        zeros.sort()
        edges = [t_min] + zeros + [t_max]
        with mp.workdps(dps):
            gaps = [(a, b) for a, b in zip(edges, edges[1:])
                    if float(zb.theta_chi(b, chi) - zb.theta_chi(a, chi)) / math.pi
                    > 1.8]
        if not gaps:
            break
        for a, b in gaps:
            for aa, bb in _bracket_zeros(chi, a + 1e-4, b - 1e-4, 0.35 / 6.0, dps):
                zeros.append(_polish_bracket(chi, aa, bb, dps))
    zeros = sorted(zeros)
    out = [zeros[0]] if zeros else []
    for t in zeros[1:]:
        if t - out[-1] > 1e-6:
            out.append(t)
    if verbose:
        print(f"      q={chi.q}: {len(out)} zeros walked to t = {t_max}")
    return out


def census_residual(chi, zeros, t_max=T_WALK):
    """N_walked - theta_chi(T)/pi: should sit at S(T) size (|.| < ~1, no +1)."""
    return len(zeros) - float(zb.theta_chi(t_max, chi) / PI)


def a1_at(gamma, chi, workdps=25):
    """(a_1, |L'|, |B(rho+1)|) at rho = 1/2 + i gamma: the displacement coefficient
    a_1 = rate_comb_chi(rho)/L'(rho) (zero_birth.disp_coeff_chi's algebra) plus the
    two factors of its conductor decomposition
    |a_1| = (q|rho|/2 pi^2) |B(rho+1)| / |L'|."""
    rho = mp.mpc("0.5", mp.mpf(gamma))
    with mp.workdps(workdps):
        Lp = mp.diff(lambda z: cb.L_chi(z, chi), rho)
        pred = cb.rate_comb_chi(rho, chi)
        B = zb.section_B(rho + 1, chi)
    return pred / Lp, float(abs(Lp)), float(abs(B))


# --------------------------------------------------------------------------
# 3. the census re-ask + ride-along closed forms
# --------------------------------------------------------------------------
def census_at(chi, K_list=(1, 2, 3), box=zb.CENSUS_BOX, max_jitter=3):
    """Argument-principle counts (target, {K: N_K}) in the box, entire-L convention
    (no +1). A zero ON the contour raises in winding_count; retry with the box
    edges jittered (the #39 off-round-edge trick, automated)."""
    (sr, tr) = box
    for j in range(max_jitter + 1):
        dt = 0.037 * j
        b_s = (sr[0] - dt / 2, sr[1] + dt / 2)
        b_t = (tr[0] + dt, tr[1] + dt)
        try:
            target = zb.winding_count(lambda s: cb.L_chi(s, chi), b_s, b_t)
            counts = {K: zb.winding_count(
                lambda s, K=K: cb.L_comb_K(s, K, chi), b_s, b_t)
                for K in K_list}
            return target, counts
        except RuntimeError:
            continue
    raise RuntimeError(f"census contour failed after {max_jitter} jitters "
                       f"({chi.label})")


def ride_along_row(chi):
    """The #43 conductor-indexed closed forms for an ODD real primitive chi:
    (L0, wobble radius L0/ln(q-1), conductor-string sigma*, string spacing,
    ground density ln(q-1)/2 pi). The string location/spacing generalize beyond
    phi(q) = 2 as the two-fastest-frequency balance of the seed D (the a = q-1
    term, coefficient chi(-1) = -1, against q^{-s} L0)."""
    q = chi.q
    L0 = zb.L0_chi(chi)
    sig, spac = zb.conductor_string(chi)
    return (float(L0), float(L0 / mp.log(q - 1)), sig, spac,
            float(mp.log(q - 1) / (2 * PI)))


# --------------------------------------------------------------------------
# 4. CSV caches (the chi6_two_component pattern: cheap re-runs, honest recompute)
# --------------------------------------------------------------------------
def write_rate_csv(rows, path=RATE_CSV):
    path.parent.mkdir(exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["q", "label", "parity", "p_star", "pred_re", "pred_im",
                    "meas_re", "meas_im", "rel_resid", "raw_rel", "c2_norm",
                    "B_re", "B_im"])
        for q, chi, pred, meas, rel, raw, c2n, B in rows:
            w.writerow([q, chi.label, chi.parity, p_star(q),
                        f"{float(mp.re(pred)):.12g}", f"{float(mp.im(pred)):.12g}",
                        f"{float(mp.re(meas)):.12g}", f"{float(mp.im(meas)):.12g}",
                        f"{rel:.6g}", f"{raw:.6g}", f"{c2n:.6g}",
                        f"{float(mp.re(B)):.12g}", f"{float(mp.im(B)):.12g}"])
    return path


def read_rate_csv(path=RATE_CSV):
    out = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.append({"q": int(r["q"]), "label": r["label"],
                        "parity": int(r["parity"]), "p_star": int(r["p_star"]),
                        "pred": complex(float(r["pred_re"]), float(r["pred_im"])),
                        "meas": complex(float(r["meas_re"]), float(r["meas_im"])),
                        "rel": float(r["rel_resid"]), "raw": float(r["raw_rel"]),
                        "c2n": float(r["c2_norm"]),
                        "B": complex(float(r["B_re"]), float(r["B_im"]))})
    return out


def write_walk_csv(rows, path=WALK_CSV):
    path.parent.mkdir(exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["q", "label", "parity", "gamma", "abs_Lp", "abs_B", "abs_a1"])
        for q, label, parity, gamma, absLp, absB, absa1 in rows:
            w.writerow([q, label, parity, f"{gamma:.12f}", f"{absLp:.8g}",
                        f"{absB:.8g}", f"{absa1:.8g}"])
    return path


def read_walk_csv(path=WALK_CSV):
    out = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.append({"q": int(r["q"]), "label": r["label"],
                        "parity": int(r["parity"]), "gamma": float(r["gamma"]),
                        "absLp": float(r["abs_Lp"]), "absB": float(r["abs_B"]),
                        "absa1": float(r["abs_a1"])})
    return out


# --------------------------------------------------------------------------
# 5. driver sections
# --------------------------------------------------------------------------
def _median(xs):
    ss = sorted(xs)
    n = len(ss)
    return ss[n // 2] if n % 2 else 0.5 * (ss[n // 2 - 1] + ss[n // 2])


def _compute_sweep(verbose=True):
    rows = []
    t0 = time.time()
    for q in range(3, Q_MAX + 1):
        got = measure_rate_q(q)
        for chi, pred, meas, rel, raw, c2n in got:
            rows.append((q, chi, pred, meas, rel, raw, c2n, section_factor(chi)))
        if verbose and got:
            worst = max(r[3] for r in got)
            print(f"   q={q:>2}: {len(got):>2} primitive chi, worst rel resid "
                  f"{worst:.1e}   [{time.time() - t0:.0f} s]")
    return rows


def _print_sweep(recompute):
    print("== the sweep: lim K(L - L_comb_K) vs (sq/2pi^2) B(s+1) at s0 = 1.3+7i, "
          f"all primitive chi, q <= {Q_MAX} ==")
    if recompute or not RATE_CSV.exists():
        rows = _compute_sweep()
        write_rate_csv(rows)
        print(f"   wrote {RATE_CSV}")
    data = read_rate_csv()
    rels = [d["rel"] for d in data]
    raws = [d["raw"] for d in data]
    c2ns = [d["c2n"] for d in data]
    print(f"   {len(data)} primitive characters across "
          f"{len(set(d['q'] for d in data))} moduli")
    print(f"   Richardson resid: median {_median(rels):.1e}, worst {max(rels):.1e}"
          f"   (raw top-checkpoint K=10q: median {_median(raws):.1e}, "
          f"worst {max(raws):.1e})")
    print(f"   resolution clock: |c_2/(Delta_1 q^2)| median {_median(c2ns):.2f} "
          f"(range {min(c2ns):.2f}..{max(c2ns):.2f}) -- the q/K expansion")
    # the exact group-mean identity (the spine of the amplitude panel)
    for q in (15, 16):
        res = mean_rate_identity_residual(q)
        print(f"   group-mean identity q={q}: |mean_chi rate - q s/2pi^2| = "
              f"{res:.1e}  (exact by orthogonality)")
    # section-factor tiers: |B - 1| set by the smallest surviving prime
    print("   section tiers |B(s0+1) - 1| by p* (smallest prime not dividing q):")
    for p in (2, 3, 5):
        sub = [abs(d["B"] - 1) for d in data if d["p_star"] == p]
        if sub:
            scale = float(p ** (-float(mp.re(S0)) - 1))
            print(f"      p*={p}: {len(sub):>3} chi, mean |B-1| = "
                  f"{sum(sub)/len(sub):.3f}  (~ p*^-2.3 = {scale:.3f})")
    return data


def _print_walks(recompute):
    print("\n== bulk zeros per conductor: the generalized Hardy-Z walk ==")
    if recompute or not WALK_CSV.exists():
        rows = []
        for q in Q_WALK:
            chi = real_primitive(q)
            eps = cb.root_number(chi)
            assert abs(eps - 1) < mp.mpf("1e-20"), (q, complex(eps))
            t0 = time.time()
            zeros = walk_zeros(chi)
            resid = census_residual(chi, zeros)
            print(f"   {chi.label:<10} ({'odd' if chi.parity else 'even'}): "
                  f"{len(zeros):>2} zeros to t={T_WALK:.0f}, census - theta/pi = "
                  f"{resid:+.2f}   [{time.time() - t0:.0f} s]")
            assert abs(resid) < 1.0, (q, resid)   # S(T)-sized, no +1
            for g in zeros:
                a1, absLp, absB = a1_at(g, chi)
                rows.append((q, chi.label, chi.parity, g, absLp, absB,
                             float(abs(a1))))
        write_walk_csv(rows)
        print(f"   wrote {WALK_CSV}")
    data = read_walk_csv()
    print(f"   sample: {len(data)} zeros across {len(set(d['q'] for d in data))} "
          "conductors (eps = +1 verified for each)")
    return data


def _print_a1_stats(walk):
    print("\n== displacement statistics: a_1 = rate_comb_chi(rho)/L'(rho) vs q ==")
    print("   q    n   med|a_1|   med|L'|   med|L'|/dens   med|B|")
    stats = {}
    for q in sorted(set(d["q"] for d in walk)):
        sub = [d for d in walk if d["q"] == q]
        dens = [_mean_density(d["gamma"], q) for d in sub]
        scaled = [d["absLp"] / rho for d, rho in zip(sub, dens)]
        stats[q] = {"n": len(sub),
                    "med_a1": _median([d["absa1"] for d in sub]),
                    "med_Lp": _median([d["absLp"] for d in sub]),
                    "med_scaled": _median(scaled),
                    "med_B": _median([d["absB"] for d in sub])}
        s = stats[q]
        print(f"   {q:>2} {s['n']:>4}   {s['med_a1']:>7.3f}   {s['med_Lp']:>7.3f}"
              f"   {s['med_scaled']:>9.3f}      {s['med_B']:.3f}")
    qs = sorted(stats)
    lo, hi = qs[0], qs[-1]
    ratio = stats[hi]["med_a1"] / stats[lo]["med_a1"]
    print(f"   growth {lo} -> {hi}: med|a_1| x{ratio:.1f} "
          f"(pure-linear x{hi/lo:.1f}; q/ln-tempered "
          f"x{(hi/lo) * math.log(lo * 25) / math.log(hi * 25):.1f})")
    print("   density-rescaled |L'| roughly q-flat -> the tempering is the local "
          "zero density ln(qt/2pi)/2pi")
    return stats


def _print_disp_certification():
    print("\n== spot certification at NEW moduli: measured comb roots vs a_1 ==")
    out = []
    for q, n_zero in ((8, 2), (15, 1)):
        chi = real_primitive(q)
        zeros = walk_zeros(chi, t_max=14.0)[:n_zero]
        for g in zeros:
            rho = mp.findroot(lambda ss: cb.L_chi(ss, chi), mp.mpc("0.5", g))
            a1 = zb.disp_coeff_chi(rho, chi)
            m45 = zb.measured_disp(rho, chi, 45)
            rel = float(abs(m45 - a1) / abs(a1))
            out.append((chi.label, float(mp.im(rho)), a1, m45, rel))
            print(f"   {chi.label} rho = 1/2+{float(mp.im(rho)):.3f}i:  "
                  f"a_1 = {complex(a1):.4f}, K=45 measured {complex(m45):.4f}  "
                  f"(rel {rel:.2f})")
    assert max(r[4] for r in out) < 0.15
    return out


def _print_census():
    print("\n== instant census at new conductors: N_K vs target, K = 1, 2, 3 ==")
    rows = []
    for q in CENSUS_QS:
        chi = real_primitive(q)
        t0 = time.time()
        target, counts = census_at(chi)
        rows.append((q, chi, target, counts))
        marks = "  ".join(f"K={K}: {n}{'=' if n == target else '!'}"
                          for K, n in sorted(counts.items()))
        print(f"   {chi.label:<10} target {target:>2}   {marks}   "
              f"[{time.time() - t0:.0f} s]")
    return rows


def _print_ride_alongs():
    print("\n== ride-alongs: the #43 closed forms across the odd walk family ==")
    print("   q     L(0,chi)   wobble L0/ln(q-1)   string sigma*   spacing   "
          "density ln(q-1)/2pi")
    rows = []
    for q in Q_WALK:
        chi = real_primitive(q)
        if chi.parity == 0:
            continue
        L0, rad, sig, spac, dens = ride_along_row(chi)
        rows.append((q, L0, rad, sig, spac, dens))
        print(f"   {q:>2}   {L0:>8.4f}   {rad:>12.3f}      {sig:>9.2f}   "
              f"{spac:>8.1f}   {dens:>10.3f}")
    print("   (L(0,chi) rational and climbing; the conductor string thins ~1/q: "
          "spacing 2pi/ln(q/(q-1)) ~ 2 pi q)")
    return rows


# --------------------------------------------------------------------------
# 6. figure
# --------------------------------------------------------------------------
def _figure(rate, walk, stats, census_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figstyle
    figstyle.enlarge()
    plt.rcParams.update({"axes.titlesize": 12.5, "legend.fontsize": 10,
                         "figure.titlesize": 15.5})

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10))
    tier_col = {2: "C0", 3: "C1", 5: "C3"}
    spine = float(abs(rl.rate_comb(S0)))

    # (0,0) amplitude: |constant| vs q, the q s/2pi^2 spine, tier colors
    ax = axes[0, 0]
    seen = set()
    for d in rate:
        c = tier_col[d["p_star"]]
        kw = {}
        if d["p_star"] not in seen:
            kw["label"] = f"$p_*={d['p_star']}$"
            seen.add(d["p_star"])
        ax.plot(d["q"], abs(d["meas"]), "o", ms=4, color=c, alpha=0.65, **kw)
    qq = list(range(3, Q_MAX + 1))
    ax.plot(qq, [spine * q for q in qq], "k-", lw=1.2,
            label=r"group mean $q\,|s|/2\pi^2$ (exact)")
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$|\lim K(L - L^{(K)}_{\rm comb})|$ at $s_0=1.3+7i$")
    ax.set_title("the amplitude law: constant $=\\frac{sq}{2\\pi^2}B(s{+}1)$,\n"
                 "linear in $q$ with an $O(1)$ section factor")
    ax.legend(loc="upper left")

    # (0,1) verification: rel resid vs q (Richardson + raw)
    ax = axes[0, 1]
    ax.semilogy([d["q"] for d in rate], [max(d["rel"], 1e-7) for d in rate],
                "o", ms=4, color="C0", alpha=0.6, label="3-pt Richardson resid")
    ax.semilogy([d["q"] for d in rate], [d["raw"] for d in rate],
                "^", ms=3.5, color="0.6", alpha=0.5, label=r"raw at $K=10q$")
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$|{\rm meas}-{\rm pred}|/|{\rm pred}|$")
    ax.set_title("confirmed conductor-by-conductor:\n$q$-uniform residuals at "
                 "matched resolution $K\\propto q$")
    ax.legend(loc="upper right")

    # (0,2) section factors in the complex plane: tiers = surviving primes
    ax = axes[0, 2]
    th = [i * 2 * math.pi / 120 for i in range(121)]
    for p in (2, 3, 5):
        r = float(p ** (-float(mp.re(S0)) - 1))
        ax.plot([1 + r * math.cos(a) for a in th], [r * math.sin(a) for a in th],
                "--", lw=1, color=tier_col[p], alpha=0.8)
    for d in rate:
        ax.plot(d["B"].real, d["B"].imag, "o", ms=4,
                color=tier_col[d["p_star"]], alpha=0.6)
    ax.plot(1, 0, "k+", ms=10, mew=1.5)
    ax.set_xlabel(r"${\rm Re}\,B(s_0{+}1)$")
    ax.set_ylabel(r"${\rm Im}\,B(s_0{+}1)$")
    ax.set_aspect("equal")
    ax.set_title("the section factor: $|B-1|$ set by the smallest\n"
                 "surviving prime $p_*$ (dashed: $p_*^{-\\sigma_0-1}$)")

    # (1,0) |a_1| vs q: scatter, medians, linear vs q/ln q curves
    ax = axes[1, 0]
    ax.semilogy([d["q"] for d in walk], [d["absa1"] for d in walk], "o", ms=3,
                color="C0", alpha=0.35, label=r"$|a_1|$ per zero")
    qs = sorted(stats)
    meds = [stats[q]["med_a1"] for q in qs]
    ax.semilogy(qs, meds, "s-", ms=7, color="C3", lw=1.5, mec="k", mew=0.5,
                label="per-$q$ median")
    q0 = qs[len(qs) // 2]
    m0 = stats[q0]["med_a1"]
    ax.semilogy(qs, [m0 * q / q0 for q in qs], ":", color="0.4", lw=1.4,
                label=r"pure linear $\propto q$")
    tbar = 25.0
    ax.semilogy(qs, [m0 * (q / q0) * math.log(q0 * tbar) / math.log(q * tbar)
                     for q in qs], "-", color="0.2", lw=1.4,
                label=r"$\propto q/\ln(q\bar t)$")
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$|a_1| = |{\rm rate}/L'(\rho)|$")
    ax.set_title("zero displacement: linear-in-$q$ amplitude,\n"
                 "log-$q$ density tempering, heavy $1/|L'|$ tail")
    ax.legend(loc="lower right", fontsize=9)

    # (1,1) L' statistics: raw growth vs density-rescaled collapse
    ax = axes[1, 1]
    qs = sorted(stats)
    ax.plot(qs, [stats[q]["med_Lp"] for q in qs], "o-", color="C0",
            label=r"median $|L'(\rho)|$")
    ax.plot(qs, [stats[q]["med_scaled"] for q in qs], "s-", color="C2",
            label=r"median $|L'|/\rho_{\rm dens}$ (rescaled)")
    dd = [float(math.log(q * tbar / (2 * math.pi)) / (2 * math.pi)) for q in qs]
    m_mid = stats[qs[len(qs) // 2]]["med_Lp"] / dd[len(qs) // 2]
    ax.plot(qs, [m_mid * d for d in dd], ":", color="0.4", lw=1.4,
            label=r"$\propto \ln(q\bar t/2\pi)$")
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$|L'|$ statistics over the walk sample")
    ax.set_title("the tempering measured: $|L'|$ grows with the local\n"
                 "density; rescaled by it, roughly $q$-flat")
    ax.legend(loc="lower right", fontsize=9)

    # (1,2) instant census at new moduli
    ax = axes[1, 2]
    xs = list(range(len(census_rows)))
    width = 0.22
    for i, K in enumerate((1, 2, 3)):
        ax.bar([x + (i - 1) * width for x in xs],
               [counts[K] for _, _, _, counts in census_rows],
               width=width, color=["C0", "C2", "C4"][i], label=f"$N_K$, $K={K}$")
    for x, (_, _, target, _) in zip(xs, census_rows):
        ax.plot([x - 2 * width, x + 2 * width], [target, target], "k--", lw=1.4)
    ax.plot([], [], "k--", label="target (entire $L$, no $+1$)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$q={q}$" for q, _, _, _ in census_rows])
    ax.set_ylabel(rf"box count, $t<{zb.CENSUS_BOX[1][1]:.0f}$")
    ax.set_title("the instant census generalizes:\nfull count at $K=1$ at every "
                 "new conductor")
    ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(r"Rate vs conductor: $\lim K(L-L^{(K)}_{\rm comb})"
                 r"=\frac{sq}{2\pi^2}B(s{+}1,\chi)$ swept over $q\leq 50$, and the"
                 r" $a_1$ statistics (issue #40)")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    figdir = _HERE / "figures"
    figdir.mkdir(exist_ok=True)
    out = figdir / "conductor_sweep.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recompute-sweep", action="store_true",
                    help="rebuild data/conductor_rate_sweep.csv (~2-3 min)")
    ap.add_argument("--recompute-zeros", action="store_true",
                    help="rebuild data/conductor_walk_a1.csv (~3-4 min)")
    args = ap.parse_args(argv)
    t0 = time.time()

    rate = _print_sweep(args.recompute_sweep)
    walk = _print_walks(args.recompute_zeros)
    stats = _print_a1_stats(walk)
    _print_disp_certification()
    census_rows = _print_census()
    _print_ride_alongs()
    _figure(rate, walk, stats, census_rows)
    print(f"({time.time() - t0:.0f} s total)")


if __name__ == "__main__":
    _main()
