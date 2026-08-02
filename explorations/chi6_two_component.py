r"""The chi_6 zero set as a two-component (crystal x GUE) spectrum -- issue #38.

EXPERIMENTAL (sub-issue of #37; same disjoint-from-the-manuscript rules as
character_bridge.py). The structural half is done there: L(s, chi_6) =
(1 + 2^{-s}) L(s, chi_3) is verified, the Euler-factor comb at sigma = 0,
t = (2m+1) pi / ln2 is confirmed as a zero set, and the K-migration splits the
ground string onto sigma = 1/2 UNION the sigma = 0 comb. This module is the
deferred SPECTRAL reading: generalize eta_two_component.py's crystal x GUE
superposition analysis to that zero set, running the vendored instruments
(gue_spacing, spectral_rigidity, zero_form_factor) on the union vs the
components.

The eta prediction, tested verbatim -- and its chi-twist
--------------------------------------------------------
eta's two components are the rigid sigma = 1 comb t = 2 pi k / ln2 and the GUE
zeta zeros on sigma = 1/2; the headline there was the p=2 lock: the comb teeth
sit at the ZETA-ZERO DENSITY MINIMA, with the measured <cos(gamma ln2)> equal to
the (negative) p=2 explicit-formula coefficient. chi_6's two components differ
in BOTH factors, and the two differences cancel:

  * the comb: zeros of 1 + 2^{-s} sit at t = (2m+1) pi/ln2 -- SAME period
    2 pi/ln2 as eta's comb, but offset by HALF a period. The teeth now sit where
    cos(t ln2) = -1 (eta's sat at cos = +1).
  * the GUE component: the explicit formula for L(s, chi_3) carries chi-twisted
    coefficients -chi_3(n) Lambda(n)/sqrt(n), and chi_3(2) = -1, so the p=2^m
    resonance <cos(m gamma ln2)> ALTERNATES in m and is POSITIVE at m = 1
    (zeta's is negative): the L-zero density now peaks where cos(t ln2) = +1.

Half-period comb offset x sign-flipped resonance = the SAME geometry: the comb
teeth sit at the L-zero density minima, again. That is the invariant form of the
"p=2 ghost" -- comb teeth avoid the partner zeros' density crests in eta, in
chi_0 mod 2, and in chi_6 alike -- and it is what this module measures.

The imprimitivity accounting behind it: chi_6(2^k) = 0, so the L(s, chi_6)
explicit formula has NO p=2 prime terms at all -- yet its zero set is
{L(chi_3) zeros} U {comb}, and each part separately carries p=2 oscillation.
The comb's own harmonics are EXACT: <cos(m tau ln2)>_teeth = (-1)^m (odd
multiples of pi/ln2 <-> the alternating sums over chi_3(2^k) = (-1)^k that
imprimitivity removed from the prime side). So the comb supplies, with the
opposite sign, the p=2 content the character deleted -- the crystalline
component IS the missing Euler factor, seen as a spectrum. (chi_0 mod 2 is the
constant-sign version of the same accounting: teeth at ALL multiples of
2 pi/ln2 <-> unsigned 2^k sums.)

A second falsifiable novelty vs eta: the CONDUCTOR prime goes dark. chi_3(3) = 0
kills the explicit-formula terms at log 3 and log 9, so the resonance scan over
the chi_3 zeros should show NOTHING at the p=3 frequencies where zeta resonates
strongly -- the arithmetic of the character read directly off the zeros.

The instruments (the issue's checklist)
---------------------------------------
  * SPACING (gue_spacing): the L(chi_3) zeros alone should be Wigner-GUE (the
    "GUE at 1/2" half of the prediction -- for zeta it was textbook, for chi_3 we
    measure it); the union deviates toward the independent-superposition law
    (reference: the same union with the comb offset re-drawn at random, which
    keeps both marginals and destroys any comb<->zero phase relation).
  * RIGIDITY (spectral_rigidity): Sigma^2(L) of the unfolded union. At this
    sample size (N ~ 400) Sigma^2 sits in Berry's SATURATED, non-universal
    regime (the chi_3 zeros alone measure flat ~0.27, matching a matched-size
    zeta sample -- GUE class at matched N; the growing GUE-log regime needs a
    much longer sample and is deferred). What IS measurable, and decisive, is
    the OFFSET comparison: the union with the comb at its PHYSICAL offset
    (pi/ln2, the locked phase) has Sigma^2 indistinguishable from the zeros
    alone and sits at the extreme LOW tail of the dephased ensemble (the same
    union with the offset drawn at random -- essentially no draw falls below
    it), while the anti-locked offset (delta = 0, teeth at the density maxima)
    and the naive independent superposition GUE(f_L L) + picket(f_c, L) (picket
    reused verbatim from eta_two_component) both sit well ABOVE. The crystal
    slots into the L-zero density minima and adds NO count variance: LENS 3's
    one-point lock re-measured as a negative two-point cross-covariance --
    consistent with the union being the SINGLE zero set of L(s, chi_6), whose
    explicit formula carries no p=2 terms at all (the components cancel each
    other's p=2 oscillation, so no independent-superposition model can fit).
  * FORM FACTOR (zero_form_factor): K(tau) of the L zeros alone -> the GUE ramp
    (small-tau slope ~ 1). The comb's crystalline (Bragg) content is smeared
    over a tau band in unfolded coordinates (the comb is periodic in t, not in
    unfolded position), and the offset comparison repeats LENS 2's verdict in
    Fourier: a DEPHASED comb adds the expected band excess over the zeros
    alone; the PHYSICAL comb adds NONE (a slight deficit, in fact) -- its Bragg
    content destructively interferes with the zeros' own p=2 oscillation, the
    form-factor face of the explicit-formula cancellation. The sharp
    t-coordinate Bragg reading is LENS 3, as in eta.

Forward-only / honesty
----------------------
The critical-line zero sample is computed HERE, blind: a Hardy-Z sign walk
(eps(chi_3) = +1, so Z_3(t) = e^{i theta_3(t)} L(1/2 + it, chi_3) is real on the
line) seeded by nothing but the smooth counting function, then bracketed-root
polish; the census is checked against theta_3(T)/pi + 1 and spot-verified by
2-D findroot on L itself (|Re - 1/2| ~ 1e-11). No zero tables are injected; the
first four heights independently reproduce character_bridge.CHI3_ZEROS (found
there by |L| minima scan). The comb is closed form. The instruments then measure
JOINT statistics of the two established families -- the forward content is the
two-component law with the chi-twisted lock, not a rediscovery of the zeros.

Run directly to validate + plot: writes explorations/figures/
chi6_two_component.png (seconds, from the cached CSV). Regenerate the zero
sample with `python chi6_two_component.py --recompute-zeros` (~1-2 min; writes
explorations/data/chi3_zeros.csv). Local/manual only -- NOT in repro.py or CI
(issue #37's CI-economy note).
"""
import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import mpmath as mp
from scipy.special import loggamma as _cloggamma

# Path shim: explorations/ module cross-importing the flat bridge/ sources as bare
# siblings (same convention as character_bridge.py; explorations/conftest.py does
# this for pytest).
_HERE = Path(__file__).resolve().parent
_BRIDGE = _HERE.parent / "bridge"
for _p in (str(_HERE), str(_BRIDGE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import character_bridge as cb   # noqa: E402  (CHI3, L_chi, polish_zero)
import cone_log_prime as clp    # noqa: E402  (von_mangoldt, prime_power_peaks)
import gue_spacing as gs        # noqa: E402  (spacings, Wigner GUE, zeta zeros)
import spectral_rigidity as sr  # noqa: E402  (unfold_to_unit_density, Sigma^2)
import zero_form_factor as zff  # noqa: E402  (spectral_form_factor, GUE ramp)
from eta_two_component import picket_number_variance  # noqa: E402  (reused verbatim)

PI = math.pi
LN2 = math.log(2.0)
COMB_PERIOD = 2.0 * PI / LN2    # tooth spacing (~9.0647) -- same as eta's comb
COMB_OFFSET = PI / LN2          # half-period offset: 1 + 2^{-s}, not 1 - 2^{1-s}
COMB_DENSITY = LN2 / (2.0 * PI)
CHI3_ZEROS_CSV = _HERE / "data" / "chi3_zeros.csv"
T_MAX_DEFAULT = 550.0           # theta_3/pi + 1 ~ 401 zeros


# ===========================================================================
# The two families
# ===========================================================================
def _teeth_at(offset: float, t_max: float, t_min: float) -> np.ndarray:
    """Lattice points offset + m * COMB_PERIOD (m >= 0) with t_min < t <= t_max."""
    m_lo = max(int(math.floor((t_min - offset) / COMB_PERIOD)) + 1, 0)
    m_hi = int(math.floor((t_max - offset) / COMB_PERIOD))
    return offset + COMB_PERIOD * np.arange(m_lo, m_hi + 1, dtype=float)


def comb_teeth(t_max: float, t_min: float = 0.0) -> np.ndarray:
    """The sigma = 0 Euler-factor zeros (1 + 2^{-s} = 0): t = (2m+1) pi/ln2.

    Period 2 pi/ln2 (eta's), offset pi/ln2 (half a period off eta's teeth); the
    crystalline component of the chi_6 two-component spectrum.
    """
    return _teeth_at(COMB_OFFSET, t_max, t_min)


def load_chi3_zeros(path: Optional[Path] = None) -> np.ndarray:
    """The cached L(1/2 + it, chi_3) zero heights (t > 0) as float64, sorted."""
    path = path or CHI3_ZEROS_CSV
    out: List[float] = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.append(float(row["gamma"]))
    return np.array(sorted(out), dtype=float)


# ===========================================================================
# Smooth counting / unfolding (theta_3), and the Hardy-Z zero walk
# ===========================================================================
def theta3(t: Any) -> np.ndarray:
    """The chi_3 Riemann-Siegel-type phase (vectorized, float64):

        theta_3(t) = Im ln Gamma(3/4 + it/2) + (t/2) ln(3/pi),

    from the completed Lambda(s) = (3/pi)^{(s+1)/2} Gamma((s+1)/2) L(s, chi_3)
    (chi_3 odd, kappa = 1). N(T) ~ theta_3(T)/pi counts the 0 < t <= T critical
    zeros up to the small S(T); d(theta_3/pi)/dt = ln(3t/2pi)/2pi is the smooth
    density.
    """
    t = np.asarray(t, dtype=float)
    return np.imag(_cloggamma(0.75 + 0.5j * t)) + 0.5 * t * math.log(3.0 / PI)


def smooth_count(t: Any) -> np.ndarray:
    """The Riemann-von Mangoldt-type smooth zero count, theta_3(t)/pi.

    NO "+1": zeta's N(T) = theta/pi + 1 + S(T) owes its +1 to the pole at s = 1
    in the argument principle, and L(s, chi_3) is entire. Measured against the
    walked census, N - theta_3/pi oscillates within ~+-0.7 of zero (S(T)-sized)
    across the whole sample, confirming both the constant and completeness.
    """
    return theta3(t) / PI


def hardy_z3(t: float, dps: int = 15) -> float:
    """Z_3(t) = e^{i theta_3(t)} L(1/2 + it, chi_3): REAL on the line.

    chi_3 is real primitive with root number eps = +1 (tau(chi_3) = i sqrt(3)),
    so Lambda(1/2 + it) is real and the phase-rotated L is too -- sign changes of
    Z_3 bracket the critical zeros. Phase and L are evaluated at matched mpmath
    precision so the imaginary part cancels to rounding (checked in tests).
    """
    with mp.workdps(dps):
        tt = mp.mpf(t)
        th = mp.im(mp.loggamma(mp.mpc(0.75, tt / 2))) + tt / 2 * mp.log(3 / mp.pi)
        return float(mp.re(mp.expj(th) * cb.L_chi(mp.mpc(0.5, tt), cb.CHI3)))


def _mean_density(t: float) -> float:
    """The smooth zero density ln(3t/2pi)/2pi, floored for the low-t grid."""
    return max(math.log(max(3.0 * t / (2.0 * PI), 1.05)) / (2.0 * PI), 0.02)


def _bracket_zeros(t_lo: float, t_hi: float, step_frac: float,
                   dps: int) -> List[Tuple[float, float]]:
    """Sign-change brackets of Z_3 on an adaptive grid over (t_lo, t_hi):
    step = step_frac / local mean density, capped at 0.45. The final grid point
    is CLAMPED to t_hi -- an overshooting grid would step past a rescan gap's
    right endpoint, which is itself a zero, and re-find it as a duplicate."""
    grid = [t_lo]
    while grid[-1] < t_hi:
        step = min(0.45, step_frac / _mean_density(grid[-1]))
        grid.append(min(grid[-1] + step, t_hi))
    vals = [hardy_z3(t, dps) for t in grid]
    return [(a, b) for a, b, va, vb in zip(grid, grid[1:], vals, vals[1:])
            if va * vb < 0]


def _polish(a: float, b: float, dps: int) -> float:
    """Bracketed (Anderson) findroot on the real Z_3 over (a, b)."""
    with mp.workdps(dps):
        t0 = mp.findroot(lambda t: hardy_z3(float(t), dps),
                         (mp.mpf(a), mp.mpf(b)), solver="anderson", maxsteps=60)
    return float(t0)


def compute_chi3_zeros(t_max: float = T_MAX_DEFAULT, t_min: float = 2.0,
                       dps: int = 15, verbose: bool = True) -> np.ndarray:
    """The bulk L(1/2 + it, chi_3) zero sample: blind Hardy-Z sign walk + polish.

    Pass 1 walks an adaptive grid at ~1/3 of the local mean spacing and polishes
    every sign change. Deficit passes then rescan (at 6x grid density, twice) any
    gap between consecutive found zeros whose theta_3-predicted count exceeds 1.8
    -- a missed CLOSE PAIR leaves an even number of interior zeros, invisible to
    sign changes on the coarse grid but visible as a counting deficit (|S(T)| is
    well below 0.8 in this range, so 1.8 only fires on genuine holes). The driver
    checks the final census against theta_3(T)/pi.
    """
    zeros = [_polish(a, b, dps) for a, b in _bracket_zeros(t_min, t_max, 0.35, dps)]
    if verbose:
        print(f"   pass 1: {len(zeros)} sign-change zeros on the coarse walk")
    for npass in (2, 3):
        zeros.sort()
        edges = [t_min] + zeros + [t_max]
        gaps = [(a, b) for a, b in zip(edges, edges[1:])
                if float(theta3(b) - theta3(a)) / PI > 1.8]
        if not gaps:
            break
        found = 0
        for a, b in gaps:
            for aa, bb in _bracket_zeros(a + 1e-4, b - 1e-4, 0.35 / 6.0, dps):
                zeros.append(_polish(aa, bb, dps))
                found += 1
        if verbose:
            print(f"   pass {npass}: rescanned {len(gaps)} deficit gaps, "
                  f"recovered {found} zeros")
    out = np.array(sorted(zeros), dtype=float)
    # Defensive collapse of re-finds: genuine consecutive zeros in this range are
    # never closer than ~1e-2, so a sub-1e-6 gap is the same root polished twice.
    return out[np.concatenate([[True], np.diff(out) > 1e-6])]


def write_chi3_zeros(zeros: np.ndarray, path: Optional[Path] = None) -> Path:
    """Cache the zero sample in the data/riemann_zeros.csv column convention."""
    path = path or CHI3_ZEROS_CSV
    path.parent.mkdir(exist_ok=True)
    src = "Hardy Z_3 sign walk + Anderson polish (chi6_two_component.py)"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "gamma", "gamma_dps", "source"])
        for i, t in enumerate(zeros, start=1):
            w.writerow([i, f"{t:.12f}", 12, src])
    return path


# ===========================================================================
# Unfolding the union
# ===========================================================================
def union_unfolded(gammas: np.ndarray, t_max: float,
                   offset: float = COMB_OFFSET) -> Tuple[np.ndarray, float]:
    """Unit-density unfolded UNION spectrum {gamma_n} U {comb} up to t_max.

    Unfold by the smooth union counting function theta_3(t)/pi + t ln2/2pi (the
    L smooth count + the comb staircase's smooth part), then the single global
    affine rescale (sr.unfold_to_unit_density). Returns (xi, f_comb) with f_comb
    the mean comb fraction COMB_DENSITY/(COMB_DENSITY + rho_L) over the band --
    the mixing fraction for the independent-superposition reference. `offset`
    exists for the surrogate resamplings; the physical comb is COMB_OFFSET.
    """
    g = np.asarray(gammas, dtype=float)
    g = g[g <= t_max]
    teeth = _teeth_at(offset, t_max, float(g.min()))
    union = np.sort(np.concatenate([g, teeth]))
    xi_raw = smooth_count(union) + union * COMB_DENSITY
    xi = sr.unfold_to_unit_density(xi_raw)
    rho_L = np.log(np.clip(3.0 * union / (2.0 * PI), 1.05, None)) / (2.0 * PI)
    f_comb = float(np.mean(COMB_DENSITY / (COMB_DENSITY + rho_L)))
    return xi, f_comb


# ===========================================================================
# LENS 1 -- packing: L zeros per comb gap, and the comb phase
# ===========================================================================
def zeros_per_gap(gammas: np.ndarray,
                  n_gaps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count chi_3 zeros in each of the first n_gaps comb gaps (tooth_m, tooth_{m+1}].

    Predicted count = log2(3 t_c / 2pi), the inter-tooth density ratio (the L-zero
    density ln(3t/2pi)/2pi over the comb density ln2/2pi) -- eta's packing law with
    the conductor q = 3 inside the log.
    """
    g = np.sort(np.asarray(gammas, dtype=float))
    teeth = COMB_OFFSET + COMB_PERIOD * np.arange(0, n_gaps + 1, dtype=float)
    counts = (np.searchsorted(g, teeth[1:], side="right")
              - np.searchsorted(g, teeth[:-1], side="right")).astype(int)
    centers = 0.5 * (teeth[:-1] + teeth[1:])
    predicted = np.log2(np.maximum(3.0 * centers / (2.0 * PI), 1e-9))
    return np.arange(1, n_gaps + 1), counts, predicted


def comb_phase(gammas: np.ndarray) -> np.ndarray:
    """phi_n = frac(gamma_n ln2 / 2pi), on eta's clock: phi = 1/2 sits ON a chi_6
    comb tooth (the half-period offset); phi = 0 is where eta's teeth sat.

    Keeping eta's phase convention makes the chi-twist legible: eta's p=2 dip at
    phi = 0 becomes chi_6's PEAK at phi = 0 and dip at phi = 1/2 -- so the teeth
    sit in the dip in both spectra.
    """
    g = np.asarray(gammas, dtype=float)
    return np.mod(g * LN2 / (2.0 * PI), 1.0)


def phase_uniformity(gammas: np.ndarray, n_bins: int = 12) -> Dict[str, Any]:
    """Comb-phase histogram + chi^2 against uniform + the leading cosine mode.

    As in eta: {gamma alpha} is asymptotically equidistributed (Hlawka/Fujii), and
    the finite-N non-uniformity IS the resonance lens -- c1 = 2 <cos(gamma ln2)>,
    here predicted POSITIVE (chi_3(2) = -1 flips zeta's sign): peak at phi = 0,
    dip at the teeth (phi = 1/2).
    """
    phi = comb_phase(gammas)
    N = phi.size
    hist, _ = np.histogram(phi, bins=n_bins, range=(0.0, 1.0))
    expected = N / n_bins
    chi2 = float(((hist - expected) ** 2 / expected).sum())
    c1 = float(2.0 * np.cos(2.0 * PI * phi).mean())
    return {"counts": hist, "chi2_per_dof": chi2 / (n_bins - 1),
            "cos1_coeff": c1, "expected_per_bin": float(expected)}


# ===========================================================================
# LENS 3 -- the chi-twisted p=2 resonance (the headline)
# ===========================================================================
def prime_resonance(gammas: np.ndarray, freq: float) -> float:
    """<cos(gamma_n * freq)> over the zero sample (the even structure factor)."""
    g = np.asarray(gammas, dtype=float)
    return float(np.cos(g * freq).mean())


def resonance_prediction(gammas: np.ndarray, n: int,
                         chi: Optional[cb.Character] = cb.CHI3) -> float:
    """Explicit-formula prediction at freq = log n for the chi zero sample:

        <cos(gamma log n)>  ~  -Re[chi(n)] (Lambda(n)/sqrt n) T_max/(2 pi N),

    the chi-twisted form of eta_two_component.resonance_prediction (chi = None
    means the trivial weight 1, i.e. zeta's zeros). chi_3(2) = -1 makes the 2-adic
    tower ALTERNATE (positive at n = 2); chi_3(3) = 0 makes the conductor prime
    DARK (nothing at n = 3, 9); Lambda kills non-prime-powers as before.
    """
    g = np.asarray(gammas, dtype=float)
    N, t_max = g.size, float(g.max())
    lam = float(clp.von_mangoldt(n)[n])
    w = 1.0 if chi is None else float(mp.re(chi(n)))
    return -w * lam / math.sqrt(n) * t_max / (2.0 * PI * N)


def comb_resonance_table(
    gammas: np.ndarray, m_max: int = 4
) -> List[Tuple[int, float, float, float, float]]:
    """The comb-frequency resonances: rows (m, omega, measured, predicted, comb).

    `measured`/`predicted` are the chi_3-zero resonance and its chi-twisted
    explicit-formula amplitude at omega = m ln2; `comb` = (-1)^m is the comb's own
    EXACT resonance (teeth at odd multiples of pi/ln2: cos(m(2j+1)pi) = (-1)^m) --
    always opposite in sign to the predicted zero resonance: the two components
    carry the p=2 oscillation with opposite phase, the imprimitivity accounting.
    """
    out: List[Tuple[int, float, float, float, float]] = []
    for m in range(1, m_max + 1):
        omega = m * LN2
        out.append((m, omega, prime_resonance(gammas, omega),
                    resonance_prediction(gammas, 2 ** m, cb.CHI3),
                    float((-1.0) ** m)))
    return out


def chi0_mod2_check(m_max: int = 3) -> List[Tuple[int, float, float, float]]:
    """The degenerate cross-check: L(s, chi_0 mod 2) = (1 - 2^{-s}) zeta(s).

    Its comb sits at t = 2 pi k/ln2 (offset 0 -- eta's very teeth, at sigma = 0
    instead of 1), its GUE component IS zeta's zeros (cached CSV; no new
    computation). Rows (m, measured, predicted, comb): the trivial-weight
    resonance reproduces eta_two_component's numbers (-0.098 at m = 1), the comb's
    own resonance is +1 exactly -- constant-sign where chi_6 alternates, and the
    teeth again sit at the partner density minima.
    """
    zg = np.asarray(gs.load_riemann_zeros(), dtype=float)
    return [(m, prime_resonance(zg, m * LN2),
             resonance_prediction(zg, 2 ** m, None), 1.0)
            for m in range(1, m_max + 1)]


def _zeta_union_sigma2(zg: np.ndarray, offset: float, L_values: np.ndarray,
                       n_windows: int) -> np.ndarray:
    """Sigma^2(L) of {zeta zeros} U {comb at `offset`}, RvM-unfolded."""
    teeth = _teeth_at(offset, float(zg.max()), float(zg.min()))
    union = np.sort(np.concatenate([zg, teeth]))
    xi_raw = gs.unfold_riemann_zeros(union) + union * COMB_DENSITY
    xi = sr.unfold_to_unit_density(xi_raw)
    s2, _ = sr.number_variance(xi, np.asarray(L_values), n_windows=n_windows)
    return np.asarray(s2)


def chi0_sigma2_lock(n_zeros: int = 400, L_values: Tuple[float, ...] = (5.0, 10.0),
                     n_draws: int = 16, n_windows: int = 2500,
                     seed: int = 38) -> Dict[str, Any]:
    """The cross-check's two-point lock, matched to the chi_6 sample size.

    For chi_0 mod 2 the PHYSICAL comb offset is 0 (teeth at 2 pi k/ln2) -- and
    zeta's p=2 coefficient is NEGATIVE, so those teeth sit at the ZETA-zero
    density minima: delta = 0 should be zeta's variance-minimizing phase exactly
    as pi/ln2 is chi_3's. Same instrument, first n_zeros cached zeta zeros.
    """
    zg = np.asarray(gs.load_riemann_zeros(), dtype=float)[:n_zeros]
    L = np.asarray(L_values, dtype=float)
    locked = _zeta_union_sigma2(zg, 0.0, L, n_windows)
    rng = np.random.default_rng(seed)
    draws = np.array([_zeta_union_sigma2(zg, float(rng.uniform(0.0, COMB_PERIOD)),
                                         L, n_windows) for _ in range(n_draws)])
    return {"L": L, "locked": locked, "dephased_mean": draws.mean(axis=0),
            "dephased_std": draws.std(axis=0, ddof=1),
            "frac_below": (draws < locked[None, :]).mean(axis=0),
            "n_draws": n_draws}


# ===========================================================================
# LENS 2 -- number variance: the lock at two-point level (offset ensemble)
# ===========================================================================
def sigma2_at_offset(gammas: np.ndarray, t_max: float, L_values: np.ndarray,
                     offset: float, n_windows: int = 3000) -> np.ndarray:
    """Sigma^2(L) of the unfolded union with the comb lattice at `offset`."""
    xi, _ = union_unfolded(gammas, t_max, offset=offset)
    s2, _ = sr.number_variance(xi, np.asarray(L_values), n_windows=n_windows)
    return np.asarray(s2)


def surrogate_sigma2(gammas: np.ndarray, t_max: float, L_values: np.ndarray,
                     n_draws: int = 32, n_windows: int = 3000,
                     seed: int = 38) -> np.ndarray:
    """The dephased ensemble: Sigma^2 curves for uniformly-random comb offsets
    (the independent-superposition null with both marginals intact). Returns the
    (n_draws, len(L_values)) array; deterministic via the seed."""
    rng = np.random.default_rng(seed)
    return np.array([sigma2_at_offset(gammas, t_max, L_values,
                                      float(rng.uniform(0.0, COMB_PERIOD)),
                                      n_windows)
                     for _ in range(n_draws)])


def union_number_variance(
    gammas: np.ndarray, t_max: float, L_values: np.ndarray,
    n_windows: int = 3000, n_draws: int = 32, seed: int = 38
) -> Dict[str, Any]:
    """Sigma^2(L): the LOCKED union vs the zeros alone, the dephased ensemble,
    the anti-locked comb, and the naive references.

    At N ~ 400 the variance sits in Berry's saturated regime (all curves flat),
    so the decisive measurement is the OFFSET comparison: `locked` (physical
    comb, pi/ln2) vs the `draws` ensemble (random offsets) vs `anti` (delta = 0,
    teeth at the L-zero density maxima). The one-point lock predicts
    locked <= every draw <= anti, i.e. the physical phase MINIMIZES the count
    variance -- the comb fills the density minima and adds nothing. `frac_below`
    is the fraction of dephased draws below the locked curve at each L
    (predicted ~0). The naive independent superposition GUE(f_L L) +
    picket(f_c, L) is kept as the reference that OVER-predicts.
    """
    g = np.asarray(gammas, dtype=float)
    g = g[g <= t_max]
    xi_alone = sr.unfold_to_unit_density(smooth_count(g))
    s2_alone, _ = sr.number_variance(xi_alone, L_values, n_windows=n_windows)
    locked = sigma2_at_offset(g, t_max, L_values, COMB_OFFSET, n_windows)
    anti = sigma2_at_offset(g, t_max, L_values, 0.0, n_windows)
    draws = surrogate_sigma2(g, t_max, L_values, n_draws, n_windows, seed)
    xi_u, f_comb = union_unfolded(g, t_max)
    f_L = 1.0 - f_comb
    gue_full = sr.number_variance_reference(zff.gue_form_factor, L_values)
    gue_L = sr.number_variance_reference(zff.gue_form_factor,
                                         f_L * np.asarray(L_values))
    superpos = np.asarray(gue_L) + picket_number_variance(L_values, f_comb)
    return {"L": np.asarray(L_values), "alone": np.asarray(s2_alone),
            "locked": locked, "anti": anti,
            "dephased_mean": draws.mean(axis=0),
            "dephased_std": draws.std(axis=0, ddof=1),
            "frac_below": (draws < locked[None, :]).mean(axis=0),
            "gue": np.asarray(gue_full), "superposition": superpos,
            "f_comb": f_comb, "n_levels": xi_u.size, "n_draws": n_draws}


# ===========================================================================
# Instrument 1 -- nearest-neighbour spacings (gue_spacing)
# ===========================================================================
def unfolded_spacings(gammas: np.ndarray, t_max: float) -> np.ndarray:
    """Unit-mean spacings of the chi_3 zeros ALONE (the GUE-at-1/2 test)."""
    g = np.asarray(gammas, dtype=float)
    g = g[g <= t_max]
    xi = sr.unfold_to_unit_density(smooth_count(g))
    return gs.rescale_to_unit_mean(gs.nearest_neighbour_spacings(xi))


def union_spacings(gammas: np.ndarray, t_max: float,
                   offset: float = COMB_OFFSET) -> np.ndarray:
    """Unit-mean spacings of the union, comb at `offset` (physical: COMB_OFFSET)."""
    xi, _ = union_unfolded(gammas, t_max, offset=offset)
    return gs.rescale_to_unit_mean(gs.nearest_neighbour_spacings(xi))


def surrogate_spacings(gammas: np.ndarray, t_max: float, n_offsets: int = 64,
                       seed: int = 38) -> np.ndarray:
    """The independent-superposition spacing reference: the same two marginals
    with the comb offset drawn uniformly in [0, period) -- any comb<->zero phase
    relation is destroyed, the densities are untouched. Concatenated spacings
    over n_offsets draws (seeded: deterministic)."""
    rng = np.random.default_rng(seed)
    out = [union_spacings(gammas, t_max, offset=float(rng.uniform(0.0, COMB_PERIOD)))
           for _ in range(n_offsets)]
    return np.concatenate(out)


def ks_distance(spacings: np.ndarray, cdf) -> float:
    """Kolmogorov-Smirnov distance of a spacing sample against a reference CDF."""
    s = np.sort(np.asarray(spacings, dtype=float))
    ecdf = np.arange(1, s.size + 1, dtype=float) / s.size
    return float(np.max(np.abs(ecdf - np.asarray(cdf(s), dtype=float))))


# ===========================================================================
# Instrument 3 -- the spectral form factor (zero_form_factor)
# ===========================================================================
def union_form_factor(gammas: np.ndarray, t_max: float, tau_max: float = 1.35,
                      n_tau: int = 2700, n_bins: int = 36, n_draws: int = 16,
                      seed: int = 38) -> Dict[str, Any]:
    """K(tau): the L zeros alone, the PHYSICAL union, and the dephased-comb mean.

    The L-alone curve is the GUE test: ramp of small-tau slope ~ 1 into the
    plateau K -> 1. The comb's Bragg content is smeared over the band tau in
    [f_comb(t_max), f_comb(t_min)] (`bragg_band`) because the comb is periodic
    in t, not in unfolded position. `band_K` holds the band-averaged K for the
    three spectra: the dephased comb ADDS a crystalline excess over the zeros
    alone, the physical comb adds NONE -- destructive interference with the
    zeros' own p=2 oscillation (LENS 2's negative cross-covariance in Fourier).
    """
    g = np.asarray(gammas, dtype=float)
    g = g[g <= t_max]
    xi_l = sr.unfold_to_unit_density(smooth_count(g))
    taus = np.linspace(0.012, tau_max, n_tau)
    K_l = zff.spectral_form_factor(xi_l, taus)

    def K_at(offset: float) -> np.ndarray:
        xi, _ = union_unfolded(g, t_max, offset=offset)
        return zff.spectral_form_factor(xi, taus)

    K_u = K_at(COMB_OFFSET)
    rng = np.random.default_rng(seed)
    K_d = np.mean([K_at(float(rng.uniform(0.0, COMB_PERIOD)))
                   for _ in range(n_draws)], axis=0)
    _, f_comb = union_unfolded(g, t_max)
    rho = np.log(np.clip(3.0 * np.array([t_max, float(g.min())]) / (2.0 * PI),
                         1.05, None)) / (2.0 * PI)
    band = COMB_DENSITY / (COMB_DENSITY + rho)          # [f at t_max, f at t_min]
    in_band = (taus >= band[0]) & (taus <= band[1])
    return {"union_binned": zff.bin_form_factor(taus, K_u, n_bins),
            "L_binned": zff.bin_form_factor(taus, K_l, n_bins),
            "dephased_binned": zff.bin_form_factor(taus, K_d, n_bins),
            "slope_L": zff.small_tau_slope(taus, K_l),
            "slope_union": zff.small_tau_slope(taus, K_u),
            "band_K": {"alone": float(K_l[in_band].mean()),
                       "locked": float(K_u[in_band].mean()),
                       "dephased": float(K_d[in_band].mean())},
            "f_comb": f_comb, "bragg_band": (float(band[0]), float(band[1]))}


# ===========================================================================
# self-validating driver
# ===========================================================================
def _print_provenance(gammas: np.ndarray) -> None:
    N, t_hi = gammas.size, float(gammas.max())
    pred = float(smooth_count(t_hi + 0.4))
    print("== the zero sample (blind Hardy-Z walk; cached in data/chi3_zeros.csv) ==")
    print(f"   N = {N} zeros of L(1/2 + it, chi_3) on 0 < t <= {t_hi:.2f}")
    print(f"   census vs smooth count theta_3/pi = {pred:.2f}:  "
          f"|N - pred| = {abs(N - pred):.2f}  (|S(T)|-sized; no +1 -- no pole)")
    for t in (float(gammas[0]), float(gammas[N // 2]), t_hi):
        z = cb.polish_zero(cb.CHI3, t, 0.5)
        assert z is not None, t
        print(f"   spot polish on L itself at t = {t:9.3f}:  "
              f"|Re - 1/2| = {abs(float(z.real) - 0.5):.1e}   "
              f"|t - cached| = {abs(float(z.imag) - t):.1e}")
    match = max(abs(a - b) for a, b in zip(gammas[:4], cb.CHI3_ZEROS))
    print(f"   first four vs character_bridge.CHI3_ZEROS (independent seed): "
          f"max |diff| = {match:.1e}")


def _print_packing(gammas: np.ndarray) -> None:
    print("\n== LENS 1: packing -- chi_3 zeros per comb gap vs log2(3t/2pi) ==")
    k, counts, pred = zeros_per_gap(gammas, 8)
    print("   comb gap m:         " + "  ".join(f"{int(c):>3d}" for c in counts)
          + "   (measured zeros in the gap)")
    print("   log2(3 t_c/2pi):    " + "  ".join(f"{p:4.1f}" for p in pred)
          + "   (predicted density ratio)")
    u = phase_uniformity(gammas)
    print("   comb phase (teeth at phi = 1/2): chi^2/dof = "
          f"{u['chi2_per_dof']:.2f} vs uniform, leading cosine c1 = "
          f"{u['cos1_coeff']:+.4f}")
    print("   -> c1 POSITIVE (eta's was negative): peak at phi = 0, dip at the")
    print("      teeth (phi = 1/2) -- the chi-twist, and LENS 3 seen in phase space.")


def _print_resonance(gammas: np.ndarray) -> None:
    print("\n== LENS 3 (headline): the chi-twisted p=2 resonance ==")
    N, t_hi = gammas.size, float(gammas.max())
    se = math.sqrt(0.5 / N)
    print(f"   N = {N} zeros, t_max = {t_hi:.1f}; sampling noise on <cos> ~ {se:.4f}.")
    print("   comb frequencies omega = m ln2: the zero resonance ALTERNATES "
          "(chi_3(2) = -1),")
    print("   the comb's own resonance is (-1)^m EXACTLY -- always opposite:")
    print("     m   omega     zeros <cos>   predicted   z-score   comb (-1)^m")
    for m, omega, meas, pred, comb in comb_resonance_table(gammas, 4):
        print(f"     {m}   {omega:7.5f}   {meas:+.4f}      {pred:+.4f}     "
              f"{meas / se:+6.2f}     {comb:+.0f}")
    print("   -> teeth sit where cos(t ln2) = -1; zero density peaks where it is +1:")
    print("      the comb teeth sit at the L-ZERO DENSITY MINIMA -- eta's lock, "
          "chi-twisted.")
    print("\n   the prime scan (the conductor prime is DARK: chi_3(3) = 0):")
    print("     n   freq=log n   chi_3(n)   measured <cos>   predicted   note")
    for n in (2, 3, 4, 5, 6, 7, 8, 9):
        w = float(mp.re(cb.CHI3(n)))
        meas = prime_resonance(gammas, math.log(n))
        pred = resonance_prediction(gammas, n, cb.CHI3)
        note = ""
        if n in (3, 9):
            note = "<== conductor: dark"
        elif n in (2, 4, 8):
            note = "<== comb (p=2)"
        elif n == 6:
            note = "not a prime power"
        print(f"     {n}   {math.log(n):8.4f}     {w:+4.0f}      {meas:+.4f}"
              f"          {pred:+.4f}    {note}")
    print("   -> resonance only at prime powers COPRIME to the conductor; zeta")
    print("      resonates at log 3 (coeff -0.13), the chi_3 zeros do not.")


def _print_crosscheck() -> None:
    print("\n== degenerate cross-check: chi_0 mod 2 = (1 - 2^{-s}) zeta ==")
    print("   comb at t = 2 pi k/ln2 (offset 0), GUE component = the cached zeta "
          "zeros:")
    print("     m   zeta <cos>   predicted   comb <cos>")
    for m, meas, pred, comb in chi0_mod2_check(3):
        print(f"     {m}   {meas:+.4f}     {pred:+.4f}     {comb:+.0f}")
    print("   -> eta_two_component's numbers verbatim (-0.098 at m = 1): constant")
    print("      sign where chi_6 alternates, teeth at the partner minima in both.")
    lk = chi0_sigma2_lock()
    print("   two-point lock, matched N = 400 zeta zeros (PHYSICAL offset = 0 "
          "there):")
    for i, Lv in enumerate(lk["L"]):
        print(f"     L = {Lv:4.1f}:  locked {lk['locked'][i]:.3f}   dephased "
              f"{lk['dephased_mean'][i]:.3f} +- {lk['dephased_std'][i]:.3f}   "
              f"draws below locked: {lk['frac_below'][i]:.2f}")
    print("   -> delta = 0 minimizes zeta's union variance exactly as pi/ln2")
    print("      minimizes chi_3's: each L parks its comb in its own minima.")


def _print_rigidity(rig: Dict[str, Any]) -> None:
    print("\n== LENS 2: number variance -- the lock at two-point level ==")
    print(f"   union of {int(rig['n_levels'])} levels, f_comb = "
          f"{rig['f_comb']:.3f}; {rig['n_draws']} dephased draws.")
    print("   Berry-saturated regime at N ~ 400 (all curves flat; the GUE-log")
    print("   growth reading needs a much longer sample -- deferred). The decisive")
    print("   comparison is across COMB OFFSETS:")
    L = rig["L"]
    print("     L    zeros alone   locked (pi/ln2)   dephased mean+-std   "
          "anti (0)   GUE(f L)+picket   below")
    for i in range(0, len(L), max(1, len(L) // 8)):
        print(f"   {L[i]:5.1f}    {rig['alone'][i]:7.3f}      {rig['locked'][i]:7.3f}"
              f"        {rig['dephased_mean'][i]:6.3f} +- {rig['dephased_std'][i]:5.3f}"
              f"     {rig['anti'][i]:6.3f}      {rig['superposition'][i]:7.3f}"
              f"        {rig['frac_below'][i]:.2f}")
    print("   -> the LOCKED union tracks the zeros ALONE (the comb adds ~no")
    print("      variance) and sits at the extreme low tail of the dephased")
    print("      ensemble; the anti-locked comb and the independent GUE+picket")
    print("      model sit well above. The crystal fills the L-zero density")
    print("      minima: LENS 3's lock as a negative two-point cross-covariance")
    print("      (the union is ONE L-function's zero set -- chi_6 has no p=2")
    print("      terms for an independent-superposition model to reproduce).")


def _print_spacing(s_l: np.ndarray, s_u: np.ndarray, s_ind: np.ndarray) -> None:
    print("\n== instrument 1: nearest-neighbour spacings ==")
    d_gue = ks_distance(s_l, gs.wigner_gue_cdf)
    d_poi = ks_distance(s_l, gs.poisson_cdf)
    print(f"   chi_3 zeros alone ({s_l.size} spacings): KS vs Wigner-GUE = "
          f"{d_gue:.3f}, vs Poisson = {d_poi:.3f}")
    print("   -> GUE-at-1/2 measured (for zeta it was textbook; for chi_3 we "
          "check it).")
    du_gue = ks_distance(s_u, gs.wigner_gue_cdf)
    print(f"   union ({s_u.size} spacings): KS vs GUE = {du_gue:.3f} -- the comb "
          "spoils pure GUE;")
    # KS of the union against the surrogate's empirical CDF
    s_sorted = np.sort(s_ind)
    def _surr_cdf(x):
        return np.searchsorted(s_sorted, x, side="right") / s_sorted.size
    d_surr = ks_distance(s_u, _surr_cdf)
    print(f"   union vs the random-offset surrogate (independent superposition, "
          f"{s_ind.size} spacings): KS = {d_surr:.3f}")
    print("   -> the union matches the INDEPENDENT superposition at spacing level:")
    print("      the p=2 lock is a ~10% one-point modulation, invisible in p(s) --")
    print("      it lives in the resonance (LENS 3) and the Sigma^2 offset test")
    print("      (LENS 2), not in the spacing law.")


def _print_form_factor(res: Dict[str, Any]) -> None:
    print("\n== instrument 3: spectral form factor K(tau) ==")
    print(f"   chi_3 zeros alone: small-tau slope = {res['slope_L']:.2f} "
          "(GUE ramp = 1, GOE = 2, Poisson = flat)")
    lo, hi = res["bragg_band"]
    bk = res["band_K"]
    print(f"   band-averaged K over the smeared comb band tau in "
          f"[{lo:.2f}, {hi:.2f}]:")
    print(f"     zeros alone {bk['alone']:.3f}   union PHYSICAL comb "
          f"{bk['locked']:.3f}   union dephased comb {bk['dephased']:.3f}")
    print("   -> the dephased comb ADDS the expected crystalline (Bragg) excess;")
    print("      the physical comb adds NONE (slight deficit): its Bragg content")
    print("      destructively interferes with the zeros' own p=2 oscillation --")
    print("      the form-factor face of the explicit-formula cancellation")
    print("      (chi_6 has no p=2 terms). Sharp t-coordinate Bragg = LENS 3.")


def _main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="chi_6 two-component spectrum (#38)")
    ap.add_argument("--recompute-zeros", action="store_true",
                    help="re-run the Hardy-Z walk and rewrite data/chi3_zeros.csv "
                         "(~1-2 min) instead of loading the cache")
    ap.add_argument("--t-max", type=float, default=T_MAX_DEFAULT,
                    help="walk ceiling for --recompute-zeros")
    args = ap.parse_args(argv)

    if args.recompute_zeros or not CHI3_ZEROS_CSV.exists():
        print(f"== computing the chi_3 zero sample up to t = {args.t_max} ==")
        zeros = compute_chi3_zeros(args.t_max)
        out = write_chi3_zeros(zeros)
        print(f"   wrote {len(zeros)} zeros -> {out}")
    gammas = load_chi3_zeros()
    t_hi = float(gammas.max())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figstyle
    figstyle.enlarge()
    plt.rcParams.update({"axes.titlesize": 12, "figure.titlesize": 14.5,
                         "legend.fontsize": 10})

    _print_provenance(gammas)
    _print_packing(gammas)
    _print_resonance(gammas)
    _print_crosscheck()
    L_values = np.linspace(0.5, 16.0, 20)
    rig = union_number_variance(gammas, t_hi, L_values)
    _print_rigidity(rig)
    s_l = unfolded_spacings(gammas, t_hi)
    s_u = union_spacings(gammas, t_hi)
    s_ind = surrogate_spacings(gammas, t_hi)
    _print_spacing(s_l, s_u, s_ind)
    ffres = union_form_factor(gammas, t_hi)
    _print_form_factor(ffres)

    # ================================ figure ================================
    fig, axg = plt.subplots(2, 3, figsize=(16.5, 9.5))

    # (a) comb phase: peak at 0, dip at the teeth (phi = 1/2) -- the chi-twist
    ax = axg[0, 0]
    phi = comb_phase(gammas)
    nb = 16
    ax.hist(phi, bins=nb, range=(0, 1), color="C0", alpha=0.7,
            label=r"comb phase $\phi_n$")
    exp = gammas.size / nb
    ax.axhline(exp, ls="--", color="0.4", lw=1, label="uniform")
    xx = np.linspace(0, 1, 200)
    c1 = phase_uniformity(gammas, nb)["cos1_coeff"]
    ax.plot(xx, exp * (1.0 + c1 * np.cos(2 * PI * xx)), color="C3", lw=2,
            label=fr"$1{{+}}c_1\cos2\pi\phi$, $c_1{{=}}{c1:+.3f}$")
    ax.axvline(0.5, ls=":", lw=1.6, color="C1")
    ax.text(0.5, 0.02, " teeth", transform=ax.get_xaxis_transform(),
            color="C1", ha="left")
    ax.set_xlabel(r"comb phase $\phi_n=\{\gamma_n\ln2/2\pi\}$ (teeth at "
                  r"$\frac{1}{2}$)")
    ax.set_ylabel("count")
    ax.set_title(r"(a) $c_1>0$: peak at $\phi{=}0$, dip at the teeth"
                 "\n" r"($\eta$'s dip was at $\phi{=}0$ -- the $\chi$-twist)")
    ax.legend(loc="lower center")

    # (b) resonance scan: chi-signed prime-power comb, conductor dark
    ax = axg[0, 1]
    omegas = np.linspace(0.45, 3.2, 1400)
    res = np.array([prime_resonance(gammas, w) for w in omegas])
    ax.plot(omegas, res, color="C0", lw=1.0)
    ax.axhline(0.0, color="0.6", lw=0.8)
    for i, (n, lg, _w) in enumerate(clp.prime_power_peaks(24)):
        w = float(mp.re(cb.CHI3(n)))
        if w == 0.0:                       # conductor prime powers 3, 9 -- dark
            col, ls, lw = "0.25", "-", 1.8
        elif (n & (n - 1)) == 0:           # the comb's p = 2 tower
            col, ls, lw = "C3", "--", 1.4
        else:
            col, ls, lw = "C2", ":", 0.8
        ax.axvline(lg, color=col, ls=ls, lw=lw, alpha=0.8)
        # predicted amplitude tick (0 for the dark conductor lines)
        ax.plot(lg, resonance_prediction(gammas, n, cb.CHI3), marker="_",
                color="k", ms=9, mew=1.6, zorder=5)
        ax.text(lg, 0.03 if i % 2 == 0 else 0.10, str(n), ha="center",
                va="bottom", fontsize=10, color=col,
                transform=ax.get_xaxis_transform())
    ax.set_xlim(float(omegas.min()), float(omegas.max()))
    ax.set_xlabel(r"probe frequency $\omega$ ($=\log n$ at prime powers)")
    ax.set_ylabel(r"$\langle\cos(\gamma_n\,\omega)\rangle$")
    ax.set_title(r"(b) LENS 3: $\chi_3$-signed peaks; conductor $p{=}3$ DARK"
                 "\n" r"(black: $\log3,\log9$ -- $\zeta$ resonates there, $\chi_3$ "
                 "does not)")

    # (c) comb harmonics: alternating lock + the chi_0 mod 2 cross-check
    ax = axg[0, 2]
    tab = comb_resonance_table(gammas, 4)
    ms = [r[0] for r in tab]
    meas = [r[2] for r in tab]
    pred = [r[3] for r in tab]
    se = math.sqrt(0.5 / gammas.size)
    ax.errorbar(ms, meas, yerr=se, fmt="o", ms=7, color="C3", capsize=3,
                label=r"$\chi_3$ zeros: $\langle\cos(m\gamma\ln2)\rangle$")
    ax.plot(ms, pred, "s--", color="C0", ms=6,
            label=r"$-\chi_3(2^m)\frac{\ln2}{2^{m/2}}\frac{T}{2\pi N}$")
    ck = chi0_mod2_check(4)
    zg_n = len(gs.load_riemann_zeros())
    ax.errorbar([m + 0.12 for m, *_ in ck], [r[1] for r in ck],
                yerr=math.sqrt(0.5 / zg_n), fmt="D", ms=5, mfc="none",
                color="0.45", capsize=2,
                label=r"$\chi_0$ mod 2 check ($\zeta$ zeros)")
    ax.plot([m + 0.12 for m, *_ in ck], [r[2] for r in ck], "x", ms=6,
            color="0.45")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xticks(ms)
    ax.set_xlabel(r"comb harmonic $m$  ($\omega=m\ln2$)")
    ax.set_ylabel(r"$\langle\cos(m\gamma\ln2)\rangle$")
    ax.set_title(r"(c) headline: ALTERNATING lock ($\chi_3(2){=}{-}1$);"
                 "\n" r"comb's own $(-1)^m$ always opposite $\to$ teeth at minima")
    ax.legend(loc="upper right")

    # (d) spacings: L alone = Wigner GUE; union = independent superposition
    ax = axg[1, 0]
    bins = np.linspace(0.0, 3.2, 33)
    ax.hist(s_l, bins=bins, density=True, color="C0", alpha=0.55,
            label=r"$\chi_3$ zeros alone")
    ax.hist(s_u, bins=bins, density=True, histtype="step", color="C1", lw=1.8,
            label="union (comb in)")
    hist_ind, edges = np.histogram(s_ind, bins=bins, density=True)
    ax.stairs(hist_ind, edges, color="0.25", ls=":", lw=1.6,
              label="random-offset surrogate")
    ss = np.linspace(0.0, 3.2, 300)
    ax.plot(ss, gs.wigner_gue_pdf(ss), color="C3", lw=2, label="Wigner GUE")
    ax.plot(ss, gs.poisson_pdf(ss), color="0.6", ls="--", lw=1.2, label="Poisson")
    ax.set_xlabel(r"unit-mean spacing $s$")
    ax.set_ylabel(r"$p(s)$")
    ax.set_title(r"(d) $\chi_3$ alone $=$ GUE; union $=$ independent"
                 "\nsuperposition (lock invisible in $p(s)$)")
    ax.legend(loc="upper right")

    # (e) number variance across comb offsets: the locked phase minimizes it
    ax = axg[1, 1]
    ax.fill_between(rig["L"], rig["dephased_mean"] - rig["dephased_std"],
                    rig["dephased_mean"] + rig["dephased_std"], color="0.75",
                    alpha=0.55, label=f"dephased comb ({rig['n_draws']} offsets, "
                                      r"$\pm$sd)")
    ax.plot(rig["L"], rig["anti"], "-", color="C3", lw=1.6,
            label=r"anti-locked ($\delta{=}0$: teeth at maxima)")
    ax.plot(rig["L"], rig["superposition"], "-.", color="0.35", lw=1.2,
            label=r"independent: GUE$(f_LL)$+picket")
    ax.plot(rig["L"], rig["alone"], "--", color="C0", lw=1.6,
            label=r"$\chi_3$ zeros alone")
    ax.plot(rig["L"], rig["locked"], "-o", color="C1", lw=1.8, ms=4,
            label=r"union, PHYSICAL comb ($\delta{=}\pi/\ln2$)")
    top = max(float(np.nanmax(rig["anti"])), float(np.nanmax(rig["superposition"])))
    ax.set_ylim(0, 1.6 * top)   # headroom so the legend clears the curves
    ax.set_xlabel(r"window length $L$ (unfolded)")
    ax.set_ylabel(r"$\Sigma^2(L)$")
    ax.set_title(r"(e) LENS 2: the physical offset MINIMIZES $\Sigma^2$"
                 "\n(locked $=$ zeros alone: the comb adds no variance)")
    ax.legend(loc="upper left", fontsize=8.5)

    # (f) form factor: dephased comb shows the Bragg excess, physical cancels it
    ax = axg[1, 2]
    cu, mu, eu = ffres["union_binned"]
    cl, ml, el = ffres["L_binned"]
    cd, md, _ed = ffres["dephased_binned"]
    ax.errorbar(cl, ml, yerr=el, fmt="o", ms=3.5, color="C0", alpha=0.85,
                label=rf"$\chi_3$ alone (slope {ffres['slope_L']:.2f})")
    ax.errorbar(cu, mu, yerr=eu, fmt="s", ms=3.5, color="C1", alpha=0.85,
                label="union, PHYSICAL comb")
    ax.plot(cd, md, "-", color="0.4", lw=1.5, label="union, dephased comb (mean)")
    tt = np.linspace(0, float(cu.max()), 200)
    ax.plot(tt, np.minimum(tt, 1.0), color="C3", lw=1.8,
            label=r"GUE: $\min(\tau,1)$")
    lo, hi = ffres["bragg_band"]
    ax.axvspan(lo, hi, color="C3", alpha=0.12,
               label=r"comb Bragg band ($m{=}1$, smeared)")
    ax.set_xlabel(r"$\tau$ (unfolded)")
    ax.set_ylabel(r"$K(\tau)$")
    ax.set_ylim(0, 2.2)
    ax.set_title("(f) dephased comb: Bragg excess over the band;\n"
                 r"physical comb: CANCELLED (rides the ramp with $\chi_3$)")
    ax.legend(loc="upper right", fontsize=8.5)

    fig.suptitle(r"The $\chi_6$ two-component spectrum: rigid $\sigma{=}0$ Euler "
                 r"comb $\times$ GUE $\sigma{=}\frac{1}{2}$ $L(\chi_3)$ zeros -- "
                 r"$\eta$'s crystal$\,\times\,$GUE law, $\chi$-twisted (issue #38)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    figdir = _HERE / "figures"
    figdir.mkdir(exist_ok=True)
    out = figdir / "chi6_two_component.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    _main()
