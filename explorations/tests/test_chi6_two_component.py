"""Tests for chi6_two_component (issue #38). OUTSIDE the default CI suite (run
with `pytest explorations/tests`). Lean: the heavy work (the Hardy-Z walk, the
surrogate ensembles) lives in the driver / behind the slow marker; these tests
validate the cached zero sample and the closed-form/identity layer.
"""
import math

import numpy as np
import mpmath as mp
import pytest

import character_bridge as cb
import chi6_two_component as c2
import gue_spacing as gs


# --------------------------------------------------------------------------
# the comb
# --------------------------------------------------------------------------
def test_comb_teeth_are_euler_factor_zeros():
    """Teeth solve 1 + 2^{-s} = 0 on sigma = 0, with eta's period, half-offset."""
    teeth = c2.comb_teeth(100.0)
    for t in teeth:
        assert abs(1 + mp.power(2, -mp.mpc(0, t))) < 1e-12
    assert np.allclose(np.diff(teeth), c2.COMB_PERIOD)
    assert abs(teeth[0] - c2.COMB_OFFSET) < 1e-12
    assert abs(c2.COMB_OFFSET - c2.COMB_PERIOD / 2) < 1e-12


def test_comb_own_resonance_is_alternating_exactly():
    """<cos(m t ln2)> over the teeth = (-1)^m: the crystal's exact harmonics."""
    teeth = c2.comb_teeth(2000.0)
    for m in (1, 2, 3):
        val = float(np.cos(teeth * m * c2.LN2).mean())
        assert abs(val - (-1.0) ** m) < 1e-10


# --------------------------------------------------------------------------
# the cached zero sample
# --------------------------------------------------------------------------
def test_cached_zeros_load_sorted_and_match_structural_seeds():
    g = c2.load_chi3_zeros()
    assert g.size >= 350
    assert np.all(np.diff(g) > 1e-2)          # sorted, no duplicates/close refinds
    for ours, ref in zip(g[:4], cb.CHI3_ZEROS):
        assert abs(ours - ref) < 1e-6         # character_bridge's independent seeds


def test_census_matches_smooth_count():
    """|N - theta_3/pi| stays S(T)-sized at the top of the sample (no +1: no pole)."""
    g = c2.load_chi3_zeros()
    pred = float(c2.smooth_count(float(g.max()) + 0.4))
    assert abs(g.size - pred) < 1.5


def test_hardy_z_is_real_and_vanishes_at_zeros():
    """|Z_3| == |L(1/2+it)| (the phase is pure rotation) and Z_3 ~ 0 at a zero."""
    t = 37.0
    with mp.workdps(15):
        L = abs(cb.L_chi(mp.mpc(0.5, t), cb.CHI3))
    assert abs(abs(c2.hardy_z3(t)) - float(L)) < 1e-10
    g = c2.load_chi3_zeros()
    assert abs(c2.hardy_z3(float(g[5]))) < 1e-6


def test_spot_zeros_polish_onto_the_line():
    """A few cached heights re-polished by 2-D findroot on L itself sit on 1/2."""
    g = c2.load_chi3_zeros()
    for t in (float(g[10]), float(g[200])):
        z = cb.polish_zero(cb.CHI3, t, 0.5)
        assert z is not None
        assert abs(float(z.real) - 0.5) < 1e-8
        assert abs(float(z.imag) - t) < 1e-6


@pytest.mark.slow
def test_bulk_zero_verification():
    """Every 20th cached zero re-polishes onto sigma = 1/2 (findroot on L)."""
    g = c2.load_chi3_zeros()
    for t in g[::20]:
        z = cb.polish_zero(cb.CHI3, float(t), 0.5)
        assert z is not None
        assert abs(float(z.real) - 0.5) < 1e-8


# --------------------------------------------------------------------------
# LENS 1 -- packing and phase
# --------------------------------------------------------------------------
def test_packing_counts_and_density_ratio():
    g = c2.load_chi3_zeros()
    k, counts, pred = c2.zeros_per_gap(g, 20)
    teeth = c2.COMB_OFFSET + c2.COMB_PERIOD * np.arange(0, 21)
    manual = int(((g > teeth[0]) & (g <= teeth[-1])).sum())
    assert int(counts.sum()) == manual
    assert np.all(pred[1:] >= pred[:-1])      # log density ratio increases
    # totals track the prediction across the sample
    assert abs(counts.sum() - pred.sum()) < 0.15 * pred.sum()


def test_phase_teeth_at_half_and_positive_c1():
    g = c2.load_chi3_zeros()
    teeth = c2.comb_teeth(500.0)
    assert np.allclose(c2.comb_phase(teeth), 0.5)     # teeth sit at phi = 1/2
    u = c2.phase_uniformity(g)
    se = math.sqrt(2.0 / g.size)                      # sd of c1 = 2<cos>
    assert u["cos1_coeff"] > 2 * se                   # the POSITIVE p=2 mode


# --------------------------------------------------------------------------
# LENS 3 -- the chi-twisted resonance
# --------------------------------------------------------------------------
def test_resonance_prediction_chi_twist():
    g = c2.load_chi3_zeros()
    p1 = c2.resonance_prediction(g, 2, cb.CHI3)
    p2 = c2.resonance_prediction(g, 4, cb.CHI3)
    p3 = c2.resonance_prediction(g, 8, cb.CHI3)
    assert p1 > 0 > p2 and p3 > 0                     # chi_3(2) = -1 alternation
    for n in (3, 9):                                  # the conductor is dark
        assert c2.resonance_prediction(g, n, cb.CHI3) == 0.0
    assert c2.resonance_prediction(g, 6, cb.CHI3) == 0.0   # Lambda(6) = 0
    zg = np.asarray(gs.load_riemann_zeros(), dtype=float)
    assert c2.resonance_prediction(zg, 2, None) < 0   # zeta's sign (eta's case)


def test_measured_lock_alternates_and_matches_prediction():
    """The headline, cheaply: measured <cos(m gamma ln2)> alternates with m and
    tracks the chi-twisted explicit-formula amplitude within 3 sigma."""
    g = c2.load_chi3_zeros()
    se = math.sqrt(0.5 / g.size)
    for m, _omega, meas, pred, comb in c2.comb_resonance_table(g, 3):
        assert meas * pred > 0                        # sign alternation measured
        assert abs(meas - pred) < 3 * se
        assert comb * pred < 0                        # comb always opposite


def test_conductor_dark_measured():
    """Measured resonance at log 3 is noise-level (zeta's there is ~ -0.13)."""
    g = c2.load_chi3_zeros()
    se = math.sqrt(0.5 / g.size)
    assert abs(c2.prime_resonance(g, math.log(3.0))) < 2.5 * se


def test_chi0_mod2_crosscheck_reproduces_eta():
    """The degenerate cross-check reproduces eta's measured p=2 numbers."""
    rows = c2.chi0_mod2_check(2)
    m1 = rows[0]
    assert abs(m1[1] - (-0.0981)) < 5e-3              # eta_two_component's -0.098
    assert abs(m1[1] - m1[2]) < 5e-3                  # measured == predicted
    assert m1[3] == 1.0                               # constant-sign comb


# --------------------------------------------------------------------------
# unfolding + LENS 2 machinery (fast pieces)
# --------------------------------------------------------------------------
def test_union_unfolded_unit_density():
    g = c2.load_chi3_zeros()
    xi, f_comb = c2.union_unfolded(g, float(g.max()))
    d = np.diff(xi)
    assert np.all(d > -1e-12)
    assert abs(d.mean() - 1.0) < 1e-9                 # exact affine rescale
    assert 0.10 < f_comb < 0.35


def test_picket_bound_reused_from_eta():
    L = np.linspace(0.2, 25.0, 400)
    v = c2.picket_number_variance(L, 0.13)
    assert np.all(v >= 0.0) and np.all(v <= 0.25 + 1e-12)


def test_spacings_gue_beats_poisson():
    g = c2.load_chi3_zeros()
    s = c2.unfolded_spacings(g, float(g.max()))
    assert abs(float(s.mean()) - 1.0) < 1e-9
    assert c2.ks_distance(s, gs.wigner_gue_cdf) < c2.ks_distance(s, gs.poisson_cdf)


@pytest.mark.slow
def test_sigma2_lock_below_dephased():
    """The two-point lock: the physical offset's Sigma^2 sits below the dephased
    ensemble mean (small version of the driver's 32-draw comparison)."""
    g = c2.load_chi3_zeros()
    t_hi = float(g.max())
    L = np.array([5.0, 12.0])
    locked = c2.sigma2_at_offset(g, t_hi, L, c2.COMB_OFFSET, n_windows=1500)
    draws = c2.surrogate_sigma2(g, t_hi, L, n_draws=8, n_windows=1500, seed=7)
    assert np.all(locked < draws.mean(axis=0))
