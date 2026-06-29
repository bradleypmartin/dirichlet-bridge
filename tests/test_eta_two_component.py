"""Unit tests for the eta two-component (crystal x GUE) spectrum (issue #132,
knowledge/sum-integral/eta-two-component-spectrum.md).

Follow-up from #118 (the eta split) / #124 / #131. The eta zero set on the t-axis is the
rigid sigma=1 prefactor comb (t=2 pi k/ln2) superposed with the GUE sigma=1/2 zeta zeros.
The checks, by lens:

* LENS 1 (packing): the comb teeth ARE the (1-2^{1-s}) zeros; zeta zeros per comb gap
  reproduce 1,2,2,3,3,3,4,3 = log2(t/2pi); the comb phase is equidistributed but its
  histogram Fourier modes ARE the p=2 resonances (Lens 1 <-> Lens 3 identity);
* LENS 3 (headline): <cos(gamma log n)> resonates at prime powers with the explicit-
  formula coefficient -(Lambda(n)/sqrt n) T/(2 pi N), zero off prime powers; the COMB
  frequencies m ln2 = log 2^m lock NEGATIVELY (the comb teeth at zeta-density minima) ->
  the two components are NOT independent (they share the prime-2 lattice);
* LENS 2 (rigidity, slow): the union number variance is GUE-class (log, bounded ripple),
  not Poisson -- the comb is an additive backbone.

Everything uses the cached Riemann zeros (deterministic) and closed-form references; no
zero-finding is seeded by the zeros (the #132 forward-only discipline).
"""
import math

import mpmath as mp
import numpy as np
import pytest

import eta_two_component as etc

GAMMAS = np.asarray(etc.load_riemann_zeros(), dtype=float)
N = GAMMAS.size
TMAX = float(GAMMAS.max())
SE = math.sqrt(0.5 / N)          # sampling noise on a <cos> estimator (~0.0158)
LN2 = math.log(2.0)


# --------------------------------------------------------------------------
# LENS 1 -- the comb is the prefactor lattice; packing = log2(t/2pi)
# --------------------------------------------------------------------------
def test_comb_teeth_are_prefactor_zeros():
    """Each comb tooth t_k = 2 pi k/ln2 is a zero of the eta prefactor 1 - 2^{1-s} on sigma=1."""
    teeth = etc.comb_teeth(40.0)
    assert teeth.size >= 3
    for t in teeth:
        s = mp.mpc(1.0, float(t))
        assert abs(1 - mp.power(2, 1 - s)) < 1e-12


def test_comb_period_and_density():
    """The comb period is 2 pi/ln2 and its density the reciprocal ln2/2pi (constant)."""
    assert etc.COMB_PERIOD == pytest.approx(2 * math.pi / LN2)
    assert etc.COMB_DENSITY == pytest.approx(LN2 / (2 * math.pi))
    assert etc.COMB_PERIOD * etc.COMB_DENSITY == pytest.approx(1.0)


def test_comb_phase_zero_at_teeth():
    """comb_phase(tooth) ~ 0 (mod 1): phi=0 sits ON a tooth."""
    teeth = etc.comb_teeth(60.0)
    phi = etc.comb_phase(teeth)
    # phase is in [0,1); a tooth is at 0 or (numerically) just below 1
    assert np.all((phi < 1e-9) | (phi > 1 - 1e-9))


def test_zeros_per_tooth_reproduces_sequence():
    """The measured zeta zeros per comb gap reproduce the issue's 1,2,2,3,3,3,4,3."""
    _, counts, pred = etc.zeros_per_tooth(GAMMAS, 8)
    assert list(counts) == [1, 2, 2, 3, 3, 3, 4, 3]
    # and the prediction log2(t_c/2pi) tracks them to ~1 zero
    assert np.all(np.abs(counts - pred) < 1.0)


# --------------------------------------------------------------------------
# LENS 3 -- the prime-power resonance and the p=2 comb lock (headline)
# --------------------------------------------------------------------------
def test_resonance_prediction_is_explicit_formula_coeff():
    """resonance_prediction(n) = -(Lambda(n)/sqrt n) T/(2 pi N); 0 off prime powers."""
    for n in (2, 3, 4, 5, 7, 8, 9):
        lam = float(mp.log(etc._prime_base(n))) if etc._prime_base(n) else 0.0
        want = -(lam / math.sqrt(n)) * TMAX / (2 * math.pi * N)
        assert etc.resonance_prediction(GAMMAS, n) == pytest.approx(want, rel=1e-9, abs=1e-12)
    assert etc.resonance_prediction(GAMMAS, 6) == pytest.approx(0.0, abs=1e-15)  # 6 not a prime power


def test_prime_resonance_matches_prediction_at_prime_powers():
    """Measured <cos(gamma log n)> matches the explicit-formula prediction at prime powers."""
    for n in (2, 3, 4, 5, 7, 8, 9):
        meas = etc.prime_resonance(GAMMAS, math.log(n))
        pred = etc.resonance_prediction(GAMMAS, n)
        assert abs(meas - pred) < 0.01            # agreement far tighter than SE


def test_no_resonance_off_prime_powers():
    """No resonance at non-prime-power frequencies (log 6, log 10) or arbitrary controls."""
    for freq in (math.log(6), math.log(10), 1.0, 2.3137):
        assert abs(etc.prime_resonance(GAMMAS, freq)) < 3 * SE


def test_comb_locks_to_p2_negative():
    """THE HEADLINE: the comb frequencies m ln2 resonate NEGATIVELY and significantly --
    the comb teeth sit at the zeta-zero density minima, so the two components are NOT
    independent. m=1 is a > 5-sigma lock matching the p=2 explicit-formula coefficient."""
    for m, _omega, meas, pred in etc.comb_resonance_table(GAMMAS, 3):
        assert meas < 0                                    # teeth at density minima
        assert abs(meas - pred) < 0.01                     # = the p=2^m coefficient
    # the fundamental is a decisive lock, not noise
    r1 = etc.prime_resonance(GAMMAS, LN2)
    assert r1 / SE < -4.0


def test_phase_modes_are_p2_resonances():
    """Lens 1 <-> Lens 3: the comb-phase histogram's j-th cosine mode IS 2x the p=2^j
    resonance. c1 = 2 <cos(gamma ln2)> exactly (the dip at phi=0 is the p=2 lock)."""
    c1 = etc.phase_uniformity(GAMMAS)["cos1_coeff"]
    assert c1 == pytest.approx(2 * etc.prime_resonance(GAMMAS, LN2), rel=1e-9)
    assert c1 < 0                                          # dip at the tooth


def test_comb_frequencies_are_p2_bragg_peaks():
    """The comb teeth's spectral content m ln2 coincides with the p=2 Bragg peaks of the
    zeta-zero structure factor (#80B/#84): log(2^m) = m ln2 for the prime powers of 2."""
    peaks = {n: lg for n, lg, _ in etc.clp.prime_power_peaks(8)}
    for m in (1, 2, 3):
        assert peaks[2 ** m] == pytest.approx(m * LN2, rel=1e-12)


# --------------------------------------------------------------------------
# LENS 2 -- long-range rigidity of the union (slow: sliding-window estimator)
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_union_number_variance_is_gue_class():
    """The union Sigma^2(L) is GUE-class: it grows only ~log L (bounded, well below
    Poisson's L) -- the rigid comb is an ADDITIVE backbone, not a class change."""
    L = np.linspace(0.5, 15.0, 30)
    rig = etc.union_number_variance(GAMMAS, TMAX, L, n_windows=1500)
    s2 = rig["sigma2"]
    big = L >= 8.0
    # far below Poisson = L (rigid, not uncorrelated)
    assert np.all(s2[big] < 0.25 * L[big])
    # bounded and positive (a real spectrum, GUE+picket scale ~ O(1), not collapsing)
    assert np.all(s2[big] > 0.05)
    assert np.nanmax(s2) < 1.0
    # the comb fraction is the expected ~0.1 (ln2/2pi over the mean union density)
    assert 0.08 < rig["f_comb"] < 0.20


@pytest.mark.slow
def test_picket_variance_is_bounded():
    """A rigid sub-lattice contributes a BOUNDED number variance (<= 1/4) -- the crystal's
    long-range stiffness, the reason the comb does not Poissonise the union."""
    L = np.linspace(0.0, 50.0, 500)
    pv = etc.picket_number_variance(L, 0.12)
    assert np.all(pv >= 0.0) and np.all(pv <= 0.25 + 1e-12)
