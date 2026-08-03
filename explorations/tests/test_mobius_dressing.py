"""Tests for mobius_dressing (issue #47). OUTSIDE the default CI suite:
pytest.ini's `testpaths = tests` excludes explorations/; run these with
`pytest explorations/tests`. Lean by design -- the module is pure
post-processing of the committed a_1 cache, so the headline measurements
themselves are cheap enough to regression-test directly.
"""
import math

import mpmath as mp
import numpy as np

import arithmetic_clock as ac
import character_bridge as cb
import mobius_dressing as md
import zero_birth as zb


# --------------------------------------------------------------------------
# the weight algebra
# --------------------------------------------------------------------------
def test_mu_of():
    """Moebius on the band n's: squarefree signs, prime-power zeros."""
    assert [md.mu_of(n) for n in (1, 2, 5, 6, 10, 30)] == [1, -1, -1, 1, 1, -1]
    assert [md.mu_of(n) for n in (4, 8, 9, 12, 16)] == [0, 0, 0, 0, 0]


def test_B_factor_matches_section():
    """The closed-form B(rho + 1) is zero_birth's section polynomial at the
    same argument (q = 3: B(s) = 1 - 2^{-s})."""
    for g in (8.039737155681, 141.5):
        closed = complex(md.B_factor(np.array([g]))[0])
        ref = complex(zb.section_B(mp.mpc(1.5, g), cb.CHI3))
        assert abs(closed - ref) < 1e-12


def test_undress_roundtrip():
    """Dressing back the undressed weight recovers a_1 exactly."""
    g = np.array([10.0, 25.0, 100.0])
    a1 = np.array([1 + 2j, -3 + 0.5j, 0.1 - 4j])
    assert np.allclose(md.undress(g, a1) * md.B_factor(g), a1)


def test_spectrum_matches_resonance_constant():
    """The vectorized spectrum equals arithmetic_clock.resonance_constant."""
    rng = np.random.default_rng(3)
    g = np.sort(rng.uniform(10, 300, 50))
    a1 = rng.normal(size=50) + 1j * rng.normal(size=50)
    for w in (0.7, ac.LN2, 2.3):
        ref = ac.resonance_constant(g, list(a1), w)
        assert abs(md.spectrum(g, a1, [w])[0] - ref) < 1e-12


# --------------------------------------------------------------------------
# the model identities (exact algebra, no data)
# --------------------------------------------------------------------------
def test_half_law_is_exact_in_the_model():
    """dressed_ref/mobius_ref = 1/2 identically on even squarefree lines:
    |mu(2)chi(2)/sqrt2 - beta| / (1/sqrt2) = 1 - 1/2."""
    for n in (2, 10, 14, 22, 26, 34):
        assert abs(md.dressed_ref(n, 1.0) / md.mobius_ref(n, 1.0) - 0.5) < 1e-12


def test_ln4_masquerade_is_exact_in_the_model():
    """The vacancy satellite equals the Landau amplitude identically at p = 2
    (satellite/Landau = 2/p): scale ln4 beta/sqrt2 = scale Lambda(4)/2."""
    assert abs(md.dressed_ref(4, 3.7) - md.landau_ref(4, 3.7)) < 1e-12


def test_cascade_stops_after_one_step():
    """ln 8 and ln 16 are dark in the dressed model too: their parents ln 4,
    ln 8 are mu = 0 vacancies, and B has a single sideband."""
    assert md.dressed_ref(8, 1.0) == 0.0
    assert md.dressed_ref(16, 1.0) == 0.0
    assert md.mobius_ref(4, 1.0) == 0.0


def test_odd_lines_undressed_in_the_model():
    """Odd-n lines are untouched by the dressing (the shift only lands on
    even frequencies ln 2m)."""
    for n in (5, 7, 35):
        assert md.dressed_ref(n, 2.0) == md.mobius_ref(n, 2.0)


# --------------------------------------------------------------------------
# the measured headline (regression on the committed a_1 cache)
# --------------------------------------------------------------------------
def _cache():
    g, a1 = md.load_weights()
    a1p = md.undress(g, a1)
    return g, a1, a1p, md.fit_scale(g, a1p)


def test_undressed_spectrum_is_mobius_not_landau():
    """The discriminators: ln 4 dark (Landau says 1.0 of its ref), ln 10 at
    full Moebius strength (Landau says 0)."""
    g, _a1, a1p, scale = _cache()
    mx4, _w = md.band_max(g, a1p, math.log(4))
    assert mx4 < 0.25 * md.landau_ref(4, scale)
    mx10, _w = md.band_max(g, a1p, math.log(10))
    assert abs(mx10 / md.mobius_ref(10, scale) - 1.0) < 0.15


def test_measured_half_law():
    """dressed/undressed on the even squarefree lines ln 2 and ln 10."""
    g, a1, a1p, _scale = _cache()
    for n in (2, 10):
        w0 = math.log(n)
        ratio = md.band_max(g, a1, w0)[0] / md.band_max(g, a1p, w0)[0]
        assert 0.42 <= ratio <= 0.62


def test_measured_ln4_satellite():
    """The dressed ln 4 band sits on its closed form (and so on Landau --
    the p = 2 masquerade) to a few percent."""
    g, a1, _a1p, scale = _cache()
    mx, _w = md.band_max(g, a1, math.log(4))
    assert abs(mx / md.dressed_ref(4, scale) - 1.0) < 0.05


def test_measured_odd_line_untouched():
    """ln 35 (odd squarefree, 2-free, Lambda = 0): present in both spectra at
    comparable strength -- dressing acts only through the factor 2."""
    g, a1, a1p, scale = _cache()
    w0 = math.log(35)
    mu_m = md.band_max(g, a1p, w0)[0]
    md_m = md.band_max(g, a1, w0)[0]
    assert mu_m > 0.75 * md.mobius_ref(35, scale)
    assert md_m / mu_m > 0.85
