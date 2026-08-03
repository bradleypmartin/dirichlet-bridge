"""Tests for conductor_clock (issue #48). OUTSIDE the default CI suite:
pytest.ini's `testpaths = tests` excludes explorations/; run these with
`pytest explorations/tests`. Lean by design -- the heavy verification (the
per-conductor K-walks, winding certification) lives in the driver, feeding
the committed data/*.csv caches; here we pin the exact model algebra and
regression-test the committed caches.
"""
import math

import mpmath as mp
import numpy as np
import pytest

import arithmetic_clock as ac
import character_bridge as cb
import conductor_clock as cc
import mobius_dressing as md
import zero_birth as zb


# --------------------------------------------------------------------------
# the experiment design
# --------------------------------------------------------------------------
def test_scaled_schedule_preserves_kappa_coverage():
    """Layer A x q/3 keeps the clock reading of every rung within rounding:
    kappa_q(K_q, t) = kappa_3(K_3, t) up to the integer round."""
    for q in (4, 7):
        for K3, Kq in zip(ac.K_SCHEDULE, cc._scaled_schedule(q)):
            assert abs(Kq - K3 * q / 3.0) <= 0.5
            t = 100.0
            assert abs(ac.kappa_of(Kq, t, q) - ac.kappa_of(K3, t, 3)) < 0.03


def test_walk_schedule_layers():
    """Every zero gets Layer A; stride zeros add the kappa ladder."""
    q = 4
    Ks_plain = cc.walk_schedule_q(1, 300.0, q)
    assert Ks_plain == sorted(set(cc.CONDUCTORS[q]["layerA"]))
    Ks_kappa = cc.walk_schedule_q(0, 300.0, q)
    top = ac.K_of_kappa(cc.KAPPA_SCHEDULE[-1], 300.0, q)
    assert top in Ks_kappa
    assert set(Ks_plain) <= set(Ks_kappa)


def test_real_primitive_picks():
    """chi_4 is the odd quadratic mod 4 (chi(2) = 0); chi_7 the Legendre
    symbol mod 7 (odd, chi(2) = +1: 2 = 3^2 mod 7)."""
    chi4, chi7 = cc.chi_q(4), cc.chi_q(7)
    assert chi4.q == 4 and chi4.parity == 1 and float(chi4(2)) == 0.0
    assert float(chi4(3)) == -1.0
    assert chi7.q == 7 and chi7.parity == 1
    assert float(chi7(2)) == 1.0 and float(chi7(3)) == -1.0


# --------------------------------------------------------------------------
# the chi-general dressing model (exact algebra, no data)
# --------------------------------------------------------------------------
def test_general_B_factor_matches_section():
    """The vectorized section factor equals zero_birth's polynomial at
    rho + 1 for both new conductors."""
    for q in (4, 7):
        chi = cc.chi_q(q)
        for g in (8.0, 141.5):
            closed = complex(md.B_factor(np.array([g]), chi)[0])
            ref = complex(zb.section_B(mp.mpc(1.5, g), chi))
            assert abs(closed - ref) < 1e-12


def test_q4_dressing_laws():
    """The #48 closed forms at q = 4: 3-divisible squarefree lines at 2/3,
    the ln9 vacancy satellite at 2/3 of Landau, ln27 dark, odd 3-free
    lines untouched, the whole 2-divisible column zero."""
    chi4 = cc.chi_q(4)
    for n in (3, 15, 21, 33):
        assert abs(md.dressed_ref(n, 1.0, chi4) / md.mobius_ref(n, 1.0, chi4)
                   - 2.0 / 3.0) < 1e-12
    assert abs(md.dressed_ref(9, 1.0, chi4) / md.landau_ref(9, 1.0)
               - 2.0 / 3.0) < 1e-12
    assert md.dressed_ref(27, 1.0, chi4) == 0.0
    for n in (5, 7, 35):
        assert md.dressed_ref(n, 1.0, chi4) == md.mobius_ref(n, 1.0, chi4)
    for n in (2, 4, 8, 10, 14, 22):
        assert md.mobius_ref(n, 1.0, chi4) == 0.0
        assert md.dressed_ref(n, 1.0, chi4) == 0.0


def test_q7_dressing_laws():
    """The full-comb rational suppressions at q = 7 and the vacancy
    satellites (1/2, 3/4, 2/3 of Landau at ln4, ln8, ln9 -- all
    non-Landau, unlike q = 3's masquerading ln4)."""
    chi7 = cc.chi_q(7)
    for n, target in ((2, 0.5), (3, 2.0 / 3.0), (5, 0.8), (6, 1.0 / 3.0),
                      (10, 0.3), (15, 7.0 / 15.0)):
        assert abs(md.dressed_ref(n, 1.0, chi7) / md.mobius_ref(n, 1.0, chi7)
                   - target) < 1e-12
    for n, target in ((4, 0.5), (8, 0.75), (9, 2.0 / 3.0)):
        assert abs(md.dressed_ref(n, 1.0, chi7) / md.landau_ref(n, 1.0)
                   - target) < 1e-12
    for n in (7, 14, 21, 35, 49):
        assert md.dressed_ref(n, 1.0, chi7) == 0.0


def test_dressing_suppression_is_chi_independent():
    """The adjudicated correction to #47's sign-flip prediction: the ln2
    suppression is 1/2 at BOTH chi_3(2) = -1 and chi_7(2) = +1 -- the
    character cancels between the section coefficient chi(a) and the
    direct line's chi(n) = chi(a) chi(n/a)."""
    r3 = md.dressed_ref(2, 1.0, cb.CHI3) / md.mobius_ref(2, 1.0, cb.CHI3)
    r7 = md.dressed_ref(2, 1.0, cc.chi_q(7)) / md.mobius_ref(2, 1.0, cc.chi_q(7))
    assert abs(r3 - 0.5) < 1e-12
    assert abs(r7 - 0.5) < 1e-12          # NOT the sign-flip 3/2


def test_q3_dressed_ref_backward_compatible():
    """The general divisor convolution reproduces #47's single-sideband
    closed form at q = 3 on every band the old model covered."""
    for n in (2, 4, 8, 16, 5, 7, 10, 14, 22, 26, 34, 35, 3, 9, 15):
        general = md.dressed_ref(n, 1.7)
        if n % 2:
            old = md.mobius_ref(n, 1.7)
        else:
            m = n // 2
            old = (1.7 * math.log(n) * md.BETA
                   * abs(md.mu_of(m)) * md.chi_weight(m) / math.sqrt(m))
        assert abs(general - old) < 1e-12


# --------------------------------------------------------------------------
# cache regressions (the committed censuses and tables)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("q", [4, 7])
def test_zero_census_cached_and_complete(q):
    """The committed census matches theta_chi/pi within S(T) noise."""
    if not cc.zeros_csv(q).exists():
        pytest.skip("census cache not built yet")
    z = cc.load_zeros(q)
    cfg = cc.CONDUCTORS[q]
    resid = len(z) - float(zb.theta_chi(cfg["t_census"], cc.chi_q(q)) / mp.pi)
    assert abs(resid) < 1.0
    assert z[0] > 0 and z[-1] <= cfg["t_census"]


@pytest.mark.parametrize("q", [4, 7])
def test_a1_cache_matches_census(q):
    """One a_1 row per census zero; spot-check one value fresh."""
    if not cc.a1_csv(q).exists():
        pytest.skip("a1 cache not built yet")
    z = cc.load_zeros(q)
    a1s = cc.a1_table_q(q, z, verbose=False)
    assert len(a1s) == len(z)
    g0, a1_cached = a1s[0]
    fresh = ac.a1_of(g0, chi=cc.chi_q(q))
    assert abs(fresh - a1_cached) < 1e-4 * max(1.0, abs(fresh))


@pytest.mark.parametrize("q", [4, 7])
def test_kwalk_cache_collapse(q):
    """The committed K-walk collapses: pre-knee saturation, asymptotic
    decay past kappa = 3 (the prediction-1 regression)."""
    if not cc.kwalk_csv(q).exists():
        pytest.skip("kwalk cache not built yet")
    pts, (c1, c2) = cc.collapse_for(q)
    cent, med = ac.collapse_binned(pts)
    post = [m for c, m in zip(cent, med) if c > 3.0]
    assert post and max(post) < 0.2
    assert 0.0 < c1 < 0.5
