"""Unit tests for warping the eta integral (warp_eta.py).

The eta-side supplement to the nonlinear warp: warp the WHOLE continuous-eta integrand
cos(pi x) x^{-s} (not just the bland power x^{-s}), and find that the headline midpoint phase
alpha = 1/2 is annihilated -- cos(pi(m+1/2)) = 0 -- while alpha = 1 survives and reproduces the
eta split. The checks:

* alpha = 1 completed -> eta(s) = (1 - 2^{1-s}) zeta(s), at O(1/K);
* alpha = 1/2 honest -> 0 (the midpoints are cos's zeros), also O(1/K);
* the load-bearing +1 (the omitted n=1 cell) is exactly warp_eta_K + warp_eta_alpha(.,.,1) = 1;
* the K -> infinity target is mp.altzeta, with the sigma=1 prefactor comb at t = 2 pi k/ln2;
* the fast evaluator matches the transparent Lerch form and the raw oscillatory integral;
* migration: the genuine zeta zeros climb onto sigma = 1/2 and the prefactor comb onto
  sigma = 1 -- the same split the additive comb (harmonic_bridge.eta_K) produces, by the warp.

The Lerch cross-check, the raw integral, and the (continuation-heavy) migration are `slow`;
the fast suite keeps the cheap building blocks -- the limit, the degeneracy, the load-bearing
identity, and the target.
"""
from __future__ import annotations

import mpmath as mp
import pytest

import harmonic_bridge as hb
import warp_eta as we

# short continuation schedule for the migration tests (warp_eta_alpha is the fast evaluator).
SHORT = [1, 2, 4, 8, 16, 32]


# --------------------------------------------------------------------------
# fast building blocks
# --------------------------------------------------------------------------
def test_alpha1_completed_limit_is_eta():
    """warp_eta_K = -warp_eta_alpha(.,.,1)+1 -> eta(s), tightening with K."""
    for s in [mp.mpc(2, 0), mp.mpc(0.7, 9), mp.mpc(0.5, 14.134725), mp.mpc(-0.5, 6)]:
        eta = mp.altzeta(s)
        e8 = abs(we.warp_eta_K(s, 8) - eta)
        e32 = abs(we.warp_eta_K(s, 32) - eta)
        assert e32 < e8                      # tightening
        assert e32 < mp.mpf("6e-2")          # O(1/K) close by K=32


def test_alpha_half_is_degenerate():
    """alpha = 1/2 honest warp -> 0: the midpoints m+1/2 are the zeros of cos(pi x)."""
    for s in [mp.mpc(2, 0), mp.mpc(0.7, 9), mp.mpc(1.3, 7)]:
        e8 = abs(we.warp_eta_alpha(s, 8, 0.5))
        e32 = abs(we.warp_eta_alpha(s, 32, 0.5))
        assert e32 < e8                      # collapsing toward 0
        assert e32 < mp.mpf("4e-2")          # already small (O(1/K) Gibbs residual)


def test_plus_one_is_load_bearing():
    """The +1 = -(omitted n=1 cell): warp_eta_K + warp_eta_alpha(.,.,1) == 1 identically."""
    for s in [mp.mpc(0.5, 14), mp.mpc(1.3, 7), mp.mpc(0, 9.0647)]:
        for K in (3, 9):
            assert abs((we.warp_eta_K(s, K) + we.warp_eta_alpha(s, K, 1.0)) - 1) < mp.mpf("1e-25")


def test_target_is_alt_zeta():
    """target_eta(s) == eta(s) == (1 - 2^{1-s}) zeta(s) -- the K -> infinity migration object."""
    for s in [mp.mpc(2, 1), mp.mpc(0.5, 14.134725), mp.mpc(1.0, 9.0647), mp.mpc(1.3, 7)]:
        assert abs(we.target_eta(s) - mp.altzeta(s)) < mp.mpf("1e-25")
        assert abs(we.target_eta(s) - (1 - mp.power(2, 1 - s)) * mp.zeta(s)) < mp.mpf("1e-22")


def test_comb_heights():
    """comb_heights lists the sigma=1 prefactor zeros t = 2 pi k/ln2 (= warp_bridge.pref_heights)."""
    import warp_bridge as wb
    hs = we.comb_heights(30.0)
    assert len(hs) == 3                                      # ~9.06, 18.13, 27.19
    for k, h in enumerate(hs, start=1):
        assert abs(h - 2 * float(mp.pi) * k / float(mp.log(2))) < 1e-9
    assert hs == wb.pref_heights(30.0)                       # same lattice, the FE mirror height


# --------------------------------------------------------------------------
# slow: evaluator cross-checks and the migration map
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_fast_matches_lerch():
    """The fast grid+moment warp_eta_alpha matches the transparent Lerch form, across sigma
    (incl. sigma <= 0; sigma exactly = 1 is skipped, where the alternating Lerch crawls)."""
    for s in [mp.mpc(2, 1), mp.mpc(0.5, 14.134725), mp.mpc(-0.5, 7), mp.mpc(1.3, 7)]:
        for K in (8, 25):
            assert abs(we.warp_eta_alpha(s, K, 1.0) - we.warp_eta_lerch(s, K, 1.0)) < mp.mpf("1e-10")
    # alpha = 1/2 too (the degenerate phase still matches its reference, both small)
    s = mp.mpc(1.3, 7)
    assert abs(we.warp_eta_alpha(s, 12, 0.5) - we.warp_eta_lerch(s, 12, 0.5)) < mp.mpf("1e-10")


@pytest.mark.slow
def test_fast_matches_raw_integral():
    """warp_eta_alpha == the literal oscillatory int_1^inf (sigma > 1) -- the independent guise."""
    s = mp.mpc(2.5, 1)
    assert abs(we.warp_eta_alpha(s, 8, 1.0) - we.warp_eta_raw(s, 8, 1.0)) < mp.mpf("1e-3")


@pytest.mark.slow
def test_zeta_zeros_climb_to_half():
    """zeta zero 1/2+14.13i: warp_eta_K migrates it onto sigma = 1/2 (the genuine zeta zero)."""
    t = dict((K, zz) for K, zz in hb.migrate(we.warp_eta_K, mp.mpc(0.5, 14.134725),
                                             schedule=SHORT) if zz is not None)
    assert 32 in t
    assert abs(float(t[32].real) - 0.5) < 0.06               # climbed onto the critical line


@pytest.mark.slow
def test_prefactor_comb_climbs_to_one():
    """The prefactor zero at t = 2 pi/ln2 (~9.06) migrates onto sigma = 1 (the eta comb)."""
    pref_t = 2 * float(mp.pi) / float(mp.log(2))             # ~ 9.0647
    t = dict((K, zz) for K, zz in hb.migrate(we.warp_eta_K, mp.mpc(1.0, pref_t),
                                             schedule=SHORT) if zz is not None)
    assert 32 in t
    assert abs(float(t[32].real) - 1.0) < 0.06               # -> sigma = 1
    assert abs(float(t[32].imag) - pref_t) < 0.2             # stays at the comb height
