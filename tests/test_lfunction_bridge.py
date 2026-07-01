"""Unit tests for the L(s, chi_4) bridge (lfunction_bridge.py).

The headline fork-2 experiment: apply the bridge's K-knob and warp to a genuine primitive
Dirichlet L-function, L(s, chi_4) = sum chi_4(n) n^{-s}, chi_4 the odd character mod 4. The
checks:

* the evaluator: L via the Hurwitz combination 4^{-s}[zeta(s,1/4)-zeta(s,3/4)] == the direct
  sum (sigma>1); anchors L(1)=pi/4 (Leibniz) and the odd-character functional equation
  Lambda(s)=Lambda(1-s) (root number W=1);
* the continuous endpoint G(s)=int sin(pi x/2)x^{-s}dx = Smom(pi/2,s) is STRUCTURED (an
  off-critical eta-type string, sigma->3/2), NOT zeroless -- the design-crux answer;
* the additive comb L_comb_K has K=0 anchor G+1/2 and converges to L at O(1/K); its zeros
  migrate from the ground string onto the SINGLE line sigma=1/2 (no companion);
* the warp of the phase-1/2 carrier sin(pi y) lands on 2^s L(s,chi_4) = L(1/2,1/2,s) (the
  Lerch tie-in), with the load-bearing +2^s the m=0 midpoint cell; alpha=1 dies (the dual of
  warp_eta's alpha=1/2 death); the fast evaluator matches the transparent Lerch/raw forms;
* the modulus surprise: warping the GENUINE chi_4 carrier sin(pi y/2) at the midpoints lands
  on a mod-8 character, 2^s(sqrt2/2) L(s, chi_{-8}).

The Lerch/raw cross-checks and the (continuation-heavy) migrations are `slow`; the fast suite
keeps the cheap building blocks -- the evaluator, anchors, endpoint, limits, and identities.
"""
from __future__ import annotations

import mpmath as mp
import pytest

import lfunction_bridge as lb

PI = mp.pi
# short continuation schedule for the migration tests (both evaluators are fast).
SHORT = [1, 2, 4, 8, 16, 32]


# --------------------------------------------------------------------------
# fast: the evaluator and its anchors
# --------------------------------------------------------------------------
def test_chi4_character():
    """chi_4(n) = sin(pi n/2): the odd character mod 4 (period 4, mean zero)."""
    assert [lb.chi4(n) for n in range(1, 9)] == [1, 0, -1, 0, 1, 0, -1, 0]


def test_L_hurwitz_matches_direct_sum():
    """L via 4^{-s}[zeta(s,1/4)-zeta(s,3/4)] == the direct sum chi_4(n) n^{-s} for sigma>1."""
    for s in [mp.mpc(3, 1), mp.mpc(2.5, 4), mp.mpc(4, 0)]:
        assert abs(lb.L_chi4(s) - lb.L_chi4_sum(s)) < mp.mpf("1e-9")


def test_anchor_L1_is_pi_over_4():
    """L(1, chi_4) = pi/4 (Leibniz) -- the removable value where the Hurwitz poles cancel."""
    assert abs(lb.L_chi4(mp.mpc(1, 0)) - PI / 4) < mp.mpf("1e-25")


def test_functional_equation():
    """Lambda(s) = Lambda(1-s): the odd-character functional equation, root number W = 1."""
    for s in [mp.mpc(2, 1), mp.mpc(0.5, 6), mp.mpc(-1, 3), mp.mpc(0.5, 18.29)]:
        assert lb.functional_eq_residual(s) < mp.mpf("1e-25")


# --------------------------------------------------------------------------
# fast: the continuous endpoint G (the design-crux answer -- structured, not zeroless)
# --------------------------------------------------------------------------
def test_endpoint_G_matches_quadosc():
    """G(s) = Smom(pi/2, s) == the oscillatory quadrature int_1^inf sin(pi x/2) x^{-s} dx."""
    for s in [mp.mpc(2, 1), mp.mpc(0.8, 10), mp.mpc(1.5, 25)]:
        assert abs(lb.G(s) - lb.G_quad(s)) < mp.mpf("1e-20")


def test_comb_endpoint_is_G_plus_half():
    """L_comb_K(., 0) is exactly the K=0 ground anchor G(s) + 1/2 (endpoint + n=1 half-weight)."""
    for s in [mp.mpc(0.5, 6), mp.mpc(1.3, 7), mp.mpc(1.8, 12)]:
        assert abs(lb.L_comb_K(s, 0) - (lb.G(s) + mp.mpf("0.5"))) < mp.mpf("1e-25")


def test_sigma_law_tracks_G_string():
    """The endpoint is STRUCTURED (eta-type), not zeroless: G has off-critical zeros whose real
    parts follow sigma_law (-> 3/2), well off the critical line."""
    zeros = lb.find_G_zeros(20.0)
    assert len(zeros) >= 3
    for z in zeros:
        assert abs(lb.G(z)) < 1e-15                        # it really is a zero of G
        assert float(z.real) > 1.3                          # off-critical (heading to 3/2)
        assert abs(float(z.real) - lb.sigma_law(float(z.imag))) < 0.05   # on the sigma-law curve


# --------------------------------------------------------------------------
# fast: convergence of both routes, and the identities they land on
# --------------------------------------------------------------------------
def test_comb_converges_to_L():
    """L_comb_K -> L(s, chi_4) as K grows (tightening, O(1/K))."""
    for s in [mp.mpc(2, 0), mp.mpc(0.5, 6.020948), mp.mpc(1.3, 7)]:
        L = lb.L_chi4(s)
        e5 = abs(lb.L_comb_K(s, 5) - L)
        e45 = abs(lb.L_comb_K(s, 45) - L)
        assert e45 < e5
        assert e45 < mp.mpf("2e-2")


def test_comb_rate_constant_reuses_rate_law():
    """The O(1/K) comb constant is rate_law.rate_comb(s) = s/2 pi^2 -- the SAME as zeta/eta
    (sin(pi x/2) = +1 at the integer endpoint x=1, so |carrier(1)| = 1 as for zeta's 1)."""
    import rate_law as rl
    for s in [mp.mpc(2, 0), mp.mpc(0.5, 6.020948), mp.mpc(1.3, 7)]:
        L = lb.L_chi4(s)
        pred = abs(rl.rate_comb(s))                             # |s / 2 pi^2|
        k160 = abs(160 * (L - lb.L_comb_K(s, 160)))
        assert abs(k160 - pred) < mp.mpf("2e-2") * pred         # K(L - L^{(K)}) -> the constant


def test_warp_limit_is_2sL():
    """warp_chi4_K = warp_chi4_alpha(.,.,1/2) + 2^s -> 2^s L(s, chi_4), tightening with K."""
    for s in [mp.mpc(2, 0), mp.mpc(0.5, 6.020948), mp.mpc(1.3, 7), mp.mpc(-0.5, 6)]:
        tgt = lb.target_2sL(s)
        e5 = abs(lb.warp_chi4_K(s, 5) - tgt)
        e45 = abs(lb.warp_chi4_K(s, 45) - tgt)
        assert e45 < e5
        assert e45 < mp.mpf("3e-2")


def test_warp_alpha1_dies():
    """alpha = 1 honest warp -> 0: the integers are the zeros of the sin(pi y) carrier
    (the dual of warp_eta's alpha=1/2 death, where cos vanishes at the midpoints)."""
    for s in [mp.mpc(2, 0), mp.mpc(0.7, 9), mp.mpc(1.3, 7)]:
        e5 = abs(lb.warp_chi4_alpha(s, 5, 1.0))
        e45 = abs(lb.warp_chi4_alpha(s, 45, 1.0))
        assert e45 < e5
        assert e45 < mp.mpf("2e-2")


def test_load_bearing_2s():
    """The +2^s = (1/2)^{-s} is the omitted m=0 midpoint cell: warp_chi4_K - warp_chi4_alpha == 2^s."""
    for s in [mp.mpc(0.5, 6.020948), mp.mpc(1.3, 7), mp.mpc(0, 9)]:
        for K in (3, 12):
            assert abs(lb.warp_chi4_K(s, K) - lb.warp_chi4_alpha(s, K, 0.5) - mp.power(2, s)) < mp.mpf("1e-25")


def test_target_2sL_is_lerch():
    """2^s L(s, chi_4) == Phi(-1, s, 1/2) = L(1/2, 1/2, s) -- the Lerch tie-in."""
    for s in [mp.mpc(2, 1), mp.mpc(0.5, 6.020948), mp.mpc(-1, 3), mp.mpc(1.3, 7)]:
        assert abs(lb.target_2sL(s) - mp.lerchphi(-1, s, mp.mpf(1) / 2)) < mp.mpf("1e-22")


def test_mod8_modulus_surprise():
    """Warping the GENUINE chi_4 carrier sin(pi y/2) at the midpoints lands on a mod-8 character:
    sum sin(pi(m+1/2)/2)(m+1/2)^{-s} == 2^s (sqrt2/2) L(s, chi_{-8})."""
    for s in [mp.mpc(2, 0), mp.mpc(2.5, 1), mp.mpc(0.5, 5)]:
        lhs = lb.midpoint_chi4_carrier_limit(s)
        rhs = mp.power(2, s) * (mp.sqrt(2) / 2) * lb.L_chi_m8(s)
        assert abs(lhs - rhs) < mp.mpf("1e-20")


# --------------------------------------------------------------------------
# slow: evaluator cross-checks and the migration maps
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_warp_fast_matches_lerch():
    """The fast grid+moment warp_chi4_alpha matches the transparent Lerch form, across sigma
    (incl. sigma <= 0; sigma exactly = 1 is skipped, where the alternating Lerch crawls)."""
    for s in [mp.mpc(2, 1), mp.mpc(0.5, 6.020948), mp.mpc(-0.5, 7), mp.mpc(1.3, 7)]:
        for K in (8, 25):
            assert abs(lb.warp_chi4_alpha(s, K, 0.5) - lb.warp_chi4_lerch(s, K, 0.5)) < mp.mpf("1e-10")


@pytest.mark.slow
def test_warp_fast_matches_raw_integral():
    """warp_chi4_alpha == the literal oscillatory int_1^inf (sigma > 1) -- the independent guise."""
    s = mp.mpc(2.5, 1)
    assert abs(lb.warp_chi4_alpha(s, 8, 0.5) - lb.warp_chi4_raw(s, 8, 0.5)) < mp.mpf("1e-3")


@pytest.mark.slow
def test_comb_zeros_born_and_migrate_to_half():
    """comb: the L(s, chi_4) zero 1/2+6.02i is continued down to a K=0 ground zero (Re ~ 0.8-1.0,
    off the critical line) and climbs onto sigma=1/2 as K grows -- the single-line migration."""
    traj = lb.hb.migrate(lb.L_comb_K, mp.mpc(0.5, 6.020948), schedule=[0] + SHORT)
    d = {K: zz for K, zz in traj if zz is not None}
    assert 0 in d and 32 in d
    assert 0.5 < float(d[0].real) < 1.1                     # ground zero, off-line (eta-type)
    assert abs(float(d[32].real) - 0.5) < 0.06              # migrated onto sigma=1/2


@pytest.mark.slow
def test_warp_zeros_migrate_to_half():
    """warp: the 2^s L(s, chi_4) zero 1/2+10.24i climbs onto sigma=1/2 (single line, no companion)."""
    traj = lb.hb.migrate(lb.warp_chi4_K, mp.mpc(0.5, 10.243770), schedule=SHORT)
    d = {K: zz for K, zz in traj if zz is not None}
    assert 32 in d
    assert abs(float(d[32].real) - 0.5) < 0.06
    assert abs(float(d[32].imag) - 10.243770) < 0.2         # stays at its height
