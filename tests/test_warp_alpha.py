"""Unit tests for the general-phase n+alpha warp -> Hurwitz zeta(s, alpha).

The general-phase follow-up: pin the n*-warp to a general phase n+alpha (not just the
midpoint n+1/2) and demonstrate the Saias-Weingartner P*L dichotomy -- only the
privileged phases alpha in {0, 1/2, 1} give zeros on clean vertical lines; every generic
alpha is Davenport-Heilbronn-scattered. The checks:

* the phase is a DC shift -- phi_K^(alpha) = (alpha - 1/2) + phi_K;
* alpha = 1/2 reproduces the midpoint warp exactly (warp_alpha == warp_bridge.warp_K, completion ==
  warp_half_K);
* the completion is the general-alpha load-bearing half-cell alpha^{-s} (-> 2^s at 1/2);
* the K=0 endpoint is phase-deformed (alpha + 1/2)^{1-s}/(s-1) -- bland only at alpha=1/2;
* the limits: honest warp -> zeta(s, 1+alpha), completed -> zeta(s, alpha);
* THE PAYOFF: zeros of zeta(s, alpha) lie on {sigma=0, sigma=1/2} for alpha in {1/2, 1}
  (clean) but scatter off-line for generic alpha (e.g. 0.3); and the warp migration lands
  on each -- the clean line for alpha=1 (a genuine zeta zero), the scattered locus for 0.3.

The full-K limits, the quadrature cross-check, and the (zero-finding / continuation) payoff
tests are marked `slow` (deselected by default); the fast suite keeps the cheap algebraic
building blocks -- the DC shift, the midpoint-warp reduction, the completion, the endpoint.
"""
from __future__ import annotations

import mpmath as mp
import pytest

import harmonic_bridge as hb
import warp_alpha as wa
import warp_bridge as wb

# coarse box + short K schedule for the (slow) zero-finding / migration payoff tests
BOX_SIG = (-0.25, 1.7)
BOX_T = (2.0, 22.0)
SHORT = [1, 2, 4, 8, 16, 32]


# --------------------------------------------------------------------------
# fast algebraic building blocks
# --------------------------------------------------------------------------
def test_warp_phi_alpha_is_dc_shift():
    """phi_K^(alpha)(u) = (alpha - 1/2) + phi_K(u): a pure DC shift of the midpoint-warp sawtooth."""
    for u in (0.17, 0.5, 0.83):
        for K in (0, 3, 8):
            for a in (0.3, 0.5, 0.9):
                lhs = wa.warp_phi_alpha(mp.mpf(u), K, a)
                rhs = (mp.mpf(a) - mp.mpf("0.5")) + wb.warp_phi(mp.mpf(u), K)
                assert abs(lhs - rhs) < mp.mpf("1e-25")


def test_alpha_half_reduces_to_phase2():
    """alpha = 1/2 (DC shift 0) reproduces warp_bridge: warp_alpha == warp_K, completion
    == warp_half_K -- the general module contains the midpoint warp as a special case."""
    for s in [mp.mpc(2, 1), mp.mpc(0.5, 14.134725), mp.mpc(1.3, 7)]:
        for K in (3, 12):
            assert abs(wa.warp_alpha(s, K, 0.5) - wb.warp_K(s, K)) < mp.mpf("1e-20")
            assert abs(wa.warp_complete_alpha(s, K, 0.5) - wb.warp_half_K(s, K)) < mp.mpf("1e-20")


def test_completion_is_the_alpha_half_cell():
    """warp_complete_alpha - warp_alpha == alpha^{-s} (the restored m=0 cell at alpha);
    at alpha = 1/2 this half-cell is (1/2)^{-s} = 2^s, the midpoint warp's load-bearing term."""
    for a in (0.3, 0.5, 0.7, 1.0):
        for s in [mp.mpc(0.6, 12), mp.mpc(1.3, 7)]:
            assert abs((wa.warp_complete_alpha(s, 5, a) - wa.warp_alpha(s, 5, a))
                       - mp.power(mp.mpf(a), -s)) < mp.mpf("1e-25")
    s = mp.mpc(0.5, 14)
    assert abs(mp.power(mp.mpf("0.5"), -s) - mp.power(2, s)) < mp.mpf("1e-25")


def test_endpoint_is_phase_deformed():
    """warp_alpha(., 0, alpha) = (alpha + 1/2)^{1-s}/(s-1): the phase offset deforms even the
    K=0 endpoint -- the bland 1/(s-1) survives ONLY at alpha = 1/2."""
    s = mp.mpc(1.3, 7)
    for a in (0.3, 0.5, 0.8):
        endpoint = mp.power(mp.mpf(a) + mp.mpf("0.5"), 1 - s) / (s - 1)
        assert abs(wa.warp_alpha(s, 0, a) - endpoint) < mp.mpf("1e-25")
    assert abs(wa.warp_alpha(s, 0, 0.5) - 1 / (s - 1)) < mp.mpf("1e-25")          # bland at 1/2
    assert abs(wa.warp_alpha(s, 0, 0.3) - 1 / (s - 1)) > mp.mpf("1e-2")           # deformed else


# --------------------------------------------------------------------------
# slow: the K -> infinity limits and the quadrature cross-check
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_honest_limit_is_zeta_one_plus_alpha():
    """The honest warp_alpha -> zeta(s, 1 + alpha) (the cells 1+alpha, 2+alpha, ...)."""
    for a in (0.3, 0.5, 0.7, 1.0):
        for s in [mp.mpc(2, 0), mp.mpc(0.6, 12.0)]:
            z = mp.zeta(s, 1 + mp.mpf(a))
            e8 = abs(wa.warp_alpha(s, 8, a) - z)
            e32 = abs(wa.warp_alpha(s, 32, a) - z)
            assert e32 < e8
            assert e32 < mp.mpf("8e-2")


@pytest.mark.slow
def test_completed_limit_is_zeta_alpha():
    """warp_complete_alpha -> zeta(s, alpha); at alpha=1/2 the target is (2^s-1)zeta(s),
    at alpha=1 it is plain zeta(s)."""
    for a in (0.3, 0.5, 0.7, 1.0):
        for s in [mp.mpc(2, 0), mp.mpc(0.6, 12.0)]:
            z = mp.zeta(s, mp.mpf(a))
            assert abs(wa.warp_complete_alpha(s, 32, a) - z) < mp.mpf("8e-2")
    s = mp.mpc(0.7, 13)
    assert abs(mp.zeta(s, mp.mpf("0.5")) - (mp.power(2, s) - 1) * mp.zeta(s)) < mp.mpf("1e-25")
    assert abs(mp.zeta(s, mp.mpf("1.0")) - mp.zeta(s)) < mp.mpf("1e-25")


@pytest.mark.slow
def test_fast_warp_matches_hurwitz():
    """The fast grid warp_alpha agrees with the transparent warp_alpha_hurwitz, across alpha,
    sigma (incl. sigma <= 0) and K."""
    for a in (0.3, 0.7):
        for s in [mp.mpc(2, 1), mp.mpc(0.5, 14.134725), mp.mpc(0, 9.0647)]:
            for K in (8, 32):
                assert abs(wa.warp_alpha(s, K, a) - wa.warp_alpha_hurwitz(s, K, a)) < mp.mpf("1e-6")


@pytest.mark.slow
def test_warp_alpha_stable_at_high_tau():
    """Regression: warp_alpha shares warp_bridge's grid+moment evaluator and inherits the
    same high-|Im s| fix (|Im s|-scaled working precision / moment count / GL degree). Above the
    old ~110 breakdown it stays bounded and matches the transparent warp_alpha_hurwitz (raised
    precision), for the clean (1/2) and a generic (0.3) phase alike."""
    for a in (0.5, 0.3):
        s = mp.mpc(0.5, 150.0)
        with mp.workdps(45):                                  # accurate reference, raised precision
            ref = wa.warp_alpha_hurwitz(s, 30, a)
        got = wa.warp_alpha(s, 30, a)                         # evaluated at the ambient dps = 30
        assert abs(got) < 50                                  # bounded -- not the old blow-up
        assert abs(got - ref) < mp.mpf("1e-7")                # and accurate to its native ~1e-9


# --------------------------------------------------------------------------
# slow: THE PAYOFF -- the clean (alpha in {1/2, 1}) vs scattered (generic) dichotomy
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_clean_alphas_land_on_vertical_lines():
    """alpha in {1/2, 1}: every zero of zeta(s, alpha) sits on a clean line ({0, 1/2})."""
    for a in (0.5, 1.0):
        zs = wa.find_zeros(lambda s: wa.target(s, a), BOX_SIG, BOX_T)
        dev, n_off, _n_gt1 = wa.clean_score(zs)
        assert len(zs) >= 1
        assert dev < 0.02 and n_off == 0          # all on sigma in {0, 1/2}


@pytest.mark.slow
def test_alpha_half_carries_the_sigma_zero_companion():
    """alpha=1/2's zeros include the sigma=0 companion (2^s-1=0 at t=2pi/ln2) that the
    plain-zeta alpha=1 lacks -- the (2^s-1)zeta factorization made visible."""
    zs = wa.find_zeros(lambda s: wa.target(s, 0.5), BOX_SIG, BOX_T)
    on_zero = [z for z in zs if abs(float(z.real)) < 0.02]
    pref_t = 2 * float(mp.pi) / float(mp.log(2))      # ~ 9.0647
    assert any(abs(float(z.imag) - pref_t) < 0.1 for z in on_zero)
    # alpha = 1 (plain zeta) has NO sigma=0 companion in the same box
    zs1 = wa.find_zeros(lambda s: wa.target(s, 1.0), BOX_SIG, BOX_T)
    assert all(abs(float(z.real)) > 0.1 for z in zs1)


@pytest.mark.slow
def test_generic_alpha_is_scattered():
    """A generic phase (alpha = 0.3) is Davenport-Heilbronn: zeros off the clean lines,
    including a known one near sigma ~ 0.28 (t ~ 14.49)."""
    zs = wa.find_zeros(lambda s: wa.target(s, 0.3), BOX_SIG, BOX_T)
    dev, n_off, _n_gt1 = wa.clean_score(zs)
    assert dev > 0.1 and n_off >= 2                       # genuinely off {0, 1/2}
    assert any(abs(float(z.real) - 0.28) < 0.05 and abs(float(z.imag) - 14.49) < 0.2
               for z in zs)


@pytest.mark.slow
def test_warp_migration_lands_on_clean_and_scattered():
    """Reuse harmonic_bridge.migrate on the general-alpha warp: alpha=1 lands a genuine zeta
    zero on sigma=1/2; alpha=0.3 lands on its scattered off-line zero (sigma ~ 0.28)."""
    # alpha = 1: warp_complete_alpha -> zeta(s); the Riemann zero migrates onto sigma = 1/2
    t1 = dict((K, zz) for K, zz in hb.migrate(
        lambda s, K: wa.warp_complete_alpha(s, K, 1.0), mp.mpc(0.5, 14.134725),
        schedule=SHORT) if zz is not None)
    assert 32 in t1 and abs(float(t1[32].real) - 0.5) < 0.05

    # alpha = 0.3: the warp tracks the scattered Davenport-Heilbronn zero, NOT sigma = 1/2
    z_off = mp.mpc("0.2799118238867328", "14.4877598801336641")
    t2 = dict((K, zz) for K, zz in hb.migrate(
        lambda s, K: wa.warp_complete_alpha(s, K, 0.3), z_off, schedule=SHORT) if zz is not None)
    assert 32 in t2
    assert abs(float(t2[32].real) - 0.28) < 0.05         # lands off the critical line
    assert abs(float(t2[32].real) - 0.5) > 0.15
