"""Unit tests for the stability companion (stability.py).

The point under test: the K-truncated interpolant zeta^{(K)} is meromorphic on the whole
plane (each harmonic is an incomplete-gamma closed form, entire in s), so it has NO abscissa of
convergence -- unlike the zeta series (sigma>1) or the eta series (sigma>0). The checks:

* no abscissa: zeta^{(K)} -> zeta at sigma = -2 (where BOTH Dirichlet series diverge), at the
  same O(1/K) rate as in the trivial region;
* Re(s) is essentially free: the convergence error is ~independent of sigma at fixed height;
* height is the cost: the fixed-K error tracks the analytic rate constant |s|/(2 pi^2 K), so it
  grows with t = Im(s) -- the convergence-side face of the birth-K law.

All cheap (closed-form zeta_K, no zero-finding); the deepest-K check is `slow`.
"""
from __future__ import annotations

import mpmath as mp
import pytest

import stability as st


def test_converges_left_of_every_abscissa():
    """zeta^{(K)} -> zeta at sigma = -2 (both series diverge there), O(1/K)."""
    e64 = st.conv_error(-2.0, 15.0, 64)
    e256 = st.conv_error(-2.0, 15.0, 256)
    assert e256 < e64                                  # still converging far left
    assert 3.0 < e64 / e256 < 5.0                      # 4x harmonics -> ~4x smaller (O(1/K))


def test_convergence_is_sigma_independent():
    """At fixed height, the error barely depends on Re(s): sigma=-2 vs sigma=2 agree."""
    e_left = st.conv_error(-2.0, 40.0, 80)
    e_right = st.conv_error(2.0, 40.0, 80)
    assert abs(e_left - e_right) / e_right < 0.1       # ~the same |s|/(2 pi^2 K)


def test_height_is_the_cost():
    """The fixed-K error tracks the analytic |s|/(2 pi^2 K) rate constant (grows with t)."""
    K = 40
    for sigma, t in [(2.0, 80.0), (0.5, 50.0)]:
        law = float(abs(st.rate_constant(mp.mpc(sigma, t))) / K)
        assert abs(st.conv_error(sigma, t, K) - law) / law < 0.25


def test_rate_constant_is_s_over_two_pi_squared():
    """rate_constant(s) == s/(2 pi^2) -- the comb leading constant (rate_law.rate_comb)."""
    for s in [mp.mpc(0.5, 14), mp.mpc(2, 0), mp.mpc(-1, 7)]:
        assert abs(st.rate_constant(s) - mp.mpc(s) / (2 * mp.pi**2)) < mp.mpf("1e-25")


@pytest.mark.slow
def test_deep_K_stays_on_the_one_over_K_law():
    """No floor / no creep at dps=30, moderate t: error keeps halving as K doubles to K=512."""
    s = (-0.5, 20.0)
    errs = [st.conv_error(s[0], s[1], K) for K in (64, 128, 256, 512)]
    for a, b in zip(errs, errs[1:]):
        assert 1.8 < a / b < 2.2                       # clean 1/K, no precision floor reached
