"""Unit tests for the side-by-side comb-vs-warp integrand illustration (issue #11).

`comb_vs_warp.py` draws the two restoration routes at the level of the integrand:
the additive comb b_K(x) = x^{-s} + s phi_K(x) x^{-s-1} (kernel added, x linear)
and the variable warp w_K(x) = (x + phi_K(x))^{-s} (kernel composed into the
argument). These checks pin the drawn integrands to the real bridge objects:

* its vectorized `phi_K` matches the canonical `warp_bridge.warp_phi`;
* at K = 0 both integrands are the bland x^{-s} (the shared start);
* the additive integrand integrates (+ the 1/2 endpoint term) to `harmonic_bridge.zeta_K`;
* the warp integrand integrates to `warp_bridge.warp_K`;
* the warp integrand collapses onto the cell-midpoint value as K grows.
"""
from __future__ import annotations

import numpy as np
import pytest
from mpmath import mp

import comb_vs_warp as cw
import harmonic_bridge as hb
import warp_bridge as wb

mp.dps = 30
S = cw.S


def test_phi_matches_canonical():
    """The numpy phi_K equals warp_bridge.warp_phi to full float precision."""
    for xv in (0.3, 1.7, 2.5, 3.14159, 4.62):
        for K in (1, 4, 16):
            mine = float(cw.phi_K(np.array([xv]), K)[0])
            ref = float(wb.warp_phi(xv, K))
            assert abs(mine - ref) < 1e-9, (xv, K, mine, ref)


def test_K0_integrands_are_bland():
    """At K = 0 (phi_0 = 0) both routes' integrands are the bland x^{-s}."""
    x = np.array([1.2, 2.7, 4.1])
    assert np.allclose(cw.comb_integrand(x, 0), x ** (-S), atol=1e-12)
    assert np.allclose(cw.warp_integrand(x, 0), x ** (-S), atol=1e-12)


@pytest.mark.slow
def test_additive_area_is_zeta_K():
    """int_1^inf [x^{-s} + s phi_K x^{-s-1}] dx + 1/2 == zeta_K(s, K).

    The smooth x^{-s} integrates to 1/(s-1); the oscillatory ripple is integrated by an
    INDEPENDENT quadrature (quadosc), not zeta_K's incomplete-gamma closed form."""
    s = mp.mpf(S)
    for K in (1, 2, 4):
        ripple = mp.quadosc(lambda x: s * wb.warp_phi(x, K) * x ** (-s - 1),
                            [1, mp.inf], period=1)
        area = 1 / (s - 1) + mp.mpf("0.5") + ripple
        assert abs(area - hb.zeta_K(s, K)) < mp.mpf("1e-6"), (K, area)


@pytest.mark.slow
def test_warp_area_is_warp_K():
    """int_1^inf (x + phi_K)^{-s} dx == warp_K(s, K). warp_raw is exactly the plotted w_K's
    area by oscillatory quadrature; warp_K is the fast grid evaluator."""
    s = mp.mpf(S)
    for K in (1, 2, 4):
        assert abs(wb.warp_raw(s, K) - wb.warp_K(s, K)) < mp.mpf("1e-3"), K


def test_warp_collapses_to_midpoint():
    """In a cell interior the warp integrand converges to (m+1/2)^{-s} as K grows."""
    xi = np.array([2.30])                  # interior of the [2, 3] cell, clear of the edges
    mid = (2.5) ** (-S)
    devs = [abs(float(cw.warp_integrand(xi, K)[0]) - mid) for K in (4, 16, 64)]
    assert devs[0] > devs[1] > devs[2], devs


def test_limit_integrands_match_large_K():
    """The K -> infinity integrand formulas are the large-K limits of the finite-K ones
    (checked in a cell interior, away from the Gibbs edges where convergence is pointwise)."""
    x = np.linspace(2.2, 2.8, 50)          # interior of the [2, 3] cell
    assert np.max(np.abs(cw.comb_integrand(x, 256) - cw.comb_integrand_limit(x))) < 1e-2
    assert np.max(np.abs(cw.warp_integrand(x, 256) - cw.warp_integrand_limit(x))) < 1e-2
