"""Unit tests for the two-phase staircase comparison (alpha = 1/2 vs alpha = 1).

`warp_phase_compare.py` contrasts the midpoint warp (alpha = 1/2, pristine x ->
half-integers) with the integer warp (alpha = 1, x + 1/2 -> counting numbers).
These checks pin the illustration to the real general-phase warp object:

* its coordinate matches the canonical `warp_alpha.warp_phi_alpha`;
* the K=0 endpoints are exactly x (alpha=1/2) and x+1/2 (alpha=1) -- the trade-off;
* both phases converge to floor(x) + alpha away from the Gibbs edges.
"""
from __future__ import annotations

import numpy as np

import warp_alpha as wa
import warp_phase_compare as wpc


def test_matches_canonical_warp_phi_alpha():
    """The coordinate equals x + warp_alpha.warp_phi_alpha to full float precision."""
    for xv in (0.3, 1.7, 2.5, 3.62):
        for K in (1, 4, 16):
            for alpha in (0.5, 1.0):
                mine = float(wpc.warp_alpha_coord(np.array([xv]), K, alpha)[0])
                ref = xv + float(wa.warp_phi_alpha(xv, K, alpha))
                assert abs(mine - ref) < 1e-9, (xv, K, alpha, mine, ref)


def test_K0_endpoints_are_the_tradeoff():
    """alpha=1/2 starts from the pristine line x; alpha=1 starts from the shifted x+1/2."""
    x = np.linspace(0, 4, 50)
    assert np.allclose(wpc.warp_alpha_coord(x, 0, 0.5), x, atol=1e-12)
    assert np.allclose(wpc.warp_alpha_coord(x, 0, 1.0), x + 0.5, atol=1e-12)


def test_both_phases_converge_to_their_staircase():
    """In a cell interior both phases approach floor(x)+alpha as K grows."""
    xi = np.linspace(2.15, 2.45, 200)
    for alpha in (0.5, 1.0):
        devs = [float(np.max(np.abs(wpc.warp_alpha_coord(xi, K, alpha)
                                    - wpc.staircase_alpha(xi, alpha))))
                for K in (2, 8, 32, 128)]
        assert devs[0] > devs[1] > devs[2] > devs[3], (alpha, devs)
        assert devs[-1] < 0.01
