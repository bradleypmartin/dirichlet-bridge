"""Unit tests for the warp-coordinate illustration.

`warp_coordinate.py` draws the warp map n*(x) = x + phi_K(x) bending from the
linear x (K=0) to the midpoint staircase floor(x)+1/2 (K -> infinity). These
checks pin the illustration to the real warp object:

* its vectorized `phi_K` matches the canonical `warp_bridge.warp_phi`;
* the cell midpoints m + 1/2 are exact fixed points at every K;
* away from the Gibbs step edges the warp converges to the staircase tread;
* K = 0 is the untouched linear coordinate.
"""
from __future__ import annotations

import numpy as np

import warp_bridge as wb
import warp_coordinate as wc


def test_phi_matches_canonical():
    """The numpy phi_K equals warp_bridge.warp_phi to full float precision."""
    for xv in (0.3, 1.7, 2.5, 3.14159, 4.62):
        for K in (1, 4, 16, 40):
            mine = float(wc.phi_K(np.array([xv]), K)[0])
            ref = float(wb.warp_phi(xv, K))
            assert abs(mine - ref) < 1e-9, (xv, K, mine, ref)


def test_midpoints_are_fixed_points():
    """Cell midpoints m + 1/2 map to themselves at every K (sin(2 pi k (m+1/2)) = 0)."""
    mids = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    for K in (1, 2, 8, 33):
        assert np.allclose(wc.warp_coord(mids, K), mids, atol=1e-9)


def test_K0_is_linear():
    """K = 0 is the bland linear coordinate n* = x."""
    x = np.linspace(0, 5, 50)
    assert np.allclose(wc.warp_coord(x, 0), x, atol=1e-12)


def test_interior_converges_to_staircase():
    """In a cell interior (clear of the Gibbs edge) the deviation shrinks with K."""
    xi = np.linspace(2.15, 2.45, 200)
    devs = [float(np.max(np.abs(wc.warp_coord(xi, K) - wc.staircase(xi))))
            for K in (2, 8, 32, 128)]
    assert devs[0] > devs[1] > devs[2] > devs[3], devs
    assert devs[-1] < 0.01
