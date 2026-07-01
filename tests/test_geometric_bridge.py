"""Unit tests for the geometric-series miniature (geometric_bridge.py).

The point under test: the whole apparatus reproduces, in elementary closed form, on the
geometric series g(s)=sum s^n=1/(1-s) with continuous shadow -1/ln s. The checks:

* exact telescoping: the full comb == (1/2)coth(L/2)-1/L == 1/(1-s) (the coth partial fraction);
* the comb g_K -> 1/(1-s) at O(1/K), with error the arc's rate law |ln s|/(2 pi^2 K) (s -> ln s);
* the load-bearing half-weight: the K=0 anchor is shadow+1/2, and every g_K shares the pole
  residue -1 at s=1;
* no abscissa: g_K converges for |s|>1 too, summing the divergent series (1+2+4+...=-1) to its
  analytic value, at the same O(1/K);
* the warp factorizes (warp_K = discrete . J_K, J_K -> 1) and matches the honest integral;
* the ascending comb runs the same harmonics backwards (asc_K -> shadow).

All cheap (elementary closed forms + a few 1-D quadratures; no zero-finding).
"""
from __future__ import annotations

import mpmath as mp
import pytest

import geometric_bridge as gb

# a spread inside and outside the unit disk, real and complex
SAMPLES = [mp.mpf("0.5"), mp.mpf("0.9"), mp.mpc("0.3", "0.4"),
           mp.mpf("2"), mp.mpf("5"), mp.mpf("-1"), mp.mpc("1.5", "2")]


def test_telescoping_is_exact():
    """The full comb telescopes to 1/(1-s): both closed forms agree to full precision."""
    for s in SAMPLES:
        assert abs(gb.geom_limit_closed(s) - gb.discrete(s)) < mp.mpf("1e-27")
        # harm_sum_closed = shadow-side identity: shadow + 1/2 - harm_sum_closed == discrete
        recon = gb.shadow(s) + mp.mpf("0.5") - gb.harm_sum_closed(s)
        assert abs(recon - gb.discrete(s)) < mp.mpf("1e-27")


def test_comb_converges_one_over_K():
    """g_K -> 1/(1-s): 4x the harmonics -> ~4x smaller error (O(1/K)), inside and outside |s|<1."""
    for s in SAMPLES:
        d = gb.discrete(s)
        e64 = abs(gb.geom_K(s, 64) - d)
        e256 = abs(gb.geom_K(s, 256) - d)
        assert e256 < e64
        assert 3.5 < float(e64 / e256) < 4.2          # clean 1/K


def test_rate_cost_is_the_arc_law_with_lns():
    """The fixed-K error tracks |ln s|/(2 pi^2 K) -- the zeta comb constant s/(2 pi^2 K), s -> ln s."""
    K = 256
    for s in SAMPLES:
        law = float(gb.rate_cost(s, K))
        err = float(abs(gb.geom_K(s, K) - gb.discrete(s)))
        assert abs(err - law) / law < 0.02            # ratio -> 1 (0.998 at K=256)


def test_K0_anchor_is_shadow_plus_half():
    """The K=0 comb anchor is the continuous shadow + the load-bearing half-weight 1/2."""
    for s in SAMPLES:
        assert abs(gb.geom_K(s, 0) - (gb.shadow(s) + mp.mpf("0.5"))) < mp.mpf("1e-27")


def test_pole_residue_minus_one_shared_by_every_K():
    """Every g_K has the same simple pole at s=1 with residue -1: (s-1) g_K(s) -> -1."""
    eps = mp.mpf("1e-6")
    for K in (0, 5, 50):
        assert abs(eps * gb.geom_K(1 + eps, K) + 1) < mp.mpf("1e-10")


def test_no_abscissa_sums_divergent_series():
    """Outside |s|<1 the classical series diverges, but g_K lands on 1/(1-s) (Abel value)."""
    # 1+2+4+8+... = -1 ; Grandi 1-1+1-... = 1/2
    for s, val in [(mp.mpf(2), mp.mpf(-1)), (mp.mpf(-1), mp.mpf("0.5"))]:
        assert abs(gb.geom_K(s, 256) - val) < mp.mpf("1e-3")
        # and it is still O(1/K) there, not a boundary
        e64 = abs(gb.geom_K(s, 64) - gb.discrete(s))
        e256 = abs(gb.geom_K(s, 256) - gb.discrete(s))
        assert 3.5 < float(e64 / e256) < 4.2


def test_warp_factorizes_and_converges():
    """warp_K = discrete . J_K -> 1/(1-s), with J_K -> 1 (inside and outside |s|<1)."""
    for s in (mp.mpf("0.5"), mp.mpc("0.3", "0.4"), mp.mpf("5"), mp.mpf("-1")):
        d = gb.discrete(s)
        assert abs(gb.warp_J(s, 32) - 1) < abs(gb.warp_J(s, 4) - 1)     # J_K -> 1
        assert abs(gb.warp_K(s, 32) - d) < abs(gb.warp_K(s, 4) - d)     # warp_K -> discrete
        assert abs(gb.warp_K(s, 32) - d) < mp.mpf("5e-3")


@pytest.mark.slow
def test_warp_factorization_matches_honest_integral():
    """The factorization is exact: the honest period-by-period int_0^inf == discrete . J_K (|s|<1)."""
    for s in (mp.mpf("0.5"), mp.mpc("0.3", "0.4")):
        assert abs(gb.warp_raw(s, 16, ncells=60) - gb.warp_K(s, 16)) < mp.mpf("1e-12")


def test_warp_midpoint_is_half_shifted_geometric():
    """The midpoint warp lands on s^{1/2}/(1-s) = sum_{m>=0} s^{m+1/2} (echo of zeta(s,1/2))."""
    for s in (mp.mpf("0.5"), mp.mpc("0.3", "0.4")):
        tgt = mp.power(mp.mpc(s), mp.mpf("0.5")) * gb.discrete(s)
        assert abs(gb.warp_mid_K(s, 32) - tgt) < mp.mpf("5e-3")


def test_ascending_comb_returns_to_shadow():
    """The same harmonics run backwards: asc_K = discrete - 1/2 + sum h_k -> -1/ln s."""
    for s in SAMPLES:
        c = gb.shadow(s)
        assert abs(gb.asc_K(s, 128) - c) < abs(gb.asc_K(s, 8) - c)      # -> shadow
        assert abs(gb.asc_K(s, 128) - c) < mp.mpf("2e-3")
