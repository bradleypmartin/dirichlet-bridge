"""Unit tests for the continuous Dirichlet-eta integral and its zero laws.

F(s) = int_1^inf cos(pi x) x^{-s} dx, the rich continuous endpoint of the bridge:

* the incomplete-gamma closed form `F_closed` matches the oscillatory-quadrature
  reference `F_quad` to full precision;
* the two-term reduction `F_asym = 1/(1-s) + (1/2)(-i pi)^{s-1} Gamma(1-s)`
  converges to F as t grows;
* the zero string sits OFF the critical line (Re -> 3/2, not 1/2);
* the three derived laws -- real part `sigma_law`, spacing `2pi/ln(t/pi)`,
  Riemann-von Mangoldt counting `(t/2pi)ln(t/pi e)+5/8` -- hold numerically.
"""
from __future__ import annotations

import math

import mpmath as mp

import cont_eta as ce


def test_closed_matches_quad():
    """Closed form == oscillatory quadrature to (near) full precision."""
    for s in [mp.mpc(2, 1), mp.mpc(1.5, 2), mp.mpc(0.7, 10), mp.mpc(1.8, 25)]:
        rel = abs((ce.F_closed(s) - ce.F_quad(s)) / ce.F_quad(s))
        assert rel < mp.mpf("1e-20"), (s, rel)


def test_asym_reduction_converges():
    """The two-term reduction tightens from t=20 to t=150 (1% by t~40)."""
    rels = []
    for t in [20, 40, 80, 150]:
        s = mp.mpc(ce.sigma_law(t), t)
        rels.append(float(abs((ce.F_asym(s) - ce.F_closed(s)) / ce.F_closed(s))))
    assert rels[0] < 2e-2
    assert rels[-1] < 5e-3
    # endpoint improvement only: the decay can wiggle at intermediate t
    assert rels[-1] < rels[0]


def test_found_points_are_zeros():
    zeros = ce.find_zeros(40.0)
    assert len(zeros) == 10                       # zeros t in (10.67, 38.87]
    for z in zeros:
        assert abs(ce.F_closed(z)) < 1e-18, (complex(z), abs(ce.F_closed(z)))


def test_zeros_off_critical_line():
    """Real parts live near 1.8-2.1 and drift DOWN toward 3/2 -- never 1/2."""
    zeros = ce.find_zeros(60.0)
    res = [float(z.real) for z in zeros]
    assert min(res) > 1.5 and max(res) < 2.2
    assert res[0] > res[-1]                      # decreasing
    assert res[-1] < 1.8                         # already well below the first


def test_spacing_law():
    """Consecutive gaps match 2pi/ln(t/pi); tight (<0.3%) once t is moderate."""
    ts = [float(z.imag) for z in ce.find_zeros(80.0)]
    for i in range(1, len(ts)):
        gap = ts[i] - ts[i - 1]
        law = ce.spacing_law(0.5 * (ts[i] + ts[i - 1]))
        tol = 2e-3 if ts[i] > 18 else 6e-3       # first gap is the loosest
        assert abs(gap - law) / gap < tol, (i, gap, law)


def test_counting_law_offset_is_five_eighths():
    """N(t) - (t/2pi)ln(t/pi e) settles on the constant 5/8."""
    ts = [float(z.imag) for z in ce.find_zeros(80.0)]
    offsets = [(i + 1) - (t / (2 * math.pi) * math.log(t / (math.pi * math.e)))
               for i, t in enumerate(ts) if t > 25]
    assert offsets, "need zeros above t=25"
    assert abs(sum(offsets) / len(offsets) - 0.625) < 5e-3
    assert max(offsets) - min(offsets) < 5e-3    # genuinely constant


def test_sigma_law_tracks_and_limits():
    """sigma_law matches the located real parts and -> 3/2 from above."""
    zeros = ce.find_zeros(60.0)
    checked = 0
    for z in zeros:
        if float(z.imag) > 50:                   # leading-order law converges ~1/ln t
            assert abs(ce.sigma_law(float(z.imag)) - float(z.real)) < 3e-3
            checked += 1
    assert checked >= 3
    assert ce.sigma_law(1e6) > 1.5               # approaches 3/2 from above
    assert abs(ce.sigma_law(1e12) - 1.5) < 0.05
