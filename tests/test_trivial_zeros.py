"""Unit tests for the trivial-zero birth study (trivial_zeros.py).

The point under test: the trivial zeros -2, -4, -6, ... are absent from the bland endpoint
1/(s-1)+1/2 (one real zero, at -1) and are restored as K grows, by the same O(1/K) tide
that drives every other family. The checks:

* convergence -- zeta^{(K)}(-2n) sits just ABOVE 0 and -> 0 like |2n|/(2 pi^2 K);
* rate-law reuse -- the trivial-zero displacement coefficient IS rate_law.disp_coeff at -2n;
* the inherited -2 zero -- continuation of the endpoint zero (-1), real and monotone onto -2;
* the birth-K law -- deep pairs (factorially deep dips) are born first, the shallow {-4,-6} last.

The cheap checks (closed-form values, displacement algebra, a short continuation) run in the
fast suite; the zero-finding birth-K measurement is `slow`.
"""
from __future__ import annotations

import mpmath as mp
import pytest

import rate_law as rl
import trivial_zeros as tz


def test_endpoint_has_one_real_zero_at_minus_one():
    """The bland K=0 endpoint 1/(s-1)+1/2 vanishes only at s=-1 -- the seed of the string."""
    import harmonic_bridge as hb
    assert tz.endpoint_zero() == -1
    assert abs(hb.zeta_K(tz.endpoint_zero(), 0)) < mp.mpf("1e-25")


def test_trivial_value_sits_above_zero_and_converges_one_over_K():
    """zeta^{(K)}(-2n) > 0 and 4x more harmonics shrink it ~4x (O(1/K))."""
    v40, v160 = float(tz.trivial_value(2, 40).real), float(tz.trivial_value(2, 160).real)
    assert v40 > 0 and v160 > 0                       # the tide is a positive lift
    assert 3.5 < v40 / v160 < 4.5                     # 4x K -> ~4x smaller


def test_trivial_value_tracks_the_2n_over_2pi2K_law():
    """zeta^{(K)}(-2n) ~ |2n|/(2 pi^2 K) for several depths."""
    for n in (1, 2, 4):
        v = float(tz.trivial_value(n, 160).real)
        assert abs(v / tz.predict_value(n, 160) - 1) < 0.02


def test_disp_coeff_trivial_is_rate_law_at_minus_2n():
    """The trivial-zero displacement coefficient IS rate_law.disp_coeff('comb_zeta', -2n)."""
    for n in (1, 2, 3, 5):
        a = tz.disp_coeff_trivial(n)
        b = rl.disp_coeff("comb_zeta", mp.mpc(-2 * n, 0))
        assert abs(a - b) < mp.mpf("1e-25")
    # real on the axis, and largest in the middle (-6/-8 hardest to settle)
    mags = [abs(float(tz.disp_coeff_trivial(n).real)) for n in (1, 2, 3, 4, 5)]
    assert max(range(len(mags)), key=mags.__getitem__) in (2, 3)   # peak at n=3 (-6) or n=4 (-8)


def test_pinch_K_law_orders_deep_pairs_first():
    """Birth-K decreases with depth: the shallow {-4,-6} clears the tide last."""
    assert tz.pinch_K_law(1) > tz.pinch_K_law(2) > tz.pinch_K_law(3)
    # the ordering is driven by factorially-deepening dips
    assert tz.valley_depth(1) < tz.valley_depth(2) < tz.valley_depth(3)


def test_inherited_branch_is_real_monotone_and_lands_on_minus_two():
    """Continue the endpoint zero (-1) up in K: real, moving left, limiting onto -2."""
    br = tz.inherited_branch(schedule=[0, 1, 5, 20, 60, 130])
    reals = [float(z.real) for _, z in br]
    assert reals[0] == -1.0                                   # starts at the endpoint zero
    assert all(abs(z.imag) < 1e-9 for _, z in br)             # stays on the real axis
    assert all(b < a for a, b in zip(reals, reals[1:]))       # moves monotonically left
    assert -2.0 < reals[-1] < -1.95                           # approaches -2 from the right


def test_predicted_zero_matches_a_real_zero_near_minus_two():
    """The leading prediction -2 + Re(a_1)/K locates the actual real zero near -2."""
    import harmonic_bridge as hb
    K = 120
    z = mp.findroot(lambda s: hb.zeta_K(s, K), mp.mpc(tz.predict_zero(1, K), 0))
    assert abs(z.imag) < 1e-9
    assert abs(float(z.real) - tz.predict_zero(1, K)) < 0.01


@pytest.mark.slow
def test_birth_K_measured_matches_the_law():
    """The measured pinch-K (conjugate pair lands on the axis) tracks |s_mid|/(2 pi^2 |zeta|)."""
    for j in (1, 2, 3):
        meas = tz.pinch_K_measured(j)
        law = tz.pinch_K_law(j)
        assert meas is not None
        assert abs(meas - law) / law < 0.2                   # within 20% across 8 <= K <= 64


@pytest.mark.slow
def test_pair_is_born_off_axis_and_pinches_onto_the_real_axis():
    """Below its birth-K the {-4,-6} pair is a conjugate pair with Im>0; it descends to Im~0."""
    fk = tz.pinch_fork(1)
    cb = fk["complex_branch"]
    assert cb, "expected an off-axis conjugate branch below the pinch"
    ims = [float(z.imag) for _, z in cb]               # ordered by increasing K
    assert ims[0] > 0.3                                # well off the axis at small K
    assert ims[-1] < ims[0]                            # descends toward the axis as K grows
    # the real prongs above the pinch head for -4 and -6
    pr = [float(z.real) for _, z in fk["prong_right"]]
    pl = [float(z.real) for _, z in fk["prong_left"]]
    assert pr and pl and pr[0] > pl[0]                 # right prong (-> -4) is right of left (-> -6)
