"""Unit tests for the Fornberg-Kolbig (1975) Jonquiere/polylogarithm reproduction.

F(x,s) = sum_{k>=1} x^k/k^s, the one-parameter polylogarithm deformation of zeta
(F(1,s)=zeta, F(-1,s)=-eta) and the direct ancestor of the bridge program:

* the polylog evaluator matches the direct sum (|x|<1) and the FK functional
  equation F(x,s)=2^{1-s}F(x^2,s)-F(-x,s) (Eq. 28);
* the x->+/-1 series (Eq. 26 and its branch-free x=-1 fold) match polylog;
* the FK asymptotic seeds (Eqs. 10/17, 19/20) and the sigma=1 log-2 comb
  1+2pi i m/log2 (Eq. 23) are where FK say they are;
* (slow) traced zero trajectories reproduce the two classes: the P=0 class tends
  to s=1, s0+(x,1,1) brushes sigma=1/2 at s1 then spirals off, the x->-1 N=1
  trajectories land on the log-2 comb, and the real-axis P=0 ones land on -2N;
* (slow) the argument principle reproduces FK's counts Z(0.1)=27, Z(-0.1)=26.
"""
from __future__ import annotations

import mpmath as mp
import pytest

import jonquiere_zeros as jz

mp.mp.dps = 30


# ----------------------------- evaluator + identities -----------------------
def test_polylog_matches_direct_sum():
    """F = polylog(s,x) equals the defining series sum for |x|<1."""
    for x in [0.1, -0.3, 0.7, -0.7, 0.9]:
        for s in [mp.mpc(0.5, 14), mp.mpc(-1, 5), mp.mpc(2, 0)]:
            a, b = jz.F(x, s), jz.F_sum(x, s)
            assert abs((a - b) / b) < mp.mpf("1e-25"), (x, complex(s))


def test_functional_equation():
    """F(x,s) = 2^{1-s} F(x^2,s) - F(-x,s)  (FK Eq. 28)."""
    for x in [0.3, 0.7, -0.5, 0.9]:
        for s in [mp.mpc(0.5, 14), mp.mpc(-1, 5), mp.mpc(1.5, 3)]:
            assert jz.functional_eq_residual(x, s) < mp.mpf("1e-14"), (x, complex(s))


def test_endpoints_zeta_and_eta():
    """F(1,s) = zeta(s) and F(-1,s) = -eta(s) (the arithmetic endpoints)."""
    for s in [mp.mpc(2, 1), mp.mpc(0.7, 10), mp.mpc(-1, 3), mp.mpc(3, 0)]:
        assert abs(jz.F(1, s) - mp.zeta(s)) < mp.mpf("1e-25"), complex(s)
        assert abs(jz.F(-1, s) + jz.eta(s)) < mp.mpf("1e-25"), complex(s)


def test_expansions_match_polylog():
    """Both x->1 series (Eq. 26 and the branch-free x=-1 fold) match polylog."""
    for x in [0.85, 0.95, 0.999]:
        for s in [mp.mpc(1, 9.06), mp.mpc(0.5, 14), mp.mpc(-3, 4)]:
            ref = jz.F(x, s)
            assert abs((jz.F_expand(x, s) - ref) / ref) < mp.mpf("1e-22")
    for x in [-0.85, -0.95, -0.999]:
        for s in [mp.mpc(1, 9.06), mp.mpc(0.5, 14), mp.mpc(-6, 4.5)]:
            ref = jz.F(x, s)
            assert abs((jz.F_expand_neg(x, s) - ref) / ref) < mp.mpf("1e-22")


def test_F_track_continuous_at_switch():
    """The tracking evaluator agrees with polylog on both sides of |x|=0.8."""
    for x in [-0.82, -0.8, 0.8, 0.82]:
        for s in [mp.mpc(0.5, 14), mp.mpc(-2, 3)]:
            ref = jz.F(x, s)
            assert abs((jz.F_track(x, s) - ref) / ref) < mp.mpf("1e-20")


# ----------------------------- seeds / asymptotes / comb --------------------
def test_asymptotes_match_seed_imag():
    """As x->0, Im of the FK seed -> the asymptote heights v+/-(P,N) (Eqs. 17,20)."""
    x = mp.mpf("1e-12")
    for P, N in [(0, 1), (1, 1), (2, 3)]:
        assert abs(mp.im(jz.seed_plus(x, P, N)) - jz.asymptote_plus(P, N)) < 1e-25
    for P, N in [(1, 1), (2, 2), (3, 4)]:
        assert abs(mp.im(jz.seed_minus(-x, P, N)) - jz.asymptote_minus(P, N)) < 1e-25


def test_comb_spacing_is_2pi_over_log2():
    """The sigma=1 comb points sit at 1 + 2pi i m/log2 ~ 1 + 9.06472 m i."""
    assert abs(jz.TWOPI_OVER_LOG2 - 9.064720283654388) < 1e-12
    for m in [1, 2, 3]:
        c = jz.comb_point(m)
        assert mp.re(c) == 1
        assert abs(mp.im(c) - m * jz.TWOPI_OVER_LOG2) < 1e-25


def test_comb_points_are_eta_factor_zeros():
    """1 + 2pi i m/log2 are exactly the zeros of the eta factor 1 - 2^{1-s}."""
    for m in [1, 2, 3, -1]:
        s = jz.comb_point(m)
        assert abs(1 - 2 ** (1 - s)) < mp.mpf("1e-24"), m


def test_alpha_counting_constants():
    """alpha+ = (2-2log2-gamma)/2pi, alpha- = (log2-gamma)/2pi  (FK Eq. 40)."""
    assert abs(float(jz.alpha_plus()) - 0.00580756) < 1e-7
    assert abs(float(jz.alpha_minus()) - 0.01845107) < 1e-7


# ----------------------------- traced trajectories (slow) -------------------
@pytest.mark.slow
def test_p0_class_tends_to_s1():
    """The P=0 trajectory s0+(x,0,1) migrates toward the pole at s=1, not a zero."""
    tr = jz.trace_zero(0, 1, +1, "0.05", "1e-4", 22)
    s_start, s_end = tr[0][1], tr[-1][1]
    assert mp.re(s_end) > mp.re(s_start)            # real part climbs toward 1
    assert abs(s_end - 1) < abs(s_start - 1)        # overall it heads to s=1
    assert mp.re(s_end) > 0.4                        # left the deep left half-plane


@pytest.mark.slow
def test_trajectory_brushes_s1():
    """s0+(x,1,1) climbs from the left half-plane to brush the critical line at
    s1 as x->1 (FK Fig. 6). That it does not *stay* is the spiral test below."""
    tr = jz.trace_zero(1, 1, +1, "0.02", "1e-6", 30)
    assert mp.re(tr[0][1]) < 0                       # starts in the left half-plane
    assert min(abs(s - jz.S1) for _x, s in tr) < mp.mpf("0.02")   # brushes s1


@pytest.mark.slow
def test_spiral_winds_out_never_lands():
    """The Fig. 7 spiral about s1: converges in, then the radius grows again."""
    sp = jz.spiral_about_s1(alpha_lo=1.5, alpha_hi=12.0, nsteps=22)
    radii = [abs(s - jz.S1) for _a, s in sp]
    imin = radii.index(min(radii))
    assert min(radii) < mp.mpf("1e-3")             # reaches s1 closely
    assert 0 < imin < len(radii) - 1               # then leaves (interior minimum)
    assert radii[-1] > 3 * min(radii)              # winds back out


@pytest.mark.slow
def test_comb_trajectories_land_on_log2_comb():
    """x->-1 N=1 trajectories s0-(x,P,1) terminate on 1 + 2pi i m/log2 (Eq. 23)."""
    for P in [1, 2, 3]:
        tr = jz.trace_zero(P, 1, -1, "0.2", "1e-6", 16)
        assert abs(tr[-1][1] - jz.comb_point(P)) < mp.mpf("1e-3"), P


@pytest.mark.slow
def test_trivial_zero_trajectories_land_on_minus_2N():
    """Real-axis x->-1 trajectories s0-(x,0,N) slide onto the trivial zeros -2N."""
    for N in [1, 2, 3]:
        x_lo = float(mp.mpf(10) ** (-N - 1))
        tr = jz.trace_zero(0, N, -1, x_lo, "1e-6", 30)
        end = tr[-1][1]
        assert abs(end + 2 * N) < mp.mpf("1e-3"), N
        assert abs(mp.im(end)) < mp.mpf("1e-3")     # stays on the real axis


@pytest.mark.slow
def test_argument_principle_zero_counts():
    """The winding number reproduces FK's counts Z(0.1)=27, Z(-0.1)=26 (Eq. 25)."""
    z_p, _ = jz.count_zeros("0.1", -11, 0, 0, 110, npts=300)
    z_m, _ = jz.count_zeros("-0.1", -11, 0, 1, 109, npts=300)
    assert round(float(z_p)) == 27
    assert round(float(z_m)) == 26
