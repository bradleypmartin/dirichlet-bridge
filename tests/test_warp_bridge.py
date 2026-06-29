"""Unit tests for the nonlinear n*-warp bridge.

The midpoint-warp stage of the bridge: warp the integration variable (n* = x + phi_K(x)) instead of
adding harmonics to the integrand, and watch where the zeros go. The checks:

* the warp endpoint is bland -- warp_K(., 0) = 1/(s-1);
* the honest int_1^inf warp limits to the *ugly* half-integer-midpoint zeta(s, 3/2);
* the load-bearing +2^s completes it to the *clean* half-shifted zeta(s, 1/2) =
  (2^s - 1) zeta(s) -- the eta factor the bridge set aside, re-derived as the lower-endpoint
  half-cell;
* the Hurwitz form matches the genuine oscillatory integral (sigma > 1);
* migration: zeta zeros climb onto Re = 1/2 (from below), and the prefactor companions
  (2^s - 1 = 0, at t = 2 pi k/ln2) climb onto Re = 0 -- the mirror, across 1/2, of
  the additive comb's eta companion on Re = 1; without the +2^s that companion does not exist.
* density bijection: the warp target zeta(s,1/2)=(2^s-1)zeta(s) carries N_1/2(T)
  zeta zeros on sigma=1/2 and N_0(T) companions on sigma=0; N_1/2 + N_0 reproduces the
  continuous-eta's Riemann-von Mangoldt N(T) to leading order, and N_0 = floor(T ln2/2pi) on the nose --
  the warp mirror (measured, not asserted-by-symmetry) of the additive comb's eta count.

The full-K limits, the quadrature cross-check, the (continuation-heavy) migration, and the
box-finder *measured* counts are marked `slow` (deselected by default, like the cone/krein
end-to-end verdicts); the fast suite keeps the cheap building blocks -- the two endpoints,
the load-bearing 2^s, and the analytic (root-finder-free) counting identity.
"""
from __future__ import annotations

import mpmath as mp
import pytest

import warp_bridge as wb

# short continuation schedule for the migration tests (warp_K is the fast grid evaluator,
# so K=32 is cheap and lands the zeros within ~0.02 of their limit lines).
SHORT = [1, 2, 4, 8, 16, 32]


# --------------------------------------------------------------------------
# fast building blocks
# --------------------------------------------------------------------------
def test_warp_endpoint_is_bland():
    """warp_K(., 0) is the bland zeta-side endpoint 1/(s-1) (no warp, no zeros)."""
    for s in [mp.mpc(2, 0), mp.mpc(0.5, 14), mp.mpc(1.3, 7)]:
        assert abs(wb.warp_K(s, 0) - 1 / (s - 1)) < mp.mpf("1e-25")


def test_half_weight_is_exactly_two_to_the_s():
    """warp_half_K = warp_K + 2^s by construction (the restored m=0 half-cell)."""
    for s in [mp.mpc(0.5, 14), mp.mpc(1.3, 7), mp.mpc(0, 9.06)]:
        assert abs(wb.warp_half_K(s, 5) - (wb.warp_K(s, 5) + mp.power(2, s))) < mp.mpf("1e-25")


def test_two_s_endpoint_is_load_bearing():
    """At s = 2 pi i/ln2 the half-shift zeta(s,1/2) vanishes but the honest limit
    zeta(s,3/2) does not -- the sigma=0 companion is a creature of the +2^s alone."""
    sp = mp.mpc(0, 2 * mp.pi / mp.log(2))                  # full precision: ~ 9.0647 i
    assert abs((mp.power(2, sp) - 1) * mp.zeta(sp)) < mp.mpf("1e-18")   # zeta(s,1/2) = 0
    assert abs(mp.zeta(sp, mp.mpf(3) / 2)) > mp.mpf("0.5")             # zeta(s,3/2) != 0
    # the raw warp (no completion) is likewise non-vanishing near the companion height
    assert abs(wb.warp_K(sp, 12)) > mp.mpf("0.5")


def test_pref_heights():
    """pref_heights lists t = 2 pi k/ln2 below the cutoff."""
    hs = wb.pref_heights(30.0)
    assert len(hs) == 3                                       # ~9.06, 18.13, 27.19
    for k, h in enumerate(hs, start=1):
        assert abs(h - 2 * float(mp.pi) * k / float(mp.log(2))) < 1e-9


# --------------------------------------------------------------------------
# fast: the density-bijection counting identity, root-finder-free
# --------------------------------------------------------------------------
def test_target_half_is_half_shift():
    """target_half(s) == (2^s - 1) zeta(s) == zeta(s, 1/2) (the K -> infinity migration object)."""
    for s in [mp.mpc(2, 1), mp.mpc(0.5, 14.134725), mp.mpc(0, 9.0647), mp.mpc(1.3, 7)]:
        assert abs(wb.target_half(s) - mp.zeta(s, mp.mpf(1) / 2)) < mp.mpf("1e-22")


def test_companion_count_is_on_the_nose():
    """N_0(T) = #{companions 2 pi k/ln2 < T} = floor(T ln2/2pi) exactly (the sigma=0 family)."""
    for T in (20.0, 40.0, 73.0):
        n0 = len(wb.pref_heights(T))
        assert n0 == int(T * float(mp.log(2)) / (2 * float(mp.pi)))      # on the nose


def test_density_bijection_sum_matches_rvm():
    """N_1/2(T) + N_0(T) reproduces the continuous-eta's Riemann-von Mangoldt N(T) to leading order.

    N_1/2 = #zeta-zeros below T, N_0 = #companions below T; their densities
    (1/2pi)ln(t/2pi) + (ln2/2pi) = (1/2pi)ln(t/pi) sum to the continuous-eta's ground density exactly, so
    the integer total tracks counting_law(T) within an O(1) slack (the 7/8-vs-5/8 constant
    offset shared with the eta side, plus rounding)."""
    import cont_eta as ce
    for T in (30.0, 40.0):
        n_half = sum(1 for n in range(1, 20) if float(mp.zetazero(n).imag) < T)
        n0 = len(wb.pref_heights(T))
        assert abs((n_half + n0) - float(ce.counting_law(T))) < 1.0    # leading-order match


# --------------------------------------------------------------------------
# slow: full-K limits, quadrature cross-check, the migration map, the measured count
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_warp_limit_is_zeta_three_half():
    """The honest int_1^inf warp -> zeta(s, 3/2) (the cell midpoints 3/2, 5/2, ...)."""
    for s in [mp.mpc(2, 0), mp.mpc(0.5, 14.134725), mp.mpc(1.3, 7)]:
        z = mp.zeta(s, mp.mpf(3) / 2)
        e8 = abs(wb.warp_K(s, 8) - z)
        e32 = abs(wb.warp_K(s, 32) - z)
        assert e32 < e8                       # tightening
        assert e32 < mp.mpf("8e-2")


@pytest.mark.slow
def test_warp_half_limit_is_half_shift():
    """warp_half_K -> zeta(s, 1/2) == (2^s - 1) zeta(s): the half-shift target."""
    for s in [mp.mpc(2, 0), mp.mpc(0.5, 14.134725), mp.mpc(1.3, 7)]:
        target = mp.zeta(s, mp.mpf(1) / 2)
        assert abs(target - (mp.power(2, s) - 1) * mp.zeta(s)) < mp.mpf("1e-25")
        e8 = abs(wb.warp_half_K(s, 8) - target)
        e32 = abs(wb.warp_half_K(s, 32) - target)
        assert e32 < e8
        assert e32 < mp.mpf("8e-2")


@pytest.mark.slow
def test_fast_warp_matches_hurwitz():
    """The fast grid+moment warp_K agrees with the transparent warp_hurwitz, across sigma
    (incl. sigma=0) and K -- and is exact (-> 1/(s-1)) at K=0."""
    for s in [mp.mpc(2, 1), mp.mpc(0.5, 14.134725), mp.mpc(0, 9.0647), mp.mpc(1.3, 7)]:
        for K in (8, 32):
            assert abs(wb.warp_K(s, K) - wb.warp_hurwitz(s, K)) < mp.mpf("1e-6")
    assert abs(wb.warp_K(mp.mpc(1.3, 7), 0) - 1 / (mp.mpc(1.3, 7) - 1)) < mp.mpf("1e-25")


@pytest.mark.slow
def test_warp_matches_raw_integral():
    """warp_K == the genuine oscillatory int_1^inf (sigma > 1) -- the independent guise."""
    s = mp.mpc(2.5, 1)                          # fast-decaying integrand for quadosc
    assert abs(wb.warp_K(s, 2) - wb.warp_raw(s, 2)) < mp.mpf("1e-3")


@pytest.mark.slow
def test_zeta_zeros_climb_to_half_from_below():
    """zeta zero 1/2+14.13i: migrates onto Re=1/2, approaching from below (warp's signature)."""
    t = dict((K, zz) for K, zz in wb.hb.migrate(wb.warp_half_K, mp.mpc(0.5, 14.134725),
                                                schedule=SHORT) if zz is not None)
    assert 1 in t and 32 in t
    assert float(t[1].real) < 0.4                 # born BELOW the critical line
    assert abs(float(t[32].real) - 0.5) < 0.05    # climbed onto it
    assert float(t[32].real) > float(t[1].real)   # monotone climb from below


@pytest.mark.slow
def test_prefactor_companion_migrates_to_zero():
    """The 2^s-1 companion at t=2pi/ln2 migrates onto Re=0 (the eta-mirror line)."""
    pref_t = 2 * float(mp.pi) / float(mp.log(2))   # ~ 9.0647
    t = dict((K, zz) for K, zz in wb.hb.migrate(wb.warp_half_K, mp.mpc(0, pref_t),
                                                schedule=SHORT) if zz is not None)
    assert 32 in t
    assert abs(float(t[32].real)) < 0.05           # -> Re = 0
    assert abs(float(t[32].imag) - pref_t) < 0.2   # stays at the prefactor height


@pytest.mark.slow
def test_measured_bijection_on_target():
    """MEASURED bijection: find every zero of the warp target zeta(s,1/2) below T=40 and
    bin it -- 6 land on sigma=1/2 (the zeta zeros), 4 on sigma=0 (the 2^s-1 companions), none
    off-line, and the total matches the continuous-eta's RvM N(40)~10.46 (so N_1/2+N_0 reproduces the ground
    count). Closes the 'mirror of eta, same heights' claim as a count, not a symmetry argument."""
    import cont_eta as ce
    counts, off, zeros = wb.count_on_lines(wb.target_half, 40.0)
    assert len(off) == 0                                       # nothing off the two lines
    assert counts[0.5] == 6                                    # zeta zeros < 40 on sigma=1/2
    assert counts[0.0] == 4                                    # companions < 40 on sigma=0
    assert counts[0.0] == len(wb.pref_heights(40.0))          # == floor(T ln2/2pi), on the nose
    assert abs((counts[0.5] + counts[0.0]) - float(ce.counting_law(40.0))) < 1.0
    # the sigma=0 companions sit exactly at 2 pi k/ln2
    comp = sorted(float(z.imag) for z in zeros if abs(float(z.real)) <= 0.06)
    for k, t in enumerate(comp, start=1):
        assert abs(t - 2 * float(mp.pi) * k / float(mp.log(2))) < 1e-3


@pytest.mark.slow
def test_measured_bijection_finite_K():
    """The literal 'large K' ask: zeros of warp_half_K(., 45) -- not the limit target -- already
    bin onto sigma=1/2 U sigma=0. To T=30 that is 3 zeta zeros + 3 companions, each within the
    O(1/K) ~ 0.02 offset of its line (so a 0.08 tolerance bins them with none off-line)."""
    counts, off, _z = wb.count_on_lines(lambda s: wb.warp_half_K(s, 45), 30.0,
                                        tol_line=0.08, tol=1e-6, dedupe=2e-3)
    assert len(off) == 0
    assert counts[0.5] == 3                                    # 14.13, 21.02, 25.01
    assert counts[0.0] == 3                                    # 9.06, 18.13, 27.19


@pytest.mark.slow
def test_warp_K_stable_at_high_tau():
    """Regression: the grid+moment warp_K used to diverge catastrophically above
    |Im s| ~ 110 -- e.g. warp_K(1/2 + 160i, 40) blew up to ~2e8 (vs the true O(1)) because the
    binomial factor binom(-s,j) ~ 10^{0.4|s|} amplified the m>M tail (zeta(s+j)-H_M) once it
    decayed past mpmath's absolute error floor. With the |Im s|-scaled working precision /
    moment count / GL degree it now stays bounded and matches the transparent warp_hurwitz
    (taken at raised precision for an accurate oscillatory-quadrature reference) across -- and
    well past -- the previously-broken range."""
    for tau in (120.0, 170.0):
        s = mp.mpc(0.5, tau)
        with mp.workdps(45):                                  # accurate reference, raised precision
            ref = wb.warp_hurwitz(s, 40)
        got = wb.warp_K(s, 40)                                 # evaluated at the ambient dps = 30
        assert abs(got) < 10                                   # bounded -- not the old ~1e8 blow-up
        assert abs(got - ref) < mp.mpf("1e-7")                 # and accurate to its native ~1e-9
