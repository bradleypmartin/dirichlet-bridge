"""Unit tests for the unified O(1/K) rate / birth-K law.

A follow-up across the bridge: one leading-rate constant and one zero-displacement
coefficient per bridge route, plus the sub-leading correction behind the
comb-overshoot / warp-no-overshoot asymmetry. The checks:

* leading rate: K(zeta - zeta^{(K)}) -> the CLOSED comb constant s/(2 pi^2) (and eta, same
  x=1 endpoint); the warp constant is bulk s(s+1)zeta(s+2,3/2)/4 pi^2 PLUS a Gibbs
  boundary layer (so full != bulk);
* the 2-term comb error model (sub-leading coefficient -1/2 of the leading) beats 1-term;
* the route-agnostic displacement law a_1 = -Delta_1/target' across all five families;
* approach side = sign(Re a_1): comb alternates (some above, some below), warp uniformly
  below; overshoot <=> Re a_1 < 0 for the comb (born above), never for the warp;
* birth-K catch_K = |Re a_1|/eps grows (on average) with the zero height;
* the continuous-eta sigma_law residual is F_asym truncation -- solving the balance exactly doesn't
  shrink it.

The continuation-heavy migrate checks and the slow exact-Richardson warp constant are
marked `slow` (deselected by default); the fast suite keeps the closed-form constants,
the displacement-coefficient identities, the signs, and the birth-K trend.
"""
from __future__ import annotations

import math

import mpmath as mp
import pytest

import rate_law as rl
import harmonic_bridge as hb
import warp_bridge as wb

# the first few zeta zeros / 2^s-1 companion heights used across the tests
ZGAM = [14.134725142, 21.022039639, 25.010857580, 30.424876126, 32.935061588, 37.586178159]
PI = float(mp.pi)
LN2 = float(mp.log(2))
PREF = [2 * PI * k / LN2 for k in (1, 2, 3)]
SHORT = [1, 2, 4, 8, 16, 32]


# --------------------------------------------------------------------------
# fast: the closed comb constant, identical for zeta and eta
# --------------------------------------------------------------------------
def test_rate_comb_is_s_over_two_pi_squared():
    """The 2-term model (s/2 pi^2)(1/K - 1/2K^2) reproduces zeta - zeta^{(K)} to O(1/K^3), and
    Richardson-extrapolating K*err removes the 1/K residual to nail the constant s/(2 pi^2)."""
    for s in [mp.mpc(2, 0), mp.mpc(0.5, 14.134725142), mp.mpc(1.3, 7)]:
        c = rl.rate_comb(s)
        # the 2-term model (sub-leading coefficient -1/2 included) matches to O(1/K^3)
        assert abs(rl.err_comb(s, 40) - (mp.zeta(s) - hb.zeta_K(s, 40))) < mp.mpf("5e-4")
        # Richardson 2*(K2 err) - (K1 err) kills the genuine 1/K residual -> the leading constant
        rich = (2 * 160 * (mp.zeta(s) - hb.zeta_K(s, 160))
                - 80 * (mp.zeta(s) - hb.zeta_K(s, 80)))
        assert abs(rich - c) < mp.mpf("1e-3")


def test_rate_comb_same_for_eta():
    """The eta comb has the SAME leading constant s/(2 pi^2) (its x=1 endpoint, |cos pi|=1)."""
    for s in [mp.mpc(2, 0), mp.mpc(0.5, 14.134725142)]:
        c = rl.rate_comb(s)
        rich = (2 * 160 * (mp.altzeta(s) - hb.eta_K(s, 160))
                - 80 * (mp.altzeta(s) - hb.eta_K(s, 80)))
        assert abs(rich - c) < mp.mpf("1e-3")


def test_err_comb_two_term_beats_one_term():
    """The 2-term model (s/2 pi^2)(1/K - 1/2K^2) is closer to zeta - zeta^{(K)} than 1-term."""
    s = mp.mpc(2, 0)
    for K in (10, 25):
        actual = mp.zeta(s) - hb.zeta_K(s, K)
        one = rl.rate_comb(s) / K
        two = rl.err_comb(s, K)
        assert abs(two - actual) < abs(one - actual)


# --------------------------------------------------------------------------
# fast: warp constant = bulk + Gibbs boundary layer (no elementary form)
# --------------------------------------------------------------------------
def test_rate_warp_bulk_form():
    """rate_warp_bulk(s) == s(s+1) zeta(s+2,3/2)/(4 pi^2) -- the second-order bulk response."""
    for s in [mp.mpc(2, 0), mp.mpc(0.5, 14.134725142)]:
        want = s * (s + 1) * mp.zeta(s + 2, mp.mpf(3) / 2) / (4 * mp.pi**2)
        assert abs(rl.rate_warp_bulk(s) - want) < mp.mpf("1e-25")


def test_rate_warp_full_differs_from_bulk():
    """The full warp constant (grid estimate) is NOT the bulk alone -- the boundary layer is real."""
    for s in [mp.mpc(2, 0), mp.mpc(1.3, 7)]:
        full = rl.rate_warp(s)                  # fast grid estimate, K=40
        bulk = rl.rate_warp_bulk(s)
        assert abs(full - bulk) > mp.mpf("2e-3")          # a genuine, sizeable gap


# --------------------------------------------------------------------------
# fast: the unified displacement coefficient a_1 = -Delta_1/target'
# --------------------------------------------------------------------------
def test_disp_coeff_is_minus_delta1_over_target_deriv():
    """a_1 = -Delta_1(rho)/target'(rho) for every route (the one route-agnostic law)."""
    cases = [("comb_zeta", mp.mpc(0.5, 14.134725142)),
             ("comb_eta", mp.mpc(0.5, 14.134725142)),
             ("eta_pref", mp.mpc(1, PREF[0])),
             ("warp_zeta", mp.mpc(0.5, 14.134725142)),
             ("warp_pref", mp.mpc(0, PREF[0]))]
    for route, rho in cases:
        a1 = rl.disp_coeff(route, rho)
        want = -rl.delta1(route, rho) / rl.target_deriv(route, rho)
        assert abs(a1 - want) < mp.mpf("1e-25")


def test_comb_zeta_coeff_closed_form():
    """For the comb zeta family a_1 reduces to the closed rho/(2 pi^2 zeta'(rho))."""
    for g in ZGAM[:3]:
        rho = mp.mpc(0.5, g)
        want = rho / (2 * mp.pi**2 * mp.zeta(rho, derivative=1))
        assert abs(rl.disp_coeff("comb_zeta", rho) - want) < mp.mpf("1e-22")


# --------------------------------------------------------------------------
# fast: approach side, overshoot criterion, birth-K trend
# --------------------------------------------------------------------------
def test_comb_alternates_warp_uniformly_below():
    """Comb Re a_1 takes BOTH signs over the first zeros; warp Re a_1 < 0 for all of them."""
    comb_signs = {rl.disp_coeff("comb_zeta", mp.mpc(0.5, g)).real > 0 for g in ZGAM}
    assert comb_signs == {True, False}                    # genuinely alternates
    for g in ZGAM:
        assert rl.disp_coeff("warp_zeta", mp.mpc(0.5, g)).real < 0   # warp always from below


def test_overshoot_criterion():
    """Comb overshoots <=> Re a_1 < 0: gamma=14 no, gamma=21 yes; warp never (born below)."""
    assert rl.overshoots(mp.mpc(0.5, 14.134725142)) is False
    assert rl.overshoots(mp.mpc(0.5, 21.022039639)) is True
    for g in ZGAM:
        rho = mp.mpc(0.5, g)
        assert rl.overshoots(rho) == (rl.disp_coeff("comb_zeta", rho).real < 0)


def test_approach_side_labels():
    """approach_side reports 'above'/'below' consistently with sign(Re a_1)."""
    rho = mp.mpc(0.5, 14.134725142)
    assert rl.approach_side("comb_zeta", rho) == "above"   # Re a_1 > 0
    assert rl.approach_side("warp_zeta", rho) == "below"   # Re a_1 < 0


def test_catch_K_grows_with_height():
    """Birth-K catch_K grows (on average) with gamma: its envelope over high zeros exceeds low.

    catch_K fluctuates zero-to-zero (it carries 1/|zeta'(rho)|), so we compare windowed maxima
    rather than demand strict monotonicity."""
    lo = max(rl.catch_K("comb_zeta", mp.mpc(0.5, float(mp.zetazero(n).imag)))
             for n in range(1, 5))               # first 4 zeros (gamma < 31)
    hi = max(rl.catch_K("comb_zeta", mp.mpc(0.5, float(mp.zetazero(n).imag)))
             for n in range(10, 16))             # zeros 10..15 (gamma ~ 50-65)
    assert hi > lo


# --------------------------------------------------------------------------
# fast: vertical migration & the Gram phase  arg(a_1) ~ pi delta_n
# --------------------------------------------------------------------------
def test_vertical_side_up_for_low_zeros():
    """Comb Im a_1 > 0 for the first zeros: the bridge zero descends onto gamma from above
    (vertical_side 'up'). This is the low-gamma uniformity that the Gram-phase finding shows is NOT universal."""
    for g in ZGAM:
        rho = mp.mpc(0.5, g)
        assert rl.disp_coeff("comb_zeta", rho).imag > 0
        assert rl.vertical_side("comb_zeta", rho) == "up"


def test_gram_phase_governs_both_axes():
    """arg(a_1) ~ pi delta_n: sign(Im a_1) = sign((-1)^n sin theta), sign(Re a_1) =
    sign((-1)^n cos theta), AND Im a_1 > 0 <=> delta_n in (0,1) (Gram-regular)."""
    for n in range(1, 13):
        g = mp.zetazero(n).imag
        a1 = rl.disp_coeff("comb_zeta", mp.mpc(0.5, g))
        th = mp.siegeltheta(g)
        assert (a1.imag > 0) == (((-1)**n) * mp.sin(th) > 0)
        assert (a1.real > 0) == (((-1)**n) * mp.cos(th) > 0)
        assert (a1.imag > 0) == (0 < rl.gram_position(g, n) < 1)


def test_axis_duality_warp_im_oscillates():
    """The axis swap: warp Re a_1 < 0 uniformly (horizontal pinned) but warp Im a_1
    takes BOTH signs over the first zeros (vertical oscillates) -- comb is the mirror."""
    warp_im_signs = {rl.disp_coeff("warp_zeta", mp.mpc(0.5, g)).imag > 0 for g in ZGAM}
    assert warp_im_signs == {True, False}                  # genuinely oscillates
    for g in ZGAM:
        assert rl.disp_coeff("warp_zeta", mp.mpc(0.5, g)).real < 0   # Re pinned below


# --------------------------------------------------------------------------
# fast: the continuous-eta sigma_law sub-leading -- exact balance ~ closed form
# --------------------------------------------------------------------------
def test_sigma_law_exact_matches_closed():
    """Solving the F_asym magnitude balance exactly lands within O(1/t^2) of the closed sigma_law."""
    import cont_eta as ce
    for t in (30.0, 60.0, 120.0):
        assert abs(rl.sigma_law_exact(t) - ce.sigma_law(t)) < 5e-4   # the balance refinement is tiny


# --------------------------------------------------------------------------
# fast: loose end 1 -- the C_warp closed form (Si-Gibbs boundary layer)
# --------------------------------------------------------------------------
def test_gibbs_profile_endpoints():
    """P(v) = (1/pi)(pi/2 - Si(2 pi v)): P(0) = 1/2 (edge argument 1), P -> 0 in the bulk."""
    assert abs(rl.gibbs_profile(mp.mpf(0)) - mp.mpf("0.5")) < mp.mpf("1e-25")
    assert abs(rl.gibbs_profile(mp.mpf(50))) < 1e-2          # decays ~cos(2 pi v)/(2 pi^2 v)


def test_c_warp_closed_beats_bulk():
    """The Gibbs boundary layer is real: the closed form is much closer to the numerical truth
    than the bulk alone, and the closed-vs-bulk gap is a sizeable, genuine boundary-layer term."""
    with mp.workdps(20):                                     # the integral needs only ~8 digits here
        s = mp.mpc(2, 0)
        closed = rl.c_warp_closed(s, Vmax=12)
        grid = rl.rate_warp(s)                               # fast grid K=40 (~2-3 digits)
        bulk = rl.rate_warp_bulk(s)
        assert abs(closed - grid) < abs(bulk - grid)         # closed beats bulk against the truth
        assert abs(closed - bulk) > mp.mpf("2e-3")           # a genuine boundary-layer gap


# --------------------------------------------------------------------------
# fast: loose end 2 -- catch_K <-> |zeta'(rho)| (Montgomery a-values)
# --------------------------------------------------------------------------
def test_montgomery_a_is_zeta_prime_abs():
    """montgomery_a(rho) = |zeta'(rho)| (the Montgomery a-value, = |Z'(gamma)| by the Gram-phase finding)."""
    rho = mp.mpc(0.5, 14.134725142)
    assert abs(rl.montgomery_a(rho) - abs(mp.zeta(rho, derivative=1))) < mp.mpf("1e-25")


def test_catch_K_abs_normalized_is_inverse_zeta_prime():
    """catch_K(mode='abs') with the |rho| height-trend divided out == 1/|zeta'(rho)| EXACTLY --
    the pure Montgomery a-value reciprocal; the hardest-to-catch zeros are the small-|zeta'| ones."""
    for g in ZGAM[:4]:
        rho = mp.mpc(0.5, g)
        norm = rl.catch_K_normalized("comb_zeta", rho, mode="abs")
        assert abs(norm - 1.0 / float(rl.montgomery_a(rho))) < 1e-9


def test_catch_K_re_is_abs_times_gram_factor():
    """catch_re/catch_abs == |Re a_1|/|a_1| == |cos arg a_1|, and arg a_1 ~ pi delta_n, so the
    Re-based birth-K carries a SECOND fluctuation beyond 1/|zeta'|: the Gram factor |cos pi delta_n|."""
    for g in ZGAM:                                            # the definitional ratio is exact
        rho = mp.mpc(0.5, g)
        a1 = rl.disp_coeff("comb_zeta", rho)
        ratio = (rl.catch_K("comb_zeta", rho, mode="re")
                 / rl.catch_K("comb_zeta", rho, mode="abs"))
        assert abs(ratio - float(abs(a1.real) / abs(a1))) < 1e-12
    for n in (10, 16, 20):                                    # |cos arg a_1| ~ |cos pi delta_n|
        g = float(mp.zetazero(n).imag)
        a1 = rl.disp_coeff("comb_zeta", mp.mpc(0.5, g))
        cos_arg = float(abs(a1.real) / abs(a1))
        cos_d = float(abs(mp.cos(mp.pi * rl.gram_position(g, n))))
        assert abs(cos_arg - cos_d) < 0.02


def test_catch_K_re_never_exceeds_abs():
    """|Re a_1| <= |a_1|, so the Re-based birth-K is always <= the |a_1|-based one."""
    for g in ZGAM:
        rho = mp.mpc(0.5, g)
        assert (rl.catch_K("comb_zeta", rho, mode="re")
                <= rl.catch_K("comb_zeta", rho, mode="abs") + 1e-12)


def test_catch_K_modes_pick_different_hard_zeros():
    """The Gram factor |cos pi delta_n| makes mode='re' and mode='abs' rank the zeros differently:
    the hardest by pure |zeta'| (abs) is not the hardest once the Gram modulation enters (re)."""
    gammas = [float(mp.zetazero(n).imag) for n in range(1, 13)]
    hardest_abs = max(range(len(gammas)),
                      key=lambda i: rl.catch_K("comb_zeta", mp.mpc(0.5, gammas[i]), mode="abs"))
    hardest_re = max(range(len(gammas)),
                     key=lambda i: rl.catch_K("comb_zeta", mp.mpc(0.5, gammas[i]), mode="re"))
    assert hardest_abs != hardest_re


# --------------------------------------------------------------------------
# fast: loose end 3 -- FE-mirror companions (same heights, broken dynamics)
# --------------------------------------------------------------------------
def test_fe_mirror_shares_heights():
    """The sigma=1 eta prefactor and sigma=0 warp companion sit at the SAME height t_k = 2 pi k/ln2
    (exact s<->1-s images as zero sets -- the p=2 comb)."""
    for k in (1, 2, 3):
        m = rl.fe_mirror(k)
        assert abs(m["t"] - 2 * PI * k / LN2) < 1e-9
        assert abs(float(m["rho_eta"].imag) - float(m["rho_warp"].imag)) < 1e-9
        assert abs(float(m["rho_eta"].real) - 1) < 1e-12
        assert abs(float(m["rho_warp"].real)) < 1e-12


def test_fe_mirror_eta_delta1_is_rigid():
    """eta's Delta_1 = -rho/2 pi^2: Re == -1/2 pi^2 for ALL k (the closed additive-comb constant)."""
    const = -1.0 / (2 * PI**2)
    for k in (1, 2, 3, 4):
        m = rl.fe_mirror(k)
        assert abs(float(m["d1_eta"].real) - const) < 1e-9
        assert abs(m["d1_eta"] - (-m["rho_eta"] / (2 * mp.pi**2))) < mp.mpf("1e-25")


def test_fe_mirror_warp_delta1_wanders():
    """warp's Delta_1 = C_warp wanders: its Re is far from eta's rigid -1/2 pi^2 AND varies with k.
    This is the entire FE-mirror break -- analytic (boundary-layer C_warp), not arithmetic."""
    const = -1.0 / (2 * PI**2)
    res = [float(rl.fe_mirror(k)["d1_warp"].real) for k in (1, 2, 3, 4)]
    for r in res:
        assert abs(r - const) > 0.1                          # far from the rigid eta constant
    assert max(res) - min(res) > 0.2                         # genuinely wanders (0.40 -> 1.64)


def test_fe_mirror_dynamics_not_mirror():
    """The migration a_1's are NOT s<->1-s mirrors: |a1_eta| != |a1_warp| at the shared heights
    (target' = ln2 zeta(rho) is FE-conjugate; the break is Delta_1 = C_warp)."""
    for k in (1, 2, 3):
        m = rl.fe_mirror(k)
        assert abs(abs(m["a1_eta"]) - abs(m["a1_warp"])) > 0.05


def test_fe_mirror_target_deriv_shared_form():
    """Both companions have target' = ln2 * zeta(rho) -- the same functional form, at the
    FE-conjugate arguments 1+it_k (eta) vs it_k (warp)."""
    ln2 = mp.log(2)                                          # full precision (the float LN2 truncates)
    for k in (1, 2):
        m = rl.fe_mirror(k)
        assert abs(m["tprime_eta"] - ln2 * mp.zeta(m["rho_eta"])) < mp.mpf("1e-25")
        assert abs(m["tprime_warp"] - ln2 * mp.zeta(m["rho_warp"])) < mp.mpf("1e-25")


# --------------------------------------------------------------------------
# slow: migrate trajectories, the exact warp constant, the truncation residual
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_disp_predicts_comb_trajectory():
    """a_1/K + a_2/K^2 predicts the hb.migrate comb trajectory -- including gamma=21's overshoot
    (ends BELOW 1/2, Re a_1 < 0) vs gamma=14 (ends above)."""
    for g, ends_below in [(14.134725142, False), (21.022039639, True)]:
        rho = mp.mpc(0.5, g)
        a1, a2 = rl.disp_coeff("comb_zeta", rho), rl.disp_coeff2(rho)
        traj = dict((K, zz) for K, zz in hb.migrate(hb.zeta_K, rho, schedule=SHORT)
                    if zz is not None)
        assert 32 in traj
        actual = float((traj[32] - rho).real)
        pred = float((a1 / 32 + a2 / 32**2).real)
        assert abs(actual - pred) < 5e-3
        assert (actual < 0) == ends_below                 # the overshoot side


@pytest.mark.slow
def test_disp_predicts_warp_trajectory():
    """a_1 = -C_warp/((2^rho-1)zeta') predicts the warp_half_K trajectory, climbing from below."""
    rho = mp.mpc(0.5, 14.134725142)
    a1 = rl.disp_coeff("warp_zeta", rho)
    assert a1.real < 0                                     # from below
    traj = dict((K, zz) for K, zz in hb.migrate(wb.warp_half_K, rho, schedule=SHORT)
                if zz is not None)
    assert 32 in traj
    actual = float((traj[32] - rho).real)
    assert abs(actual - float((a1 / 32).real)) < 1e-2     # leading prediction


@pytest.mark.slow
def test_disp_predicts_warp_companion():
    """warp_pref a_1 predicts the sigma=0 companion (2^s-1=0) trajectory."""
    rho = mp.mpc(0, PREF[0])                               # t ~ 9.0647
    a1 = rl.disp_coeff("warp_pref", rho)
    traj = dict((K, zz) for K, zz in hb.migrate(wb.warp_half_K, rho, schedule=SHORT)
                if zz is not None)
    assert 32 in traj
    actual = float((traj[32] - rho).real)
    assert abs(actual - float((a1 / 32).real)) < 1e-2


@pytest.mark.slow
def test_rate_warp_richardson_matches_grid_sign():
    """The slow exact (Richardson of warp_hurwitz) warp constant agrees with the fast grid
    estimate in sign and to ~5%."""
    for s in [mp.mpc(2, 0), mp.mpc(1.3, 7)]:
        fast = rl.rate_warp(s)                            # grid, K=40
        slow = rl.rate_warp(s, richardson=True)          # exact, extrapolated
        assert abs(fast - slow) < mp.mpf("0.05") * abs(slow) + mp.mpf("2e-3")


@pytest.mark.slow
def test_im_a1_flips_at_gram_failures():
    """The Gram-phase headline: comb Im a_1 > 0 is NOT universal -- it IS Gram's law. It flips negative
    exactly at the first Gram-law failures (n=127,136,196; gamma ~ 282,296,391), where the
    Gram-interval position delta_n leaves (0,1); it stays > 0 for every regular zero below.

    Pure analytic check (disp_coeff at the true zeros) -- no zero-finding seeded by them."""
    for n in (50, 100, 120, 125, 126):                     # all regular -> vertical up
        g = mp.zetazero(n).imag
        assert rl.disp_coeff("comb_zeta", mp.mpc(0.5, g)).imag > 0
        assert 0 < rl.gram_position(g, n) < 1
    for n in (127, 136, 196):                               # Gram-law failures -> flip
        g = mp.zetazero(n).imag
        assert rl.disp_coeff("comb_zeta", mp.mpc(0.5, g)).imag < 0
        assert not (0 < rl.gram_position(g, n) < 1)


@pytest.mark.slow
def test_vertical_migration_matches_im_a1():
    """Post-hoc check (forward-only): the ACTUAL bridge-zero vertical displacement Im(s_K-rho)
    from hb.migrate matches the analytic Im(a_1)/K. migrate is seeded from rho purely to
    VERIFY the prediction, never to inform it (the rate law is derived forward)."""
    rho = mp.mpc(0.5, 14.134725142)
    a1 = rl.disp_coeff("comb_zeta", rho)
    traj = dict((K, zz) for K, zz in hb.migrate(hb.zeta_K, rho, schedule=SHORT)
                if zz is not None)
    assert 32 in traj
    actual = float((traj[32] - rho).imag)
    assert actual > 0                                       # ascends onto gamma from above
    assert abs(actual - float((a1 / 32).imag)) < 5e-3


@pytest.mark.slow
def test_sigma_law_residual_is_truncation():
    """The residual to the TRUE continuous-eta zeros is the F_asym truncation: solving the balance exactly
    (sigma_law_exact) leaves essentially the same residual as the closed sigma_law."""
    import cont_eta as ce
    zeros = ce.find_zeros(60.0)
    for z in zeros:
        if float(z.imag) < 25:
            continue
        t, st = float(z.imag), float(z.real)
        res_law = abs(st - ce.sigma_law(t))
        res_exact = abs(st - rl.sigma_law_exact(t))
        assert abs(res_law - res_exact) < 0.1 * res_law   # refining the balance barely moves it


# --------------------------------------------------------------------------
# slow: loose ends -- the C_warp closed form vs the exact numerical truth; the pair-correlation tie
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_c_warp_closed_matches_richardson():
    """loose end 1: the Si-Gibbs closed form C_warp = rate_warp_bulk + boundary integral matches
    the exact numerical rate_warp(richardson=True) to that truth's own ~5-digit accuracy (the
    closed form is exact; the residual is the Richardson 1/K^2 truncation, larger at high t)."""
    for s in [mp.mpc(2, 0), mp.mpc(1.3, 7), mp.mpc(0.5, 14.134725142)]:
        closed = rl.c_warp_closed(s)
        truth = rl.rate_warp(s, richardson=True)
        assert abs(closed - truth) < mp.mpf("5e-3")


@pytest.mark.slow
def test_c_warp_closed_quadratic_subtraction_is_required():
    """The quadratic counterterm -P^2 zeta_aa'' is NOT optional: dropping it (keeping only the
    linear counterterm, the issue's first guess) double-counts the boundary quadratic and
    overshoots C_warp by a large factor -- so the +bulk/-quadratic bookkeeping is essential."""
    s = mp.mpc(2, 0)
    z32 = mp.zeta(s, mp.mpf(3) / 2)
    D1 = -s * mp.zeta(s + 1, mp.mpf(3) / 2)
    D2 = s * (s + 1) * mp.zeta(s + 2, mp.mpf(3) / 2)

    def naive_integrand(v):                                  # linear counterterm only (no -P^2 D2)
        p = rl.gibbs_profile(v)
        return (mp.zeta(s, mp.mpf(3) / 2 - p) - z32 + p * D1
                + mp.zeta(s, mp.mpf(3) / 2 + p) - z32 - p * D1)

    naive = rl.rate_warp_bulk(s) + mp.quad(naive_integrand, [mp.mpf(j) for j in range(31)])
    truth = rl.rate_warp(s, richardson=True)
    assert abs(rl.c_warp_closed(s) - truth) < mp.mpf("5e-3")  # the correct form is right
    assert abs(naive - truth) > 0.5 * abs(truth)             # the naive form is grossly wrong


@pytest.mark.slow
def test_warp_disp_coeff_closed_matches_grid():
    """disp_coeff with the closed-form Delta_1 (closed=True) reproduces the grid-based
    warp displacement coefficient -- the closed C_warp drives the same migration (~2-3 digit grid)."""
    rho = mp.mpc(0.5, 14.134725142)
    a_grid = rl.disp_coeff("warp_zeta", rho)
    a_closed = rl.disp_coeff("warp_zeta", rho, closed=True)
    assert abs(a_grid - a_closed) < mp.mpf("5e-2")


@pytest.mark.slow
def test_small_zeta_prime_zeros_are_near_degenerate():
    """loose end 2 tie: the hardest-to-catch (small |zeta'|) zeros ARE the near-degenerate
    (closely-spaced) ones -- |zeta'| correlates with the relative neighbour spacing (~0.74 over
    the first 80 zeros), so small |zeta'| <=> a closely-spaced pair (the pair-correlation link)."""
    import numpy as np
    N = 60
    g = [float(mp.zetazero(n).imag) for n in range(1, N + 2)]
    zp, rel = [], []
    for n in range(2, N):
        gamma = g[n - 1]
        gap = min(gamma - g[n - 2], g[n] - gamma)
        mean_sp = 2 * math.pi / math.log(gamma / (2 * math.pi))
        zp.append(float(rl.montgomery_a(mp.mpc(0.5, gamma))))
        rel.append(gap / mean_sp)
    corr = float(np.corrcoef(zp, rel)[0, 1])
    assert corr > 0.4
