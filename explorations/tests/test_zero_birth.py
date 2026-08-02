"""Tests for zero_birth (issue #39). OUTSIDE the default CI suite:
pytest.ini's `testpaths = tests` excludes explorations/; run these with
`pytest explorations/tests`. Lean by design -- the heavy verification (the birth
flow, the census, the migrations) lives in the driver, local/manual only.
"""
import mpmath as mp
import pytest

import character_bridge as cb
import zero_birth as zb


# --------------------------------------------------------------------------
# the seed object and the amplitude family
# --------------------------------------------------------------------------
def test_L0_special_values():
    """L(0, chi) = -(1/q) sum chi(a) a; matches the Hurwitz combination at s=0;
    vanishes for even chi (the s = 0 trivial zero)."""
    assert abs(zb.L0_chi(cb.CHI3) - mp.mpf(1) / 3) < 1e-25
    assert abs(zb.L0_chi(cb.CHI4) - mp.mpf(1) / 2) < 1e-25
    assert abs(zb.L0_chi(cb.CHI5_QUAD)) < 1e-25
    assert abs(zb.L0_chi(cb.CHI3) - cb.L_chi(0, cb.CHI3)) < 1e-20


def test_projector_through_the_evaluator():
    """F_K^lam at lam = 0 is identically zero for non-principal chi -- the
    projector, measured through the actual warp machinery (not just algebra)."""
    for s in (mp.mpc(1.3, 7), mp.mpc(0.5, 6)):
        assert abs(zb.F_lam(s, 0, 0, cb.CHI3)) < 1e-24
        assert abs(zb.F_lam(s, 0, 0, cb.CHI5_I)) < 1e-24


def test_seed_tangent_linear_and_K_independent():
    """F_K^lam/lam -> D(s) linearly in lam, with the SAME limit at K = 1 and 2."""
    s = mp.mpc(2, 1)
    D = zb.seed_D(s, cb.CHI3)
    e2 = abs(zb.G_lam(s, 1, mp.mpf("0.01"), cb.CHI3) - D)
    e3 = abs(zb.G_lam(s, 1, mp.mpf("0.001"), cb.CHI3) - D)
    assert 8 < float(e2 / e3) < 12                     # linear in lam
    e2b = abs(zb.G_lam(s, 2, mp.mpf("0.01"), cb.CHI3) - D)
    assert float(e2b) < 3 * float(e2)                  # K-independent tangent


def test_lam1_is_the_raw_warp_family():
    """lam = 1 reproduces character_bridge.L_warp_K exactly (same machinery,
    same psis): the K >= 1 family is convention-free."""
    s = mp.mpc(2, 1)
    for K in (1, 3):
        assert abs(zb.F_lam(s, K, 1, cb.CHI3)
                   - cb.L_warp_K(s, K, cb.CHI3)) < 1e-22


def test_seed_geometry_first_order():
    """The first seed of chi_3 sits within O((L0/ln p)^2) of the displaced-comb
    prediction i t_1 - 3^{-i t_1} L0 / ln 2."""
    pred = zb.seed_prediction(cb.CHI3, 1)
    z = mp.findroot(lambda s: zb.seed_D(s, cb.CHI3), pred)
    assert abs(z - pred) < 0.15
    assert abs(zb.seed_D(z, cb.CHI3)) < 1e-20


def test_conductor_string_prediction():
    """The odd-chi conductor seed string: sigma* = ln L0 / ln(q/p), spacing
    2 pi / ln(q/p), and a genuine D-zero sits there (q = 3: -2.87 + 15.21i)."""
    sig_c, spacing = zb.conductor_string(cb.CHI3)
    assert abs(sig_c - (-2.7095)) < 0.001
    assert abs(spacing - 15.4962) < 0.001
    z = mp.findroot(lambda s: zb.seed_D(s, cb.CHI3), mp.mpc(sig_c, spacing))
    assert abs(z - mp.mpc(-2.869, 15.211)) < 0.01
    assert abs(zb.seed_D(z, cb.CHI3)) < 1e-20


def test_rival_tangent_degenerate():
    """The cell-location rival tangent (1+s) q^{-s} L0: identically zero for even
    chi, zero only at s = -1 for odd chi -- no seed string either way."""
    s = mp.mpc(0.3, 11)
    assert abs(zb.rival_tangent(s, cb.CHI5_QUAD)) < 1e-25
    assert abs(zb.rival_tangent(s, cb.CHI3)) > 1e-3
    assert abs(zb.rival_tangent(mp.mpc(-1), cb.CHI3)) < 1e-25


# --------------------------------------------------------------------------
# census instrument
# --------------------------------------------------------------------------
def test_winding_count_known_zeros():
    """Argument-principle box count on a polynomial with known roots."""
    def f(s):
        return (s - mp.mpc(0.3, 5)) * (s - mp.mpc(-0.2, 12)) * (s - mp.mpc(3, 40))
    assert zb.winding_count(f, (-1.0, 2.0), (1.0, 20.0)) == 2
    assert zb.winding_count(f, (-1.0, 4.0), (1.0, 50.0)) == 3


def test_theta_chi_matches_chi6_module():
    """The general primitive-chi theta reduces to chi6_two_component.theta3."""
    import chi6_two_component as c6
    for t in (10.0, 100.0):
        assert abs(float(zb.theta_chi(t, cb.CHI3)) - float(c6.theta3(t))) < 1e-9


# --------------------------------------------------------------------------
# ground-string geometry
# --------------------------------------------------------------------------
def test_axis_roots_are_ground_zeros():
    """Every phase-equation root is a genuine E_chi zero, exactly on sigma = 0."""
    for chi in (cb.CHI3, cb.CHI4, cb.CHI6):
        roots = zb.axis_roots(chi.q, 25.0)
        assert roots                                    # nonempty
        for t in roots[:3]:
            assert abs(cb.comb_endpoint(mp.mpc(0, t), chi)) < 1e-18


def test_certify_axis_chi3():
    """Box count == axis-root count: the whole chi_3 ground string is on the axis
    (2 roots below t ~ 20)."""
    nb, na = zb.certify_axis(cb.CHI3, (0.53, 20.29))
    assert nb == na == 2


def test_section_skeleton_phi2():
    """For phi(q) = 2, B = 1 - p^{-s}: zeros exactly the sigma = 0 comb."""
    t1 = 2 * mp.pi / mp.log(2)
    assert abs(zb.section_B(mp.mpc(0, t1), cb.CHI3)) < 1e-25


# --------------------------------------------------------------------------
# displacement law
# --------------------------------------------------------------------------
def test_disp_coeff_chi_matches_measurement():
    """K(s_K - rho) at K = 40 lands within the O(1/K) remainder of a_1."""
    rho = cb.polish_zero(cb.CHI3, cb.CHI3_ZEROS[0])
    a1 = zb.disp_coeff_chi(rho, cb.CHI3)
    m = zb.measured_disp(rho, cb.CHI3, 40)
    assert abs(m - a1) / abs(a1) < 0.15


def test_migrate_robust_chi6_tooth():
    """The K = 45 -> 35 tooth step that stalls hb.migrate's plain-secant root
    step converges under the Muller local triple (the chi_6 lock walk)."""
    traj = zb.migrate_robust(lambda ss, K: cb.L_comb_K(ss, K, cb.CHI6),
                             mp.mpc(0, 4.532360141827194), [27, 35, 45])
    d = {K: z for K, z in traj if z is not None}
    assert 35 in d and 27 in d
    assert abs(float(d[45].imag) - 4.549) < 0.01


# --------------------------------------------------------------------------
# the birth flow (slow: findroot chains through the lam schedule)
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_birth_flow_lands_on_seed():
    """A raw K = 1 zero of the chi_3 family flows onto a seed (a zero of D) as
    lam -> 0, and the flow never jumps."""
    z0 = mp.findroot(lambda s: zb.F_lam(s, 1, 1, cb.CHI3), mp.mpc(0.34, 8.26))
    traj = zb.flow_lambda(z0, 1, cb.CHI3)
    lam_end, z_end = traj[-1]
    assert lam_end == 0.0 and z_end is not None
    assert abs(zb.seed_D(z_end, cb.CHI3)) < 1e-8
    assert abs(z_end - mp.mpc(0.340, 8.914)) < 0.01     # the m = 1 main seed
