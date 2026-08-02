"""Tests for character_bridge (issue #37). OUTSIDE the default CI suite:
pytest.ini's `testpaths = tests` excludes explorations/; run these with
`pytest explorations/tests`. Keep them lean regardless -- the heavy verification
(migrations, box scans) lives in the driver, which is local/manual only.
"""
import mpmath as mp
import pytest

import character_bridge as cb
import harmonic_bridge as hb
import lfunction_bridge as lfb


# --------------------------------------------------------------------------
# characters
# --------------------------------------------------------------------------
def test_character_axioms():
    """Multiplicativity + periodicity + parity/conductor bookkeeping for the cast."""
    for chi in cb.CAST:
        us = cb.units(chi.q)
        for a in us:
            assert abs(chi(a + chi.q) - chi(a)) < 1e-25          # periodicity
            for b in us:
                assert abs(chi(a * b) - chi(a) * chi(b)) < 1e-25  # multiplicativity
        for n in range(chi.q):
            if n and cb.math.gcd(n, chi.q) > 1:
                assert chi(n) == 0
    assert (cb.CHI3.parity, cb.CHI4.parity, cb.CHI5_I.parity) == (1, 1, 1)
    assert cb.CHI5_QUAD.parity == 0
    assert cb.CHI3.primitive and cb.CHI5_I.primitive
    assert not cb.CHI6.primitive and cb.CHI6.conductor == 3
    assert cb.CHI5_I.conj(2) == mp.mpc(0, -1) and cb.CHI5_I(2) == mp.mpc(0, 1)


def test_orthogonality_and_projector():
    """sum_a chi(a) = 0 for every non-principal chi -- the prediction-3 projector."""
    for q in (3, 4, 5, 6, 7):
        chars = [cb.induced(cb.CHI3, 6)] if q == 6 else cb.characters(q)
        for chi in chars:
            if not chi.principal:
                assert cb.projector_residual(chi) < 1e-25
    assert cb.projector_residual(cb.characters(5)[0]) == pytest.approx(4.0)


# --------------------------------------------------------------------------
# L(s, chi) evaluator + anchors
# --------------------------------------------------------------------------
def test_L_special_values():
    assert abs(cb.L_chi(1, cb.CHI4) - mp.pi / 4) < 1e-25
    assert abs(cb.L_chi(2, cb.CHI4) - mp.catalan) < 1e-25
    assert abs(cb.L_chi(1, cb.CHI3) - mp.pi / (3 * mp.sqrt(3))) < 1e-25


def test_L_against_direct_sum_and_euler():
    s = mp.mpc(2.5, 1)
    for chi in [cb.CHI3, cb.CHI5_I, cb.CHI6]:
        ref = cb.L_sum(s, chi, terms=30000)
        assert abs(cb.L_chi(s, chi) - ref) / abs(ref) < 1e-8
    e = cb.L_euler(mp.mpc(3), cb.CHI5_I, pmax=300)
    assert abs(cb.L_chi(mp.mpc(3), cb.CHI5_I) - e) / abs(e) < 1e-6


def test_functional_equation_primitive():
    """Lambda(s) = eps Lambda-bar(1-s) with |eps| = 1, including the COMPLEX chi_5."""
    for chi in [cb.CHI3, cb.CHI4, cb.CHI5_I, cb.CHI5_QUAD]:
        assert abs(abs(cb.root_number(chi)) - 1) < 1e-25
        for s in [mp.mpc(2, 1), mp.mpc(0.5, 5)]:
            assert cb.functional_eq_residual(s, chi) < 1e-20


def test_chi4_matches_lfunction_bridge():
    """The q=4 spine identity reproduces lfunction_bridge's L_chi4 exactly."""
    for s in [mp.mpc(2, 0), mp.mpc(0.5, 6.020948), mp.mpc(1.3, 7)]:
        assert abs(cb.L_chi(s, cb.CHI4) - lfb.L_chi4(s)) < 1e-24


def test_imprimitive_factorization():
    """L(s, chi_6) = (1 + 2^{-s}) L(s, chi_3); Euler comb zeros at sigma=0."""
    for s in [mp.mpc(1.3, 7), mp.mpc(0.5, 10), mp.mpc(2, 1)]:
        lhs = cb.L_chi(s, cb.CHI6)
        rhs = (1 + mp.power(2, -s)) * cb.L_chi(s, cb.CHI3)
        assert abs(lhs - rhs) < 1e-24
    t0 = mp.pi / mp.log(2)                    # first prefactor zero, exact height
    assert abs(cb.L_chi(mp.mpc(0, t0), cb.CHI6)) < 1e-24


def test_q2_precedent_identity():
    """2^{-s} zeta(s,1/2) = (1-2^{-s}) zeta(s) = L(s, chi_0 mod 2): the in-repo case."""
    s = mp.mpc(1.3, 7)
    assert abs(mp.power(2, -s) * mp.zeta(s, mp.mpf("0.5"))
               - (1 - mp.power(2, -s)) * mp.zeta(s)) < 1e-25


# --------------------------------------------------------------------------
# route A: the residue-class comb
# --------------------------------------------------------------------------
def test_comb_reduces_to_zeta_K_at_alpha_1():
    """zeta_comb_K(s, K, 1) == harmonic_bridge.zeta_K(s, K) term by term."""
    for s in [mp.mpc(2, 1), mp.mpc(0.5, 14.13), mp.mpc(-0.5, 3)]:
        assert abs(cb.zeta_comb_K(s, 12, 1) - hb.zeta_K(s, 12)) < 1e-24


def test_emom_lower_limit_against_quadrature():
    """The lower-limit moment identity, once, at moderate precision."""
    w, a, al = 2 * mp.pi * 3, mp.mpc(2.3, 1.1), mp.mpf(1) / 3
    quad = mp.quadosc(lambda y: mp.exp(1j * w * y) * y ** (-a), [al, mp.inf],
                      period=float(2 * mp.pi / w))
    assert abs(cb.Emom_a(w, a, al) - quad) < 1e-20


def test_hurwitz_comb_converges_with_alpha_rate():
    """O(1/K) with the alpha-generalized constant (s/2pi^2) alpha^{-s-1}."""
    s = mp.mpc(1.3, 7)
    for al in [mp.mpf(1) / 3, mp.mpf(2) / 5]:
        e40 = abs(cb.zeta_comb_K(s, 40, al) - mp.zeta(s, al))
        e80 = abs(cb.zeta_comb_K(s, 80, al) - mp.zeta(s, al))
        assert 1.8 < float(e40 / e80) < 2.2                      # O(1/K)
        pred = abs(cb.rate_comb_alpha(s, al))
        assert abs(float(80 * e80 / pred) - 1) < 0.05


def test_L_comb_converges_with_chi_rate():
    """L_comb_K -> L with the q-dependent constant (sq/2pi^2) sum chi(a) a^{-s-1}."""
    s = mp.mpc(1.3, 7)
    for chi in [cb.CHI3, cb.CHI5_I, cb.CHI6]:
        L = cb.L_chi(s, chi)
        e60 = abs(cb.L_comb_K(s, 60, chi) - L)
        pred = abs(cb.rate_comb_chi(s, chi))
        assert abs(float(60 * e60 / pred) - 1) < 0.05


def test_comb_endpoint_entire_and_ground_string():
    """E_chi finite through s=1 (non-principal); phi(q)=2 ground zeros EXACTLY sigma=0."""
    for chi in [cb.CHI3, cb.CHI5_I]:
        vals = [cb.comb_endpoint(mp.mpc(1) + mp.mpf("1e-8") * d, chi) for d in (1, -1)]
        assert all(mp.isfinite(v) and abs(v) < 10 for v in vals)
        assert abs(vals[0] - vals[1]) < 1e-6                     # no jump across s=1
    with mp.workdps(40):
        z = mp.findroot(lambda s: cb.comb_endpoint(s, cb.CHI3), mp.mpc(0.0, 8.957))
        assert abs(z.real) < mp.mpf("1e-35")                     # exactly on sigma=0
        z5 = mp.findroot(lambda s: cb.comb_endpoint(s, cb.CHI5_I), mp.mpc(0.3, 7.3))
        assert abs(z5.real) > 0.05                               # phi(q)>2: scattered


# --------------------------------------------------------------------------
# route B: the warp combination
# --------------------------------------------------------------------------
def test_warp_combination_converges():
    """L_warp_K -> L(s, chi): the linear combination of warp_alpha objects."""
    s = mp.mpc(2, 1)
    L = cb.L_chi(s, cb.CHI3)
    e8 = abs(cb.L_warp_K(s, 8, cb.CHI3) - L)
    e16 = abs(cb.L_warp_K(s, 16, cb.CHI3) - L)
    assert float(e16) < float(e8) < 1e-2
    # DC-always-on endpoint: small, entire, NOT zero
    edc = cb.warp_endpoint_dc(s, cb.CHI3)
    assert 1e-6 < abs(edc) < 10


# --------------------------------------------------------------------------
# zero data
# --------------------------------------------------------------------------
def test_stored_zero_heights_polish():
    """Every stored migration target re-polishes onto its line."""
    for chi, zs in [(cb.CHI3, cb.CHI3_ZEROS[:2]), (cb.CHI5_I, cb.CHI5_I_ZEROS[:2])]:
        for t in zs:
            z = cb.polish_zero(chi, t)
            assert z is not None and abs(float(z.real) - 0.5) < 1e-10


@pytest.mark.slow
def test_migration_lands_on_half():
    """chi_3 first zero: the comb migration reaches sigma ~ 1/2 by K=45 (O(1/K))."""
    traj = hb.migrate(lambda ss, K: cb.L_comb_K(ss, K, cb.CHI3),
                      mp.mpc(0.5, cb.CHI3_ZEROS[0]), schedule=cb.K_SCHEDULE)
    d = {K: z for K, z in traj if z is not None}
    assert 45 in d and abs(float(d[45].real) - 0.5) < 0.01
    assert abs(float(d[1].real) - 0.5) > 0.2                     # born far off-line
