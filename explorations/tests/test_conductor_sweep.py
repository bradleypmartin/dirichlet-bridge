"""Tests for conductor_sweep + the characters_any CRT constructor (issue #40).
OUTSIDE the default CI suite (pytest.ini's `testpaths = tests`); run with
`pytest explorations/tests`. Lean by design -- the heavy verification (the full
q <= 50 sweep, the 12-conductor walks) lives in the driver, local/manual only.
"""
import math

import mpmath as mp
import pytest

import character_bridge as cb
import conductor_sweep as cs
import rate_law as rl
import zero_birth as zb


# --------------------------------------------------------------------------
# characters_any: the CRT constructor (the issue #40 build item)
# --------------------------------------------------------------------------
def _mobius(n):
    out = 1
    for _, e in cb._factorize(n):
        if e > 1:
            return 0
        out = -out
    return out


def test_counts_and_primitive_counts():
    """|chars| = phi(q) and #(conductor == q) = (mu * phi)(q) for q <= 36."""
    for q in range(3, 37):
        chars = cb.characters_any(q)
        phi = len(cb.units(q))
        assert len(chars) == phi, q
        want = sum(_mobius(d) * len(cb.units(q // d))
                   for d in range(1, q + 1) if q % d == 0)
        got = sum(1 for c in chars if c.conductor == q)
        assert got == want, q
    # q = 2 mod 4 has no primitive characters at all
    for q in (6, 10, 14, 22):
        assert cs.primitive_characters(q) == []


def test_cyclic_moduli_reproduce_characters():
    """Same character SET as the old cyclic-only constructor."""
    for q in (3, 4, 5, 7, 9, 18):
        us = cb.units(q)

        def key(c):
            return tuple((round(complex(c(a)).real, 12),
                          round(complex(c(a)).imag, 12)) for a in us)

        assert sorted(key(c) for c in cb.characters(q)) == \
            sorted(key(c) for c in cb.characters_any(q))


def test_non_cyclic_structure():
    """q = 8: conductors {1,4,8,8}, one even + one odd primitive; q = 12: {1,3,4,12};
    exact multiplicativity and orthogonality at q = 16."""
    ch8 = cb.characters_any(8)
    assert sorted(c.conductor for c in ch8) == [1, 4, 8, 8]
    assert sorted(c.parity for c in ch8 if c.conductor == 8) == [0, 1]
    assert sorted(c.conductor for c in cb.characters_any(12)) == [1, 3, 4, 12]
    ch16 = cb.characters_any(16)
    us = cb.units(16)
    for ci in ch16:
        for a in us[:4]:
            for b in us:
                assert abs(ci(a * b) - ci(a) * ci(b)) < 1e-28
    for i, ci in enumerate(ch16):
        for j, cj in enumerate(ch16):
            ip = mp.fsum(ci(a) * mp.conj(cj(a)) for a in us)
            assert abs(ip - (len(us) if i == j else 0)) < 1e-28


def test_functional_equation_at_new_moduli():
    """L machinery runs off characters_any objects: |eps| = 1 and Lambda(s) =
    eps Lambda-bar(1-s) at the non-cyclic moduli."""
    for q in (8, 12, 15):
        for chi in cs.primitive_characters(q):
            assert abs(abs(cb.root_number(chi)) - 1) < 1e-25
            assert cb.functional_eq_residual(mp.mpc(2, 1), chi) < 1e-20


# --------------------------------------------------------------------------
# the rate measurement
# --------------------------------------------------------------------------
def test_rate_measured_vs_predicted_q8():
    """The sweep cell at the first non-cyclic modulus: rel resid < 1e-3."""
    rows = cs.measure_rate_q(8)
    assert len(rows) == 2
    for chi, pred, meas, rel, raw, c2n in rows:
        assert rel < 1e-3
        assert raw < 3e-2
        assert 0.1 < c2n < 2.0        # the q^2-scaled second-order coefficient


def test_group_mean_identity():
    """(1/phi(q)) sum_chi rate_comb_chi = q s/2pi^2 exactly (orthogonality)."""
    for q in (8, 15):
        assert cs.mean_rate_identity_residual(q) < 1e-25


def test_section_tier():
    """p_star and the tier scale |B - 1| ~ p*^{-sigma0-1}."""
    assert cs.p_star(15) == 2 and cs.p_star(16) == 3 and cs.p_star(48) == 5
    for q, lo, hi in ((15, 0.05, 0.6), (16, 0.02, 0.25), (48, 0.005, 0.08)):
        for chi in cs.primitive_characters(q):
            dev = abs(complex(cs.section_factor(chi)) - 1)
            assert lo < dev < hi, (q, chi.label, dev)


# --------------------------------------------------------------------------
# the generalized Hardy-Z walk
# --------------------------------------------------------------------------
def test_hardy_z_real_and_walk_census_q8():
    """Z_chi real (eps = +1, Gauss), and the short walk's census matches
    theta_chi/pi with NO +1 (entire L)."""
    chi = cs.real_primitive(8)
    assert chi.parity == 0                      # even preferred at q = 8
    assert abs(cb.root_number(chi) - 1) < 1e-25
    with mp.workdps(20):
        for t in (3.3, 8.1):
            v = mp.expj(zb.theta_chi(mp.mpf(t), chi)) \
                * cb.L_chi(mp.mpc("0.5", t), chi)
            assert abs(mp.im(v)) < 1e-15
    zeros = cs.walk_zeros(chi, t_max=16.0)
    assert abs(cs.census_residual(chi, zeros, 16.0)) < 1.0
    # spot-verify one walked height is a true zero of L itself
    rho = mp.findroot(lambda ss: cb.L_chi(ss, chi), mp.mpc("0.5", zeros[0]))
    assert abs(float(rho.real) - 0.5) < 1e-10
    assert abs(float(rho.imag) - zeros[0]) < 1e-6


def test_a1_decomposition():
    """|a_1| = (q|rho|/2pi^2) |B(rho+1)| / |L'| -- the three stored factors agree."""
    chi = cs.real_primitive(5)
    g = 6.648452945                            # CHI5_QUAD_ZEROS[0], reproduced blind
    a1, absLp, absB = cs.a1_at(g, chi)
    rho = mp.mpc("0.5", g)
    lhs = float(abs(a1))
    rhs = 5 * float(abs(rl.rate_comb(rho))) * absB / absLp
    assert abs(lhs - rhs) / lhs < 1e-6


@pytest.mark.slow
def test_displacement_transfers_to_new_modulus():
    """measured K(s_K - rho) vs a_1 at q = 8 (constructor -> comb -> displacement,
    end to end at a non-cyclic modulus)."""
    chi = cs.real_primitive(8)
    zeros = cs.walk_zeros(chi, t_max=9.0)
    rho = mp.findroot(lambda ss: cb.L_chi(ss, chi), mp.mpc("0.5", zeros[0]))
    a1 = zb.disp_coeff_chi(rho, chi)
    m25 = zb.measured_disp(rho, chi, 25)
    assert abs(m25 - a1) / abs(a1) < 0.12


@pytest.mark.slow
def test_census_instant_at_q8():
    """The #39 instant census generalizes: N_1 == target for chi_8."""
    target, counts = cs.census_at(cs.real_primitive(8), K_list=(1,),
                                  box=((-0.97, 2.03), (0.53, 20.29)))
    assert counts[1] == target
