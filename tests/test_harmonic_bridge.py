"""Unit tests for the harmonic interpolation zeta^{(K)} / eta^{(K)}.

The additive-comb stage of the bridge: restore discreteness one Fourier harmonic at a time and watch
the zeros migrate. The interpolants run from a bland/ground endpoint (K=0) to the
genuine Dirichlet object (K -> infinity). The checks:

* the incomplete-gamma moments C/S match oscillatory quadrature;
* both endpoints are exact -- zeta_K(.,0) = 1/(s-1)+1/2, eta_K(.,0) = -F(s)+1/2
  (the continuous-eta object + the load-bearing n=1 half-weight);
* zeta_K / eta_K (and the Bose twins) converge to zeta / eta as K grows;
* the two zeta routes (Euler-Maclaurin sawtooth, Abel-Plana Bose) are identical
  *term by term* -- one comb in two integral guises;
* the migration map: zeta zeros are born off-line at K=1 and slide onto Re=1/2;
  eta zeros split from the K=0 ground string onto Re=1/2 (zeta) and Re=1 (prefactor).
"""
from __future__ import annotations

import mpmath as mp

import cont_eta as ce
import harmonic_bridge as hb

# short, fast continuation schedule for the migration tests
SHORT = [1, 2, 4, 8, 16, 32]


def test_moments_match_quadosc():
    """C/S moments == oscillatory quadrature; Cmom(pi, s) is exactly F(s)."""
    for (w, a) in [(2 * mp.pi, mp.mpc(2, 1)), (3 * mp.pi, mp.mpc(0.7, 10))]:
        per = 2 * mp.pi / w
        cq = mp.quadosc(lambda x: mp.cos(w * x) * x ** (-a), [1, mp.inf], period=per)
        sq = mp.quadosc(lambda x: mp.sin(w * x) * x ** (-a), [1, mp.inf], period=per)
        assert abs(hb.Cmom(w, a) - cq) < mp.mpf("1e-20")
        assert abs(hb.Smom(w, a) - sq) < mp.mpf("1e-20")
    # the continuous-eta identity Cmom(pi, s) == F_closed(s)
    s = mp.mpc(1.8, 25)
    assert abs(hb.Cmom(mp.pi, s) - ce.F_closed(s)) < mp.mpf("1e-25")


def test_zeta_endpoint():
    """zeta_K(., 0) is the bland continuous endpoint 1/(s-1) + 1/2."""
    for s in [mp.mpc(2, 0), mp.mpc(0.5, 14), mp.mpc(1.3, 7)]:
        assert abs(hb.zeta_K(s, 0) - (1 / (s - 1) + mp.mpf("0.5"))) < mp.mpf("1e-25")


def test_eta_endpoint_is_minus_F_plus_half():
    """eta_K(., 0) is exactly -F(s) + 1/2 (the continuous-eta object + n=1 half-weight)."""
    for s in [mp.mpc(0.5, 14), mp.mpc(1.8, 12), mp.mpc(1.3, 7)]:
        assert abs(hb.eta_K(s, 0) - (-ce.F_closed(s) + mp.mpf("0.5"))) < mp.mpf("1e-25")


def test_half_weight_relocates_the_ground_zeros():
    """At a continuous-eta F-zero (F=0) the eta ground state equals +1/2, not 0 -- so the
    ground string is NOT the continuous-eta string; the half-weight has moved it."""
    z_fzero = ce.find_zeros(20.0)[0]            # first F-zero, t ~ 10.67, Re ~ 2.08
    assert abs(ce.F_closed(z_fzero)) < 1e-18    # it is a zero of F
    assert abs(hb.eta_K(z_fzero, 0) - mp.mpf("0.5")) < 1e-15   # ... but eta_K(.,0) = 1/2 there


def test_zeta_converges_to_zeta():
    """zeta_K -> zeta as K grows (tightening), to ~1e-2 by K=40."""
    for s in [mp.mpc(2, 0), mp.mpc(0.5, 14.134725), mp.mpc(1.3, 7)]:
        z = mp.zeta(s)
        e5 = abs(hb.zeta_K(s, 5) - z)
        e40 = abs(hb.zeta_K(s, 40) - z)
        assert e40 < e5
        assert e40 < mp.mpf("4e-2")


def test_eta_converges_to_eta():
    """eta_K -> eta (= altzeta) as K grows."""
    for s in [mp.mpc(2, 0), mp.mpc(0.5, 14.134725), mp.mpc(1.3, 7)]:
        e = mp.altzeta(s)
        e5 = abs(hb.eta_K(s, 5) - e)
        e40 = abs(hb.eta_K(s, 40) - e)
        assert e40 < e5
        assert e40 < mp.mpf("4e-2")


def test_routes_identical_term_by_term():
    """The headline unification: the k-th sawtooth mode == the k-th Bose mode."""
    for s in [mp.mpc(2, 0), mp.mpc(0.5, 14.13), mp.mpc(1.3, 7)]:
        for k in (1, 2, 5):
            assert abs(hb.zeta_term(s, k) - hb.zeta_term_bose(s, k)) < mp.mpf("1e-20")


def test_eta_bose_twin_converges():
    """The odd-pi Abel-Plana comb (route b) also -> eta, by a different anchor."""
    s = mp.mpc(1.3, 7)
    e = mp.altzeta(s)
    e3 = abs(hb.eta_K_bose(s, 3) - e)
    e15 = abs(hb.eta_K_bose(s, 15) - e)
    assert e15 < e3
    assert e15 < mp.mpf("8e-2")


def test_zeta_zero_born_then_migrates_to_half():
    """zeta zero 1/2+14.13i: absent at K=0 (born off-line at K=1), slides onto 1/2."""
    traj = hb.migrate(hb.zeta_K, mp.mpc(0.5, 14.134725), schedule=SHORT)
    d = {K: zz for K, zz in traj if zz is not None}
    assert 1 in d and 32 in d
    assert float(d[1].real) > 0.6            # born well off the critical line
    assert abs(float(d[32].real) - 0.5) < 0.02   # has migrated onto it
    # the zeroless endpoint: no K=0 zero to continue to
    full = hb.migrate(hb.zeta_K, mp.mpc(0.5, 14.134725), schedule=[0] + SHORT)
    assert dict(full).get(0) is None


def test_eta_zeros_split_to_two_lines():
    """From the K=0 ground string, eta zeros go to Re=1/2 (zeta) or Re=1 (prefactor)."""
    crit = hb.migrate(hb.eta_K, mp.mpc(0.5, 14.134725), schedule=SHORT)
    pref_t = 2 * float(mp.pi) / float(mp.log(2))      # ~ 9.0647
    pref = hb.migrate(hb.eta_K, mp.mpc(1.0, pref_t), schedule=SHORT)
    c, p = dict(crit), dict(pref)
    assert abs(float(c[32].real) - 0.5) < 0.02       # -> critical line
    assert abs(float(p[32].real) - 1.0) < 0.02       # -> prefactor line
    assert float(p[32].real) > 0.8                   # distinct destinations


def test_eta_ground_string_below_fzeros():
    """Continued to K=0 the eta zeros sit near Re~0.7 -- far below the continuous-eta Re~1.8
    string -- confirming the half-weight pulled the ground state toward criticality."""
    traj = hb.migrate(hb.eta_K, mp.mpc(0.5, 14.134725), schedule=[0] + SHORT)
    g = dict(traj).get(0)
    assert g is not None
    assert float(g.real) < 1.0               # ground zero well below the continuous-eta ~1.8
    assert abs(hb.eta_K(g, 0)) < 1e-12       # it really is a ground-state zero
