"""Unit tests for the Epstein / lattice-zeta reverse-direction foil (epstein_zeros.py).

The sharpest single foil in the literature for the bridge's directional claim: Travenec-Samaj
(2022) use the spatial dimension d as a continuous knob on the hypercubic Epstein zeta
Z_d(s) = sum'_{n in Z^d} |n|^{-2s} and watch a zero pair get BORN OFF the critical line
Re s = d/4 and migrate AWAY -- the opposite arrow to our zeroless -> born -> ONTO 1/2. The checks:

* evaluator: Z_d(s) via the Terras/theta completed formula matches the direct lattice sum
  (Re s > d/2) and the closed forms Z_1(s) = 2 zeta(2s), Z_2(s) = 4 zeta(s) beta(s);
* the Epstein functional equation Lambda_d(s) = Lambda_d(d/2-s) and the pole residue
  Res_{s=d/2} Z_d = pi^{d/2}/Gamma(d/2);
* the critical dimension d* = 9.24555... as the root of g(d) = Lambda_d(d/4);
* (slow) the pitchfork: the on-line branch t_1(d) slides to 0 as d -> d*, and for d > d* the
  real pair sigma_-(d) -> 0, sigma_+(d) -> d/2 migrates to the strip boundaries;
* (slow) an independent argument-principle count: 0 real zeros in the strip for d < d*, 2 for d > d*;
* the contrast with the bridge's K-knob (harmonic_bridge.zeta_K): the zeta_K zero climbs ONTO 1/2
  -- born from a zeroless endpoint and arriving on the line, the reverse of the Epstein pair.

The zero-tracking / argument-principle checks (~40 mpmath quadratures each) are `slow`; the fast
suite keeps the evaluator, the closed forms, the functional equation, the residue, d*, and the
K-knob contrast.
"""
import mpmath as mp
import pytest

import epstein_zeros as ez

mp.mp.dps = 30


# --------------------------------------------------------------------------
# fast: the evaluator, closed forms, functional equation, residue
# --------------------------------------------------------------------------
def test_terras_vs_direct_sum():
    """Z_d(s) (Terras completed formula) == the direct lattice sum for Re s > d/2."""
    assert abs((ez.epstein_Z(mp.mpc(3, 0), 2) - ez.epstein_Z_direct(mp.mpc(3, 0), 2, 60))
               / ez.epstein_Z(mp.mpc(3, 0), 2)) < mp.mpf("1e-6")
    assert abs((ez.epstein_Z(mp.mpc(3, 0), 3) - ez.epstein_Z_direct(mp.mpc(3, 0), 3, 30))
               / ez.epstein_Z(mp.mpc(3, 0), 3)) < mp.mpf("1e-4")


def test_closed_form_d1():
    """Z_1(s) = 2 zeta(2s) (the 1D lattice {+/- n})."""
    for s in [mp.mpc(1.2, 0.4), mp.mpc(0.9, 3.0), mp.mpc(-0.5, 2.0)]:
        assert abs(ez.epstein_Z(s, 1) - ez.Z1_closed(s)) < mp.mpf("1e-25")


def test_closed_form_d2():
    """Z_2(s) = 4 zeta(s) beta(s) (Gaussian integers; beta = L(s, chi_4))."""
    for s in [mp.mpc(1.6, 0.3), mp.mpc(1.2, 2.0), mp.mpc(0.7, 5.0)]:
        assert abs(ez.epstein_Z(s, 2) - ez.Z2_closed(s)) < mp.mpf("1e-25")


def test_functional_equation():
    """Lambda_d(s) = Lambda_d(d/2 - s) -- zeros symmetric about the critical line Re s = d/4."""
    worst = 0.0
    for d in [3, 5.5, 9, 12]:
        for s in [mp.mpc(0.7, 2.3), mp.mpc(-1.0, 4.0)]:
            worst = max(worst, float(ez.functional_eq_residual(s, d)))
    assert worst < 1e-25


def test_pole_residue():
    """Res_{s=d/2} Z_d(s) = pi^{d/2}/Gamma(d/2) (the leading Weyl/volume term), via 2-point
    Richardson on e*Z_d(d/2+e) (mp.limit near the pole is ~600x slower)."""
    for d in [2, 3, 5, 9.5]:
        assert abs(ez.pole_residue_numeric(d) - ez.pole_residue(d)) < mp.mpf("1e-6")


def test_critical_dimension():
    """d* = root of g(d) = Lambda_d(d/4) reproduces Travenec-Samaj's 9.24555..."""
    dstar = ez.critical_dimension()
    assert abs(float(dstar) - 9.2455524667) < 1e-8
    assert abs(float(ez.center_value(dstar))) < 1e-20        # g(d*) vanishes


def test_lambda_real_on_critical_line():
    """Lambda_d is real on the critical line s = d/4 + iy (functional eq + Schwarz), so its
    on-line zeros are real-root problems -- what crit_line_value / lowest_critical_zero use."""
    assert abs(mp.im(ez.epstein_Lambda(mp.mpf(7) / 4, 7))) < mp.mpf("1e-25")      # y = 0
    assert abs(mp.im(ez.epstein_Lambda(mp.mpc(mp.mpf(7) / 4, 2.5), 7))) < mp.mpf("1e-25")


# --------------------------------------------------------------------------
# fast: the K-knob contrast (harmonic_bridge.zeta_K) -- zeroless ground -> ONTO 1/2
# --------------------------------------------------------------------------
def test_kknob_contrast_born_onto_half():
    """The reverse arrow: the bridge's zeta_K zero begins off the line (K=1; K=0 zeroless) and
    climbs ONTO Re=1/2 as K grows -- where the Epstein pair leaves the line."""
    traj = ez.kknob_trajectory(14.134725)
    assert traj[0][0] == 1
    assert float(traj[0][1].real) > 0.7                       # born off the line
    assert abs(float(traj[-1][1].real) - 0.5) < 0.03          # onto 1/2


# --------------------------------------------------------------------------
# slow: the pitchfork -- zero-tracking across the dimension knob
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_online_branch_descends_to_dstar():
    """On-line branch: the lowest critical-line zero t_1(d) decreases monotonically toward 0 as
    d rises to d* (the pair slides down the critical line to collide at the real center)."""
    crit = ez.critical_branch([3, 5, 7, 8.5, 9, 9.2, 9.24])
    ts = [t for _d, t in crit]
    assert len(ts) == 7
    assert all(ts[i] > ts[i + 1] for i in range(len(ts) - 1))   # strictly decreasing in d
    assert ts[0] > 4.0 and ts[-1] < 0.2                          # 5.2 (d=3) -> ~0.12 (d=9.24)


@pytest.mark.slow
def test_real_pair_migrates_to_strip_edges():
    """Off-critical real pair (d > d*): sigma_-(d) -> 0 and sigma_+(d) -> d/2 as d grows -- the
    zeros migrate AWAY from the center to the strip boundaries (Travenec-Samaj Sec. 3)."""
    real = ez.real_branch([9.25, 9.4, 10, 12, 17, 25])
    assert len(real) == 6
    d_last, sm_last, sp_last = real[-1]                          # d = 25
    assert sm_last < 0.01                                        # sigma_- -> lower edge 0
    assert abs(sp_last - d_last / 2) < 0.01                      # sigma_+ -> upper edge d/2
    sms = [sm for _d, sm, _sp in real]
    assert all(sms[i] > sms[i + 1] for i in range(len(sms) - 1))  # sigma_- strictly falling


@pytest.mark.slow
def test_argument_principle_pitchfork():
    """Independent certificate (winding number, no root-finding): a real-axis strip holds 0 zeros
    for d < d* (the pair is on the critical line) and exactly 2 for d > d* (the born real pair)."""
    n_below = ez.count_zeros(8.0, 0.15, 8.0 / 2 - 0.15, -0.6, 0.6, npts=80)
    n_above = ez.count_zeros(12.0, 0.15, 12.0 / 2 - 0.15, -0.6, 0.6, npts=80)
    assert abs(round(float(n_below))) == 0
    assert round(float(n_above)) == 2
