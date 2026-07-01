"""Unit tests for the Hurwitz/Lerch zero-trajectory driver (hurwitz_lerch_zeros.py).

The closest *published* deform-and-track analogue of the bridge: Garunkstis-Steuding (2007)
vary the Hurwitz shift alpha (1 -> 1/2) and watch zeta(s, alpha)'s zeros move, calling a zero
"stable" iff its trajectory starts AND ends on Re s = 1/2. The Lerch family L(lambda, alpha, s)
contains the bridge's four objects at the corners of the (lambda, alpha) square, and its three
deformation paths are the three published ancestors. The checks:

* evaluators: zeta(s, alpha) = mpmath.zeta(s, alpha) and L(lambda, alpha, s) = mpmath.lerchphi(
  e^{2 pi i lambda}, s, alpha) each match their direct sums (sigma>1); the z -> 1 (integer lambda)
  blow-up of lerchphi is handled by the Hurwitz fallback lerchphi(1, s, a) = zeta(s, a);
* the four unification identities at the corners: (1,1)=zeta, (1,1/2)=(2^s-1)zeta, (1/2,1)=eta,
  (1/2,1/2)=2^s L(s, chi_4);
* the Hurwitz alpha-trajectories (fast, via mp.zeta): a STABLE zero returns to Re=1/2, an unstable
  one lands on the sigma=0 companion comb s = 2 pi i m / log 2;
* (slow, lerchphi) the Lerch diagonal lambda=alpha glues a zeta zero to a 2^s L(s,chi_4) zero on
  sigma=1/2, and the alpha=1 polylog edge carries it to eta's sigma=1 comb;
* the contrast with the bridge's K-knob (harmonic_bridge.zeta_K): a zeroless-endpoint ground zero
  climbs ONTO 1/2 -- the arrow points onto the line, not on/off between two zero-rich ends.

The Lerch-family trajectory tracks (lerchphi ~ 0.1 s/eval, ~26 s each) are `slow`; the fast suite
keeps the evaluators, identities, the cheap Hurwitz sweep, and the K-knob contrast.
"""
from __future__ import annotations

import mpmath as mp
import pytest

import hurwitz_lerch_zeros as hl

PI = mp.pi
TWOPI_OVER_LOG2 = float(2 * mp.pi / mp.log(2))     # 9.0647...: the sigma=0 / sigma=1 comb spacing


# --------------------------------------------------------------------------
# fast: the evaluators and the z -> 1 fallback
# --------------------------------------------------------------------------
def test_hurwitz_matches_sum():
    """zeta(s, alpha) = mpmath.zeta == the direct sum sum_{m>=0}(m+alpha)^{-s} for sigma>1."""
    for s, a in [(mp.mpc(3, 1), 0.3), (mp.mpc(4, 0), 0.7)]:
        assert abs(hl.zeta_hurwitz(s, a) - hl.zeta_hurwitz_sum(s, a, terms=20000)) < mp.mpf("1e-7")


def test_lerch_matches_sum():
    """L(lambda, alpha, s) = mpmath.lerchphi == the direct sum (|z|=1 converges for sigma>1)."""
    for lam, a in [(0.25, 0.7), (0.75, 0.4)]:
        s = mp.mpc(4, 0.5)
        assert abs(hl.lerch(lam, a, s) - hl.lerch_sum(lam, a, s)) < mp.mpf("1e-9")


def test_lerch_z1_fallback():
    """z=1 (integer lambda): lerchphi(1, s, a) = zeta(s, a); mpmath diverges at z=1, we splice it."""
    for s in [mp.mpc(0.7, 12), mp.mpc(2, 1)]:
        for a in [1.0, 0.5]:
            assert abs(hl.lerch(1, a, s) - mp.zeta(s, mp.mpf(a))) < mp.mpf("1e-25")


def test_chi4_leibniz_anchor():
    """The local L_chi4 evaluator's removable value L(1, chi_4) = pi/4 (Leibniz)."""
    assert abs(hl.L_chi4(mp.mpc(1, 0)) - PI / 4) < mp.mpf("1e-25")


# --------------------------------------------------------------------------
# fast: the four unification identities at the corners of the (lambda, alpha) square
# --------------------------------------------------------------------------
def test_unification_corners():
    """L(lambda, alpha, s) == the independent evaluator (zeta / (2^s-1)zeta / eta / 2^s L(chi_4))
    at each corner -- the family contains all four bridge objects."""
    for s in [mp.mpc(0.7, 12.0), mp.mpc(2, 1)]:
        for lam, a, _name, tgt in hl.UNIFICATION:
            assert abs(hl.lerch(lam, a, s) - tgt(s)) < mp.mpf("1e-25")


def test_unification_chi4_is_lerch_half_half():
    """The least-obvious corner (1/2, 1/2) -> 2^s L(s, chi_4) = Phi(-1, s, 1/2) (issue #20)."""
    for s in [mp.mpc(0.5, 6.020948), mp.mpc(2, 1)]:
        assert abs(hl.lerch(mp.mpf(1) / 2, mp.mpf(1) / 2, s) - 2 ** s * hl.L_chi4(s)) < mp.mpf("1e-25")


# --------------------------------------------------------------------------
# fast: the Hurwitz alpha-sweep (mp.zeta) -- the Garunkstis-Steuding stable/unstable dichotomy
# --------------------------------------------------------------------------
def test_hurwitz_stable_zero():
    """The 2nd Riemann zero is STABLE: rho(alpha) starts and ends on Re=1/2 (lands on the 1st
    zeta zero at alpha=1/2), though it bulges off the line in between."""
    traj = hl.track_hurwitz(float(mp.zetazero(2).imag))
    assert hl.classify(traj) == "stable"
    end = traj[-1][1]
    assert abs(float(end.real) - 0.5) < 0.02
    assert abs(float(end.imag) - 14.134725) < 0.05          # -> 1st zeta zero height


def test_hurwitz_sigma0_companion():
    """The 1st Riemann zero is UNSTABLE: its trajectory lands on the sigma=0 companion comb
    s = 2 pi i / log 2 = 9.0647i (the half-shift's (2^s-1) zeros) -- ties to warp_bridge.py."""
    traj = hl.track_hurwitz(float(mp.zetazero(1).imag))
    assert hl.classify(traj) == "sigma=0"
    end = traj[-1][1]
    assert abs(float(end.real)) < 0.02
    assert abs(float(end.imag) - TWOPI_OVER_LOG2) < 0.05


# --------------------------------------------------------------------------
# fast: the K-knob contrast (harmonic_bridge.zeta_K) -- zeroless ground -> onto 1/2
# --------------------------------------------------------------------------
def test_kknob_contrast_born_onto_half():
    """The bridge's OWN motion (the contrast): the zeta_K zero for a Riemann zero begins at K=1
    off the line (K=0 is zeroless) and climbs ONTO Re=1/2 as K grows -- onto, not on/off."""
    import harmonic_bridge as hb
    traj = hb.migrate(hb.zeta_K, mp.mpc(0.5, float(mp.zetazero(3).imag)),
                      schedule=[1, 2, 4, 8, 16, 32])
    d = {K: zz for K, zz in traj if zz is not None}
    assert 1 in d and float(d[1].real) > 0.7                 # off-line ground (born from zeroless K=0)
    assert 32 in d and abs(float(d[32].real) - 0.5) < 0.06   # climbs onto the critical line


# --------------------------------------------------------------------------
# slow: the Lerch-family paths (lerchphi ~ 0.1 s/eval)
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_lerch_diagonal_lands_on_chi4():
    """Diagonal lambda=alpha: 1 -> 1/2 (Garunkstis-Tamosiunas: zeta -> 2^s L(s, chi_4)). The 1st
    zeta zero stays glued to sigma=1/2 the whole way (primitive L -> NO companion) and lands on
    the 1st L(s, chi_4) zero at 6.0209i."""
    params = [(v, v) for v in hl._linspace(1.0, 0.5, 26)]
    traj = hl.track_lerch_path(params, float(mp.zetazero(1).imag))
    assert len(traj) == 26
    for _p, s in traj:
        assert abs(float(s.real) - 0.5) < 0.02               # never leaves the line
    end = traj[-1][1]
    assert abs(float(end.imag) - 6.020948) < 0.05


@pytest.mark.slow
def test_lerch_polylog_edge_to_eta_comb():
    """alpha=1 edge lambda: 1 -> 1/2 (the Fornberg-Kolbig polylog knob: zeta -> eta). The 1st zeta
    zero climbs to eta's companion comb on sigma=1 (not sigma=0) at 9.0647i -- the SAME comb
    jonquiere_zeros.py reaches, here as one edge of the Lerch square."""
    params = [(v, mp.mpf(1)) for v in hl._linspace(1.0, 0.5, 26)]
    traj = hl.track_lerch_path(params, float(mp.zetazero(1).imag))
    end = traj[-1][1]
    assert abs(float(end.real) - 1.0) < 0.02                 # companion on sigma=1
    assert abs(float(end.imag) - TWOPI_OVER_LOG2) < 0.05


@pytest.mark.slow
def test_kknob_trajectory_full():
    """The module wrapper hl.kknob_trajectory over its full schedule reaches sigma=1/2 (K=89)."""
    traj = hl.kknob_trajectory(float(mp.zetazero(3).imag))
    assert traj[0][0] == 1                                    # begins at K=1 (K=0 zeroless)
    assert float(traj[0][1].real) > 0.7                       # off-line ground
    assert abs(float(traj[-1][1].real) - 0.5) < 0.03          # onto 1/2 by K=89
