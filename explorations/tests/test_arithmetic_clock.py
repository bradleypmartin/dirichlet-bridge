"""Tests for arithmetic_clock (issue #44). OUTSIDE the default CI suite:
pytest.ini's `testpaths = tests` excludes explorations/; run these with
`pytest explorations/tests`. Lean by design -- the heavy verification (the
400-zero K-walk, the lambda flow, the census certification) lives in the
driver, local/manual only, feeding the committed data/*.csv caches.
"""
import math

import mpmath as mp
import numpy as np
import pytest

import arithmetic_clock as ac
import character_bridge as cb
import chi6_two_component as c6
import zero_birth as zb


# --------------------------------------------------------------------------
# the clock variable
# --------------------------------------------------------------------------
def test_kappa_roundtrip():
    """K_of_kappa realizes at least the requested clock reading, tightly."""
    for kap in (0.5, 1.0, 4.0):
        for t in (8.0, 100.0, 550.0):
            K = ac.K_of_kappa(kap, t)
            assert ac.kappa_of(K, t) >= kap
            assert ac.kappa_of(K - 1, t) < kap or K == 1


def test_clock_band_edge_is_conductor_times_height():
    """kappa = 1 at K = q t/2 pi: the stationary-phase band edge of the a = 1
    Hurwitz moment (frequency 2 pi K meets the integrand phase t at y = 1/q)."""
    t = 300.0
    assert abs(ac.kappa_of(ac.Q * t / (2 * math.pi), t) - 1.0) < 1e-12


# --------------------------------------------------------------------------
# the walk
# --------------------------------------------------------------------------
def test_walk_zero_matches_measured_disp():
    """A single-K walk at the first chi_3 zero reproduces zero_birth's
    measured_disp (same root, independently seeded machinery)."""
    g = cb.CHI3_ZEROS[0]
    a1 = ac.a1_of(g)
    (K, s_K), = ac.walk_zero(g, a1, [40])
    ref = zb.measured_disp(mp.mpc(0.5, g), cb.CHI3, 40)
    assert s_K is not None
    assert abs(complex(K * (s_K - mp.mpc(0.5, g))) - complex(ref)) < 1e-6


def test_walk_zero_descends_toward_a1():
    """Down the schedule at a low zero, K(s_K - rho) approaches a_1 as K grows
    (the displacement law, kappa >> 1 here)."""
    g = cb.CHI3_ZEROS[0]
    a1 = ac.a1_of(g)
    walk = dict(ac.walk_zero(g, a1, [12, 45]))
    d45 = abs(complex(45 * (walk[45] - mp.mpc(0.5, g))) - a1)
    d12 = abs(complex(12 * (walk[12] - mp.mpc(0.5, g))) - a1)
    assert d45 < d12
    assert d45 / abs(a1) < 0.15


# --------------------------------------------------------------------------
# the resonance instruments
# --------------------------------------------------------------------------
def test_resonance_constant_linearizes_field_prediction():
    """C_omega is the exact derivative of the field prediction at 1/K = 0:
    field(K) - [R_inf + C/K] = O(1/K^2) on synthetic data."""
    rng = np.random.default_rng(7)
    g = np.sort(rng.uniform(10, 200, 60))
    a1 = [complex(rng.normal(), rng.normal()) for _ in g]
    omega = ac.LN2
    r_inf = float(np.cos(g * omega).mean())
    errs = []
    for K in (100, 200):
        lin = r_inf + ac.resonance_constant(g, a1, omega) / K
        errs.append(abs(ac.field_prediction(g, a1, omega, K) - lin))
    assert errs[0] / errs[1] > 3.0          # O(1/K^2): ratio ~ 4 per doubling


def test_resonance_on_locked_comb():
    """A ln2-comb set gamma = 2 pi j/ln 2 is exactly phase-locked: resonance
    +1 at every 2-adic mode, order parameter 1."""
    g = np.array([2 * math.pi * j / ac.LN2 for j in range(1, 40)])
    for m in (1, 2, 3):
        assert abs(c6.prime_resonance(g, m * ac.LN2) - 1.0) < 1e-9


def test_field_prediction_reduces_to_measured_at_inf():
    """field_prediction at K = inf-like values reproduces the unperturbed
    resonance."""
    g = np.linspace(20, 40, 11)
    a1 = [1 + 1j] * 11
    r0 = c6.prime_resonance(g, ac.LN2)
    assert abs(ac.field_prediction(g, a1, ac.LN2, 10 ** 9) - r0) < 1e-8


def test_von_mangoldt():
    """Lambda(n) on the named C-spectrum bands: ln p at prime powers, 0 else."""
    assert abs(ac._von_mangoldt(8) - math.log(2)) < 1e-12
    assert abs(ac._von_mangoldt(9) - math.log(3)) < 1e-12
    assert abs(ac._von_mangoldt(13) - math.log(13)) < 1e-12
    assert ac._von_mangoldt(6) == 0.0
    assert ac._von_mangoldt(12) == 0.0


def test_c_band_table_finds_planted_line():
    """A synthetic Im a_1 oscillating at ln 5 puts a band-limited line at ln 5
    in the C spectrum, and the control bands stay at their floor."""
    rng = np.random.default_rng(11)
    g = np.sort(rng.uniform(20, 500, 300))
    a1 = [complex(0.0, 3.0 * math.sin(x * math.log(5))) for x in g]
    bands = ac.c_band_table(g, a1, half=0.01, dw=1e-3)
    by = {}
    for lb, _w0, mx, _wm, _ref in bands:
        by.setdefault(lb, []).append(mx)
    assert by["ln5"][0] > 2.0                        # the planted line
    assert max(by["ctl"]) < 0.5 * by["ln5"][0]       # controls at the floor


# --------------------------------------------------------------------------
# census repair
# --------------------------------------------------------------------------
def test_dedup_patch_flags_collisions(monkeypatch):
    """Two walks landing on one root: the farther source loses it, the rescue
    scan (stubbed) restores it as 'patched'."""
    rows = [(0, 10.0, 45, 0.51, 10.20, "ok"),
            (1, 11.5, 45, 0.51, 10.20, "ok"),        # collided with row 0
            (2, 13.0, 45, 0.52, 13.10, "ok")]
    monkeypatch.setattr(ac, "_local_rescue",
                        lambda K, sc, lo, hi, taken, **kw: mp.mpc(0.5, 11.62))
    repairs = ac.dedup_patch(rows, verbose=False)
    assert repairs == {45: (1, 1)}
    patched = [r for r in rows if r[5] == "patched"]
    assert len(patched) == 1
    assert patched[0][0] == 1                        # the farther source zero
    assert abs(patched[0][4] - 11.62) < 1e-9


def test_dedup_patch_no_false_positives():
    """Distinct roots untouched."""
    rows = [(0, 10.0, 45, 0.51, 10.20, "ok"),
            (1, 11.5, 45, 0.52, 11.60, "ok")]
    assert ac.dedup_patch(list(rows), verbose=False) == {}


# --------------------------------------------------------------------------
# the collapse instruments
# --------------------------------------------------------------------------
def test_collapse_points_amplification():
    """A = K(s_K - rho)/a_1 comes out 1 when the walk lands exactly on the
    first-order prediction."""
    a1 = complex(0.4, 2.0)
    K, g = 50, 100.0
    s_K = complex(0.5, g) + a1 / K
    rows = [(0, g, K, s_K.real, s_K.imag, "ok")]
    (t, KK, kap, A), = ac.collapse_points(rows, {0: a1})
    assert abs(A - 1) < 1e-12
    assert abs(kap - ac.kappa_of(K, g)) < 1e-12


def test_collapse_fit_recovers_powerlaw():
    """The c1/kappa + c2/kappa^2 fit recovers planted coefficients."""
    kk = np.geomspace(1.0, 8.0, 40)
    pts = [(100.0, 1, k, 1 + (0.1 / k + 0.5 / k ** 2)) for k in kk]
    c1, c2 = ac.collapse_fit(pts)
    assert abs(c1 - 0.1) < 1e-6 and abs(c2 - 0.5) < 1e-6


# --------------------------------------------------------------------------
# the lambda instruments
# --------------------------------------------------------------------------
def test_lam_instruments_locked_flow():
    """A flow ending on the exact ln2 comb reads rho_1 = 1, cos = +1 at the
    seed end and reports every lambda level."""
    teeth = [2 * math.pi * j / ac.LN2 for j in range(1, 6)]
    flows = {i: [(1.0, 0.5, t + 0.3), (0.5, 0.4, t + 0.1), (0.0, 0.0, t)]
             for i, t in enumerate(teeth)}
    out = ac.lam_instruments(flows)
    assert [r[0] for r in out] == [1.0, 0.5, 0.0]
    lam0 = out[-1]
    assert abs(lam0[2][1][0] - 1.0) < 1e-9          # <cos> = +1 at the seeds
    assert abs(lam0[2][1][1] - 1.0) < 1e-9          # order parameter 1
    assert lam0[1] == 5


def test_lam_instruments_drops_lost_rows():
    """nan gammas (honest losses) leave the census, not the instrument."""
    flows = {0: [(1.0, 0.5, 12.0), (0.5, float("nan"), float("nan"))],
             1: [(1.0, 0.5, 15.0), (0.5, 0.4, 15.1)],
             2: [(1.0, 0.5, 18.0), (0.5, 0.4, 18.2)],
             3: [(1.0, 0.5, 21.0), (0.5, 0.4, 21.3)]}
    out = ac.lam_instruments(flows)
    ns = {lam: n for lam, n, _m, _c3 in out}
    assert ns[1.0] == 4 and ns[0.5] == 3


def test_lam_fate_split():
    """Trajectories are classified by endpoint: landed on the seed set vs lost
    mid-flow, and the fate filter restricts the census accordingly."""
    seed_tr = [(1.0, 0.5, 12.0), (0.0, 0.1, 12.5)]
    exit_tr = [(1.0, 0.5, 15.0), (0.012, -2.8, 15.4),
               (0.0, float("nan"), float("nan"))]
    assert ac.flow_fate(seed_tr) == "seed"
    assert ac.flow_fate(exit_tr) == "exit"
    flows = {0: seed_tr, 1: exit_tr, 2: seed_tr, 3: seed_tr}
    all_rows = ac.lam_instruments(flows)
    seed_rows = ac.lam_instruments(flows, fate="seed")
    n_all = {lam: n for lam, n, _m, _c in all_rows}
    n_seed = {lam: n for lam, n, _m, _c in seed_rows}
    assert n_all[1.0] == 4 and n_seed[1.0] == 3
