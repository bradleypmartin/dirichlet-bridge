"""Tests for lambda_census (issue #49) + the zero_birth machinery it added
(dps bucketing, Muller flow steps). OUTSIDE the default CI suite: run with
`pytest explorations/tests`. Lean by design -- the heavy verification (the
certified t <= 120 census and its flows) lives in the driver, local/manual
only, feeding the committed data/*.csv caches.
"""
import math

import mpmath as mp
import numpy as np

import character_bridge as cb
import lambda_census as lc
import warp_alpha as wa
import zero_birth as zb


# --------------------------------------------------------------------------
# the grid bucketing (zero_birth)
# --------------------------------------------------------------------------
def test_bucket_dps_ceilings():
    """dps rounds UP to the bucket ceiling, multiples stay put."""
    assert zb._bucket_dps(15) == 15
    assert zb._bucket_dps(16) == 20
    assert zb._bucket_dps(20) == 20
    assert zb._bucket_dps(21) == 25
    assert zb._bucket_dps(53) == 55


def test_bucketed_evaluator_matches_warp_alpha():
    """warp_component_lam at lam = 1 still equals warp_alpha.warp_alpha at a
    height where the guard digits (and hence the bucketing) are active."""
    s = mp.mpc(0.7, 45.0)
    a = mp.mpf(1) / 3
    d = abs(zb.warp_component_lam(s, 1, 1, a) - wa.warp_alpha(s, 1, a))
    assert float(d) < 1e-8


def test_grid_reuse_within_bucket():
    """Two nearby heights in one dps bucket share a moment grid (the #49
    speedup); before the fix each height rebuilt every grid."""
    zb._GRID_LAM.clear()
    with mp.workdps(15):
        zb.warp_component_lam(mp.mpc(0.5, 50.0), 1, 0.5, mp.mpf(1) / 3)
        n1 = len(zb._GRID_LAM)
        zb.warp_component_lam(mp.mpc(0.5, 51.0), 1, 0.5, mp.mpf(1) / 3)
        n2 = len(zb._GRID_LAM)
    assert n1 == n2 == 1


def test_muller_flow_step_matches_secant():
    """The Muller local-triple flow polish lands on the same root as the
    committed secant path (first raw zero, one lam step)."""
    z0 = mp.mpc(0.338225885, 8.263532249)          # lamraw idx 0
    (l_m, z_m), = zb.flow_lambda(z0, 1, cb.CHI3, schedule=[1.0],
                                 muller=True)[:1]
    (l_s, z_s), = zb.flow_lambda(z0, 1, cb.CHI3, schedule=[1.0])[:1]
    assert z_m is not None and z_s is not None
    assert abs(z_m - z_s) < 1e-6


# --------------------------------------------------------------------------
# the precision band policy (the ambient-18 error bubble)
# --------------------------------------------------------------------------
def test_band_dps_uniform_above_onset():
    """Everything above the guard-onset region runs at 24 (the measured
    ambient-18 error lobes at t ~ 40-77 and 84-89 are not modeled lobe by
    lobe); the validated low band stays at 18."""
    assert lc.band_dps(2.03, 10.23) == 18
    assert lc.band_dps(26.63, 34.83) == 18
    assert lc.band_dps(34.83, 43.03) == 24      # touches the lobe region
    assert lc.band_dps(51.23, 59.43) == 24
    assert lc.band_dps(84.03, 92.23) == 24      # the second lobe
    assert lc.band_dps(108.63, 116.83) == 24


def test_flow_dps_bands():
    """Flows: 15 below t = 30 (#44), 18 to t = 36, 24 above."""
    assert lc.flow_dps(8.3) == 15
    assert lc.flow_dps(33.6) == 18
    assert lc.flow_dps(50.0) == 24
    assert lc.flow_dps(84.9) == 24
    assert lc.flow_dps(100.0) == 24


# --------------------------------------------------------------------------
# the certified scan (on the cheap seed object)
# --------------------------------------------------------------------------
def test_certified_scan_on_seed_D():
    """The Muller scan + winding certification on D(s) (a 3-term exponential
    polynomial -- cheap): every window certifies OK, and the main-comb seeds
    land near the predicted teeth 2 pi m/ln 2."""
    fn = lc._census_fn("seed")
    zeros = []
    for lo, hi in lc.windows_for(2.03, 20.03, height=9.0):
        zs, n_wind, _rf, status = lc.certify_window(fn, lc.BOX_SIG, (lo, hi),
                                                    dsig=0.25, dt=0.5)
        assert status == "OK"
        assert len(zs) == n_wind
        zeros.extend(zs)
    main = [z for z in zeros if float(z.real) > -1.5]
    teeth = [2 * math.pi * m / math.log(2) for m in (1, 2)]   # 9.06, 18.13
    assert len(main) == 2
    for z, t in zip(sorted(main, key=lambda w: float(w.imag)), teeth):
        assert abs(float(z.imag) - t) < 0.8
    # the odd-chi conductor string contributes its sparse far-left seed
    cond = [z for z in zeros if float(z.real) <= -1.5]
    assert len(cond) == 1
    assert abs(float(cond[0].imag) - 15.21) < 0.5


def test_windows_cover_contiguously():
    """Window edges tile (t_lo, t_max] with no gap or overlap."""
    wins = lc.windows_for(2.03, 50.0)
    assert wins[0][0] == 2.03
    assert abs(wins[-1][1] - 50.0) < 1e-9
    for (lo1, hi1), (lo2, hi2) in zip(wins, wins[1:]):
        assert hi1 == lo2


# --------------------------------------------------------------------------
# the instruments
# --------------------------------------------------------------------------
def test_rvm_count_matches_44_box():
    """The smooth count over the #44 box (2, 36.2) sits near the measured 11
    (the raw K = 1 census carries RvM density)."""
    n = lc.rvm_count(2.03, 36.21)
    assert 10.0 < n < 13.0


def test_predicted_fraction_law():
    """f(t) = ln q / ln(q t/2 pi): decreasing past the crossover, clamped to 1
    below it, and consistent with seed/RvM density ratio."""
    assert lc.predicted_fraction(2.0) == 1.0
    f36, f120 = lc.predicted_fraction(36.0), lc.predicted_fraction(120.0)
    assert 0 < f120 < f36 < 1
    assert abs(f36 - math.log(3) / math.log(3 * 36.0 / (2 * math.pi))) < 1e-12


def test_seed_count_density():
    """Mean-motion bookkeeping: ln q/2 pi per unit height."""
    assert abs(lc.seed_count(0.0, 2 * math.pi) - math.log(3)) < 1e-12


def test_census_fates_and_fraction_windows():
    """Synthetic flows: fates split, seeds matched, window fractions counted."""
    seeds = [(0, 0.1, 9.0, "main"), (1, -2.8, 15.2, "conductor")]
    flows = {
        0: [(1.0, 0.3, 8.9), (0.0, 0.1, 9.0)],                    # seed-fated
        1: [(1.0, 0.2, 10.5), (0.012, -2.9, 15.0),
            (0.0, float("nan"), float("nan"))],                    # expelled
        2: [(1.0, 0.4, 14.8), (0.0, -2.8, 15.2)],                  # seed-fated
    }
    fates = lc.census_fates(flows, seeds)
    assert fates[0][0] == "seed" and fates[0][2] == 0
    assert fates[1][0] == "exit" and fates[1][2] is None
    assert fates[2][0] == "seed" and fates[2][2] == 1
    assert fates[2][3] < 1e-9
    wins = [(2.03, 12.0, 0, 0, 0, "OK"), (12.0, 20.0, 0, 0, 0, "OK")]
    fr = lc.fraction_windows(fates, wins)
    assert fr[0][2] == 2 and fr[0][3] == 1          # flows 0, 1; one seeded
    assert fr[1][2] == 1 and fr[1][3] == 1          # flow 2
    assert abs(fr[0][4] - 0.5) < 1e-12


def test_birth_phase_split_reads_fate():
    """A synthetic census whose seed-fated flows are phase-locked at birth and
    whose expelled flows are anti-phased is fully threshold-readable."""
    lock = 2 * math.pi / math.log(2)
    flows = {}
    for i in range(4):                               # locked, seed-fated
        t = (i + 1) * lock
        flows[i] = [(1.0, 0.3, t), (0.0, 0.0, t)]
    for i in range(4, 8):                            # anti-phased, expelled
        t = (i + 0.5) * lock / 2
        flows[i] = [(1.0, 0.3, math.pi / math.log(2) * (2 * i + 1)),
                    (0.0, float("nan"), float("nan"))]
    split, acc = lc.birth_phase_split(flows)
    assert len(split["seed"]) == 4 and len(split["exit"]) == 4
    assert all(c > 0.99 for _i, _t, c in split["seed"])
    assert all(c < -0.99 for _i, _t, c in split["exit"])
    assert acc == 1.0


def test_dedupe_merges_close_roots():
    zs = [mp.mpc(0.5, 9.0), mp.mpc(0.5 + 1e-5, 9.0), mp.mpc(0.5, 12.0)]
    out = lc._dedupe(zs)
    assert len(out) == 2
