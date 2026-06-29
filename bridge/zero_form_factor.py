"""Spectral form factor of the Riemann zeros: the GUE face (small tau) and the
prime face (large u) of one Fourier transform (issue #84).

This is the bridge between the repo's two prime/zero forks:

  * #8  / gue-spacing.md            -- "the Riemann zeros are textbook GUE",
                                        measured by nearest-neighbour SPACINGS.
  * #80B / log-prime-diffraction.md -- the prime comb in the DIFFRACTION
                                        (structure factor) of the zero set.

Both are the same object -- the Fourier transform of the zero point set -- seen
in two coordinates:

  FORM FACTOR  K(tau)  of the *unfolded* zeros (unit mean density). Conjugate
      variable tau = "time" in units of the Heisenberg time T_H = 2 pi rhobar.
      Reproduces the universal RMT ramp-and-plateau
          K_GUE(tau)  = min(|tau|, 1),
      and the SLOPE at small tau distinguishes the symmetry class:
          GUE: K ~ 1*tau   (unitary, broken time-reversal -- the zeros),
          GOE: K ~ 2*tau   (orthogonal, time-reversal invariant).
      Recovering slope 1 (not 2) is a SECOND, two-point-correlation-based
      confirmation of the unitary class, complementing the surmise fit in #8.

  STRUCTURE FACTOR  S(u)  of the *raw* zeros (varying density). Conjugate
      variable u to the raw ordinates gamma. Sharp Bragg peaks at u = log(p^k)
      (the prime comb of #80B): the resolved short periodic orbits = the primes.

The two faces are the SAME transform up to the local unfolding rescale
u = 2 pi rho(gamma) tau. The catch -- and the real content of the bridge -- is
that rho(gamma) varies across the sample, so a fixed prime u = log p maps to a
WHOLE RANGE of tau. Unfolding therefore *sharpens* the universal ramp (the
diagonal / all-orbits average) while *smearing* the arithmetic comb (the
individual short orbits), and vice versa. The diagonal -> off-diagonal crossover
(Berry 1986) is the trade-off between the two faces, not a single shared axis.

Caveat (the repo's recurring trap): a form factor from only ~2000 zeros is noisy;
the ramp emerges only after unfolding + binning/averaging, the same
"wrong-unfold / under-averaged-statistic fakes the answer" hazard flagged in
#36/#48/#49. Nothing here is new about zeta (Montgomery 1973; Berry 1986;
Bogomolny-Keating 1996); the deliverable is the in-repo bridge plus the
independent-statistic confirmation of the GUE class.

Driver main() writes zero_form_factor.{png,txt}. See knowledge/statistics/zero-form-factor.md.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

# repo root + sibling source dirs (maass/gue_spacing, cone_interface/cone_log_prime)
# on sys.path so the bare cross-group imports resolve when run directly as a
# script (pytest gets these from the root conftest.py). See #107.
_ROOT = Path(__file__).resolve().parents[1]
for _d in ("", "maass", "krein", "prime_zero", "legacy", "cone_interface"):
    _p = str(_ROOT / _d) if _d else str(_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gue_spacing import load_riemann_zeros, unfold_riemann_zeros  # noqa: E402
import cone_log_prime as P  # noqa: E402

OUTPUT_PNG = _ROOT / "zero_form_factor.png"
OUTPUT_TXT = _ROOT / "zero_form_factor.txt"
OUTPUT_HIGH_PNG = _ROOT / "zero_form_factor_high.png"
OUTPUT_HIGH_TXT = _ROOT / "zero_form_factor_high.txt"

# Fixed-height windows for the #89 slope-vs-height trend. W0 is the top of the
# canonical cache (free); the higher windows are the committed CSVs written by
# `maass/compute_riemann_zeros.py --start N` (skipped with a note if absent).
LOW_WINDOW_N = 1000
HIGH_WINDOW_CSVS = (
    _ROOT / "data" / "riemann_zeros_window_n100000.csv",
    _ROOT / "data" / "riemann_zeros_window_n1000000.csv",
)


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


def spectral_window(positions: np.ndarray, kind: str = "hann") -> np.ndarray:
    """Smooth taper over a spectrum, evaluated at fractional positions in [0, 1].

    Tapering the two ends of the spectral range suppresses the truncation
    (Gibbs) leakage of the disconnected part -- |sum w_n e^{2 pi i wtilde tau}|
    has a tall DC spike near tau = 0 whose sidelobes would otherwise bury the
    ramp. A Hann taper pushes that leakage down so the connected ramp shows
    through for tau >~ 0.02.

    `positions` should be the unfolded ordinates mapped to [0, 1]:
    p_n = (wtilde_n - wtilde_0) / (wtilde_{N-1} - wtilde_0).
    """
    p = np.asarray(positions, dtype=float)
    if kind == "boxcar":
        return np.ones_like(p)
    if kind == "hann":
        return np.sin(math.pi * p) ** 2
    if kind == "gaussian":
        # centred Gaussian, ~3 sigma to each edge
        return np.exp(-0.5 * ((p - 0.5) / 0.17) ** 2)
    raise ValueError(f"unknown window {kind!r}")


# ---------------------------------------------------------------------------
# The measured form factor
# ---------------------------------------------------------------------------


def spectral_form_factor(
    unfolded: np.ndarray,
    taus: np.ndarray,
    window: str = "hann",
) -> np.ndarray:
    """Spectral form factor of an unfolded (unit-density) level sequence:

        K(tau) = | sum_n w_n exp(2 pi i wtilde_n tau) |^2 / sum_n w_n^2,

    with w_n a smooth spectral window. The normalisation by sum w_n^2 makes the
    DIAGONAL (self) term equal 1, so the plateau (large tau, off-diagonal
    averaging to zero) sits at K -> 1 and the GUE ramp is K = tau exactly.

    Near tau = 0 (and tau in Z) the disconnected term -- the Fourier transform
    of the smooth mean density / the unfolded lattice harmonics -- dominates;
    read the ramp only for tau bounded away from 0 and 1.
    """
    wt = np.asarray(unfolded, dtype=float)
    taus = np.atleast_1d(np.asarray(taus, dtype=float))
    span = wt[-1] - wt[0]
    pos = (wt - wt[0]) / span if span > 0 else np.zeros_like(wt)
    w = spectral_window(pos, window)
    # (n_tau, N) phase matrix -- fine for ~2000 levels x a few thousand tau.
    phase = np.exp(2j * math.pi * np.outer(taus, wt))
    amp = (w[None, :] * phase).sum(axis=1)
    return np.abs(amp) ** 2 / np.sum(w * w)


def bin_form_factor(
    taus: np.ndarray, K: np.ndarray, n_bins: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average the (exponentially noisy) raw K(tau) into `n_bins` equal-width
    tau bins. Returns (centres, means, standard errors). Each raw K(tau) is ~
    chi^2 with unit relative fluctuation, so binning ~M samples cuts the noise
    by ~sqrt(M) -- the only way the ramp becomes legible at N ~ 2000 zeros.
    """
    taus = np.asarray(taus, dtype=float)
    K = np.asarray(K, dtype=float)
    edges = np.linspace(taus.min(), taus.max(), n_bins + 1)
    idx = np.clip(np.digitize(taus, edges) - 1, 0, n_bins - 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n_bins, np.nan)
    errs = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = K[idx == b]
        if sel.size:
            means[b] = sel.mean()
            errs[b] = sel.std(ddof=1) / math.sqrt(sel.size) if sel.size > 1 else 0.0
    return centres, means, errs


def small_tau_slope(
    taus: np.ndarray, K: np.ndarray, tau_lo: float = 0.05, tau_hi: float = 0.4
) -> float:
    """Least-squares slope through the origin of K vs tau over [tau_lo, tau_hi],
    the class discriminant: ~1 for GUE, ~2 for GOE. Fitting through the origin
    matches K(0) = 0 for both surmises and is robust to the additive plateau
    offset that creeps in as tau -> 1."""
    taus = np.asarray(taus, dtype=float)
    K = np.asarray(K, dtype=float)
    m = (taus >= tau_lo) & (taus <= tau_hi) & np.isfinite(K)
    x, y = taus[m], K[m]
    return float((x @ y) / (x @ x))


# ---------------------------------------------------------------------------
# RMT reference form factors (Fourier transforms of the pair correlation --
# NOT the Wigner nearest-neighbour surmises of gue_spacing.py)
# ---------------------------------------------------------------------------


def gue_form_factor(tau: np.ndarray | float) -> np.ndarray | float:
    """K_GUE(tau) = min(|tau|, 1): the unitary-class ramp (slope 1) and
    plateau. The Fourier transform of the sine-kernel cluster function
    Y_2(s) = (sin(pi s)/(pi s))^2 (Dyson/Montgomery)."""
    return np.minimum(np.abs(np.asarray(tau, dtype=float)), 1.0)


def goe_form_factor(tau: np.ndarray | float) -> np.ndarray | float:
    """K_GOE(tau): the orthogonal-class form factor (beta = 1),

        |tau| <= 1 :  2|tau| - |tau| ln(1 + 2|tau|),
        |tau| >  1 :  2 - |tau| ln((2|tau|+1)/(2|tau|-1)).

    Small-tau slope is 2 (K ~ 2|tau| - 2 tau^2 + ...), twice the GUE ramp --
    the class discriminant. Continuous at |tau| = 1 (both give 2 - ln 3 ~ 0.901)
    and -> 1 as |tau| -> inf."""
    t = np.abs(np.asarray(tau, dtype=float))
    small = 2.0 * t - t * np.log1p(2.0 * t)
    denom = np.where(t > 1.0, 2.0 * t - 1.0, 1.0)  # keep the log argument > 0
    large = 2.0 - t * np.log((2.0 * t + 1.0) / denom)
    return np.where(t <= 1.0, small, large)


def poisson_form_factor(tau: np.ndarray | float) -> np.ndarray | float:
    """K_Poisson(tau) = 1: uncorrelated levels have no ramp (the cluster
    function vanishes), so the form factor is flat at the plateau value -- the
    absence of the small-tau dip is the absence of level repulsion."""
    return np.ones_like(np.asarray(tau, dtype=float))


# ---------------------------------------------------------------------------
# The Heisenberg / unfolding bridge between the two faces
# ---------------------------------------------------------------------------


def zero_density(gammas: np.ndarray) -> np.ndarray:
    """Mean spectral density of the zeros at height gamma:
    rho(gamma) = (1/2 pi) log(gamma / 2 pi). (d/dgamma of the smooth N(T).)"""
    g = np.asarray(gammas, dtype=float)
    return np.log(g / (2.0 * math.pi)) / (2.0 * math.pi)


def heisenberg_bridge(gammas: np.ndarray) -> dict:
    """The map between the unfolded form-factor time tau and the raw structure-
    factor frequency u: u = 2 pi rho(gamma) tau, so the Heisenberg time tau = 1
    sits at u_H = 2 pi rhobar = log(gammabar / 2 pi), and a fixed prime
    u = log p sits at tau_p = log p / (2 pi rho(gamma)).

    Because rho varies across the sample, tau_p spans a RANGE (returned as
    tau_logp_{lo,hi}); that spread is exactly why the prime comb smears in the
    unfolded form factor but the universal ramp does not. Returns the mean and
    the spread of u_H and of the first-prime time."""
    rho = zero_density(gammas)
    two_pi = 2.0 * math.pi
    u_h = two_pi * rho
    tau_log2 = math.log(2.0) / u_h
    return {
        "rho_bar": float(rho.mean()),
        "u_heisenberg": float(u_h.mean()),
        "u_heisenberg_lo": float(u_h.min()),
        "u_heisenberg_hi": float(u_h.max()),
        "tau_log2": float(tau_log2.mean()),
        "tau_log2_lo": float(tau_log2.min()),
        "tau_log2_hi": float(tau_log2.max()),
    }


# ---------------------------------------------------------------------------
# The high-height window: where the two faces collapse onto one tau axis (#89)
# ---------------------------------------------------------------------------
#
# #84 had to keep the GUE ramp and the prime comb on SEPARATE axes: the bridge
# u = 2 pi rho(gamma) tau has a height-dependent rho, so across the full cached
# cache (gamma 14 .. 2515, rho 0.13 .. 0.95, a factor of ~7) a single prime
# u = log p smears over a whole band tau ~ 0.12 .. 0.86.
#
# Take instead a NARROW WINDOW of consecutive zeros high on the critical line.
# rho(gamma) = (1/2 pi) log(gamma / 2 pi) grows only logarithmically, so across
# ~1500 zeros at gamma ~ 75000 it is constant to ~0.05%. The bridge is then a
# single CONSTANT rescale u = (2 pi rhobar) tau, and the prime Bragg peaks land
# at SHARP points tau_p = log(p^k) / (2 pi rhobar) ON the universal ramp -- the
# diagonal -> off-diagonal crossover (Berry 1986) in one figure. The slope of
# the ramp also sharpens toward the unitary 1 as the finite-height bias recedes.


def window_taper(n: int, kind: str = "hann") -> np.ndarray:
    """A smooth taper over `n` consecutive levels, centred on the window
    (positions 0 .. n-1 mapped to [0, 1]). Unlike the gamma=0-centred Gaussian
    of `cone_log_prime.gaussian_window` (which annihilates a high window), this
    tapers a block of zeros wherever it sits on the critical line."""
    if n <= 1:
        return np.ones(max(n, 0))
    pos = np.arange(n, dtype=float) / (n - 1)
    if kind == "boxcar":
        return np.ones(n)
    if kind == "hann":
        return np.sin(math.pi * pos) ** 2
    raise ValueError(f"unknown taper {kind!r}")


def window_structure_factor(
    gammas: np.ndarray, u: np.ndarray, taper: str = "hann", normalize: bool = True
) -> np.ndarray:
    """Structure factor S(u) = |sum_n w_n cos(gamma_n u)|^2 of a window of raw
    zero ordinates, with a window-centred taper w_n. Bragg peaks at u = log(p^k)
    (the prime comb, #80B) -- the same object as `cone_log_prime`'s, but tapered
    in INDEX so it works for a block of zeros at any height (not only near 0)."""
    g = np.asarray(gammas, dtype=float)
    u = np.atleast_1d(np.asarray(u, dtype=float))
    w = window_taper(g.size, taper)
    amp = (w[None, :] * np.cos(np.outer(u, g))).sum(axis=1)
    S = amp ** 2
    if normalize and S.size and S.max() > 0:
        S = S / S.max()
    return S


def constant_density_diagnostics(gammas: np.ndarray) -> dict:
    """How constant is rho(gamma) across the window? Returns the mean density
    rhobar, its min/max, the fractional spread (rho_hi - rho_lo)/rhobar, and the
    constant rescale u_per_tau = 2 pi rhobar that maps the form-factor time tau
    to the structure-factor frequency u. A small spread is what licenses the
    single-axis crossover; the full cache has spread ~ 1.5 (useless), a high
    window ~ 5e-4."""
    g = np.asarray(gammas, dtype=float)
    rho = zero_density(g)
    rho_bar = float(rho.mean())
    return {
        "n": int(g.size),
        "gamma_lo": float(g.min()),
        "gamma_hi": float(g.max()),
        "gamma_mid": float(0.5 * (g.min() + g.max())),
        "rho_bar": rho_bar,
        "rho_lo": float(rho.min()),
        "rho_hi": float(rho.max()),
        "rho_spread_frac": float((rho.max() - rho.min()) / rho_bar),
        "u_per_tau": 2.0 * math.pi * rho_bar,
    }


def _is_prime(n: int) -> bool:
    """True for a genuine prime (not a higher prime power). Used only to weight
    the figure's labels -- primes carry the explicit-formula weight Lambda(n)/sqrt(n)
    most strongly; the prime powers p^k (k>=2) are the weaker secondary peaks."""
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def prime_taus(rho_bar: float, nmax: int = 32):
    """Prime-power peak positions mapped into the form-factor time tau at a
    constant density: tau_{p^k} = log(p^k) / (2 pi rhobar). Returns a list of
    (n, tau, weight) for prime powers 2 <= n <= nmax, the weight being the
    explicit-formula coefficient Lambda(n)/sqrt(n)."""
    upt = 2.0 * math.pi * rho_bar
    return [(n, lg / upt, amp) for n, lg, amp in P.prime_power_peaks(nmax)]


def analyze_high_window(
    gammas: np.ndarray,
    tau_max: float = 2.2,
    n_tau: int = 9000,
    n_bins: int = 28,
    nmax_primes: int = 32,
) -> dict:
    """Form factor + structure factor of one fixed-height window, on a SHARED
    tau axis. The unfolded ordinates are re-centred before the phase sum (the
    absolute count N(gamma) ~ 1e5 is a common phase that cancels in |.|^2 -- but
    re-centring keeps the float64 phase honest). The structure factor is sampled
    on the SAME tau grid via the constant rescale u = u_per_tau * tau, so its
    prime peaks are directly comparable to the ramp."""
    g = np.asarray(gammas, dtype=float)
    diag = constant_density_diagnostics(g)
    upt = diag["u_per_tau"]

    unfolded = unfold_riemann_zeros(g)
    unfolded = unfolded - unfolded.mean()
    taus = np.linspace(1.0e-4, tau_max, n_tau)
    K = spectral_form_factor(unfolded, taus, window="hann")
    centres, means, errs = bin_form_factor(taus, K, n_bins)
    # Slope on the raw (averaged-in-the-fit) K: through-origin LS over many
    # points is lower-variance than fitting the handful of bins in [0.05, 0.4].
    slope_raw = small_tau_slope(taus, K, 0.05, 0.4)
    slope_binned = small_tau_slope(centres, means, 0.05, 0.4)

    # Structure factor on the prime band of the SAME tau axis (u = u_per_tau*tau).
    # Restrict to where the small primes live (tau <~ 0.3 at these heights) so the
    # normalisation is set by the prime peaks, not a far-out noise spike, and the
    # background median is the genuine inter-peak floor (the prominence reference).
    tau_S = np.linspace(0.02, 0.35, 8000)
    S = window_structure_factor(g, upt * tau_S, taper="hann")

    return {
        "gammas": g,
        "diag": diag,
        "taus": taus,
        "K": K,
        "bin_centres": centres,
        "bin_means": means,
        "bin_errs": errs,
        "slope": slope_raw,
        "slope_binned": slope_binned,
        "tau_S": tau_S,
        "S": S,
        "prime_taus": prime_taus(diag["rho_bar"], nmax_primes),
    }


def peak_resolution(res: dict) -> list:
    """For each predicted prime tau_p inside the structure-factor band, the
    (interpolated) S(tau_p) and its prominence -- S(tau_p) over the BAND MEAN of
    S -- the quantitative 'the prime peaks land on the predicted tau' check. The
    mean (not the median) is the reference: between the sharp peaks of a near-
    perfect zero set S ~ 0, so the median collapses and only the mean is a stable
    O(1) scale. A resolved Bragg peak has prominence > 1; a smeared/absent one
    sits at the mean. The S(tau_p) values themselves track the explicit-formula
    weight Lambda(n)/sqrt(n) -- the primes dominate, the prime powers are weaker."""
    tau_S, S = res["tau_S"], res["S"]
    ref = float(np.mean(S))
    out = []
    for n, tp, amp in res["prime_taus"]:
        if not (tau_S[0] <= tp <= tau_S[-1]):
            continue
        s_at = float(np.interp(tp, tau_S, S))
        out.append((n, tp, s_at, (s_at / ref) if ref > 0 else float("inf")))
    return out


def load_high_windows() -> list:
    """Ordered fixed-height windows (low -> high) for the #89 trend. W0 is the
    top `LOW_WINDOW_N` of the canonical cache (a low, mildly-varying-density
    anchor); the higher windows come from the committed window CSVs if present.
    Returns a list of {label, gammas}."""
    windows: list = []
    cached = np.asarray(load_riemann_zeros(), dtype=float)
    if cached.size >= LOW_WINDOW_N:
        windows.append(
            {"label": f"cache top {LOW_WINDOW_N}", "gammas": cached[-LOW_WINDOW_N:]}
        )
    for path in HIGH_WINDOW_CSVS:
        if path.exists():
            g = np.asarray(load_riemann_zeros(path), dtype=float)
            windows.append({"label": path.stem, "gammas": g})
    return windows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _compute_all(
    n_bins: int = 28, tau_max: float = 2.2, n_tau: int = 9000
) -> dict:
    """Load the zeros, build the form factor + its binning + the prime face."""
    gammas = np.asarray(load_riemann_zeros(), dtype=float)
    unfolded = unfold_riemann_zeros(gammas)

    taus = np.linspace(1.0e-4, tau_max, n_tau)
    K = spectral_form_factor(unfolded, taus, window="hann")
    centres, means, errs = bin_form_factor(taus, K, n_bins)
    slope = small_tau_slope(centres, means, 0.05, 0.4)

    # prime face: the raw structure factor (#80B) on the u axis.
    u = np.linspace(0.45, 3.35, 6000)
    S = P.zero_structure_factor(gammas, u, gamma_c=700.0)

    return {
        "gammas": gammas,
        "unfolded": unfolded,
        "taus": taus,
        "K": K,
        "bin_centres": centres,
        "bin_means": means,
        "bin_errs": errs,
        "slope": slope,
        "u": u,
        "S": S,
        "bridge": heisenberg_bridge(gammas),
    }


def build_summary_text(res: dict) -> str:
    g = res["gammas"]
    b = res["bridge"]
    slope = res["slope"]
    # slope distances to the two surmises
    d_gue, d_goe = abs(slope - 1.0), abs(slope - 2.0)
    verdict = "GUE (unitary)" if d_gue < d_goe else "GOE (orthogonal)"

    lines: list[str] = []
    lines.append(
        "Spectral form factor of the Riemann zeros: the GUE face (small tau) "
        "and the prime face (large u) of one Fourier transform (issue #84)."
    )
    lines.append("")
    lines.append(
        f"Input: {len(g)} cached nontrivial zeros, "
        f"{g.min():.3f} <= gamma <= {g.max():.3f}; unfolded by the "
        "Riemann-von Mangoldt N(T) to unit mean density."
    )
    lines.append("")
    lines.append("GUE FACE -- form factor K(tau) of the unfolded zeros:")
    lines.append(
        f"  Hann-windowed, {len(res['taus'])} tau-samples binned into "
        f"{len(res['bin_centres'])} bins."
    )
    lines.append(
        f"  Small-tau slope (fit on K vs tau, 0.05 <= tau <= 0.4): "
        f"{slope:.3f}."
    )
    lines.append(
        f"    GUE ramp slope = 1, GOE ramp slope = 2.  "
        f"|slope-1| = {d_gue:.3f},  |slope-2| = {d_goe:.3f}  ->  {verdict}."
    )
    lines.append("")
    lines.append(
        "  K(tau) binned (centre, mean, sem) with the surmises "
        "(tau < 0.05 omitted -- the disconnected DC spike):"
    )
    header = ("tau", "K measured", "sem", "K_GUE", "K_GOE", "K_Poisson")
    rows = []
    for c, m, e in zip(res["bin_centres"], res["bin_means"], res["bin_errs"]):
        if not math.isfinite(m) or c < 0.05:
            continue
        rows.append(
            (
                f"{c:.3f}",
                f"{m:.3f}",
                f"{e:.3f}",
                f"{float(gue_form_factor(c)):.3f}",
                f"{float(goe_form_factor(c)):.3f}",
                f"{float(poisson_form_factor(c)):.3f}",
            )
        )
    widths = [max(len(header[i]), max(len(r[i]) for r in rows)) for i in range(len(header))]
    fmt = "  ".join(f"{{:>{w}}}" for w in widths)
    lines.append("  " + fmt.format(*header))
    for r in rows:
        lines.append("  " + fmt.format(*r))
    lines.append("")
    lines.append("PRIME FACE -- structure factor S(u) of the raw zeros (#80B):")
    lines.append(
        "  Sharp Bragg peaks at u = log(p^k) (the prime comb); see "
        "knowledge/prime-cone/log-prime-diffraction.md."
    )
    lines.append("")
    lines.append("THE BRIDGE -- u = 2 pi rho(gamma) tau (height-dependent):")
    lines.append(f"  mean density rhobar = {b['rho_bar']:.4f}.")
    lines.append(
        f"  Heisenberg time tau = 1  <->  u_H = 2 pi rhobar = "
        f"{b['u_heisenberg']:.3f}  "
        f"(ranges {b['u_heisenberg_lo']:.3f} .. {b['u_heisenberg_hi']:.3f} "
        "across the sample)."
    )
    lines.append(
        f"  First prime u = log 2 = {math.log(2):.3f}  <->  tau = "
        f"{b['tau_log2']:.3f}  "
        f"(ranges {b['tau_log2_lo']:.3f} .. {b['tau_log2_hi']:.3f})."
    )
    lines.append(
        "  The wide tau-range of a single prime is why the comb smears in the "
        "unfolded form factor while the universal ramp stays sharp -- the two "
        "faces are genuinely different views, not one shared axis."
    )
    lines.append("")
    lines.append(
        "Interpretation. The unfolded zeros reproduce the universal GUE ramp-"
        "and-plateau, and the small-tau slope picks the unitary class over the "
        "orthogonal one -- an independent, two-point-correlation confirmation "
        "of the #8 nearest-neighbour-spacing result. The #80B prime comb is the "
        "large-u face of the same transform; the diagonal (universal ramp) -> "
        "off-diagonal (prime peaks) crossover is Berry's quantum-chaos picture. "
        "Noisy at ~2000 zeros: the ramp needs unfolding + binning. See "
        "knowledge/statistics/zero-form-factor.md."
    )
    return "\n".join(lines) + "\n"


def plot_form_factor(res: dict) -> None:
    """Three panels: (a) the form factor + ramp/plateau, (b) the small-tau
    slope (GUE vs GOE), (c) the prime face on the u axis with the bridge."""
    taus = res["taus"]
    K = res["K"]
    centres, means, errs = res["bin_centres"], res["bin_means"], res["bin_errs"]
    b = res["bridge"]

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.0))

    # Exclude the disconnected DC spike (tau < 0.05) from the binned markers.
    show = centres >= 0.05
    centres, means, errs = centres[show], means[show], errs[show]
    raw = taus >= 0.03

    # ---- (a) the form factor: ramp + plateau ----------------------------
    ax = axes[0]
    grid = np.linspace(0, taus.max(), 400)
    ax.plot(taus[raw], K[raw], color="0.75", lw=0.5, alpha=0.6,
            label=f"raw $K(\\tau)$ ({len(taus)} samples)")
    ax.errorbar(centres, means, yerr=errs, fmt="o", ms=4, color="#1f5fa8",
                ecolor="#1f5fa8", elinewidth=1, capsize=2, zorder=5,
                label="binned $K(\\tau)$")
    ax.plot(grid, gue_form_factor(grid), color="#c2452d", lw=2.0,
            label=r"GUE  $\min(|\tau|,1)$")
    ax.plot(grid, goe_form_factor(grid), color="#2a8a3e", lw=2.0, ls="--",
            label="GOE")
    ax.plot(grid, poisson_form_factor(grid), color="gray", lw=1.2, ls=":",
            label="Poisson $=1$")
    ax.axvline(1.0, color="0.4", lw=0.8, ls="-.")
    ax.text(1.0, 0.05, " Heisenberg\n time $\\tau=1$", fontsize=7.5,
            color="0.3", ha="left", va="bottom")
    ax.text(0.97, 0.40,
            "bridge to panel (c):  $u=2\\pi\\rho(\\gamma)\\,\\tau$\n"
            fr"$\tau{{=}}1\!\to\!u_H{{\approx}}{b['u_heisenberg']:.1f}$;  "
            fr"$\log2\!\to\!\tau{{\approx}}{b['tau_log2']:.2f}$",
            transform=ax.transAxes, fontsize=7.2, color="0.25",
            ha="right", va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.8", lw=0.5))
    ax.set_xlim(0, taus.max())
    ax.set_ylim(0, 1.6)
    ax.set_xlabel(r"$\tau$  (time / Heisenberg time)")
    ax.set_ylabel(r"form factor  $K(\tau)$")
    ax.set_title("(a) GUE FACE: unfolded zeros give the\n"
                 "RMT ramp-and-plateau", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=7.6)

    # ---- (b) the small-tau slope: GUE vs GOE ----------------------------
    ax = axes[1]
    m = centres <= 0.6
    ax.errorbar(centres[m], means[m], yerr=errs[m], fmt="o", ms=5,
                color="#1f5fa8", ecolor="#1f5fa8", elinewidth=1, capsize=2,
                zorder=5, label="binned $K(\\tau)$")
    xs = np.linspace(0, 0.6, 50)
    ax.plot(xs, xs, color="#c2452d", lw=2.0, label=r"GUE slope $=1$")
    ax.plot(xs, 2.0 * xs, color="#2a8a3e", lw=2.0, ls="--",
            label=r"GOE slope $=2$")
    ax.plot(xs, res["slope"] * xs, color="0.35", lw=1.4, ls=":",
            label=fr"fit slope $={res['slope']:.2f}$")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$K(\tau)$")
    ax.set_title("(b) the small-$\\tau$ slope picks the class\n"
                 "(unitary 1, not orthogonal 2)", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=8)

    # ---- (c) the prime face on the u axis -------------------------------
    ax = axes[2]
    u, S = res["u"], res["S"]
    ax.plot(u, S, color="#1f5fa8", lw=1.0)
    ax.fill_between(u, 0, S, color="#1f5fa8", alpha=0.12)
    for n, lg, _ in P.prime_power_peaks(26):
        ax.axvline(lg, color="#2a8a3e", lw=0.8, ls=":", alpha=0.7)
        ax.text(lg, 1.04, str(n), ha="center", va="bottom", fontsize=7.0,
                color="#1d6e30")
    ax.set_xlim(u.min(), u.max())
    ax.set_ylim(-0.02, 1.12)
    ax.set_xlabel(r"$u$  (conjugate to the raw ordinates $\gamma$)")
    ax.set_ylabel(r"structure factor  $S(u)$")
    ax.set_title("(c) PRIME FACE: same transform, raw zeros\n"
                 r"$S(u)$ combs at $u=\log p^k$ (#80B)", fontsize=9.5)

    fig.suptitle(
        "Spectral form factor of the Riemann zeros: the GUE ramp (small "
        r"$\tau$) and the prime comb (large $u$) of one transform (issue #84)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT_PNG, dpi=150)
    plt.close(fig)


# #84's slope from the full variable-density cache (gamma 14 .. 2515), the
# low-height baseline the windows are measured against (knowledge note / #84).
FULL_CACHE_SLOPE_84 = 0.855


def build_high_summary_text(windows: list, analyses: list) -> str:
    lines: list = []
    lines.append(
        "Spectral form factor at high height (#89, follow-up to #84): a narrow "
        "fixed-height window of zeros collapses the GUE ramp and the prime comb "
        "onto ONE tau axis."
    )
    lines.append("")
    lines.append(
        "Why a window. The density rho(gamma) = (1/2pi) log(gamma/2pi) grows "
        "only logarithmically, so a narrow block of consecutive zeros high on "
        "the critical line has nearly CONSTANT density. The unfolding bridge "
        "u = 2 pi rho(gamma) tau then becomes a single constant rescale, and a "
        "prime u = log p maps to a SHARP tau_p (not the band tau ~ 0.12 .. 0.86 "
        "that #84 saw smeared across the factor-of-7 density variation of the "
        "full cache). That is what lets the diagonal (universal ramp) and the "
        "off-diagonal (prime peaks) sit in one figure."
    )
    lines.append("")

    # per-window slope-vs-height table
    header = ("window", "N", "gamma_mid", "rhobar", "rho spread", "slope", "slope(bin)")
    rows = []
    for w, a in zip(windows, analyses):
        d = a["diag"]
        rows.append(
            (
                w["label"],
                str(d["n"]),
                f"{d['gamma_mid']:.0f}",
                f"{d['rho_bar']:.3f}",
                f"{d['rho_spread_frac']:.1e}",
                f"{a['slope']:.3f}",
                f"{a['slope_binned']:.3f}",
            )
        )
    widths = [max(len(header[i]), max(len(r[i]) for r in rows)) for i in range(len(header))]
    fmt = "  ".join(f"{{:>{w}}}" for w in widths)
    lines.append("SLOPE vs HEIGHT (GUE ramp slope = 1, GOE = 2):")
    lines.append("  " + fmt.format(*header))
    for r in rows:
        lines.append("  " + fmt.format(*r))
    lines.append(
        f"  #84 baseline (full variable-density cache, gamma 14..2515): "
        f"slope = {FULL_CACHE_SLOPE_84:.3f}."
    )
    lines.append(
        "  The fixed-height windows remove the variable-density smear; the "
        "slope is read on the cleaner ramp (raw through-origin LS over "
        "0.05 <= tau <= 0.4)."
    )
    lines.append("")

    # showcase (highest) window: the prime peaks on the ramp
    show = analyses[-1]
    d = show["diag"]
    lines.append(
        f"SHOWCASE WINDOW (highest): N = {d['n']} zeros, gamma in "
        f"{d['gamma_lo']:.0f} .. {d['gamma_hi']:.0f}, rho constant to "
        f"{d['rho_spread_frac']:.1e} (rhobar = {d['rho_bar']:.4f}, so "
        f"u = {d['u_per_tau']:.3f} * tau)."
    )
    lines.append(
        "  Prime peaks on the ramp: predicted tau_{p^k} = log(p^k)/(2 pi rhobar), "
        "with the windowed structure factor S(tau_p) and its prominence over the "
        "background median (a resolved Bragg peak has prominence >> 1):"
    )
    h2 = ("p^k", "tau_p", "S(tau_p)", "prominence")
    r2 = [
        (str(n), f"{tp:.4f}", f"{s_at:.3f}", f"{prom:.1f}")
        for n, tp, s_at, prom in peak_resolution(show)
    ]
    if r2:
        w2 = [max(len(h2[i]), max(len(r[i]) for r in r2)) for i in range(len(h2))]
        f2 = "  ".join(f"{{:>{w}}}" for w in w2)
        lines.append("  " + f2.format(*h2))
        for r in r2:
            lines.append("  " + f2.format(*r))
    lines.append("")
    lines.append(
        "Interpretation. At fixed high height the unfolding is a clean rescale: "
        "the universal GUE ramp (the diagonal/all-orbits average) and the sharp "
        "prime peaks at tau_p (the off-diagonal short orbits) live on the same "
        "tau axis -- Berry's diagonal -> off-diagonal crossover in a single "
        "figure, which #84 could not draw because rho varied across its sample. "
        "The ramp slope is read on the cleaner (constant-density) ramp. Ceiling: "
        "float64 keeps the phase honest to gamma ~ 1e6 and mpmath.zetazero costs "
        "~0.5 s/zero, so the reachable height is set by patience, not precision; "
        "the window width trades statistics (more zeros) against density "
        "constancy (a wider gamma-span), the #84 trade-off now quantified by the "
        "'rho spread' column. See knowledge/statistics/zero-form-factor.md."
    )
    return "\n".join(lines) + "\n"


def plot_high_window(windows: list, analyses: list) -> None:
    """Three panels: (a) the showcase window's GUE ramp + slope, (b) the
    single-axis crossover (ramp + the resolved prime comb), (c) the slope
    climbing toward the unitary 1 as the window height grows."""
    show = analyses[-1]
    d = show["diag"]
    taus, K = show["taus"], show["K"]
    centres, means, errs = show["bin_centres"], show["bin_means"], show["bin_errs"]

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.0))

    smin = centres >= 0.05
    cc, mm, ee = centres[smin], means[smin], errs[smin]
    raw = taus >= 0.03

    # ---- (a) the showcase ramp + plateau --------------------------------
    ax = axes[0]
    grid = np.linspace(0, taus.max(), 400)
    ax.plot(taus[raw], K[raw], color="0.78", lw=0.5, alpha=0.6,
            label=f"raw $K(\\tau)$ ({len(taus)} samples)")
    ax.errorbar(cc, mm, yerr=ee, fmt="o", ms=4, color="#1f5fa8",
                ecolor="#1f5fa8", elinewidth=1, capsize=2, zorder=5,
                label="binned $K(\\tau)$")
    ax.plot(grid, gue_form_factor(grid), color="#c2452d", lw=2.0,
            label=r"GUE  $\min(|\tau|,1)$")
    ax.plot(grid, goe_form_factor(grid), color="#2a8a3e", lw=2.0, ls="--",
            label="GOE")
    ax.plot(grid, show["slope"] * grid, color="0.35", lw=1.3, ls=":",
            label=fr"fit slope $={show['slope']:.2f}$")
    ax.axvline(1.0, color="0.4", lw=0.8, ls="-.")
    ax.set_xlim(0, taus.max())
    ax.set_ylim(0, 1.6)
    ax.set_xlabel(r"$\tau$  (time / Heisenberg time)")
    ax.set_ylabel(r"form factor  $K(\tau)$")
    ax.set_title(fr"(a) high window $\gamma\!\approx\!{d['gamma_mid']:.0f}$:"
                 "\nramp-and-plateau, $\\rho$ const to "
                 fr"${d['rho_spread_frac']:.0e}$", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=7.6)

    # ---- (b) the single-axis crossover ----------------------------------
    # At constant rho the bridge u = (2 pi rhobar) tau is ONE rescale, so the
    # diagonal/universal ramp (left axis, red) and the off-diagonal/arithmetic
    # prime peaks (right axis, blue, at tau_p marked in green) live on one tau
    # axis -- the figure #84 could not draw. Zoom to where the small primes sit.
    ax = axes[1]
    tau_S, S = show["tau_S"], show["S"]
    tau_view = 0.30
    band = cc <= tau_view
    ax.plot([0, tau_view], [0, tau_view], color="#c2452d", lw=2.0, zorder=4,
            label=r"GUE ramp $K=\tau$ (diagonal)")
    ax.errorbar(cc[band], mm[band], yerr=ee[band], fmt="o", ms=5,
                color="#c2452d", ecolor="#c2452d", elinewidth=1, capsize=2,
                zorder=5, label=r"binned $K(\tau)$")
    ax.set_xlim(0, tau_view)
    ax.set_ylim(0, 0.42)
    ax.set_xlabel(r"$\tau = u / (2\pi\bar\rho)$  (constant-$\rho$ rescale)")
    ax.set_ylabel(r"form factor  $K(\tau)$  (diagonal / ramp)", color="#c2452d")
    ax.tick_params(axis="y", labelcolor="#c2452d")

    axr = ax.twinx()
    axr.fill_between(tau_S, 0, S, color="#1f5fa8", alpha=0.18, zorder=1)
    axr.plot(tau_S, S, color="#1f5fa8", lw=0.8, zorder=2)
    axr.set_ylim(0, 1.32)
    axr.set_ylabel(r"structure factor  $S(\tau)$  (off-diagonal / primes)",
                   color="#1f5fa8")
    axr.tick_params(axis="y", labelcolor="#1f5fa8")
    for n, tp, amp in show["prime_taus"]:
        if tp >= tau_view:
            continue
        is_p = _is_prime(n)
        axr.axvline(tp, color="#2a8a3e", lw=0.8 if is_p else 0.5, ls=":",
                    alpha=0.65 if is_p else 0.30, zorder=0)
        if is_p:  # label primes only; prime powers stay faint unlabelled lines
            axr.text(tp, 1.17, str(n), ha="center", va="bottom", fontsize=7.6,
                     color="#1d6e30", fontweight="bold")
    axr.text(0.012, 1.17, "$p$:", fontsize=7.0, color="#1d6e30", va="bottom")
    ax.set_title("(b) the crossover in ONE figure: universal ramp\n"
                 r"(diagonal) + sharp prime peaks at $\tau_p=\log p^k/2\pi\bar\rho$",
                 fontsize=9.5)
    ax.legend(loc="upper left", fontsize=7.6)

    # ---- (c) slope vs height --------------------------------------------
    ax = axes[2]
    xs = np.array([np.log10(a["diag"]["gamma_mid"]) for a in analyses])
    ys = np.array([a["slope"] for a in analyses])
    ax.axhline(1.0, color="#c2452d", lw=2.0, label="GUE $=1$")
    ax.axhline(2.0, color="#2a8a3e", lw=2.0, ls="--", label="GOE $=2$")
    ax.axhline(FULL_CACHE_SLOPE_84, color="0.55", lw=1.0, ls=":",
               label=fr"#84 full cache $={FULL_CACHE_SLOPE_84:.2f}$")
    ax.plot(xs, ys, "o-", color="#1f5fa8", ms=7, lw=1.4,
            label="fixed-height windows")
    for a, x, y in zip(analyses, xs, ys):
        ax.annotate(fr"$\gamma\!\approx\!{a['diag']['gamma_mid']:.0f}$",
                    (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=6.8, color="0.3")
    ax.set_xlim(min(xs) - 0.5, max(xs) + 0.6)
    ax.set_ylim(0.0, 2.2)
    ax.set_xlabel(r"$\log_{10}\,\gamma_{\rm mid}$  (window height)")
    ax.set_ylabel(r"small-$\tau$ slope of $K(\tau)$")
    ax.set_title("(c) the slope sharpens toward the unitary 1\n"
                 "as the window climbs the critical line", fontsize=9.5)
    ax.legend(loc="center right", fontsize=7.6)

    fig.suptitle(
        "Spectral form factor at high height: the GUE ramp and the prime comb "
        r"on one $\tau$ axis (issue #89)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT_HIGH_PNG, dpi=150)
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    res = _compute_all()
    text = build_summary_text(res)
    OUTPUT_TXT.write_text(text)
    print(text)
    print(f"Wrote {OUTPUT_TXT.name}.")
    plot_form_factor(res)
    print(f"Wrote {OUTPUT_PNG.name} ({time.time() - t0:.1f}s).")

    # The #89 high-height follow-up: needs the committed window CSVs.
    windows = load_high_windows()
    if len(windows) >= 2:
        t1 = time.time()
        analyses = [analyze_high_window(w["gammas"]) for w in windows]
        htext = build_high_summary_text(windows, analyses)
        OUTPUT_HIGH_TXT.write_text(htext)
        print(htext)
        print(f"Wrote {OUTPUT_HIGH_TXT.name}.")
        plot_high_window(windows, analyses)
        print(f"Wrote {OUTPUT_HIGH_PNG.name} ({time.time() - t1:.1f}s).")
    else:
        print("(#89 high-height windows not found -- run "
              "`maass/compute_riemann_zeros.py --start 100000 --n-zeros 1500` "
              "and `--start 1000000 --n-zeros 1500` to generate them.)")


if __name__ == "__main__":
    main()
