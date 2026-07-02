"""Radial log-prime superlattice and the diffraction route to the Riemann zeros
(in its source project, a sibling of cone_aperiodic / cone_bloch and the
original harmonic-functions-and-zeta arc -- neither lives in this repo).

VENDORED NOTE (this repo): copied from the originating research project as a
dependency of `eta_two_component.py`, which uses only `von_mangoldt` and
`prime_power_peaks`. This file does run standalone on the cached zeros. Kept
whole so the vendored code matches its source; see CLAUDE.md "Gotchas".

Why the cone: the critical-line eigenfunction of the sign-changing cone is

    u(r) = r^{-1/2} cos(tau log r)                         (cone_log_mode.py)

a plain cosine in the natural log-coordinate t = log r. So the principled way to
inject prime structure into the cone is *radially*, with interfaces / scatterers
placed at the radii r = p (i.e. t = log p) -- a Kronig-Penney "superlattice" in
log r. The observable that ties this to zeta is not the transfer/band spectrum
(that is just integrable Bragg physics) but the **diffraction /
Fourier spectrum** of the arrangement, where the Riemann-von Mangoldt *explicit
formula* plants the connection between primes and zeros.

The duality is asymmetric -- and the asymmetry is the point:

  FORWARD  (zeros -> primes), Dyson's "the zeros are a 1-D quasicrystal":
      the point set of zero ordinates {gamma_n} has a genuine PURE-POINT
      diffraction. Its structure factor

          S(u) = ( sum_{gamma>0} w(gamma) cos(gamma u) )^2

      (a windowed Fourier transform of a *finite* sample of zeros) shows sharp
      Bragg peaks at u = log(p^k) -- the log of every PRIME POWER -- with peak
      amplitude proportional to the von Mangoldt weight Lambda(n)/sqrt(n), and
      *no* peak at composites with two or more distinct prime factors
      (Lambda(n) = 0 there). This is the explicit formula read as diffraction.

  REVERSE  (primes -> zeros):
      the finite log-prime superlattice transform (a smoothed partial sum of the
      Dirichlet series of -zeta'/zeta)
          T(tau) = sum_{n<=N} Lambda(n) n^{-1/2} taper(n) e^{-i tau log n}
      develops resonance bumps whose locations already match the first zeros
      gamma_n to ~0.1 -- the finite prime arrangement "sees" the zeros. But the
      bumps are BOUNDED: a partial sum cannot diverge, and the Dirichlet series
      does not converge on the critical line. The zeros are genuine simple POLES
      only of the analytic continuation -zeta'/zeta(1/2 + i tau) (residue 1 at
      each simple zero of zeta). So the asymmetry is sharp: the zero set is a true
      pure-point quasicrystal (forward), whereas the prime set encodes the zeros
      as POLES of a continuation, which the finite Fourier transform approaches
      but never reaches. (Getting an artifact-free reverse hinges on a correct
      von Mangoldt weight -- composites with two distinct prime factors carry
      Lambda = 0; including them scrambles the alignment.)

Cone reading: a sign-changing cone whose radial interfaces sit at r = p resonates,
in the critical-line variable tau, exactly at the Riemann zeros -- but only through
the continuation, the same way -zeta'/zeta encodes them.

All functions are float64 / numpy except the reverse-direction -zeta'/zeta probe,
which lazily imports mpmath. Run as a script to (re)draw figures/cone_log_prime.png.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RIEMANN_ZEROS_CSV = DATA_DIR / "riemann_zeros.csv"

FIGDIR = Path(__file__).resolve().parent / "figures"


# --------------------------------------------------------------------------- #
# Arithmetic: the von Mangoldt weight and the predicted Bragg peaks.
# --------------------------------------------------------------------------- #
def von_mangoldt(nmax: int) -> np.ndarray:
    """Lambda(n) for n = 0 .. nmax.

    Lambda(n) = log p if n = p^k (a prime power, k >= 1), else 0. Lambda(0) =
    Lambda(1) = 0. A sieve of Eratosthenes flags composites (so primes are the
    unflagged entries); when p is found prime we stamp log p onto every power
    p, p^2, ... <= nmax. (Note: "unmarked => prime" alone is WRONG -- non-prime-power
    composites like 6, 10, 15 are never stamped by the power loop and would be
    mis-read as prime; the explicit composite sieve is what makes Lambda(6)=0.)
    """
    if nmax < 0:
        raise ValueError("nmax must be >= 0")
    lam = np.zeros(nmax + 1)
    composite = np.zeros(nmax + 1, dtype=bool)
    for p in range(2, nmax + 1):
        if composite[p]:
            continue
        composite[p * p:nmax + 1:p] = True   # mark multiples (Eratosthenes)
        lp = float(np.log(p))
        pk = p
        while pk <= nmax:                    # stamp the prime powers p^k
            lam[pk] = lp
            pk *= p
    return lam


def is_prime_power(n: int) -> bool:
    """True iff n = p^k for a single prime p and some k >= 1 (n >= 2)."""
    if n < 2:
        return False
    return bool(von_mangoldt(n)[n] > 0.0)


def prime_power_peaks(nmax: int) -> List[Tuple[int, float, float]]:
    """The predicted forward-diffraction Bragg peaks up to integer nmax.

    Returns a list of (n, log n, Lambda(n)/sqrt(n)) for every prime power
    2 <= n <= nmax, i.e. the peak position u = log n and its predicted amplitude
    weight (the explicit-formula coefficient).
    """
    lam = von_mangoldt(nmax)
    out: List[Tuple[int, float, float]] = []
    for n in range(2, nmax + 1):
        if lam[n] > 0.0:
            out.append((n, float(np.log(n)), float(lam[n] / np.sqrt(n))))
    return out


# --------------------------------------------------------------------------- #
# Data: the cached Riemann zeros.
# --------------------------------------------------------------------------- #
def load_riemann_zeros(path: Optional[Path] = None) -> np.ndarray:
    """Load the cached imaginary ordinates gamma_n (float64), ascending."""
    path = Path(path) if path is not None else RIEMANN_ZEROS_CSV
    gammas: List[float] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            gammas.append(float(row["gamma"]))
    return np.array(sorted(gammas), dtype=float)


# --------------------------------------------------------------------------- #
# FORWARD: the zero-set structure factor (Dyson's quasicrystal).
# --------------------------------------------------------------------------- #
def gaussian_window(gammas: np.ndarray, gamma_c: float) -> np.ndarray:
    """A smooth Gaussian taper exp(-(gamma/gamma_c)^2) on the zero heights.

    Tapering the finite zero sample suppresses the truncation (Gibbs) ripple so
    the Bragg peaks come out as clean bumps of width ~1/gamma_c instead of being
    buried in sidelobes.
    """
    return np.exp(-(np.asarray(gammas, dtype=float) / gamma_c) ** 2)


def zero_structure_factor(
    gammas: np.ndarray,
    u: np.ndarray,
    gamma_c: float = 700.0,
    normalize: bool = True,
) -> np.ndarray:
    """Diffraction intensity S(u) of the symmetric zero set {+/- gamma_n}.

        S(u) = ( sum_{gamma>0} w(gamma) cos(gamma u) )^2,   w = Gaussian window.

    This is |sum_{+/-gamma} w e^{i gamma u}|^2 / 4 -- the structure factor of the
    real, even point set of zero ordinates. Peaks at u = log(prime power).
    """
    gammas = np.asarray(gammas, dtype=float)
    u = np.atleast_1d(np.asarray(u, dtype=float))
    w = gaussian_window(gammas, gamma_c)
    # (len(u), N) cosine matrix; fine for ~2000 zeros x a few thousand u.
    amp = (w[None, :] * np.cos(np.outer(u, gammas))).sum(axis=1)
    S = amp ** 2
    if normalize and S.size and S.max() > 0:
        S = S / S.max()
    return S


def peak_amplitudes_at_prime_powers(
    gammas: np.ndarray,
    nmax: int = 13,
    gamma_c: float = 700.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the (un-normalized) structure factor at u = log n for prime powers.

    Returns (ns, sqrtS, weight) where sqrtS = sqrt(S(log n)) is the measured peak
    amplitude and weight = Lambda(n)/sqrt(n) is the explicit-formula prediction.
    The two are proportional (the diffraction *is* the explicit formula).
    """
    peaks = prime_power_peaks(nmax)
    ns = np.array([n for n, _, _ in peaks])
    logs = np.array([lg for _, lg, _ in peaks])
    weight = np.array([w for _, _, w in peaks])
    S = zero_structure_factor(gammas, logs, gamma_c=gamma_c, normalize=False)
    return ns, np.sqrt(S), weight


# --------------------------------------------------------------------------- #
# REVERSE: the log-prime superlattice transform and the -zeta'/zeta continuation.
# --------------------------------------------------------------------------- #
def vonmangoldt_transform(
    taus: np.ndarray,
    nmax: int,
    riesz: bool = True,
) -> np.ndarray:
    """The naive finite log-prime superlattice transform (truncated -zeta'/zeta):

        T(tau) = | sum_{2<=n<=nmax} Lambda(n) n^{-1/2} taper(n) e^{-i tau log n} |.

    With `riesz` a log-linear (Riesz) taper (1 - log n / log(nmax+1)) is applied
    to soften the truncation. This does NOT cleanly resolve the zeros -- the
    Dirichlet series diverges on the critical line -- which is exactly the point
    (see `log_deriv_zeta_abs` for the direction that does).
    """
    taus = np.atleast_1d(np.asarray(taus, dtype=float))
    n = np.arange(2, nmax + 1)
    lam = von_mangoldt(nmax)[2:]
    logn = np.log(n)
    coef = lam / np.sqrt(n)
    if riesz:
        coef = coef * np.clip(1.0 - logn / np.log(nmax + 1), 0.0, None)
    phase = np.exp(-1j * np.outer(taus, logn))
    return np.abs((coef[None, :] * phase).sum(axis=1))


def log_deriv_zeta_abs(taus: np.ndarray, dps: int = 25) -> np.ndarray:
    """| -zeta'/zeta(1/2 + i tau) | via mpmath (the analytic continuation).

    The zeros rho = 1/2 + i gamma are simple zeros of zeta, hence simple POLES of
    -zeta'/zeta, so this blows up at tau = gamma_n. This is the rigorous reverse
    direction -- the zeros appear as poles of the continuation, not as Bragg peaks
    of the (divergent) Dirichlet series. mpmath is imported lazily.
    """
    import mpmath as mp

    taus = np.atleast_1d(np.asarray(taus, dtype=float))
    with mp.workdps(dps):
        out = np.empty(taus.shape, dtype=float)
        for i, t in enumerate(taus.ravel()):
            s = mp.mpf(1) / 2 + 1j * mp.mpf(float(t))
            val = -mp.zeta(s, derivative=1) / mp.zeta(s)
            out.ravel()[i] = float(abs(val))
    return out


# --------------------------------------------------------------------------- #
# Cone tie-in: the radial interface radii of the log-prime superlattice.
# --------------------------------------------------------------------------- #
def superlattice_radii(nmax: int, weighted: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Interface radii r = n (so t = log r = log n) and their weights for the cone.

    With `weighted`, only prime powers carry weight (Lambda(n) = log p); otherwise
    every prime power is a unit scatterer. A critical-line cone mode cos(tau log r)
    sampled at these radii sums to Re T(tau) -- the reverse transform above -- so
    the cone's log-prime superlattice resonates (through the continuation) at the
    Riemann zeros.
    """
    lam = von_mangoldt(nmax)
    ns = np.array([n for n in range(2, nmax + 1) if lam[n] > 0.0])
    if weighted:
        wts = np.array([lam[n] / np.sqrt(n) for n in ns])
    else:
        wts = np.ones_like(ns, dtype=float)
    return ns.astype(float), wts


# --------------------------------------------------------------------------- #
# Figure.
# --------------------------------------------------------------------------- #
def _make_figure() -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGDIR.mkdir(exist_ok=True)
    out = FIGDIR / "cone_log_prime.png"

    gammas = load_riemann_zeros()
    gamma_c = 700.0

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.0))

    # ---- (a) forward diffraction: zeros -> primes ------------------------
    ax = axes[0]
    u = np.linspace(0.45, 3.35, 6000)
    S = zero_structure_factor(gammas, u, gamma_c=gamma_c)
    ax.plot(u, S, color="#1f5fa8", lw=1.0)
    ax.fill_between(u, 0, S, color="#1f5fa8", alpha=0.12)
    # prime-power peaks (green) and a few composites that DON'T peak (red).
    for n, lg, _ in prime_power_peaks(26):
        ax.axvline(lg, color="#2a8a3e", lw=0.8, ls=":", alpha=0.7)
        ax.text(lg, 1.04, str(n), ha="center", va="bottom",
                fontsize=7.5, color="#1d6e30")
    for n in (6, 10, 12, 14, 15):
        lg = np.log(n)
        ax.axvline(lg, color="#c2452d", lw=0.8, ls="--", alpha=0.7)
        ax.text(lg, -0.085, str(n), ha="center", va="top",
                fontsize=7.5, color="#9c2c1c")
    ax.set_xlim(u.min(), u.max())
    ax.set_ylim(-0.02, 1.12)
    ax.set_xlabel(r"$u$  (conjugate to the zero ordinates $\gamma$)")
    ax.set_ylabel(r"structure factor  $S(u)$")
    ax.set_title("(a) FORWARD: the zeros are a 1-D quasicrystal\n"
                 r"$S(u)$ Bragg-peaks at $u=\log p^k$ (green); "
                 "composites (red) absent", fontsize=9.5)

    # ---- (b) peak amplitude proportional to Lambda(n)/sqrt(n) ------------
    ax = axes[1]
    ns, sqrtS, weight = peak_amplitudes_at_prime_powers(gammas, nmax=29, gamma_c=gamma_c)
    # least-squares slope through the origin, for the guide line.
    slope = float((weight @ sqrtS) / (weight @ weight))
    corr = float(np.corrcoef(weight, sqrtS)[0, 1])
    ax.scatter(weight, sqrtS, c="#7a1fa8", s=28, zorder=3)
    for n, x, y in zip(ns, weight, sqrtS):
        ax.annotate(str(int(n)), (x, y), textcoords="offset points",
                    xytext=(4, 3), fontsize=7.5, color="#5a1580")
    xs = np.linspace(0, weight.max() * 1.05, 50)
    ax.plot(xs, slope * xs, color="0.5", lw=1.0, ls="--",
            label=fr"$\sqrt{{S}}\,\propto\,\Lambda(n)/\sqrt{{n}}$  ($r={corr:.4f}$)")
    ax.set_xlabel(r"explicit-formula weight  $\Lambda(n)/\sqrt{n}$")
    ax.set_ylabel(r"measured peak amplitude  $\sqrt{S(\log n)}$")
    ax.set_title("(b) the diffraction IS the explicit formula\n"
                 "peak heights track the von Mangoldt weights", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=8.5)

    # ---- (c) reverse: primes -> zeros needs the continuation -------------
    ax = axes[2]
    taus = np.linspace(8.0, 38.0, 1400)
    zlz = log_deriv_zeta_abs(taus)
    ax.semilogy(taus, zlz, color="#c2452d", lw=1.0,
                label=r"$|-\zeta'/\zeta(\frac{1}{2}+i\tau)|$ (continuation: poles)")
    # naive finite superlattice transform, rescaled to share the axis.
    T = vonmangoldt_transform(taus, nmax=2000)
    T = T / T.max() * np.median(zlz) * 4.0
    ax.semilogy(taus, T, color="#1f5fa8", lw=0.9, alpha=0.7,
                label=r"finite $\sum\Lambda(n)n^{-1/2}e^{-i\tau\log n}$ (bounded bumps)")
    for g in gammas[gammas < taus.max()]:
        if g > taus.min():
            ax.axvline(g, color="0.35", lw=0.7, ls=":", alpha=0.6)
    ax.set_xlim(taus.min(), taus.max())
    ax.set_xlabel(r"$\tau$  (the critical-line variable)")
    ax.set_ylabel(r"$|\,\cdot\,|$  (log scale)")
    ax.set_title("(c) REVERSE: primes $\\to$ zeros\n"
                 r"poles of $-\zeta'/\zeta$ sit on $\gamma_n$ (dotted); "
                 "the finite sum only bumps", fontsize=9.5)
    ax.legend(loc="upper right", fontsize=7.8)

    fig.suptitle("Radial log-prime superlattice on the sign-changing cone: "
                 "the diffraction route between primes and the Riemann zeros",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    path = _make_figure()
    print("wrote", path)
