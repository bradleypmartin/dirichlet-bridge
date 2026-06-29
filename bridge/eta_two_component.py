r"""The eta zero set as a two-component (crystal x GUE) spectrum.

Follow-up to the eta migration (harmonic_bridge.py), the rate law (rate_law.py), and the
Gram-phase duality. The eta interpolant's ground string splits onto TWO lines:

  * sigma = 1  -- the prefactor (1 - 2^{1-s}) zeros: a PERFECT ARITHMETIC COMB,
                  teeth at  t = 2 pi k / ln2,  period 2 pi/ln2 ~ 9.0647, perfectly rigid;
  * sigma = 1/2 -- the genuine zeta zeros: a GUE-random sequence, packing  log2(t/2pi)
                  per comb tooth (the ratio of the two counting densities).

So the eta zero set, projected onto the t-axis, is a RIGID CRYSTALLINE COMB superposed
with a GUE sequence -- a genuine two-component spectrum. This module asks the fine-structure
question the density bijection (which only fixes the COUNTS) cannot: are the two
components INDEPENDENT ("GUE + a deterministic backbone"), or do the zeta zeros "see" the
comb? Three lenses, the third decisive:

LENS 1 -- the packing (one-point).  `zeros_per_tooth` reproduces the measured
  1,2,2,3,3,3,4,3 zeta zeros per comb gap and pins it to log2(t/2pi); the comb-phase
  phi_n = frac(gamma_n ln2/2pi) (`comb_phase`) is asymptotically EQUIDISTRIBUTED
  (Hlawka/Fujii) -- to leading order the zeta zeros are spread uniformly across the comb
  cell, so at the one-point level the union looks like an independent superposition.

LENS 2 -- the rigidity (two-point, long range).  The number variance Sigma^2(L) of the
  UNION (`union_number_variance`). A rigid comb contributes a BOUNDED, oscillating Sigma^2
  (a picket fence: variance frac(L)(1-frac(L)) <= 1/4); the GUE part contributes
  ~(1/pi^2) log L. The measured union Sigma^2 tracks  GUE-log + a bounded picket ripple
  -- i.e. the comb does NOT change the long-range rigidity CLASS, it just adds a bounded
  backbone. At the bulk/smooth level the union is the independent superposition the
  one-point picture suggested ("GUE + rigid backbone", number variances ADD).

LENS 3 -- the p=2 resonance (the headline; the components are NOT independent).  The comb
  teeth sit at  t = 2 pi k/ln2, so as a point process the comb's spectral content lives at
  the angular frequencies  omega = m * ln2,  m in Z -- which are EXACTLY the p = 2
  prime-power frequencies  log(2^m). And by the explicit formula the zeta-zero density
  carries an oscillation at every prime-power frequency,

      rho_osc(T) = -(1/pi) sum_p sum_k (log p) p^{-k/2} cos(k T log p),

  so probing the zeros at any frequency omega = log(p^k) resonates with amplitude the
  explicit-formula coefficient (`prime_resonance`):

      <cos(gamma_n log n)>  ~  -(Lambda(n)/sqrt(n)) * T_max / (2 pi N)   (n = p^k),
                            ~  0                                          (otherwise).

  At the COMB frequencies (the powers of 2) this LOCKS the comb to the zeta zeros: the
  measured <cos(m gamma ln2)> matches the p=2^m coefficient to ~4 digits (-0.098 at
  m=1), and the sign is NEGATIVE -- the comb teeth (cos = +1) sit at the zeta-zero density
  MINIMA. So the sigma=1 comb is not an independent backbone: it is the p=2 sub-comb of
  the zeta zeros' OWN prime structure (the first Bragg peak of the structure
  factor is at log2). The two components share the prime-2 lattice -- the comb is the
  "p=2 ghost".

The synthesis: the eta two-component spectrum is a rigid crystal superposed with the GUE
zeta zeros, additive (independent) in the smooth bulk -- BUT phase-locked at the comb
frequency through the prime 2. That resonance is the same object as the explicit formula,
the structure factor / form factor, and the prime peaks the repo has been tracking all
along; it is where the discrete<->continuous bridge's eta side meets the prime<->zero arc.

Forward-only / honesty. The sigma=1/2 family IS zeta's
zeros (the eta/warp targets factor through zeta) and the sigma=1 comb is the closed-form
lattice 2 pi k/ln2; both are REFERENCE sequences here, not outputs of any zero-finding.
The two families were established blind (warp_bridge.count_on_lines, the
local-minimum-seeded finder -- no zero locations injected). This module takes those families and
measures their JOINT statistics: the genuinely forward content is the two-component law
and the p=2 lock, NOT a rediscovery of the zeros. No hb.migrate (homotopy seeded from a
known zero) is used.

Run directly to validate + plot: writes bridge/figures/eta_two_component.png.
"""
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# This module sits in a flat source dir alongside its bare cross-imports; put that
# directory on sys.path so a direct ``python eta_two_component.py`` run resolves them
# (pytest gets the same path from the root conftest.py).
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from gue_spacing import load_riemann_zeros, unfold_riemann_zeros  # noqa: E402
import spectral_rigidity as sr  # noqa: E402  (number_variance + GUE reference)
from zero_form_factor import gue_form_factor, poisson_form_factor  # noqa: E402
import cone_log_prime as clp  # noqa: E402  (prime_power_peaks, zero_structure_factor)

PI = math.pi
LN2 = math.log(2.0)
COMB_PERIOD = 2.0 * PI / LN2          # t-spacing of the sigma=1 prefactor zeros (~9.0647)
COMB_DENSITY = LN2 / (2.0 * PI)       # constant density of the comb (~0.1103 per unit t)


# ===========================================================================
# The two families
# ===========================================================================
def comb_teeth(t_max: float, t_min: float = 0.0) -> np.ndarray:
    """The sigma=1 prefactor zeros (1 - 2^{1-s} = 0): t_k = 2 pi k / ln2, the rigid comb.

    A perfect arithmetic lattice of constant density ln2/2pi; the crystalline component of
    the eta two-component spectrum. Returns the teeth with t_min < t_k <= t_max.
    """
    k_lo = int(math.floor(t_min / COMB_PERIOD)) + 1
    k_hi = int(math.floor(t_max / COMB_PERIOD))
    return np.array([COMB_PERIOD * k for k in range(k_lo, k_hi + 1)], dtype=float)


def comb_phase(gammas: np.ndarray) -> np.ndarray:
    """Comb-interval position phi_n = frac(gamma_n ln2 / 2pi) in [0, 1).

    phi = 0 sits ON a comb tooth (t = 2 pi k/ln2); phi = 1/2 is the cell midpoint. By
    Hlawka/Fujii equidistribution {gamma_n alpha} is uniform mod 1 for any fixed alpha,
    so phi_n is asymptotically uniform -- the leading-order one-point independence. The
    p=2 explicit-formula term (LENS 3) is the sub-leading modulation: a deficit at phi=0.
    """
    g = np.asarray(gammas, dtype=float)
    return np.mod(g * LN2 / (2.0 * PI), 1.0)


# ===========================================================================
# LENS 1 -- packing: zeta zeros per comb tooth vs log2(t/2pi)
# ===========================================================================
def zeros_per_tooth(
    gammas: np.ndarray, n_gaps: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count zeta zeros in each of the first `n_gaps` comb gaps (t_k, t_{k+1}].

    Returns (k, counts, predicted) with k = 1..n_gaps, counts[i] = #{gamma in the gap
    between teeth k and k+1, i.e. (2 pi k/ln2, 2 pi(k+1)/ln2]}, predicted[i] =
    log2(t_center/2pi) the inter-tooth density ratio (the zeta-zero density
    (1/2pi)ln(t/2pi) over the comb density ln2/2pi). Reproduces the measured 1,2,2,3,3,3,4,3.
    """
    g = np.sort(np.asarray(gammas, dtype=float))
    teeth = COMB_PERIOD * np.arange(1, n_gaps + 2)        # t_1, t_2, ..., t_{n_gaps+1}
    counts = (np.searchsorted(g, teeth[1:], side="right")
              - np.searchsorted(g, teeth[:-1], side="right")).astype(int)
    centers = 0.5 * (teeth[:-1] + teeth[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        predicted = np.log2(np.maximum(centers, 1e-9) / (2.0 * PI))
    return np.arange(1, n_gaps + 1), counts, predicted


def phase_uniformity(gammas: np.ndarray, n_bins: int = 12) -> Dict[str, float]:
    """Equidistribution diagnostic for the comb phase phi_n (LENS 1): histogram counts and
    a chi^2 against uniform, plus the first cosine Fourier coefficient.

    The comb phase is ASYMPTOTICALLY equidistributed (Weyl/Hlawka: {gamma alpha} uniform
    mod 1), which is what makes the LENS-1 count log2(t/2pi) work. But the finite-N
    histogram is NOT flat -- and its non-uniformity IS LENS 3: the histogram's j-th cosine
    Fourier mode is  c_j = (2/N) sum cos(2 pi j phi_n) = 2 <cos(gamma * j ln2)>, the p=2^j
    resonance. So a large chi^2/dof here is the p=2 prime-power comb seen in phase space, not
    a failure of equidistribution. Returns per-bin counts, reduced chi^2, and the leading
    c1 = 2 <cos(gamma ln2)> (the p=2 dip at phi=0, predicted negative).
    """
    phi = comb_phase(gammas)
    N = phi.size
    hist, _ = np.histogram(phi, bins=n_bins, range=(0.0, 1.0))
    expected = N / n_bins
    chi2 = float(((hist - expected) ** 2 / expected).sum())
    c1 = float(2.0 * np.cos(2.0 * PI * phi).mean())
    return {"counts": hist, "chi2_per_dof": chi2 / (n_bins - 1),
            "cos1_coeff": c1, "expected_per_bin": float(expected)}


# ===========================================================================
# LENS 3 -- the p=2 resonance: <cos(gamma log n)> vs the explicit formula
# ===========================================================================
def prime_resonance(gammas: np.ndarray, freq: float) -> float:
    """The resonance amplitude  <cos(gamma_n * freq)>  of the zeta-zero set at frequency `freq`.

    By the explicit formula this is ~0 unless freq = log(p^k) (a prime power), where it
    resonates with the p^k Bragg term. The comb frequencies are freq = m ln2 = log(2^m),
    so the comb probes exactly the p=2 family. (Conjugate sin part is ~0 by the cosine
    symmetry of the zero set -- this is the real, even structure factor.)
    """
    g = np.asarray(gammas, dtype=float)
    return float(np.cos(g * freq).mean())


def resonance_prediction(gammas: np.ndarray, n: int) -> float:
    """Explicit-formula prediction for the resonance at freq = log n (n a prime power):

        <cos(gamma log n)>  ~  -(Lambda(n)/sqrt(n)) * T_max / (2 pi N),

    Lambda(n)/sqrt(n) = log p / p^{k/2} the explicit-formula coefficient. Integrating the
    oscillatory density -(1/pi) sum (log p) p^{-k/2} cos(kT log p) against cos(T log n)
    over [0, T_max] keeps only the matched p=2^.. /p^k term (others incommensurate -> 0).
    Non-prime-powers have Lambda(n) = 0 -> prediction 0.
    """
    g = np.asarray(gammas, dtype=float)
    N, t_max = g.size, float(g.max())
    lam = clp.von_mangoldt(n)[n]                  # Lambda(n): log p if n=p^k else 0
    coeff = float(lam) / math.sqrt(n)
    return -coeff * t_max / (2.0 * PI * N)


def comb_resonance_table(
    gammas: np.ndarray, m_max: int = 4
) -> List[Tuple[int, float, float, float]]:
    """The comb-frequency resonances: for m = 1..m_max, the comb harmonic omega = m ln2 =
    log(2^m). Returns (m, omega, measured <cos>, predicted p=2^m amplitude). The decisive
    LENS-3 table -- the comb teeth lock to the p=2 prime powers of the zeta zeros.
    """
    out: List[Tuple[int, float, float, float]] = []
    for m in range(1, m_max + 1):
        omega = m * LN2
        meas = prime_resonance(gammas, omega)
        pred = resonance_prediction(gammas, 2 ** m)
        out.append((m, omega, meas, pred))
    return out


# ===========================================================================
# LENS 2 -- long-range rigidity: number variance of the union
# ===========================================================================
def union_unfolded(gammas: np.ndarray, t_max: float) -> Tuple[np.ndarray, float]:
    """Unit-density unfolded UNION spectrum {gamma_n} U {2 pi k/ln2} on the t-axis up to t_max.

    Unfold every union point t by the smooth union counting function
        N_union(t) = N_zeta_smooth(t) + (t ln2)/(2pi)
    (the RvM smooth count of the zeros + the comb staircase), then a single global affine
    rescale to exactly unit mean density (sr.unfold_to_unit_density). Returns (xi, f_comb)
    where f_comb = mean comb fraction COMB_DENSITY/(COMB_DENSITY + rho_zeta) over the band
    (the Berry-Robnik mixing fraction for the independent-superposition reference).
    """
    g = np.asarray(gammas, dtype=float)
    g = g[g <= t_max]
    teeth = comb_teeth(t_max, t_min=float(g.min()))
    union = np.sort(np.concatenate([g, teeth]))
    # smooth union counting at each union point: RvM smooth (reuse unfold_riemann_zeros,
    # which maps t -> N_zeta_smooth(t)) + the comb staircase's smooth part t*ln2/2pi.
    xi_raw = unfold_riemann_zeros(union) + union * COMB_DENSITY
    xi = sr.unfold_to_unit_density(xi_raw)
    rho_zeta = np.log(np.clip(union, 2.0 * PI + 1e-9, None) / (2.0 * PI)) / (2.0 * PI)
    f_comb = float(np.mean(COMB_DENSITY / (COMB_DENSITY + rho_zeta)))
    return xi, f_comb


def picket_number_variance(L: np.ndarray, frac: float) -> np.ndarray:
    """Number variance of a rigid sub-lattice occupying density fraction `frac` of a
    unit-density spectrum: over a window of L unfolded units it holds frac*L levels with
    picket-fence variance  {frac L}(1 - {frac L})  (bounded by 1/4). The bounded backbone
    contribution to the union Sigma^2."""
    x = np.mod(frac * np.asarray(L, dtype=float), 1.0)
    return x * (1.0 - x)


def union_number_variance(
    gammas: np.ndarray, t_max: float, L_values: np.ndarray, n_windows: int = 4000
) -> Dict[str, np.ndarray]:
    """Sigma^2(L) of the unfolded union vs the references (LENS 2).

    Returns measured union Sigma^2, the GUE reference (number_variance_reference, the
    form factor against the Fejer kernel -- the zeta part alone), the independent-
    superposition prediction  Sigma^2_GUE(f_zeta L) + picket(f_comb, L), and Poisson = L.
    The measured curve tracking GUE-log + a bounded picket ripple is the "GUE + rigid
    backbone" verdict: the comb does not change the long-range rigidity class.
    """
    xi, f_comb = union_unfolded(gammas, t_max)
    f_zeta = 1.0 - f_comb
    s2, mean = sr.number_variance(xi, L_values, n_windows=n_windows)
    gue_full = sr.number_variance_reference(gue_form_factor, L_values)
    gue_zeta = sr.number_variance_reference(gue_form_factor, f_zeta * np.asarray(L_values))
    superpos = np.asarray(gue_zeta) + picket_number_variance(L_values, f_comb)
    return {"L": np.asarray(L_values), "sigma2": s2, "mean_count": mean,
            "gue": np.asarray(gue_full), "superposition": superpos,
            "poisson": np.asarray(L_values, dtype=float),
            "f_comb": f_comb, "n_levels": xi.size}


# ===========================================================================
# self-validating driver
# ===========================================================================
def _print_packing(gammas: np.ndarray) -> None:
    print("== LENS 1: packing -- zeta zeros per comb tooth vs log2(t/2pi) ==")
    k, counts, pred = zeros_per_tooth(gammas, 8)
    print("   comb gap k:         " + "  ".join(f"{int(c):>3d}" for c in counts)
          + "   (measured zeta zeros in the gap)")
    print("   log2(t_c/2pi):      " + "  ".join(f"{p:4.1f}" for p in pred)
          + "   (predicted density ratio)")
    u = phase_uniformity(gammas)
    print(f"   comb-phase: ASYMPTOTICALLY equidistributed (so the count works), but the")
    print(f"   finite-N histogram is NOT flat: chi^2/dof = {u['chi2_per_dof']:.2f} (uniform=1), "
          f"leading cosine c1 = {u['cos1_coeff']:+.4f}")
    print("   (= 2<cos(g ln2)>, the p=2 dip at phi=0). The histogram's j-th mode IS the")
    print("   p=2^j resonance -> the phase non-uniformity is exactly LENS 3 (see below).")


def _print_resonance(gammas: np.ndarray) -> None:
    print("\n== LENS 3 (headline): the p=2 resonance -- <cos(gamma*omega)> vs explicit formula ==")
    N, t_max = gammas.size, float(gammas.max())
    se = math.sqrt(0.5 / N)
    print(f"   N = {N} zeros, gamma_max = {t_max:.1f}; sampling noise on <cos> ~ {se:.4f}.")
    print("   COMB frequencies omega = m ln2 = log(2^m) -- the comb teeth lock to p=2^m:")
    print("     m   omega=m ln2   measured <cos>   predicted (p=2^m)   z-score")
    for m, omega, meas, pred in comb_resonance_table(gammas, 4):
        print(f"     {m}   {omega:9.5f}    {meas:+.4f}          {pred:+.4f}          "
              f"{meas/se:+6.2f}")
    print("   -> NEGATIVE and matching the p=2 explicit-formula coefficient: the comb teeth")
    print("      (cos=+1) sit at the zeta-zero density MINIMA. The components are NOT")
    print("      independent -- they share the prime-2 lattice (the comb is the p=2 ghost).")
    print("\n   The resonance is a PRIME-POWER comb (every log p^k, not just the comb's p=2):")
    print("     n   freq=log n   Lambda(n)/sqrt n   measured <cos>   predicted   prime power?")
    for n in (2, 3, 4, 5, 6, 7, 8, 9):
        lam = float(clp.von_mangoldt(n)[n])
        meas = prime_resonance(gammas, math.log(n))
        pred = resonance_prediction(gammas, n)
        base = _prime_base(n)
        is_pp = f"yes (p={base})" if base else "no"
        tag = "  <== comb (p=2)" if (base == 2) else ""
        print(f"     {n}   {math.log(n):8.4f}    {lam/math.sqrt(n):8.4f}        "
              f"{meas:+.4f}        {pred:+.4f}     {is_pp}{tag}")
    print("   -> resonance ONLY at prime powers (log6 ~ 0); the comb's teeth are exactly")
    print("      the powers of 2 (n = 2,4,8,...), so the comb sees the p=2 sub-comb.")


def _prime_base(n: int) -> int:
    """The prime p if n = p^k is a prime power (k >= 1), else 0. (Small-n printout helper.)"""
    if n < 2:
        return 0
    p = n
    d = 2
    while d * d <= p:
        if p % d == 0:
            while p % d == 0:
                p //= d
            return d if p == 1 else 0
        d += 1
    return n  # n itself prime


def _print_rigidity(rig: Dict[str, np.ndarray]) -> None:
    print("\n== LENS 2: long-range rigidity -- number variance Sigma^2(L) of the union ==")
    print(f"   union of {int(rig['n_levels'])} levels (zeta zeros + comb), mean comb "
          f"fraction f_comb = {rig['f_comb']:.3f}.")
    L = rig["L"]
    print("     L     union Sigma^2   GUE (zeta only)   GUE+picket superpos   Poisson=L")
    for i in range(0, len(L), max(1, len(L) // 8)):
        print(f"   {L[i]:5.1f}     {rig['sigma2'][i]:7.3f}        {rig['gue'][i]:7.3f}"
              f"          {rig['superposition'][i]:7.3f}            {rig['poisson'][i]:6.1f}")
    print("   -> the union Sigma^2 grows ~log L (GUE class) + a bounded picket ripple, NOT")
    print("      ~L (Poisson): the rigid comb is an ADDITIVE backbone, it does not change")
    print("      the GUE long-range class. Bulk-independent; the correlation is LENS 3's")
    print("      sharp p=2 resonance, invisible to the smooth Sigma^2. (Finite-sample: the")
    print("      density nearly doubles over the cached zeros -- the usual unfolding caveat.)")


def _main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gammas = np.asarray(load_riemann_zeros(), dtype=float)

    _print_packing(gammas)
    _print_resonance(gammas)
    L_values = np.linspace(0.5, 18.0, 70)
    rig = union_number_variance(gammas, float(gammas.max()), L_values)
    _print_rigidity(rig)

    # ================================ figure ================================
    fig, axg = plt.subplots(2, 2, figsize=(14, 9.5))

    # panel (a): packing -- zeta zeros per comb tooth vs log2(t/2pi)
    ax = axg[0, 0]
    k, counts, pred = zeros_per_tooth(gammas, 14)
    ax.bar(k, counts, width=0.8, color="#1f5fa8", alpha=0.75,
           label="measured zeta zeros / comb gap")
    ax.plot(k, pred, "-o", color="#c2452d", ms=4, label=r"$\log_2(t_c/2\pi)$ (density ratio)")
    ax.set_xlabel("comb gap $k$  (between teeth $2\\pi k/\\ln2$)")
    ax.set_ylabel("zeta zeros in the gap")
    ax.set_title("(a) LENS 1: packing $=\\log_2(t/2\\pi)$ zeros per tooth", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")

    # panel (b): comb-phase equidistribution (one-point independence) + the p=2 dip
    ax = axg[0, 1]
    phi = comb_phase(gammas)
    nb = 16
    ax.hist(phi, bins=nb, range=(0, 1), color="#8a4fb0", alpha=0.7, label="comb phase $\\phi_n$")
    exp = gammas.size / nb
    ax.axhline(exp, ls="--", color="0.4", lw=1, label="uniform")
    xx = np.linspace(0, 1, 200)
    c1 = phase_uniformity(gammas, nb)["cos1_coeff"]
    ax.plot(xx, exp * (1.0 + c1 * np.cos(2 * PI * xx)), color="#c2452d", lw=2,
            label=fr"$1{{+}}c_1\cos2\pi\phi$, $c_1{{=}}{c1:+.3f}$ (p=2 dip)")
    ax.set_xlabel(r"comb phase $\phi_n=\{\gamma_n\ln2/2\pi\}$  ($\phi=0$ at a tooth)")
    ax.set_ylabel("count")
    ax.set_title("(b) phase equidistributed but w/ a p=2 dip at $\\phi{=}0$ ($=$ LENS 3)",
                 fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower center")

    # panel (c): the p=2 resonance -- <cos(gamma omega)> vs omega, prime-power comb
    ax = axg[1, 0]
    omegas = np.linspace(0.45, 3.2, 1400)
    res = np.array([prime_resonance(gammas, w) for w in omegas])
    ax.plot(omegas, res, color="#1f5fa8", lw=1.0)
    ax.axhline(0.0, color="0.6", lw=0.8)
    for n, lg, _w in clp.prime_power_peaks(24):
        is2 = (n & (n - 1)) == 0
        ax.axvline(lg, color=("#c2452d" if is2 else "#2a8a3e"),
                   lw=(1.4 if is2 else 0.8), ls=(":" if not is2 else "--"),
                   alpha=0.85 if is2 else 0.55)
        ax.text(lg, 0.02, str(n), ha="center", va="bottom", fontsize=7,
                color=("#c2452d" if is2 else "#1d6e30"))
    ax.set_xlim(omegas.min(), omegas.max())
    ax.set_xlabel(r"probe frequency $\omega$ (=$\log n$ at prime powers)")
    ax.set_ylabel(r"$\langle\cos(\gamma_n\,\omega)\rangle$")
    ax.set_title(r"(c) LENS 3: resonance at every $\log p^k$; red = comb's $p{=}2$ "
                 "($\\log2,\\log4,\\log8$)", fontsize=9.0)

    # panel (d): the headline -- measured vs predicted p=2 comb-frequency resonance
    ax = axg[1, 1]
    tab = comb_resonance_table(gammas, 4)
    ms = [r[0] for r in tab]
    meas = [r[2] for r in tab]
    pred = [r[3] for r in tab]
    se = math.sqrt(0.5 / gammas.size)
    ax.errorbar(ms, meas, yerr=se, fmt="o", ms=7, color="#c2452d", capsize=3,
                label=r"measured $\langle\cos(m\gamma\ln2)\rangle$")
    ax.plot(ms, pred, "s--", color="#1f5fa8", ms=6,
            label=r"explicit formula $-\frac{\ln2}{\pi 2^{m/2}}\frac{T}{2N}$")
    ax.axhline(0.0, color="0.6", lw=0.8, label="independent (no lock)")
    ax.set_xticks(ms)
    ax.set_xlabel(r"comb harmonic $m$  ($\omega=m\ln2=\log2^m$)")
    ax.set_ylabel(r"resonance $\langle\cos(m\gamma\ln2)\rangle$")
    ax.set_title(r"(d) headline: comb $\equiv$ $p{=}2$ ghost (teeth at density minima)",
                 fontsize=9.5)
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(r"The $\eta$ two-component spectrum: rigid $\sigma{=}1$ comb "
                 r"$\times$ GUE $\sigma{=}\frac{1}{2}$ zeros -- additive in bulk, "
                 r"$p{=}2$-locked at the comb frequency", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    figdir = Path(__file__).resolve().parent / "figures"
    figdir.mkdir(exist_ok=True)
    out = figdir / "eta_two_component.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    _main()
