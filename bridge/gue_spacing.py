"""Nearest-neighbour spacing comparison for the modular-surface Maass
spectrum vs Riemann zeros.

VENDORED NOTE (this repo): copied from the originating research project as a
dependency of `eta_two_component.py`, which uses only `load_riemann_zeros` and
`unfold_riemann_zeros`; `explorations/chi6_two_component.py` additionally uses
the Wigner-GUE/Poisson pdfs+cdfs and the spacing helpers
(`nearest_neighbour_spacings`, `rescale_to_unit_mean`). The Maass arc below needs
`data/maass_eigenvalues_psl2z.csv`, which is NOT shipped here -- running this
file as a script will fail at that load. Kept whole so the vendored code
matches its source; see CLAUDE.md "Gotchas".

Operates directly on two ordinate lists -- no Selberg zeta evaluation
needed -- and KS-tests the *unfolded* nearest-neighbour spacings
against two reference distributions:

    p_GOE(s)     = (pi / 2) s exp(-pi s^2 / 4)       -- Wigner GOE,
    p_GUE(s)     = (32 / pi^2) s^2 exp(-4 s^2 / pi)  -- Wigner GUE,
    p_Poisson(s) = exp(-s)                          -- uncorrelated.

The GOE surmise is the orthogonal-class reference (a real / time-
reversal-invariant chaotic Laplacian); it is the correct positive
control for a non-arithmetic G_q Maass spectrum. GUE is
the unitary class of the Riemann zeros.

Riemann zeros are the textbook Montgomery-Odlyzko / Bohigas-Giannoni-
Schmit (BGS) chaotic-GUE sequence: at N = 2000, KS vs GUE is ~ 0.04
and KS vs Poisson is ~ 0.32. Decisive.

The modular-surface Maass spectrum is **not** BGS-chaotic GUE despite
the geodesic flow on PSL_2(Z) \\ H being classically chaotic. The
extra commuting Hecke operators (Sarnak 1995; Bogomolny-Georgeot-
Giannoni-Schmit 1992) make every Maass form a joint eigenfunction of
an infinite-dimensional commutative algebra of operators, supplying
enough good quantum numbers to wash out level repulsion. Empirically
the NN spacings sit closer to Poisson than to GUE -- both per parity
and pooled. KS vs Poisson < KS vs GUE in every case; the per-parity
histograms hug the e^{-s} curve and miss the Wigner-GUE peak at
s ~ 1. Pooling makes it even more Poisson-like, since two independent
arithmetic-chaos sectors superpose at random.

LMFDB-completeness caveat. The 2,202-form level-1 dump in
`data/maass_eigenvalues_psl2z.csv` has parity-completeness only up
to tau ~ 99.58; above that the odd-sector certification drops out
in a ~10.6-wide band (73 consecutive even forms with zero odds
between them, statistically impossible if the dump were parity-
complete). `safe_tau_cutoff` detects the first such hole and the
driver truncates automatically. The final analysis uses N ~ 611
LMFDB forms (~270 even, ~340 odd).

Helpers exposed for tests:

* `unfold_riemann_zeros(gammas)`         - Riemann-von Mangoldt N(T).
* `unfold_modular_surface_taus(taus)`    - level-1 Selberg/Weyl T^2/12.
* `nearest_neighbour_spacings(unfolded)` - consecutive diffs.
* `rescale_to_unit_mean(spacings)`       - second-pass empirical
  rescaling so that <s> = 1; the GUE surmise lives at unit mean and at
  finite N the analytic unfolding has an O(1) residual we explicitly
  do not tune subleading terms for.
* `wigner_goe_pdf(s)`, `wigner_goe_cdf(s)` - GOE surmise + closed-form CDF.
* `wigner_gue_pdf(s)`, `wigner_gue_cdf(s)` - GUE surmise + closed-form CDF.
* `poisson_pdf(s)`, `poisson_cdf(s)`     - Poisson baseline + CDF.
* `safe_tau_cutoff(rows)`                - largest tau before the first
  LMFDB parity-completeness gap.

Driver `main()` emits:

* `gue_spacing.png` - 2-panel histogram (Maass top, Riemann bottom),
  with Wigner GUE overlay and Poisson reference on each panel.
* `gue_spacing.txt` - summary table with sample sizes, unfolded mean /
  std, and KS distances vs the GUE surmise (and Maass-vs-Riemann via
  ks_2samp).
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.special import erf

from maass_loader import load_maass_eigenvalues


_ROOT = Path(__file__).resolve().parents[1]
RIEMANN_ZEROS_CSV = _ROOT / "data" / "riemann_zeros.csv"
OUTPUT_PNG = _ROOT / "gue_spacing.png"
OUTPUT_TXT = _ROOT / "gue_spacing.txt"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_riemann_zeros(path: Path | None = None) -> list[float]:
    """Return the cached Riemann gamma_n as float64. Full-precision
    decimal strings stay in the CSV for callers who need them."""
    path = path or RIEMANN_ZEROS_CSV
    out: list[float] = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.append(float(row["gamma"]))
    return out


def split_taus_by_parity(rows: list[dict]) -> dict[int, np.ndarray]:
    """Group LMFDB Maass rows into per-parity tau arrays. LMFDB
    `symmetry` is 0 for even (cos expansion) and 1 for odd (sin
    expansion); these diagonalise different self-adjoint sectors and
    their eigenvalues do not repel each other across the split."""
    out: dict[int, list[float]] = {0: [], 1: []}
    for r in rows:
        out[int(r["symmetry"])].append(float(r["tau"]))
    return {p: np.array(sorted(vs), dtype=float) for p, vs in out.items()}


def safe_tau_cutoff(rows: list[dict],
                    max_within_parity_gap: float = 5.0) -> float:
    """Find the largest tau such that within each parity sub-spectrum,
    consecutive eigenvalues are within `max_within_parity_gap` of each
    other. Beyond this cutoff, the LMFDB maass_rigor table loses
    parity-completeness (the full level-1 dump has a ~10.6-wide gap
    in the odd sub-spectrum around tau ~ 99.6 -- 73 consecutive even
    forms with zero odds between them, which is statistically
    impossible for two independent GUE sub-spectra of comparable
    density).

    The default 5.0 threshold is ~20x the expected within-parity
    nearest-neighbour spacing at tau ~ 100 (Weyl gives mean within-
    parity spacing ~ 24 / T, i.e. ~0.24 at T = 100), so it triggers
    only on genuine data holes, not on natural fluctuation.
    """
    parity = split_taus_by_parity(rows)
    cutoffs: list[float] = []
    for arr in parity.values():
        diffs = np.diff(arr)
        big = np.where(diffs > max_within_parity_gap)[0]
        if len(big):
            cutoffs.append(float(arr[big[0]]))
    return min(cutoffs) if cutoffs else float("inf")


# ---------------------------------------------------------------------------
# Unfolding
# ---------------------------------------------------------------------------


def unfold_riemann_zeros(gammas: Iterable[float]) -> np.ndarray:
    """Riemann-von Mangoldt counting function applied pointwise:

        N(T) = (T / 2pi) log(T / (2 pi e)) + 7/8 + S(T),

    keeping only the two smooth leading terms plus the +7/8 constant.
    S(T) is mean-zero, O(log T), and washes out of nearest-neighbour
    spacings over the cached 2000 zeros.
    """
    arr = np.asarray(list(gammas), dtype=float)
    two_pi = 2.0 * math.pi
    return arr / two_pi * np.log(arr / (two_pi * math.e)) + 7.0 / 8.0


def unfold_modular_surface_taus(taus: Iterable[float]) -> np.ndarray:
    """Level-1 PSL_2(Z) Selberg/Weyl counting function applied
    pointwise, keeping the leading and first subleading terms:

        N(T) ~ T^2 / 12  -  (T / pi) log T  +  O(T).

    The O(T) residual is responsible for a sizeable absolute gap between
    N(tau_100) and the true count 100 at the LMFDB cutoff tau ~ 46,
    but is locally affine so it has
    negligible effect on the *spacing shape* we are testing.
    """
    arr = np.asarray(list(taus), dtype=float)
    return arr * arr / 12.0 - arr / math.pi * np.log(arr)


def nearest_neighbour_spacings(unfolded: np.ndarray) -> np.ndarray:
    """s_n = tilde t_{n+1} - tilde t_n for an ascending unfolded
    sequence."""
    arr = np.asarray(unfolded, dtype=float)
    return np.diff(arr)


def rescale_to_unit_mean(spacings: np.ndarray) -> np.ndarray:
    """Empirical second-pass unfolding: divide out the sample mean so
    the resulting sequence has <s> = 1. Equivalent to a local affine
    correction to N(T) that absorbs whatever subleading drift the
    analytic counting function failed to capture. The Wigner GUE
    surmise lives at unit mean, so this is the apples-to-apples
    comparison space.
    """
    arr = np.asarray(spacings, dtype=float)
    return arr / arr.mean()


# ---------------------------------------------------------------------------
# Reference distributions
# ---------------------------------------------------------------------------


def wigner_gue_pdf(s: np.ndarray | float) -> np.ndarray | float:
    """Wigner surmise for the 2x2 GUE -- the canonical NN spacing
    comparison curve for L-function zeros (Montgomery-Dyson)."""
    s = np.asarray(s, dtype=float)
    return 32.0 / math.pi ** 2 * s * s * np.exp(-4.0 * s * s / math.pi)


def wigner_gue_cdf(s: np.ndarray | float) -> np.ndarray | float:
    """Closed-form CDF of the Wigner GUE surmise:

        F(s) = erf(2 s / sqrt(pi))  -  (4 s / pi) exp(-4 s^2 / pi).

    Derivation: integrate s'^2 exp(-a s'^2) by parts with a = 4/pi.
    F(0) = 0, F(inf) = 1. Used as the scipy.stats.kstest reference.
    """
    s = np.asarray(s, dtype=float)
    sqrt_pi = math.sqrt(math.pi)
    return erf(2.0 * s / sqrt_pi) - 4.0 * s / math.pi * np.exp(
        -4.0 * s * s / math.pi
    )


def wigner_goe_pdf(s: np.ndarray | float) -> np.ndarray | float:
    """Wigner surmise for the 2x2 GOE -- the NN spacing curve for a
    *real, time-reversal-invariant* chaotic Laplacian (Dyson beta = 1).
    This is the correct positive-control reference for a non-arithmetic
    Maass spectrum (Delta = -y^2(d_xx + d_yy) is real, so a generic
    chaotic surface lands in the orthogonal class):

        p_GOE(s) = (pi / 2) s exp(-pi s^2 / 4).

    Contrast GUE (~ s^2 at small s, the unitary class of the Riemann
    zeros): GOE repels only linearly (~ s)."""
    s = np.asarray(s, dtype=float)
    return 0.5 * math.pi * s * np.exp(-0.25 * math.pi * s * s)


def wigner_goe_cdf(s: np.ndarray | float) -> np.ndarray | float:
    """Closed-form CDF of the Wigner GOE surmise:

        F(s) = 1 - exp(-pi s^2 / 4).

    A clean Rayleigh CDF (the GOE surmise *is* a Rayleigh law). F(0) = 0,
    F(inf) = 1, mean 1, Var = 4/pi - 1 ~= 0.273. Used as a
    scipy.stats.kstest reference alongside the GUE and Poisson curves."""
    s = np.asarray(s, dtype=float)
    return 1.0 - np.exp(-0.25 * math.pi * s * s)


def poisson_pdf(s: np.ndarray | float) -> np.ndarray | float:
    """Uncorrelated baseline p(s) = exp(-s). No level repulsion, finite
    at s = 0. For PSL_2(Z) Maass forms this is closer to the empirical
    NN distribution than GUE -- the arithmetic-chaos anomaly: Hecke
    operators commute with the Laplacian, supplying enough good quantum
    numbers that the spectrum sits closer to Poisson than the BGS
    chaotic-GUE expectation (Sarnak 1995; Bogomolny-Georgeot-Giannoni-
    Schmit 1992)."""
    s = np.asarray(s, dtype=float)
    return np.exp(-s)


def poisson_cdf(s: np.ndarray | float) -> np.ndarray | float:
    """CDF F(s) = 1 - exp(-s) of the unit-mean Poisson NN spacing
    distribution. Used as the second reference (alongside Wigner GUE)
    in the KS comparison table."""
    s = np.asarray(s, dtype=float)
    return 1.0 - np.exp(-s)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _format_table(rows: list[tuple[str, ...]], header: tuple[str, ...]) -> str:
    widths = [
        max(len(header[i]), max(len(r[i]) for r in rows)) for i in range(len(header))
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    out = [fmt.format(*header)]
    out.append("  ".join("-" * w for w in widths))
    for r in rows:
        out.append(fmt.format(*r))
    return "\n".join(out)


def _spacing_summary(label: str, raw: np.ndarray, unfolded: np.ndarray,
                    spacings: np.ndarray) -> tuple[str, ...]:
    return (
        label,
        f"{len(raw)}",
        f"{raw.min():.3f}",
        f"{raw.max():.3f}",
        f"{unfolded[-1]:.2f}",
        f"{spacings.mean():.4f}",
        f"{spacings.std(ddof=1):.4f}",
    )


KS_HEADER = ("comparison", "KS vs GOE", "p(GOE)", "KS vs GUE", "p(GUE)",
             "KS vs Poisson", "p(Poisson)")


def _ks_row(label: str, sample: np.ndarray) -> tuple[str, ...]:
    """One-sample KS distances against three references: the Wigner GOE
    surmise (the orthogonal class -- the correct prediction for a real
    chaotic surface Laplacian), the Wigner GUE surmise (the unitary
    class of the Riemann zeros), and the Poisson baseline. Showing all
    three is essential: the arithmetic-chaos anomaly pushes PSL_2(Z) /
    arithmetic-G_q Maass spectra toward Poisson, while a non-arithmetic
    G_q spectrum should land on GOE (repulsion). GOE vs GUE differ at
    small s (~ s vs ~ s^2); both reject Poisson."""
    goe = stats.kstest(sample, wigner_goe_cdf)
    gue = stats.kstest(sample, wigner_gue_cdf)
    poi = stats.kstest(sample, poisson_cdf)
    return (
        label,
        f"{goe.statistic:.4f}", f"{goe.pvalue:.2e}",
        f"{gue.statistic:.4f}", f"{gue.pvalue:.2e}",
        f"{poi.statistic:.4f}", f"{poi.pvalue:.2e}",
    )


def build_summary_text(
    taus: np.ndarray, gammas: np.ndarray,
    unfolded_taus: np.ndarray, unfolded_gammas: np.ndarray,
    spacings_maass: np.ndarray, spacings_riemann: np.ndarray,
    rescaled_maass: np.ndarray, rescaled_riemann: np.ndarray,
    rescaled_parity: dict[int, np.ndarray],
) -> str:
    """Build the contents of gue_spacing.txt."""
    lines: list[str] = []
    lines.append(
        "Nearest-neighbour spacing comparison: modular Maass spectrum "
        "vs Riemann zeros vs Wigner GUE surmise."
    )
    lines.append("")
    lines.append("Inputs:")
    lines.append(
        f"  Maass tau_n : {len(taus)} LMFDB rigorous level-1 eigenvalues, "
        f"{taus.min():.3f} <= tau <= {taus.max():.3f}."
    )
    lines.append(
        f"  Riemann gamma_n : first {len(gammas)} nontrivial zeros via "
        f"mpmath.zetazero, {gammas.min():.3f} <= gamma <= {gammas.max():.3f}."
    )
    lines.append("")

    header = ("spectrum", "N", "min", "max", "N(T_max)", "mean spacing",
              "std spacing")
    rows = [
        _spacing_summary("Maass tau",  taus,   unfolded_taus,   spacings_maass),
        _spacing_summary("Riemann gamma", gammas, unfolded_gammas,
                          spacings_riemann),
    ]
    lines.append("Analytic-unfolding statistics (mean should be ~1; "
                 "drift is the unfolding's finite-N error):")
    lines.append(_format_table(rows, header))
    lines.append("")

    # Unfolding-imperfection diagnostic: compare N(T_max) to actual count.
    lines.append("Unfolding sanity check (asymptotic N(T) vs actual count):")
    lines.append(
        f"  Maass:   N(tau_{len(taus)}) = {unfolded_taus[-1]:7.2f}  vs  "
        f"actual = {len(taus)}  (gap is the O(T) Selberg residual)."
    )
    lines.append(
        f"  Riemann: N(gamma_{len(gammas)}) = {unfolded_gammas[-1]:7.2f}  vs  "
        f"actual = {len(gammas)}  (gap is the S(T) ~ O(log T) residual)."
    )
    lines.append("")

    lines.append(
        "Rescaled spacings (second pass: divide by sample mean so <s> = 1, "
        "matching the Wigner GUE convention):"
    )
    rescaled_header = ("spectrum", "N spacings", "mean", "std",
                       "CV = std/mean")
    rescaled_rows = [
        ("Maass tau (rescaled)", f"{len(rescaled_maass)}",
         f"{rescaled_maass.mean():.4f}", f"{rescaled_maass.std(ddof=1):.4f}",
         f"{rescaled_maass.std(ddof=1) / rescaled_maass.mean():.4f}"),
        ("Riemann gamma (rescaled)", f"{len(rescaled_riemann)}",
         f"{rescaled_riemann.mean():.4f}",
         f"{rescaled_riemann.std(ddof=1):.4f}",
         f"{rescaled_riemann.std(ddof=1) / rescaled_riemann.mean():.4f}"),
        ("Wigner GOE surmise (reference)", "-", "1.0000", "0.5227", "0.5227"),
        ("Wigner GUE surmise (reference)", "-", "1.0000", "0.4221", "0.4221"),
    ]
    lines.append(_format_table(rescaled_rows, rescaled_header))
    lines.append("")

    lines.append(
        "Per-parity Maass statistics (even and odd Maass forms "
        "diagonalise different self-adjoint sectors; cross-sector "
        "spacings do not exhibit level repulsion, so pooling them "
        "Berry-Robnik-mixes the two GUE clouds and inflates the small-s "
        "tail):"
    )
    parity_header = ("parity", "N spacings", "mean", "std", "CV")
    parity_rows = []
    for p_label, p in [("even (LMFDB symmetry=0)", 0),
                       ("odd  (LMFDB symmetry=1)", 1)]:
        rs = rescaled_parity[p]
        parity_rows.append(
            (p_label, f"{len(rs)}", f"{rs.mean():.4f}",
             f"{rs.std(ddof=1):.4f}",
             f"{rs.std(ddof=1) / rs.mean():.4f}")
        )
    parity_rows.append(
        ("Wigner GOE surmise (reference)", "-", "1.0000", "0.5227", "0.5227")
    )
    parity_rows.append(
        ("Wigner GUE surmise (reference)", "-", "1.0000", "0.4221", "0.4221")
    )
    lines.append(_format_table(parity_rows, parity_header))
    lines.append("")

    ks_rows = [
        _ks_row("Maass pooled (rescaled <s>=1)",   rescaled_maass),
        _ks_row("Maass even-only (rescaled)",      rescaled_parity[0]),
        _ks_row("Maass odd-only  (rescaled)",      rescaled_parity[1]),
        _ks_row("Riemann (rescaled <s>=1)",        rescaled_riemann),
    ]
    lines.append(
        "Kolmogorov-Smirnov distances vs the Wigner GOE and GUE surmises "
        "and the Poisson baseline (smaller KS = better fit):"
    )
    lines.append(_format_table(ks_rows, KS_HEADER))
    lines.append("")
    two_sample = stats.ks_2samp(rescaled_maass, rescaled_riemann)
    lines.append(
        f"Two-sample KS (Maass pooled vs Riemann, rescaled): "
        f"{two_sample.statistic:.4f}  p={two_sample.pvalue:.2e}"
    )
    lines.append("")

    lines.append("")
    lines.append(
        "Interpretation. Riemann recovers the textbook Montgomery-"
        "Odlyzko match to Wigner GUE (small KS vs GUE, large KS vs "
        "Poisson). The Maass per-parity spacings sit closer to the "
        "Poisson baseline than to Wigner GUE -- the arithmetic-chaos "
        "anomaly: PSL_2(Z) Maass forms are joint eigenfunctions of "
        "the Laplacian and the infinite Hecke algebra, so the extra "
        "good quantum numbers wash out level repulsion. "
        "Bogomolny-Georgeot-Giannoni-Schmit (1992) and Sarnak (1995) "
        "predicted exactly this: arithmetic surfaces violate the BGS "
        "GUE expectation."
    )
    return "\n".join(lines) + "\n"


def plot_spacing_histograms(
    rescaled_maass: np.ndarray,
    rescaled_parity: dict[int, np.ndarray],
    rescaled_riemann: np.ndarray,
    n_bins: int = 20, s_max: float = 3.0,
) -> None:
    """Three stacked histograms in rescaled (<s> = 1) coordinates:

    Panel 1: Maass spacings, pooled across both parities. Shows the
             cross-parity Berry-Robnik leak: excess density at s -> 0.
    Panel 2: Maass spacings, even and odd subspectra overlaid. Each
             subspectrum should match GUE up to finite-N noise.
    Panel 3: Riemann zeros. Reference Montgomery-Dyson sequence.

    All panels carry the Wigner GUE surmise (red) and the Poisson
    baseline (grey dashed). Saves to OUTPUT_PNG.
    """
    s_grid = np.linspace(0.0, s_max, 400)
    goe_curve = wigner_goe_pdf(s_grid)
    gue_curve = wigner_gue_pdf(s_grid)
    poisson_curve = poisson_pdf(s_grid)

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    ax_pool, ax_parity, ax_riem = axes

    # Panel 1: pooled Maass.
    ax_pool.hist(
        rescaled_maass, bins=n_bins, range=(0.0, s_max), density=True,
        color="C0", alpha=0.55, edgecolor="black", linewidth=0.5,
        label=f"empirical pooled ({len(rescaled_maass)} spacings)",
    )
    ax_pool.set_title(
        r"Maass $\tau_n$ pooled — arithmetic-chaos: tracks Poisson, "
        "not GUE"
    )

    # Panel 2: per-parity Maass overlay (both even and odd). Step
    # histograms instead of filled bars so the two subspectra remain
    # legible where they overlap.
    parity_styles = {
        0: ("C0", "even (LMFDB symmetry $=0$)"),
        1: ("C4", "odd  (LMFDB symmetry $=1$)"),
    }
    for p, sp in rescaled_parity.items():
        color, label = parity_styles[p]
        ax_parity.hist(
            sp, bins=n_bins, range=(0.0, s_max), density=True,
            histtype="step", color=color, lw=2.0,
            label=f"{label} ({len(sp)} spacings)",
        )
    ax_parity.set_title(
        r"Maass $\tau_n$ split by parity — each parity sub-spectrum "
        "is also Poisson-leaning (Hecke quantum numbers)"
    )

    # Panel 3: Riemann.
    ax_riem.hist(
        rescaled_riemann, bins=n_bins, range=(0.0, s_max), density=True,
        color="C2", alpha=0.55, edgecolor="black", linewidth=0.5,
        label=f"empirical ({len(rescaled_riemann)} spacings)",
    )
    ax_riem.set_title(
        r"Riemann $\gamma_n$ — textbook GUE (Montgomery–Odlyzko)"
    )

    for ax in (ax_pool, ax_parity, ax_riem):
        ax.plot(s_grid, goe_curve, color="C1", lw=2.0,
                label="Wigner GOE surmise")
        ax.plot(s_grid, gue_curve, color="C3", lw=2.0,
                label="Wigner GUE surmise")
        ax.plot(s_grid, poisson_curve, color="gray", lw=1.2,
                ls="--", label="Poisson $e^{-s}$ (uncorrelated)")
        ax.set_ylabel("density $p(s)$")
        ax.set_xlim(0.0, s_max)
        ax.set_ylim(0.0, 1.3)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    ax_riem.set_xlabel(
        "rescaled nearest-neighbour spacing $s$ (analytic unfold + "
        r"$\langle s\rangle\!=\!1$)"
    )
    fig.suptitle(
        "Nearest-neighbour spacings: modular Maass spectrum vs Riemann "
        "zeros vs Wigner GUE",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150)
    plt.close(fig)


def _rescaled_spacings(taus: np.ndarray) -> np.ndarray:
    """Convenience: analytic Selberg/Weyl unfold, NN spacings, then
    second-pass rescale to mean = 1."""
    return rescale_to_unit_mean(
        nearest_neighbour_spacings(unfold_modular_surface_taus(taus))
    )


def main() -> None:
    rows = load_maass_eigenvalues()
    # Apply parity-completeness truncation. The LMFDB maass_rigor table
    # is per-parity complete only up to a cutoff (~tau = 99.58 for the
    # full 2202-form level-1 dump); above that the data leaks
    # progressively into a pooled-but-not-per-parity regime. For the
    # GUE comparison we need per-parity completeness, so we truncate.
    tau_cut = safe_tau_cutoff(rows)
    if tau_cut < float("inf"):
        kept_rows = [r for r in rows if r["tau"] <= tau_cut]
        print(
            f"Truncating Maass spectrum at tau <= {tau_cut:.3f} "
            f"(parity-completeness cutoff): kept {len(kept_rows)} of "
            f"{len(rows)} LMFDB forms."
        )
        rows = kept_rows
    taus = np.array([r["tau"] for r in rows], dtype=float)
    gammas = np.array(load_riemann_zeros(), dtype=float)

    unfolded_taus = unfold_modular_surface_taus(taus)
    unfolded_gammas = unfold_riemann_zeros(gammas)

    spacings_maass = nearest_neighbour_spacings(unfolded_taus)
    spacings_riemann = nearest_neighbour_spacings(unfolded_gammas)
    rescaled_maass = rescale_to_unit_mean(spacings_maass)
    rescaled_riemann = rescale_to_unit_mean(spacings_riemann)

    parity_taus = split_taus_by_parity(rows)
    rescaled_parity = {p: _rescaled_spacings(t)
                       for p, t in parity_taus.items()}

    text = build_summary_text(
        taus, gammas, unfolded_taus, unfolded_gammas,
        spacings_maass, spacings_riemann,
        rescaled_maass, rescaled_riemann,
        rescaled_parity,
    )
    OUTPUT_TXT.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {OUTPUT_TXT.name}.")

    t0 = time.time()
    plot_spacing_histograms(rescaled_maass, rescaled_parity, rescaled_riemann)
    print(f"Wrote {OUTPUT_PNG.name} ({time.time() - t0:.1f}s).")


if __name__ == "__main__":
    main()
