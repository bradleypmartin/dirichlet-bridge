"""mobius_dressing.py -- the C(omega) sideband anomaly resolved (issue #47).

#44 measured the convergence-constant spectrum C(omega) = -omega <sin(gamma
omega) Im a_1> over the 400 cached chi_3 zeros and read it as a chi-twisted
LANDAU line spectrum with a structured 2-adic anomaly: ln 2 halved, ln 4 on the
Lambda(n)/sqrt(n) prediction to 0.1%, ln 8 nearly extinguished. #47 asked
whether the anomaly is the section factor B(rho+1) = 1 - 2^{-3/2} e^{-i gamma
ln 2} dressing the spectrum. Answer: yes -- and dividing B out does NOT restore
a Landau spectrum. It reveals the spectrum was never Landau:

1.  THE UNDRESSED SPECTRUM IS MOEBIUS. With a_1' = a_1/B(rho+1) the weight is
    proportional to s/L'(rho): the Perron kernel of 1/L(s,chi) = sum mu(n)
    chi(n) n^{-s} (the explicit formula for the chi-twisted Mertens function
    M_chi(x) = sum_rho x^rho/(rho L'(rho)), Titchmarsh par. 14.27 for zeta;
    Gonek/Ng's discrete moments sum x^rho/zeta'(rho) are the same object).
    Measured: lines at ln n exactly for SQUAREFREE n coprime to 3 -- amplitude
    scale * ln n * |mu(n) chi_3(n)|/sqrt(n) fits all fourteen squarefree bands
    n = 2, 5, 7, 10, 11, 13, 14, 17, 19, 22, 23, 26, 34, 35 -- while the
    prime-POWER bands ln 4, ln 8, ln 16 are dark (mu = 0). On primes,
    ln p |mu(p)| = Lambda(p): the Moebius and Landau spectra COINCIDE on every
    prime line, which is why #44's odd-prime scale fit could not tell them
    apart. The discriminators are exactly the bands #44 flagged as anomalous
    (ln 4, ln 8) plus the "non-Landau" peak it noticed at ln 10.

2.  THE DRESSING LAW IS A ONE-SIDED +ln2 SHIFT, COEFFICIENT beta = 2^{-3/2}.
    a_1 = B a_1' convolves the Moebius line measure with delta - beta
    delta_{+ln 2} (the shift is UP only: the measure's line content sits in the
    conjugate channel e^{+i gamma ln n}, so e^{-i gamma ln 2} lands on
    ln n + ln 2; no satellites appear at ln p - ln 2, as measured). On the
    Moebius comb this gives a closed-form dressed spectrum:

      * every even squarefree line ln 2m keeps EXACTLY half its amplitude:
        the satellite of the odd parent m lands on it destructively, and
        |mu(2)chi(2)/sqrt2 - beta| / |1/sqrt2| = 1 - 1/2 exactly (measured
        0.50-0.54 across n = 2, 10, 14, 22, 26, 34);
      * ln 4 hosts the ln-2 line's satellite in a mu(4) = 0 VACANCY:
        amplitude scale * ln 4 * beta/sqrt2 -- which equals the Landau
        reference scale * Lambda(4)/2 IDENTICALLY (satellite/Landau = 2/p at
        p = 2). #44's "ln 4 on the Landau scale to 0.1%" was this masquerade:
        the agreement is an algebraic identity of p = 2, not evidence of a
        Lambda spectrum;
      * ln 8 = satellite of ln 4: parent empty (mu(4) = 0), so the cascade
        stops after one step (B has a single sideband) -- ln 8, ln 16 dark.

3.  WHAT THE PHASES CERTIFY. The 1/2-law and the ln-4 vacancy amplitude are
    RELATIVE-phase measurements (parent vs satellite, forced real-positive by
    the data); the ABSOLUTE sign of each line against mu(n)chi_3(n) is not
    resolved at this window (T ~ 550: a sub-resolution band-center offset
    rotates the quadrature by ~1 rad, and the measured sin-quadrature signs
    agree with mu chi only 7/13 -- consistent with scrambling).

Falsifiable #48 predictions logged here (the conductor sweep): for a phi(q)=2
modulus with section prime p_B = q - 1, the dressing frequency is ln p_B with
beta = p_B^{-3/2}; the even-line suppression 1/2 generalizes to
|mu(p_B)chi(p_B) - 1/p_B| (a 1 + 1/p_B ENHANCEMENT where mu(p_B)chi(p_B) =
-1), and the vacancy satellite at ln p_B^2 sits at 2/p_B of the Landau
reference -- Landau-exact ONLY at p_B = 2. At q = 4 (p_B = 3): ln-3m lines at
2/3, ln 9 satellite at 2/3 of Landau.

Honest framing: experimental mathematics; no RH/GRH claims. The M(x) explicit
formula and 1/zeta'-weighted discrete moments are classical (Titchmarsh Thm
14.27; Gonek; Conrey-Ghosh-Gonek; Ng; Hughes-Keating-O'Connell; the Dirichlet
negative moments are Pearce-Crump 2026); Landau's formula is the unweighted
reference, chi-twisted by Banks/Fujii. Per the 2026-08-02 adversarial lit
pass (verdicts + verified BibTeX in README/references-37.bib) no published
analog was found for the spectral READING -- the Moebius line spectrum
measured from a zero sum, chi-twisted, with a finite-Euler-product section
dressing it by exact rational factors, here an instrument on the K-knob's own
displacement constant. Nearest classical cousin, to cite and distinguish: the
Linnik-Sprindzuk theorem (mu(k)/phi(k) from an UNWEIGHTED zero-sum limit at a
rational twist -- no 1/L' weight, a limit not a spectrum).

Pure post-processing of the committed a_1 cache (arithmetic_clock_a1.csv):
runs in seconds, no zero computation.
"""
import math
import sys
from pathlib import Path
from typing import List, Tuple

_HERE = Path(__file__).resolve().parent
_BRIDGE = _HERE.parent / "bridge"
for _p in (str(_HERE), str(_BRIDGE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import arithmetic_clock as ac  # noqa: E402
import character_bridge as cb  # noqa: E402
import chi6_two_component as c6  # noqa: E402

LN2 = math.log(2)
BETA = 2.0 ** (-1.5)                    # |2^{-(rho+1)}| at sigma = 1/2
FIG_PATH = _HERE / "figures" / "mobius_dressing.png"

# the band classes (n values; frequencies ln n)
PRIMES = [5, 7, 11, 13]                 # scale anchors: mu and Lambda agree
PRIMES_HI = [17, 19, 23]                # checked, not used in the scale fit
TOWER = [2, 4, 8, 16]                   # the 2-adic tower (the #44 anomaly)
EVEN_SQF = [10, 14, 22, 26, 34]         # mu-only lines: Lambda(2p) = 0
ODD_SQF = [35]                          # 5*7: mu-only, and 2-free (no dressing)
CONDUCTOR = [3, 9, 6, 15, 21, 33]       # 3 | n: chi_3 kills the channel
LOWER_SB = [math.log(5) - LN2, math.log(7) - LN2, math.log(11) - LN2]

W_LO, W_HI, DW = 0.4, 3.65, 1e-3        # the sweep grid (past ln 34, ln 35)


# --------------------------------------------------------------------------
# 1. the weight, dressed and undressed
# --------------------------------------------------------------------------
def load_weights():
    """(gammas, a_1) from the committed #44 cache, as numpy arrays."""
    a1s = ac.a1_table(c6.load_chi3_zeros())
    g = np.array([gg for gg, _a in a1s])
    a1 = np.array([a for _g, a in a1s])
    return g, a1


def B_factor(g, chi=cb.CHI3):
    """B(rho + 1) = sum_a chi(a) a^{-3/2} e^{-i gamma ln a} on the critical
    line (zero_birth.section_B at rho + 1, vectorized in gamma). q = 3 gives
    the #47 single-sideband 1 - 2^{-3/2} e^{-i gamma ln 2}; general q gives
    the full section comb. Never vanishes at sigma = 3/2: the a >= 2 tail is
    bounded by sum a^{-3/2} - 1 < 1 for every q <= 7 section, so the division
    in `undress` is safe."""
    g = np.asarray(g, dtype=float)
    out = np.zeros(g.shape, dtype=complex)
    for a in cb.units(chi.q):
        out += complex(chi(a)) * a ** -1.5 * np.exp(-1j * g * math.log(a))
    return out


def undress(g, a1, chi=cb.CHI3):
    """a_1' = a_1 / B(rho+1): the pure s/L' (Perron-of-1/L) weight."""
    return np.asarray(a1) / B_factor(g, chi)


# --------------------------------------------------------------------------
# 2. the spectrum, vectorized, and the band instrument
# --------------------------------------------------------------------------
def spectrum(g, a1, ws):
    """C(omega) = -omega <sin(gamma omega) Im a_1> on a frequency grid
    (identical to arithmetic_clock.resonance_constant, vectorized in omega)."""
    ws = np.asarray(ws, dtype=float)
    ia = np.asarray([complex(a).imag for a in a1])
    return -ws * (np.sin(np.outer(ws, np.asarray(g))) @ ia) / len(ia)


def band_max(g, a1, w0, half=0.02, dw=4e-4):
    """(max|C|, argmax omega) over [w0 - half, w0 + half] -- lines are
    window-limited (lobe ~ pi/T ~ 0.006 at T ~ 550), so band maxima."""
    ws = np.arange(w0 - half, w0 + half, dw)
    Cs = spectrum(g, a1, ws)
    i = int(np.argmax(np.abs(Cs)))
    return float(abs(Cs[i])), float(ws[i])


# --------------------------------------------------------------------------
# 3. the two line models
# --------------------------------------------------------------------------
def mu_of(n):
    """The Moebius function (trial division; the band n's are tiny)."""
    out, m = 1, int(n)
    p = 2
    while p * p <= m:
        if m % p == 0:
            m //= p
            if m % p == 0:
                return 0
            out = -out
        p += 1
    return -out if m > 1 else out


def chi_weight(n, chi=cb.CHI3):
    """|chi(n)| as a float: 1 on units mod q, 0 on the ramified columns."""
    return float(abs(chi(n)))


def fit_scale(g, a1p, anchors=None):
    """One common amplitude scale, fitted on odd-prime anchor lines where the
    Moebius and Landau spectra coincide (ln p |mu(p)| = Lambda(p)); pass
    conductor-appropriate `anchors` (primes coprime to q) beyond q = 3."""
    num = den = 0.0
    for p in (PRIMES if anchors is None else anchors):
        mx, _wm = band_max(g, a1p, math.log(p))
        num += mx
        den += math.log(p) / math.sqrt(p)
    return num / den


def mobius_ref(n, scale, chi=cb.CHI3):
    """The Moebius-spectrum line amplitude scale * ln n |mu(n) chi(n)|/sqrt n
    (the -omega prefactor in C supplies the ln n)."""
    return scale * math.log(n) * abs(mu_of(n)) * chi_weight(n, chi) / math.sqrt(n)


def landau_ref(n, scale):
    """The chi-weighted Landau amplitude scale * Lambda(n)/sqrt(n) -- #44's
    reference, kept as the foil (identical to mobius_ref on primes)."""
    return scale * ac._von_mangoldt(n) / math.sqrt(n)


def dressed_ref(n, scale, chi=cb.CHI3):
    """The dressed-model line amplitude: the Moebius line measure convolved
    with the FULL section comb B(rho+1) = sum_a chi(a) a^{-3/2} e^{-i gamma
    ln a} (a over the units 1 <= a < q). The a-term shifts the parent line
    ln(n/a) up onto ln n with coefficient chi(a)/a; chi(a) chi(n/a) = chi(n)
    factors out of every term, so the amplitude closes rationally and
    chi-INDEPENDENTLY inside the |.|:

        scale * ln n * |chi(n)| * | sum_{a | n, 1 <= a < q, (a, q) = 1}
                                    mu(n/a) / a |  / sqrt(n).

    q = 3 reproduces #47 exactly: odd lines untouched (only a = 1 divides),
    even squarefree lines at |mu(n) + mu(n/2)/2| = 1/2 of Moebius, the ln 4
    vacancy satellite |mu(2)|/2 = 1/2 (= Landau by the p = 2 identity), ln 8
    dark. The chi-cancellation is the correction to #47's logged sign-flip
    prediction (|mu(2)chi(2) - 1/2|): the satellite coefficient chi(a) and
    the direct line's chi(n) = chi(a) chi(n/a) share the character factor, so
    the suppression NEVER flips to an enhancement -- adjudicated by the q = 7
    measurement in conductor_clock.py."""
    if chi_weight(n, chi) == 0.0:
        return 0.0
    tot = 0.0
    for a in range(1, chi.q):
        if n % a == 0 and math.gcd(a, chi.q) == 1:
            tot += mu_of(n // a) / a
    return scale * math.log(n) * abs(tot) / math.sqrt(n)


# --------------------------------------------------------------------------
# 4. the measurement tables
# --------------------------------------------------------------------------
def band_table(g, a1, a1p, scale):
    """Rows (class, n, omega, undressed, dressed, mobius_ref, landau_ref,
    dressed_ref) over every named band."""
    rows = []
    for cls, ns in (("prime", PRIMES + PRIMES_HI), ("tower", TOWER),
                    ("even-sqf", EVEN_SQF), ("odd-sqf", ODD_SQF),
                    ("conductor", CONDUCTOR)):
        for n in ns:
            w0 = math.log(n)
            mu_meas, _w = band_max(g, a1p, w0)
            md_meas, _w = band_max(g, a1, w0)
            rows.append((cls, n, w0, mu_meas, md_meas, mobius_ref(n, scale),
                         landau_ref(n, scale), dressed_ref(n, scale)))
    return rows


def control_table(g, a1, a1p):
    """The floor: #44's low-range controls plus lower-sideband bands ln p - ln2
    (one-sidedness: the dressing must put NOTHING there)."""
    rows = []
    for w0 in ac.C_CONTROLS:
        rows.append(("ctl", w0, band_max(g, a1p, w0)[0], band_max(g, a1, w0)[0]))
    for w0 in LOWER_SB:
        rows.append(("p-ln2", w0, band_max(g, a1p, w0)[0],
                     band_max(g, a1, w0)[0]))
    return rows


# --------------------------------------------------------------------------
# 5. self-validation
# --------------------------------------------------------------------------
def validate(bands, ctls, scale):
    """The falsifiable checks; returns (n_pass, n_fail) and prints each."""
    checks = []  # type: List[Tuple[str, bool]]
    by_n = {r[1]: r for r in bands}     # band n values are unique across classes

    # V1: every squarefree band on the Moebius reference (leakage/noise widen
    # the tolerance as lines densify past omega ~ 3)
    for r in bands:
        cls, n, _w, mu_m, _md, mref, _lr, _dr = r
        if cls in ("prime", "even-sqf", "odd-sqf") or n == 2:
            ok = 0.85 <= mu_m / mref <= 1.20
            checks.append((f"V1 undressed ln{n} on Moebius ref "
                           f"({mu_m / mref:.2f})", ok))
    # V2: the prime-power vacancies are dark undressed (vs the LANDAU claim)
    for n in (4, 8, 16):
        _c, _n, _w, mu_m, _md, _mr, lref, _dr = by_n[n]
        checks.append((f"V2 undressed ln{n} dark ({mu_m / lref:.2f} of "
                       f"Landau ref)", mu_m / lref < 0.35))
    # V3: the exact-1/2 dressing law on even squarefree lines
    for n in [2] + EVEN_SQF:
        _c, _n, _w, mu_m, md_m, _mr, _lr, _dr = by_n[n]
        checks.append((f"V3 dressed/undressed ln{n} = 1/2 "
                       f"({md_m / mu_m:.2f})", 0.44 <= md_m / mu_m <= 0.60))
    # V4: odd lines undressed by the dressing (ratio ~ 1)
    for n in PRIMES + PRIMES_HI + ODD_SQF:
        _c, _n, _w, mu_m, md_m, _mr, _lr, _dr = by_n[n]
        checks.append((f"V4 dressed/undressed ln{n} = 1 "
                       f"({md_m / mu_m:.2f})", 0.88 <= md_m / mu_m <= 1.12))
    # V5: the ln-4 vacancy satellite at its closed form (= Landau, the p = 2
    # masquerade), ln 8 dark dressed
    _c, _n, _w, _mu, md4, _mr, _lr, dr4 = by_n[4]
    checks.append((f"V5 dressed ln4 = beta-satellite = Landau "
                   f"({md4 / dr4:.3f})", abs(md4 / dr4 - 1) < 0.05))
    _c, _n, _w, _mu, md8, _mr, lr8, _dr = by_n[8]
    checks.append((f"V5 dressed ln8 dark ({md8 / lr8:.2f} of Landau ref)",
                   md8 / lr8 < 0.35))
    # V6: one-sidedness -- nothing at ln p - ln 2 (below the worst low-range
    # control floor, dressed and undressed)
    floor = max(m for tag, _w, mu_m, md_m in ctls if tag == "ctl"
                for m in (mu_m, md_m))
    for tag, w0, mu_m, md_m in ctls:
        if tag == "p-ln2":
            checks.append((f"V6 no satellite at {w0:.3f} "
                           f"(und {mu_m:.2f}/drs {md_m:.2f} <= floor "
                           f"{floor:.2f})", max(mu_m, md_m) <= 1.2 * floor))
    # V7: conductor stays dark through the undressing (low-range bands; the
    # high-omega 3|n bands ln 21, ln 33 sit in dense-line leakage, reported
    # in the table but not gated)
    for n in (3, 9, 6, 15):
        _c, _n, _w, mu_m, md_m, _mr, _lr, _dr = by_n[n]
        checks.append((f"V7 conductor ln{n} dark (und {mu_m:.2f})",
                       max(mu_m, md_m) < 0.35))

    n_fail = 0
    for label, ok in checks:
        print(f"   [{'PASS' if ok else 'FAIL'}] {label}")
        n_fail += not ok
    return len(checks) - n_fail, n_fail


# --------------------------------------------------------------------------
# 6. driver
# --------------------------------------------------------------------------
def _print_bands(bands, scale):
    print(f"== the band table (scale {scale:.3f}, fitted on ln5/7/11/13 "
          "undressed) ==")
    print("   class      n    omega   undress   dress   Moebius   Landau"
          "   dressed-model")
    for cls, n, w0, mu_m, md_m, mref, lref, dref in bands:
        print(f"   {cls:>9} {n:>4}   {w0:.4f}   {mu_m:7.3f}  {md_m:6.3f}"
              f"   {mref:7.3f}  {lref:7.3f}   {dref:7.3f}")


def _print_ctls(ctls):
    print("\n== the floor: controls and the (absent) lower sidebands ==")
    for tag, w0, mu_m, md_m in ctls:
        lb = "ctl      " if tag == "ctl" else "ln p - ln2"
        print(f"   {lb} {w0:.4f}: undressed {mu_m:.3f}, dressed {md_m:.3f}")


def _figure(g, a1, a1p, bands, scale):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figstyle
    figstyle.enlarge()
    plt.rcParams["axes.titlesize"] = 11.5
    plt.rcParams["figure.titlesize"] = 15

    UND, DRS = "tab:blue", "tab:orange"      # entity colors, fixed throughout
    ws = np.arange(W_LO, W_HI, DW)
    Cu = spectrum(g, a1p, ws)
    Cd = spectrum(g, a1, ws)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axA, axB = axes[0]
    axC, axD = axes[1]

    # (A) the undressed spectrum vs the Moebius comb
    axA.plot(ws, np.abs(Cu), "-", color=UND, lw=0.7)
    first = True
    for n in range(2, 40):
        r = mobius_ref(n, scale)
        if r > 0:
            axA.plot([math.log(n)] * 2, [0, r], color="0.25", lw=2.2,
                     alpha=0.65, solid_capstyle="butt",
                     label="Moebius model" if first else None)
            first = False
    for n in (4, 8, 9, 16):
        axA.plot(math.log(n), landau_ref(n, scale), "x", ms=9, color="tab:red",
                 label="Landau-only lines (absent)" if n == 4 else None)
    axA.set_xlabel(r"test frequency $\omega$")
    axA.set_ylabel(r"$|C'(\omega)|$")
    axA.set_title("Undressed a_1/B: a Moebius line spectrum "
                  r"($\ln n$, squarefree $n$, $3 \nmid n$)")
    axA.legend(fontsize=9, loc="upper left")

    # (B) the dressed spectrum vs the dressed model (shift-convolved comb)
    axB.plot(ws, np.abs(Cd), "-", color=DRS, lw=0.7)
    first = True
    for n in range(2, 40):
        r = dressed_ref(n, scale)
        if r > 0:
            axB.plot([math.log(n)] * 2, [0, r], color="0.25", lw=2.2,
                     alpha=0.65, solid_capstyle="butt",
                     label="dressed model" if first else None)
            first = False
    axB.annotate("ln4: satellite in the\nmu(4)=0 vacancy\n(= Landau at p=2 "
                 "only)", xy=(math.log(4), dressed_ref(4, scale)),
                 xytext=(0.55, 2.45), fontsize=9,
                 arrowprops=dict(arrowstyle="->", lw=0.8))
    axB.annotate("even lines\nhalved exactly", xy=(math.log(10), 1.54),
                 xytext=(2.35, 2.6), fontsize=9,
                 arrowprops=dict(arrowstyle="->", lw=0.8))
    axB.set_xlabel(r"test frequency $\omega$")
    axB.set_ylabel(r"$|C(\omega)|$")
    axB.set_title(r"Dressed a_1: the same comb convolved with "
                  r"$\delta - 2^{-3/2}\,\delta_{+\ln 2}$")
    axB.legend(fontsize=9, loc="upper left")

    # (C) measured / model per band, both spectra
    xs, yu, yd, labels = [], [], [], []
    for cls, n, w0, mu_m, md_m, mref, _lr, dref in bands:
        if mref > 0:
            xs.append(w0)
            yu.append(mu_m / mref)
            yd.append(md_m / dref if dref > 0 else np.nan)
            labels.append(n)
    axC.axhline(1.0, color="0.5", lw=0.8)
    axC.axhspan(0.85, 1.15, color="0.9", zorder=0)
    axC.plot(xs, yu, "o", color=UND, ms=7, label="undressed / Moebius model")
    axC.plot(xs, yd, "s", color=DRS, ms=6, label="dressed / dressed model")
    for x, y, n in zip(xs, yu, labels):
        axC.annotate(str(n), (x, y), textcoords="offset points",
                     xytext=(0, 7), fontsize=8, ha="center")
    axC.set_ylim(0.0, 1.6)
    axC.set_xlabel(r"line frequency $\ln n$")
    axC.set_ylabel("measured / model")
    axC.set_title("Every squarefree line on its model (band = 8-14% high "
                  "past $\\omega \\sim 3$: leakage)")
    axC.legend(fontsize=9, loc="lower left")

    # (D) the tower: the masquerade unmasked
    ns = TOWER
    x = np.arange(len(ns))
    wd = 0.2
    for off, vals, color, lb in (
            (-1.5, [next(r[3] for r in bands if r[1] == n) for n in ns], UND,
             "undressed (measured)"),
            (-0.5, [next(r[4] for r in bands if r[1] == n) for n in ns], DRS,
             "dressed (measured)"),
            (0.5, [landau_ref(n, scale) for n in ns], "tab:red",
             "Landau model"),
            (1.5, [mobius_ref(n, scale) for n in ns], "0.35",
             "Moebius model")):
        axD.bar(x + off * wd, vals, wd * 0.92, color=color, label=lb)
    axD.set_xticks(x)
    axD.set_xticklabels([f"ln{n}" for n in ns])
    axD.set_ylabel("band max amplitude")
    axD.set_title("The 2-adic tower: dressed ln4 lands on Landau by the "
                  "p = 2 identity 2/p = 1")
    axD.legend(fontsize=9)

    fig.suptitle("The Moebius spectrum under the section dressing: "
                 "C(omega) of the chi_3 K-family displacement constant "
                 "(issue #47)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    print(f"\nfigure -> {FIG_PATH}")


def _main():
    g, a1 = load_weights()
    a1p = undress(g, a1)
    print(f"{len(g)} cached zeros to t = {g.max():.1f}; "
          f"<Im a_1'> = {np.mean(a1p.imag):+.3f} (the n = 1 DC term)")
    scale = fit_scale(g, a1p)
    bands = band_table(g, a1, a1p, scale)
    _print_bands(bands, scale)
    ctls = control_table(g, a1, a1p)
    _print_ctls(ctls)
    print("\n== self-validation ==")
    n_pass, n_fail = validate(bands, ctls, scale)
    print(f"   {n_pass} passed, {n_fail} failed")
    _figure(g, a1, a1p, bands, scale)
    return n_fail


if __name__ == "__main__":
    sys.exit(_main())
