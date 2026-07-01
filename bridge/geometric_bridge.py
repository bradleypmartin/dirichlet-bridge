r"""The geometric series: the discrete <-> continuous bridge, in miniature.

A deliberately elementary companion to the arc (issue #30 / epic #29). Everything the
bridge does with zeta -- the bland continuous endpoint that keeps the pole, the harmonic
comb that restores discreteness, the nonlinear warp, the O(1/K) rate, the no-abscissa
stability -- is here reproduced on the simplest possible Dirichlet-type object, the
**geometric series**, where every step is a *closed form in elementary functions* and there
are **no special functions and no zeros** at all.

Honest framing (read this first)
--------------------------------
Nothing in this module is new mathematics. The one identity that does the work is the
classical partial-fraction (Mittag-Leffler) expansion of the hyperbolic cotangent -- equivalently
the generating function of the Bernoulli numbers -- and the fact that the finite-K interpolants
sum the *divergent* geometric series (1+2+4+... = -1, Grandi 1-1+1-... = 1/2) to their analytic
values is textbook Abel/Euler summability (Hardy, *Divergent Series*, 1949; DLMF Ch. 24, Sec. 2.10).
The module's only purpose is **illustrative**: it is a fully-transparent consistency check of the
whole apparatus, and it makes the arc's subtlest point -- convergence *beyond the classical domain*
-- completely elementary. It has no zeros, so it advances none of the zero-birth/migration story;
that is exactly why it lives here as a miniature, not in the main arc.

The two objects and the shadow
------------------------------
With the ratio s as the variable and f(x) = s^x = e^{L x}, L = ln s (principal branch):

  * discrete      g(s)  = sum_{n>=0} s^n      = 1/(1-s)         (|s|<1; analytic elsewhere)
  * continuous    C(s)  = int_0^inf s^x dx    = -1/ln s         (|s|<1; analytic elsewhere)

Both keep the pole at s=1 with residue -1 (near s=1, -1/ln s = -1/(s-1) - 1/2 + O(s-1)); the
half-weight below restores the constant.

The comb (Euler-Maclaurin, the additive route  continuous -> discrete)
---------------------------------------------------------------------
Euler-Maclaurin with the periodic-Bernoulli sawtooth {x}-1/2 = -sum_k sin(2 pi k x)/(pi k) and
int_0^inf sin(2 pi k x) e^{L x} dx = 2 pi k/(L^2 + 4 pi^2 k^2) gives, term by term,

    g_K(s) = -1/ln s + 1/2 - sum_{k=1}^K  2 ln s/((ln s)^2 + 4 pi^2 k^2)      (`geom_K`)

-- the K=0 anchor is the continuous shadow + the load-bearing half-weight 1/2 (it cancels the
shadow's -1/2 at the pole, so *every* g_K shares the residue -1). The full comb TELESCOPES
exactly: sum_{k>=1} 2L/(L^2+4 pi^2 k^2) = (1/2)coth(L/2) - 1/L (`harm_sum_closed`), so

    g_inf(s) = 1/2 - (1/2)coth(ln s/2) = 1/(1-s)                              (`geom_limit_closed`)

on the nose -- the comb reconstructs the discrete sum. (`geom_K` -> `discrete`.)

The rate is the arc's rate law with s -> ln s
---------------------------------------------
The k-th harmonic is 2L/(L^2+4 pi^2 k^2) ~ (ln s)/(2 pi^2 k^2): the SAME 1/k^2 sawtooth-tail as
the zeta comb's `zeta_term` ~ s/(2 pi^2 k^2), but with the per-mode constant s replaced by ln s.
So the error is exactly the arc's O(1/K) law with that substitution (`rate_cost`):

    |g_K(s) - 1/(1-s)| = |ln s|/(2 pi^2) (1/K - 1/(2K^2) + ...).

The "height / cost" coordinate that was |s| ~ t for zeta is |ln s| here (log-distance of s from
the pole/unit circle). And -- the stability corollary, made trivial -- every g_K is meromorphic on
all of C (one pole at s=1), so there is **no abscissa**: g_K converges for |s|>1 too, summing the
divergent series to their analytic values (`no_abscissa` in __main__). The classical |s|<1 domain
is irrelevant to the interpolants.

The warp (nonlinear route  continuous -> discrete) -- it factorizes
------------------------------------------------------------------
Warping the integration variable x -> x + phi_K(x) with phi_K = sum_{k=1}^K sin(2 pi k x)/(pi k)
(the same sawtooth as warp_bridge) and the integer-phase DC offset -1/2 pins each cell [m,m+1]
onto its integer m. Because s^{m + rest} = s^m . s^{rest}, the cell index factors straight out of
the sum:

    warp_K(s) = int_0^inf s^{x - 1/2 + phi_K(x)} dx = (1/(1-s)) . J_K(s),
    J_K(s)    = int_0^1 s^{u - 1/2 + phi_K(u)} du  ->  1                       (`warp_J`, `warp_K`)

so the whole warp convergence is the single scalar J_K -> 1 -- no Hurwitz zeta, no moment
machinery (contrast warp_bridge). The midpoint (phase-1/2) warp lands on the half-shifted geometric
sum s^{1/2}/(1-s) = sum_{m>=0} s^{m+1/2} (`warp_mid_K`), the elementary echo of zeta(s,1/2).

The reverse (ascending comb  discrete -> continuous)
----------------------------------------------------
The same harmonic family runs backwards: from the discrete sum, *add* the h_k to recover the
shadow (`asc_K` -> C(s)), so continuous<->discrete is one comb read in either direction.

Run directly to validate + plot: writes bridge/figures/geometric_bridge.png.
"""
import sys
from pathlib import Path

import mpmath as mp

# This module is self-contained (no bridge cross-imports), but sits in the flat source dir;
# keep the sys.path header for symmetry with its siblings and so a direct run resolves figstyle.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

mp.mp.dps = 30

PI = mp.pi


# --------------------------------------------------------------------------
# the two endpoints and the shadow
# --------------------------------------------------------------------------
def discrete(s):
    """The discrete geometric closed form  g(s) = sum_{n>=0} s^n = 1/(1-s)."""
    return 1 / (1 - mp.mpc(s))


def shadow(s):
    """The continuous shadow  C(s) = int_0^inf s^x dx = -1/ln s  (the K=0 comb base, sans 1/2)."""
    return -1 / mp.log(mp.mpc(s))


# --------------------------------------------------------------------------
# the comb (Euler-Maclaurin sawtooth)  continuous -> discrete
# --------------------------------------------------------------------------
def harm(s, k):
    """The k-th comb harmonic  2 ln s/((ln s)^2 + 4 pi^2 k^2)  (~ (ln s)/(2 pi^2 k^2))."""
    L = mp.log(mp.mpc(s))
    return 2 * L / (L ** 2 + 4 * PI ** 2 * k ** 2)


def geom_K(s, K):
    """g_K(s) = -1/ln s + 1/2 - sum_{k=1}^K harm(s,k).  K=0: shadow+1/2;  K->inf: 1/(1-s).

    Inlines `harm` with L = ln s computed once (the per-term reference is `harm`); valid for
    all s != 0, 1 -- meromorphic on all of C, so there is no abscissa of convergence.
    """
    s = mp.mpc(s)
    L = mp.log(s)
    tot = -1 / L + mp.mpf("0.5")
    for k in range(1, K + 1):
        tot -= 2 * L / (L ** 2 + 4 * PI ** 2 * k ** 2)
    return tot


def harm_sum_closed(s):
    """Closed form of the full harmonic sum  sum_{k>=1} harm(s,k) = (1/2)coth(ln s/2) - 1/ln s.

    The classical coth partial-fraction (Mittag-Leffler) expansion -- the identity that makes the
    comb telescope. (DLMF 4.36 / the Bernoulli generating function.)
    """
    L = mp.log(mp.mpc(s))
    return mp.mpf("0.5") * mp.coth(L / 2) - 1 / L


def geom_limit_closed(s):
    """The telescoped limit  1/2 - (1/2)coth(ln s/2)  -- equals `discrete` exactly."""
    L = mp.log(mp.mpc(s))
    return mp.mpf("0.5") * (1 - mp.coth(L / 2))


# --------------------------------------------------------------------------
# the reverse: the ascending comb  discrete -> continuous (same harmonics, added)
# --------------------------------------------------------------------------
def asc_K(s, K):
    """c_K(s) = 1/(1-s) - 1/2 + sum_{k=1}^K harm(s,k).  K=0: discrete-1/2;  K->inf: -1/ln s.

    The comb read backwards: from the discrete sum, ADD the same h_k to recover the shadow.
    """
    s = mp.mpc(s)
    L = mp.log(s)
    tot = discrete(s) - mp.mpf("0.5")
    for k in range(1, K + 1):
        tot += 2 * L / (L ** 2 + 4 * PI ** 2 * k ** 2)
    return tot


# --------------------------------------------------------------------------
# the warp (nonlinear route) -- it factorizes into a single scalar J_K -> 1
# --------------------------------------------------------------------------
def warp_phi(u, K):
    """phi_K(u) = sum_{k=1}^K sin(2 pi k u)/(pi k): the K-harmonic sawtooth (== warp_bridge.warp_phi)."""
    if K <= 0:
        return mp.mpf(0)
    return mp.fsum(mp.sin(2 * PI * k * u) / (PI * k) for k in range(1, K + 1))


def warp_J(s, K, offset=mp.mpf("-0.5")):
    """The single-period factor  J_K(s) = int_0^1 s^{u + offset + phi_K(u)} du.

    offset = -1/2 pins cells onto integers (J_K -> 1); offset = 0 pins onto midpoints
    (J_K -> s^{1/2}). This is the whole content of the warp: the cell index factors out.
    """
    s = mp.mpc(s)
    off = mp.mpf(offset)
    return mp.quad(lambda u: mp.power(s, u + off + warp_phi(u, K)), [0, 1])


def warp_K(s, K):
    """Integer-pinning warp  int_0^inf s^{x-1/2+phi_K} dx = (1/(1-s)) . J_K(s,K) -> 1/(1-s)."""
    return discrete(s) * warp_J(s, K, mp.mpf("-0.5"))


def warp_mid_K(s, K):
    """Midpoint (phase-1/2) warp  int_0^inf s^{x+phi_K} dx = (1/(1-s)) . int_0^1 s^{u+phi_K} du.

    Limit s^{1/2}/(1-s) = sum_{m>=0} s^{m+1/2} -- the half-shifted geometric sum (the elementary
    echo of the zeta warp's zeta(s,1/2)).
    """
    return discrete(s) * warp_J(s, K, mp.mpf(0))


def warp_raw(s, K, ncells=120):
    """Honest period-by-period sum of the warp integral (|s|<1 only) -- an independent check
    that the factorization warp_K = discrete . J_K is exact.

    int_0^inf s^{x-1/2+phi_K} dx = sum_{m>=0} int_0^1 s^{m+u-1/2+phi_K(u)} du.
    """
    s = mp.mpc(s)
    return mp.fsum(
        mp.quad(lambda u, m=m: mp.power(s, m + u - mp.mpf("0.5") + warp_phi(u, K)), [0, 1])
        for m in range(ncells)
    )


# --------------------------------------------------------------------------
# the rate / cost law -- the arc's O(1/K) with s -> ln s
# --------------------------------------------------------------------------
def rate_cost(s, K):
    """Leading comb error  |ln s|/(2 pi^2 K)  (the zeta rate constant s/(2 pi^2 K) with s->ln s)."""
    return abs(mp.log(mp.mpc(s))) / (2 * float(PI) ** 2 * K)


# --------------------------------------------------------------------------
# self-validating driver
# --------------------------------------------------------------------------
# a spread of s: inside/outside the unit disk, real/complex, and the two divergent-series showcases.
SAMPLES = [
    ("0.5      (|s|<1)", mp.mpf("0.5")),
    ("0.9      (near pole)", mp.mpf("0.9")),
    ("0.3+0.4i (|s|<1)", mp.mpc("0.3", "0.4")),
    ("2        (1+2+4..=-1)", mp.mpf("2")),
    ("5        (|s|>1)", mp.mpf("5")),
    ("-1       (Grandi=1/2)", mp.mpf("-1")),
]


def _print_telescoping():
    print("== exact telescoping: full comb == (1/2)coth(L/2)-1/L == 1/(1-s) ==")
    print("   s                    |harm_sum_closed - partialsum(4000)|   |geom_limit - 1/(1-s)|")
    for lbl, s in SAMPLES:
        L = mp.log(mp.mpc(s))
        big = mp.fsum(2 * L / (L ** 2 + 4 * PI ** 2 * k ** 2) for k in range(1, 4001))
        e_hs = float(abs(harm_sum_closed(s) - big))
        e_tel = float(abs(geom_limit_closed(s) - discrete(s)))
        print(f"   {lbl:<20} {e_hs:>34.2e}   {e_tel:>20.2e}")


def _print_convergence():
    print("\n== comb g_K -> 1/(1-s): O(1/K) with cost |ln s| (ratio -> 1) ==")
    print("   s                    |ln s|    K     |g_K - D|    |ln s|/(2pi^2 K)   ratio")
    for lbl, s in SAMPLES:
        d = discrete(s)
        for K in [8, 64, 256]:
            err = float(abs(geom_K(s, K) - d))
            law = float(rate_cost(s, K))
            print(f"   {lbl:<20} {float(abs(mp.log(mp.mpc(s)))):>6.3f}  {K:>4}  "
                  f"{err:>10.3e}   {law:>14.3e}   {err/law:>6.3f}")


def _print_warp_and_reverse():
    print("\n== warp (cont->disc): warp_K = D . J_K -> D;  ascending comb (disc->cont): asc_K -> C ==")
    print("   s                    |warp_32 - D|   |J_32 - 1|   |warp_raw-warp|   |asc_128 - C|")
    for lbl, s in SAMPLES:
        d, c = discrete(s), shadow(s)
        ew = float(abs(warp_K(s, 32) - d))
        ej = float(abs(warp_J(s, 32) - 1))
        # honest-integral cross-check only where int_0^inf converges (|s|<1)
        raw = "     --      " if abs(complex(s)) >= 1 else f"{float(abs(warp_raw(s, 16) - warp_K(s, 16))):.2e}"
        ea = float(abs(asc_K(s, 128) - c))
        print(f"   {lbl:<20} {ew:>12.2e}   {ej:>9.2e}   {raw:>13}    {ea:>10.2e}")
    print("   midpoint warp -> s^{1/2}/(1-s):", end="  ")
    s = mp.mpf("0.5")
    tgt = mp.power(mp.mpc(s), mp.mpf("0.5")) * discrete(s)
    print(f"s=0.5  |warp_mid_32 - s^(1/2)/(1-s)| = {float(abs(warp_mid_K(s, 32) - tgt)):.2e}")


def _print_no_abscissa():
    print("\n== no abscissa: the interpolant sums the DIVERGENT series to its analytic value ==")
    print("   the classical series diverges for |s|>=1, but g_K still lands on 1/(1-s):")
    for lbl, s, note in [("2", mp.mpf(2), "1+2+4+8+...   = -1"),
                         ("3", mp.mpf(3), "1+3+9+27+...  = -1/2"),
                         ("-1", mp.mpf(-1), "1-1+1-1+...   = 1/2 (Grandi)"),
                         ("-2", mp.mpf(-2), "1-2+4-8+...   = 1/3")]:
        d = discrete(s)
        g = geom_K(s, 256)
        print(f"   s={lbl:>3}  {note:<22}  D={complex(d):>10.5g}  g_256={complex(g):>22.10g}"
              f"  |err|={float(abs(g - d)):.1e}")
    # residue -1 at the pole is shared by every K (the load-bearing +1/2): (s-1)g_K -> -1
    eps = mp.mpf("1e-4")
    res = [(K, float(abs(eps * geom_K(1 + eps, K) + 1))) for K in (0, 5, 50)]
    print("   pole residue (s-1)g_K -> -1 for every K (share the pole):  "
          + ", ".join(f"K={K}:{r:.1e}" for K, r in res))


def _main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figstyle
    figstyle.enlarge()
    # long formula titles/suptitle on a wide 3-panel figure -- keep them from overrunning.
    plt.rcParams.update({"axes.titlesize": 14, "figure.titlesize": 15})

    _print_telescoping()
    _print_convergence()
    _print_warp_and_reverse()
    _print_no_abscissa()

    # ================================ figure ================================
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    # panel 0: the solution migrates in the complex plane, shadow (K=0) -> discrete
    migr = [(r"$s=-1$ (Grandi)", mp.mpf(-1), "C0"),
            (r"$s=0.3+0.4i$", mp.mpc("0.3", "0.4"), "C1"),
            (r"$s=2$", mp.mpf(2), "C2"),
            (r"$s=-0.5$", mp.mpf("-0.5"), "C3")]
    Ks = list(range(0, 40)) + [40, 50, 70, 100, 150, 250]
    for lbl, s, col in migr:
        xy = [geom_K(s, K) for K in Ks]
        ax[0].plot([float(v.real) for v in xy], [float(v.imag) for v in xy], "-", lw=1, color=col)
        ax[0].plot(float(xy[0].real), float(xy[0].imag), "s", mfc="none", mec=col, ms=9)  # K=0
        d = discrete(s)
        ax[0].plot(float(d.real), float(d.imag), "*", color=col, ms=16, label=lbl)
    ax[0].set_xlabel(r"$\mathrm{Re}$"); ax[0].set_ylabel(r"$\mathrm{Im}$")
    ax[0].set_title(r"solution migrates: shadow ($K{=}0$, sq) $\to\ \frac{1}{1-s}$ (star)")
    ax[0].legend(loc="center left", fontsize=12); ax[0].grid(alpha=0.3)

    # panel 1: universal O(1/K), cost |ln s|
    KK = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for lbl, s in SAMPLES:
        d = discrete(s)
        ax[1].loglog(KK, [float(abs(geom_K(s, K) - d)) for K in KK], "-o", ms=3, lw=0.8,
                     label=f"$s={lbl.split()[0]}$")
    ax[1].loglog(KK, [float(rate_cost(mp.mpf("0.5"), K)) for K in KK], "k:", lw=1.6)
    ax[1].set_xlabel("K"); ax[1].set_ylabel(r"$|g_K(s)-\frac{1}{1-s}|$")
    ax[1].set_title(r"comb error $O(1/K)$, every $s$ (dotted $\frac{|\ln s|}{2\pi^2 K}$)")
    ax[1].legend(loc="lower left", fontsize=11, ncol=2)

    # panel 2: no abscissa -- error smooth THROUGH the classical |s|=1 boundary
    svals = [x / 100.0 for x in range(-300, 600)
             if abs(x / 100.0 - 1.0) > 0.02 and abs(x / 100.0) > 0.02]
    for K, col in [(16, "C0"), (64, "C1"), (256, "C2")]:
        es = [float(abs(geom_K(mp.mpf(sv), K) - discrete(mp.mpf(sv)))) for sv in svals]
        ax[2].semilogy(svals, es, ".", ms=2, color=col, label=f"$K={K}$")
    ax[2].axvspan(-1, 1, alpha=0.08, color="green")
    ax[2].axvline(1.0, ls="--", color="0.5", lw=1, label=r"pole $s{=}1$ (cont$=$disc)")
    ax[2].text(0.0, min(ax[2].get_ylim()) * 4, "classical\n$|s|<1$", ha="center",
               fontsize=12, color="green")
    ax[2].set_xlabel(r"$s$ (real axis)"); ax[2].set_ylabel(r"$|g_K(s)-\frac{1}{1-s}|$")
    ax[2].set_title(r"no abscissa: converges for all $s$ (cost $\propto|\ln s|$)")
    ax[2].legend(loc="upper right", fontsize=11)

    fig.suptitle(r"The geometric series in miniature: $\sum_{n\geq0}s^n=\frac{1}{1-s}$ "
                 r"vs shadow $-\frac{1}{\ln s}$, comb $g_K=-\frac{1}{\ln s}+\frac{1}{2}"
                 r"-\sum_{k=1}^K\frac{2\ln s}{(\ln s)^2+4\pi^2k^2}$")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    figdir = Path(__file__).resolve().parent / "figures"
    figdir.mkdir(exist_ok=True)
    out = figdir / "geometric_bridge.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    _main()
