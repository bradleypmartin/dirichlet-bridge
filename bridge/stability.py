r"""Where the truncated-K interpolants are stable: no abscissa, height is the cost.

A stability companion to the bridge. A natural worry, raised by the contrast with the
Dirichlet series, is "how far into the plane do the K-truncated objects actually hold
together?" The zeta series sum n^{-s} converges only for Re s > 1, the eta series only
for Re s > 0; one might expect the interpolants zeta^{(K)} to inherit some such barrier.
They do not. This module makes the stability story explicit and visual.

The point
---------
Each finite-K interpolant is NOT a partial sum of a Dirichlet series. It is

    zeta^{(K)}(s) = 1/(s-1) + 1/2 + sum_{k=1}^K (s/pi k) S(2 pi k, s+1),

and every harmonic S(omega, a) = int_1^inf sin(omega x) x^{-a} dx is an incomplete-gamma
CLOSED FORM (harmonic_bridge._Emom), entire in a for the fixed nonzero argument. So each
zeta^{(K)} is meromorphic on the WHOLE plane -- one pole at s=1, no zeros barrier, no
abscissa of convergence. It is its own analytic continuation, strictly better than either
Dirichlet series. The K=0 endpoint 1/(s-1) already shows this (entire but for the pole);
every harmonic preserves it.

So "where is it stable?" has an unusual answer that inverts the Dirichlet intuition:

  * Re(s) is essentially FREE. zeta^{(K)} converges to zeta(s) at the same O(1/K) rate at
    sigma = -2 as at sigma = 2 (panel a). There is no half-plane boundary; the curves for
    sigma in {-2, -1/2, 1/2, 2} lie on top of each other. (Contrast: the zeta series needs
    sigma>1, the eta series sigma>0 -- both shaded for reference.)

  * HEIGHT t = Im(s) is the cost. The leading rate constant is Delta_1 = s/(2 pi^2)
    (rate_law.rate_comb), so the fixed-K error grows ~ |s| ~ t: deeper up the strip needs
    proportionally more harmonics (panel b, overlaid with the |s|/(2 pi^2 K) law). This is
    the convergence-side face of the birth-K / catch-K law (rate_law.catch_K): higher,
    denser zeros take more harmonics to resolve. It depends on t (and on |s|), not on a
    vertical line in sigma.

The only genuine limit is finite precision, and it too is set by HEIGHT, not Re(s): at very
large t the closed-form moments carry e^{pi t/2}-scale cancellations, so a fixed working
precision eventually floors (the warp evaluator documents this biting near |Im s| ~ 110 and
fixes it by scaling dps / term-count / grid with |Im s|; see warp_bridge). But the O(1/K)
convergence is so slow that at moderate t and dps=30 that floor sits far below the achievable
error -- zeta^{(K)} descends cleanly with no creep through K in the hundreds (panel b stays on
the law). In EXACT arithmetic there is no divergence at any K: more harmonics always help.

Run directly to validate + plot: writes bridge/figures/stability.png.
"""
import sys
from pathlib import Path

import mpmath as mp

# This module sits in a flat source dir alongside its bare cross-imports; put that
# directory on sys.path so a direct ``python stability.py`` run resolves them
# (pytest gets the same path from the root conftest.py).
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import harmonic_bridge as hb  # noqa: E402  (zeta_K -- the additive-comb closed-form interpolant)

mp.mp.dps = 30

PI = mp.pi


def rate_constant(s):
    """The comb leading rate constant Delta_1 = s/(2 pi^2): zeta(s)-zeta^{(K)}(s) ~ -Delta_1/K.

    The same closed form as rate_law.rate_comb; reproduced here so this stability companion
    has no heavy cross-imports. Its magnitude ~ |s| ~ t is exactly why HEIGHT, not Re(s),
    sets how many harmonics a given accuracy needs.
    """
    return mp.mpc(s) / (2 * PI**2)


def conv_error(sigma, t, K):
    """|zeta^{(K)}(sigma+it) - zeta(sigma+it)| -- the interpolant's convergence error."""
    s = mp.mpc(sigma, t)
    return float(abs(hb.zeta_K(s, K) - mp.zeta(s)))


# --------------------------------------------------------------------------
# self-validating driver
# --------------------------------------------------------------------------
def _print_no_abscissa(t=15.0, Ks=(8, 32, 128, 512)):
    """zeta^{(K)} converges at sigma<0 (where BOTH Dirichlet series diverge), same O(1/K)."""
    print(f"== no abscissa: zeta^(K) -> zeta at t={t:.0f} for sigma far left of any series barrier ==")
    print("   sigma      " + "".join(f"K={k:<6d}" for k in Ks) + " rate ~1/K?")
    for sigma in (-2.0, -0.5, 0.5, 2.0):
        errs = [conv_error(sigma, t, K) for K in Ks]
        ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
        tag = "yes" if all(r > 3.0 for r in ratios) else "NO"   # each 4x-in-K should ~4x-shrink
        cells = "".join(f"{e:.2e} " for e in errs)
        print(f"   {sigma:>5.1f}   {cells}  ({tag}; 4x K -> {ratios[-1]:.1f}x drop)")
    print("   (Dirichlet series needs sigma>1; eta series sigma>0 -- the interpolant needs neither.)")


def _print_height_is_cost(K=40, sigma=2.0, ts=(5.0, 20.0, 40.0, 70.0, 100.0)):
    """At fixed K and sigma deep in the trivial region, the error still grows ~|s|~t."""
    print(f"\n== height is the cost: fixed K={K}, sigma={sigma} (trivial for the series at every t) ==")
    print("   t        |zeta_K - zeta|   |s|/(2 pi^2 K) law    ratio")
    for t in ts:
        e = conv_error(sigma, t, K)
        law = float(abs(rate_constant(mp.mpc(sigma, t))) / K)
        print(f"   {t:6.0f}    {e:.3e}        {law:.3e}          {e/law:5.2f}")
    print("   (error tracks the analytic |s|/(2 pi^2 K) rate constant -- a height slowdown, not")
    print("    a Re(s) barrier; ratio ~1 means the leading law captures it.)")


def _validate():
    # no abscissa: clean O(1/K) at sigma = -2 (both series diverge there)
    e_lo, e_hi = conv_error(-2.0, 15.0, 64), conv_error(-2.0, 15.0, 256)
    assert e_hi < e_lo, "zeta_K must keep converging at sigma=-2"
    assert 3.0 < e_lo / e_hi < 5.0, "4x more harmonics should ~4x shrink the error (O(1/K))"
    # sigma-independence: sigma=-2 and sigma=2 errors agree to ~the |s| ratio (both ~|s|/2pi^2 K)
    e_left, e_right = conv_error(-2.0, 40.0, 80), conv_error(2.0, 40.0, 80)
    assert abs(e_left - e_right) / e_right < 0.1, "convergence is ~independent of Re(s)"
    # height scaling: error tracks |s|/(2 pi^2 K) within 25%
    K, sigma, t = 40, 2.0, 80.0
    law = float(abs(rate_constant(mp.mpc(sigma, t))) / K)
    assert abs(conv_error(sigma, t, K) - law) / law < 0.25, "height law must capture the error"
    print("\n[validate] no-abscissa + sigma-independence + height-law checks passed.")


def _main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figstyle
    figstyle.enlarge()

    _print_no_abscissa()
    _print_height_is_cost()
    _validate()

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # ---- panel 0: no abscissa -- error vs K collapses across sigma incl. sigma<0 ----
    Ks = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    t0 = 15.0
    sig_styles = [(-2.0, "C3", "o"), (-0.5, "C1", "s"), (0.5, "C0", "^"), (2.0, "C2", "D")]
    guide0 = None
    for sigma, col, mk in sig_styles:
        errs = [conv_error(sigma, t0, K) for K in Ks]
        ax[0].loglog(Ks, errs, "-" + mk, color=col, ms=5, lw=1,
                     label=rf"$\sigma={sigma:+.1f}$")
        guide0 = errs[0]               # last curve drawn (sigma=+2.0) anchors the 1/K guide
    ax[0].loglog(Ks, [guide0 * Ks[0] / K for K in Ks], ":", color="0.5", label=r"$\propto 1/K$")
    ax[0].set_xlabel("K (harmonics)")
    ax[0].set_ylabel(rf"$|\zeta^{{(K)}}-\zeta|$ at $t={t0:.0f}$")
    ax[0].set_title(r"no abscissa: convergence is the same for every $\sigma$")
    ax[0].legend(loc="lower left", ncol=2)
    ax[0].text(0.97, 0.95, "Dirichlet series:\n  $\\zeta$ needs $\\sigma>1$,  $\\eta$ needs $\\sigma>0$\n"
               r"interpolant: works at $\sigma=-2$ too", transform=ax[0].transAxes,
               ha="right", va="top", fontsize=12,
               bbox=dict(boxstyle="round", fc="0.95", ec="0.7"))

    # ---- panel 1: height is the cost -- fixed-K error vs t tracks |s|/(2 pi^2 K) ----
    K1 = 40
    ts = [5 + 5 * k for k in range(0, 21)]                 # 5 .. 105
    for sigma, col, mk in [(0.5, "C0", "^"), (2.0, "C2", "D")]:
        errs = [conv_error(sigma, t, K1) for t in ts]
        ax[1].plot(ts, errs, mk, color=col, ms=5, label=rf"$\sigma={sigma}$ (measured)")
    law = [float(abs(rate_constant(mp.mpc(2.0, t))) / K1) for t in ts]
    ax[1].plot(ts, law, "-", color="0.4", lw=1.4, label=r"$|s|/(2\pi^2 K)$ law")
    ax[1].set_xlabel(r"height $t=\mathrm{Im}\,s$")
    ax[1].set_ylabel(rf"$|\zeta^{{(K)}}-\zeta|$ at $K={K1}$")
    ax[1].set_title(r"height is the cost: error grows $\propto|s|\sim t$")
    ax[1].legend(loc="upper left")

    fig.suptitle(r"Stability of the $K$-truncated interpolants: meromorphic on all of $\mathbb{C}$ "
                 r"(no abscissa); the cost is height $t$, not $\mathrm{Re}\,s$")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    figdir = Path(__file__).resolve().parent / "figures"
    figdir.mkdir(exist_ok=True)
    out = figdir / "stability.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    _main()
