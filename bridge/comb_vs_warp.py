r"""The two restoration routes, side by side at the level of the integrand (issue #11).

The bridge restores discreteness from the bland endpoint int_1^inf x^{-s} dx = 1/(s-1) to
zeta(s) by two different routes (developed in harmonic_bridge.py and warp_bridge.py). Both
inject the *same* K-harmonic discreteness kernel

    phi_K(x) = sum_{k=1}^K sin(2 pi k x)/(pi k)        (`warp_bridge.warp_phi`),

the Fourier partial sum of the sawtooth 1/2 - {x} (-> 1/2 - {x} as K -> infinity). The two
routes differ only in *where* phi_K enters the integrand -- and that is exactly the thing a
first-time reader wants to see drawn. This driver draws it.

Route 1 -- the ADDITIVE comb (harmonic_bridge.zeta_K).  Leave the variable x alone; add the
kernel to the integrand. Euler-Maclaurin on zeta with g(x) = x^{-s} gives

    zeta^{(K)}(s) = int_1^inf [ x^{-s} + s phi_K(x) x^{-s-1} ] dx + 1/2,

since sum_k s/(pi k) int sin(2 pi k x) x^{-s-1} dx = int s phi_K(x) x^{-s-1} dx (the +1/2 is
the constant Euler-Maclaurin endpoint term, a boundary term, not part of the integrand). So
the plotted integrand is

    b_K(x) = x^{-s} + s phi_K(x) x^{-s-1} = x^{-s} ( 1 + s phi_K(x)/x ):

the smooth x^{-s} carrying an *added* K-harmonic ripple. LINEAR in phi_K; x is untouched.

Route 2 -- the variable WARP (warp_bridge.warp_K).  Leave the integrand's shape alone;
compose the kernel into the argument, x -> x + phi_K(x):

    W_K(s) = int_1^inf ( x + phi_K(x) )^{-s} dx,   plotted integrand  w_K(x) = (x+phi_K(x))^{-s}.

NONLINEAR in phi_K. As K -> infinity, x + phi_K(x) -> floor(x) + 1/2, so each unit cell
collapses onto its midpoint and w_K -> (floor(x)+1/2)^{-s}, a midpoint staircase -- the route
that lands on the half-shifted zeta(s, 1/2) of warp_bridge.

Both start from the *same* bland integrand: at K = 0, phi_0 = 0, so b_0 = w_0 = x^{-s}.
The two OBJECTS differ by a boundary constant -- the comb's is 1/(s-1) + 1/2 (the n=1
half-weight), the warp's the bare 1/(s-1). Restoring harmonics then pulls them apart --
by ADDITION on the left, by COMPOSITION on the right. (`warp_coordinate.py` draws the warp
*coordinate* n* = x + phi_K; this driver draws the two *integrands* the routes integrate.)

Run directly to validate + plot: writes bridge/figures/comb_vs_warp.png.
"""
import sys
from pathlib import Path

import numpy as np
from mpmath import mp

# This module sits in a flat source dir alongside its bare cross-imports; put that
# directory on sys.path so a direct ``python comb_vs_warp.py`` run resolves them
# (pytest gets the same path from the root conftest.py).
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import harmonic_bridge as hb  # noqa: E402  (the additive comb zeta_K, for the area cross-check)
import warp_bridge as wb      # noqa: E402  (the canonical warp_phi, warp_K, warp_raw)

mp.dps = 30

# The real point we draw and validate at. sigma = 2 > 1 keeps both integrands honestly
# integrable on [1, inf) (so "area = the object" is a convergent statement, not just a
# continuation), gives a fast-decaying picture whose structure lives in [1, 5], and is in no
# way special for zeta_K or warp_K. The composition-vs-addition contrast is identical for
# complex s -- a real s just lets us draw real-valued integrands cleanly.
S = 2.0


# --------------------------------------------------------------------------
# the two integrands (vectorized numpy twins of the mpmath route definitions)
# --------------------------------------------------------------------------
def phi_K(x, K):
    """phi_K(x) = sum_{k=1}^K sin(2 pi k x)/(pi k): the shared K-harmonic sawtooth kernel.

    A vectorized numpy twin of warp_bridge.warp_phi (validated against it in _validate)."""
    x = np.asarray(x, dtype=float)
    if K <= 0:
        return np.zeros_like(x)
    k = np.arange(1, K + 1)
    return (np.sin(2 * np.pi * np.outer(x, k)) / (np.pi * k)).sum(axis=1)


def comb_integrand(x, K, s=S):
    """Route 1 (additive): b_K(x) = x^{-s} + s phi_K(x) x^{-s-1} -- kernel ADDED, x linear."""
    x = np.asarray(x, dtype=float)
    return x ** (-s) + s * phi_K(x, K) * x ** (-s - 1)


def warp_integrand(x, K, s=S):
    """Route 2 (warp): w_K(x) = (x + phi_K(x))^{-s} -- kernel COMPOSED into the argument."""
    x = np.asarray(x, dtype=float)
    return (x + phi_K(x, K)) ** (-s)


def comb_integrand_limit(x, s=S):
    """K -> infinity additive integrand: x^{-s} + s (1/2 - {x}) x^{-s-1} (full sawtooth ripple)."""
    x = np.asarray(x, dtype=float)
    return x ** (-s) + s * (0.5 - (x - np.floor(x))) * x ** (-s - 1)


def warp_integrand_limit(x, s=S):
    """K -> infinity warp integrand: (floor(x) + 1/2)^{-s} -- the collapsed midpoint staircase."""
    x = np.asarray(x, dtype=float)
    return (np.floor(x) + 0.5) ** (-s)


# --------------------------------------------------------------------------
# validation -- the plotted integrands really integrate to the bridge objects
# --------------------------------------------------------------------------
def _validate():
    s = mp.mpf(S)

    # 1. the numpy kernel matches the canonical warp_bridge.warp_phi to full float precision.
    for xv in (0.3, 1.7, 2.5, 3.14159, 4.62):
        for K in (1, 4, 16):
            mine = float(phi_K(np.array([xv]), K)[0])
            ref = float(wb.warp_phi(xv, K))
            assert abs(mine - ref) < 1e-9, (xv, K, mine, ref)

    # 2. at K = 0 both integrands are the bland x^{-s} (phi_0 = 0): the shared start.
    for xv in (1.2, 2.7, 4.1):
        assert abs(float(comb_integrand(xv, 0)) - xv ** (-S)) < 1e-12
        assert abs(float(warp_integrand(xv, 0)) - xv ** (-S)) < 1e-12

    # 3. ADDITIVE: int_1^inf [x^{-s} + s phi_K x^{-s-1}] dx + 1/2 == zeta_K(s, K). The smooth
    #    x^{-s} integrates to 1/(s-1); the oscillatory ripple is integrated here by an
    #    INDEPENDENT quadrature (quadosc), not by zeta_K's incomplete-gamma closed form.
    for K in (1, 2, 4):
        ripple = mp.quadosc(lambda x: s * wb.warp_phi(x, K) * x ** (-s - 1),
                            [1, mp.inf], period=1)
        area = 1 / (s - 1) + mp.mpf("0.5") + ripple
        ref = hb.zeta_K(s, K)
        assert abs(area - ref) < mp.mpf("1e-6"), (K, area, ref)

    # 4. WARP: int_1^inf (x + phi_K)^{-s} dx == warp_K(s, K). warp_raw is exactly this integral
    #    by oscillatory quadrature (the plotted w_K's area); warp_K is the fast grid evaluator.
    for K in (1, 2, 4):
        area = wb.warp_raw(s, K)          # mp.quadosc((x+phi_K)^{-s}, [1, inf]) -- the plotted w_K
        ref = wb.warp_K(s, K)
        assert abs(area - ref) < mp.mpf("1e-3"), (K, area, ref)

    # 5. the warp integrand collapses onto the midpoint value (m+1/2)^{-s} in a cell interior
    #    as K grows (away from the Gibbs edges); the additive integrand does NOT (it oscillates
    #    about x^{-s} instead). A one-line witness of "compose -> staircase" vs "add -> ripple".
    xi = np.array([2.30])                  # interior of the [2, 3] cell, clear of the edges
    mid = (2.5) ** (-S)
    warp_dev = [abs(float(warp_integrand(xi, K)[0]) - mid) for K in (4, 16, 64)]
    assert warp_dev[0] > warp_dev[1] > warp_dev[2], warp_dev

    print("comb_vs_warp validation OK:")
    print("  numpy phi_K == warp_bridge.warp_phi (<1e-9); K=0 integrands == x^-s;")
    print("  additive area + 1/2 == zeta_K and warp area == warp_K (s={}, K=1,2,4);".format(S))
    print("  warp integrand -> (2.5)^-s in the [2,3] cell as K=4,16,64: dev "
          + " -> ".join("{:.4f}".format(d) for d in warp_dev) + ".")


# --------------------------------------------------------------------------
# figure -- the two integrands, side by side
# --------------------------------------------------------------------------
def _figure():
    # matplotlib deferred to here (as in warp_eta/geometric_bridge): importing this module
    # for its integrands -- e.g. under pytest -- shouldn't pay the pyplot/backend load.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figstyle
    figstyle.enlarge()
    # the panel titles carry the two integrand formulas; keep them a touch smaller than the
    # global style so they stay within each panel's width (as the other dense drivers do).
    plt.rcParams["axes.titlesize"] = 13
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)

    x = np.linspace(1.0, 5.0, 9000)
    base = x ** (-S)
    Ks = [2, 4, 8, 16]
    mids = np.array([1.5, 2.5, 3.5, 4.5])

    # ---- panel (a): ADDITIVE comb -- kernel added on top of x^{-s}, variable untouched ----
    ax[0].plot(x, base, "-", color="0.55", lw=1.8, label=r"$K=0$  ($x^{-s}$, bland)")
    for K, c in zip(Ks, plt.cm.viridis(np.linspace(0.12, 0.82, len(Ks)))):
        ax[0].plot(x, comb_integrand(x, K), color=c, lw=1.1, label="$K={}$".format(K))
    ax[0].plot(x, comb_integrand_limit(x), "--", color="k", lw=1.6,
               label=r"$K\to\infty$  ($+\,s(\frac{1}{2}-\{x\})x^{-s-1}$)")
    ax[0].set_title(r"Additive comb:  $x^{-s} + s\,\phi_K(x)\,x^{-s-1}$"
                    "\n" r"(kernel added; $x$ linear)")
    ax[0].set_xlabel(r"$x$")
    ax[0].set_ylabel(r"integrand")
    ax[0].set_xlim(1, 5)
    ax[0].legend(fontsize=12, loc="upper right")
    ax[0].text(2.35, 0.74, "harmonics ride\non top of $x^{-s}$", fontsize=12, color="0.25",
               ha="left", va="top",
               bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.7))

    # ---- panel (b): variable WARP -- kernel composed into the argument, measure pulled in ----
    ax[1].plot(x, base, "-", color="0.55", lw=1.8, label=r"$K=0$  ($x^{-s}$, bland)")
    for K, c in zip(Ks, plt.cm.plasma(np.linspace(0.12, 0.78, len(Ks)))):
        ax[1].plot(x, warp_integrand(x, K), color=c, lw=1.1, label="$K={}$".format(K))
    # the K -> infinity midpoint staircase, drawn as a step (constant (m+1/2)^{-s} per cell)
    xs_step = np.linspace(1.0, 5.0, 4001)
    ax[1].step(xs_step, warp_integrand_limit(xs_step), where="post", linestyle="--",
               color="k", lw=1.6, label=r"$K\to\infty$  (midpoint staircase)")
    ax[1].plot(mids, mids ** (-S), "o", color="k", ms=6, zorder=6)
    for m in mids:
        ax[1].axvline(m, ls=":", lw=0.8, color="0.7")
    ax[1].set_title(r"Variable warp:  $(x+\phi_K(x))^{-s}$"
                    "\n" "(kernel composed in; $x$ warped)")
    ax[1].set_xlabel(r"$x$")
    ax[1].set_xlim(1, 5)
    ax[1].legend(fontsize=12, loc="upper right")
    ax[1].text(2.35, 0.74, r"measure pulled onto" "\n" r"midpoints $m+\frac{1}{2}$", fontsize=12,
               color="0.25", ha="left", va="top",
               bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.7))

    fig.suptitle(r"Two routes restore discreteness from one kernel $\phi_K$:  "
                 r"add it to the integrand (left) vs. warp the variable (right)  $[s=2]$")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    figdir = Path(__file__).resolve().parent / "figures"
    figdir.mkdir(exist_ok=True)
    out = figdir / "comb_vs_warp.png"
    fig.savefig(out, dpi=150)
    print("wrote {}".format(out))


if __name__ == "__main__":
    _validate()
    _figure()
