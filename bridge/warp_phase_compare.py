r"""Two clean staircases: the midpoint warp (alpha=1/2) vs the integer warp (alpha=1).

A companion to warp_coordinate.py that answers a natural question: why does the warp
converge to the *half-integers* (1/2, 3/2, 5/2, ...) instead of the counting numbers
(1, 2, 3, ...)? Both are reachable -- they are the alpha = 1/2 and alpha = 1 cases of
the general-phase warp (warp_alpha.py), whose cells collapse onto floor(x) + alpha:

    n*(x) = x + phi_K^(alpha)(x),   phi_K^(alpha)(x) = (alpha - 1/2) + phi_K(x)
                                                     -> alpha - {x}   as K -> infinity,

so n* -> floor(x) + alpha (`warp_alpha.warp_phi_alpha`). The trade-off is the whole point:

  * alpha = 1/2 (midpoint).  The displacement is the PURE sawtooth phi_K (zero mean, no
    constant term), so K = 0 is exactly the bland linear coordinate n* = x -- the pristine
    1/(s-1) endpoint, "linear x with no trig at all". The price: the staircase lands on the
    HALF-integers, and the completed target is the half-shifted (2^s - 1) zeta(s) = zeta(s, 1/2)
    (with its sigma = 0 companion line).

  * alpha = 1 (integers).  The displacement carries a constant +1/2 (the k = 0 / DC Fourier
    coefficient), so it is "always on" -- at K = 0 the coordinate is already n* = x + 1/2, a
    SHIFTED line, not the bland x. The reward: the staircase lands on the COUNTING NUMBERS and
    the target is plain zeta(s) = zeta(s, 1), no prefactor and no sigma = 0 companion.

The crux (panel c): a zero-mean periodic correction added to x can only ever reach
floor(x) + 1/2. Landing on the integers REQUIRES a non-trigonometric constant 1/2, which
sacrifices the pristine "pure x" start. You can have a pure-x endpoint OR integer steps --
not both. The half-integer midpoint is the unique clean staircase reachable from a pure-x
start by pure trigonometric corrections, which is exactly why it is the bridge's headline.

Run directly to validate + plot: writes bridge/figures/warp_phase_compare.png.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# This module sits in a flat source dir alongside its bare cross-imports; put that
# directory on sys.path so a direct ``python warp_phase_compare.py`` run resolves them
# (pytest gets the same path from the root conftest.py).
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import warp_alpha as wa            # noqa: E402  (the canonical warp_phi_alpha, for the cross-check)
from warp_coordinate import phi_K  # noqa: E402  (the validated numpy sawtooth partial sum)
import figstyle                    # noqa: E402  (larger fonts for the embedded figure)


def warp_alpha_coord(x, K, alpha):
    """The general-phase warp coordinate n*(x) = x + (alpha - 1/2) + phi_K(x) -> floor(x)+alpha."""
    return np.asarray(x, dtype=float) + (alpha - 0.5) + phi_K(x, K)


def staircase_alpha(x, alpha):
    """The K -> infinity limit floor(x) + alpha."""
    return np.floor(np.asarray(x, dtype=float)) + alpha


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def _validate():
    # 1. matches the canonical warp_alpha.warp_phi_alpha to full float precision.
    for xv in (0.3, 1.7, 2.5, 3.62):
        for K in (1, 4, 16):
            for alpha in (0.5, 1.0):
                mine = float(warp_alpha_coord(np.array([xv]), K, alpha)[0])
                ref = xv + float(wa.warp_phi_alpha(xv, K, alpha))
                assert abs(mine - ref) < 1e-9, (xv, K, alpha, mine, ref)

    # 2. the K = 0 endpoints: alpha=1/2 is the pristine line x; alpha=1 is the shifted x + 1/2.
    x = np.linspace(0, 4, 50)
    assert np.allclose(warp_alpha_coord(x, 0, 0.5), x, atol=1e-12)
    assert np.allclose(warp_alpha_coord(x, 0, 1.0), x + 0.5, atol=1e-12)

    # 3. interior convergence to floor(x)+alpha for both phases (deviation shrinks with K).
    for alpha in (0.5, 1.0):
        xi = np.linspace(2.15, 2.45, 200)
        devs = [float(np.max(np.abs(warp_alpha_coord(xi, K, alpha) - staircase_alpha(xi, alpha))))
                for K in (2, 8, 32, 128)]
        assert devs[0] > devs[1] > devs[2] > devs[3], (alpha, devs)
    print("warp_phase_compare validation OK:")
    print("  matches warp_alpha.warp_phi_alpha (<1e-9); K=0 endpoints are x (a=1/2) and x+1/2 "
          "(a=1); both converge to floor(x)+alpha.")


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
def _panel_warp(ax, alpha, k0_label, target_label, cmap):
    Ks = [1, 2, 4, 8, 16]
    colors = cmap(np.linspace(0.18, 0.85, len(Ks)))
    x = np.linspace(0.0, 3.0, 5000)
    ax.plot(x, warp_alpha_coord(x, 0, alpha), ":", color="0.55", lw=1.6, label=k0_label)
    for K, c in zip(Ks, colors):
        ax.plot(x, warp_alpha_coord(x, K, alpha), color=c, lw=1.0, label="$K={}$".format(K))
    ax.plot(x, staircase_alpha(x, alpha), "--", color="k", lw=1.8, label=target_label)
    # the cell levels floor(x)+alpha as faint guides
    for lvl in np.arange(0, 4) + alpha:
        ax.axhline(lvl, color="0.85", lw=0.7, zorder=0)
    ax.set_xlim(0, 3); ax.set_ylim(0, 3.2)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$n^*(x)$")
    ax.legend(fontsize=11, loc="upper left")


def _figure():
    figstyle.enlarge()
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    # (a) alpha = 1/2: pristine x -> half-integers -> (2^s-1) zeta
    _panel_warp(ax[0], 0.5,
                r"$K=0$:  $n^*=x$  (pristine $1/(s-1)$)",
                r"$K\to\infty$:  half-integers",
                plt.cm.viridis)
    ax[0].set_title(r"$\alpha=\frac{1}{2}$ (midpoint): pure $x$ $\to$ half-integers"
                    "\n" r"target $\zeta(s,\frac{1}{2})=(2^s-1)\zeta(s)$", fontsize=13)

    # (b) alpha = 1: x + 1/2 -> counting numbers -> zeta
    _panel_warp(ax[1], 1.0,
                r"$K=0$:  $n^*=x+\frac{1}{2}$  (DC offset on)",
                r"$K\to\infty$:  counting numbers",
                plt.cm.copper)
    ax[1].set_title(r"$\alpha=1$ (integer): $x+\frac{1}{2}$ $\to$ counting numbers"
                    "\n" r"target $\zeta(s,1)=\zeta(s)$", fontsize=13)

    # (c) the displacements: why not both -- the DC (k=0) term
    xs = np.linspace(0.0, 2.0, 5000)
    K = 16
    dh = (0.5 - 0.5) + phi_K(xs, K)     # alpha = 1/2 displacement (zero mean)
    di = (1.0 - 0.5) + phi_K(xs, K)     # alpha = 1   displacement (mean 1/2)
    ax[2].plot(xs, dh, color="C0", lw=1.1, label=r"$\alpha=\frac{1}{2}$:  $\phi_K$  (mean $0$)")
    ax[2].plot(xs, di, color="C1", lw=1.1, label=r"$\alpha=1$:  $\frac{1}{2}+\phi_K$  (mean $\frac{1}{2}$)")
    ax[2].plot(xs, 0.5 - (xs - np.floor(xs)), "--", color="C0", lw=1.4, alpha=0.7)
    ax[2].plot(xs, 1.0 - (xs - np.floor(xs)), "--", color="C1", lw=1.4, alpha=0.7)
    ax[2].axhline(0.0, color="C0", lw=0.8, ls=":")
    ax[2].axhline(0.5, color="C1", lw=0.8, ls=":")
    ax[2].annotate(r"the $+\frac{1}{2}$ DC term", xy=(1.5, 0.5), xytext=(1.05, 0.95),
                   fontsize=11, color="C1",
                   arrowprops=dict(arrowstyle="->", color="C1", lw=0.9))
    ax[2].set_title(r"why not both: integers need a constant $\frac{1}{2}$ (the $k{=}0$ DC term)",
                    fontsize=13)
    ax[2].set_xlabel(r"$x$"); ax[2].set_ylabel(r"displacement $\phi_K^{(\alpha)}(x)$")
    ax[2].legend(loc="lower left")

    fig.suptitle(r"Two clean staircases: pure-$x$ start forces the half-integers "
                 r"($\alpha=\frac{1}{2}$); the counting numbers ($\alpha=1$) cost a baked-in $\frac{1}{2}$")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    figdir = Path(__file__).resolve().parent / "figures"
    figdir.mkdir(exist_ok=True)
    out = figdir / "warp_phase_compare.png"
    fig.savefig(out, dpi=150)
    print("wrote {}".format(out))


if __name__ == "__main__":
    _validate()
    _figure()
