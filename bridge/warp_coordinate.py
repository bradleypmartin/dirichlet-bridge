r"""What the warp coordinate looks like: linear x -> midpoint staircase, harmonic by harmonic.

An illustration for the nonlinear midpoint warp of warp_bridge.py. That driver studies
the *zeros* of the warped object; this one just draws the warp itself, so the geometry the
zeros come from is visible at a glance.

The warp coordinate is

    n*(x) = x + phi_K(x),     phi_K(x) = sum_{k=1}^K sin(2 pi k x)/(pi k)   (`warp_bridge.warp_phi`),

with phi_K the K-harmonic Fourier partial sum of the sawtooth 1/2 - {x}. The endpoints:

  * K = 0:        phi_0 = 0, so n* = x -- the bland LINEAR coordinate (no discreteness).
  * K -> infinity: phi_K -> 1/2 - {x}, so n* -> floor(x) + 1/2 -- a MIDPOINT STAIRCASE that
                  collapses each unit cell [m, m+1] onto its midpoint m + 1/2.

So restoring Fourier harmonics literally bends the straight line into a staircase: the cell
midpoints m + 1/2 are exact fixed points at every K (sin(2 pi k (m+1/2)) = 0), and between
them the curve flattens onto the tread as K grows. Because the sawtooth has a unit jump at
each integer, the partial sums carry the classic ~9% Gibbs overshoot there -- the little
wobble-and-jump riding each step edge, which never fully disappears (it only narrows). That
boundary layer is exactly what makes the warp's 1/K rate constant non-elementary in
rate_law.py (the Si-Gibbs C_warp integral).

Run directly to validate + plot: writes bridge/figures/warp_coordinate.png.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# This module sits in a flat source dir alongside its bare cross-imports; put that
# directory on sys.path so a direct ``python warp_coordinate.py`` run resolves them
# (pytest gets the same path from the root conftest.py).
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import warp_bridge as wb  # noqa: E402  (the canonical warp_phi, for the cross-check)


def phi_K(x, K):
    """Vectorized phi_K(x) = sum_{k=1}^K sin(2 pi k x)/(pi k) (the K-harmonic sawtooth).

    A fast numpy twin of warp_bridge.warp_phi (validated against it in __main__).
    """
    x = np.asarray(x, dtype=float)
    if K <= 0:
        return np.zeros_like(x)
    k = np.arange(1, K + 1)
    return (np.sin(2 * np.pi * np.outer(x, k)) / (np.pi * k)).sum(axis=1)


def warp_coord(x, K):
    """The warp coordinate n*(x) = x + phi_K(x)."""
    return np.asarray(x, dtype=float) + phi_K(x, K)


def staircase(x):
    """The K -> infinity limit: the midpoint staircase floor(x) + 1/2."""
    x = np.asarray(x, dtype=float)
    return np.floor(x) + 0.5


def ideal_sawtooth(x):
    """The limit displacement 1/2 - {x} that phi_K converges to."""
    x = np.asarray(x, dtype=float)
    return 0.5 - (x - np.floor(x))


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def _validate():
    # 1. the numpy phi_K matches the canonical warp_bridge.warp_phi to full float precision.
    for xv in (0.3, 1.7, 2.5, 3.14159, 4.62):
        for K in (1, 4, 16, 40):
            mine = float(phi_K(np.array([xv]), K)[0])
            ref = float(wb.warp_phi(xv, K))
            assert abs(mine - ref) < 1e-9, (xv, K, mine, ref)

    # 2. cell midpoints m + 1/2 are EXACT fixed points of the warp at every K.
    mids = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    for K in (1, 2, 8, 33):
        assert np.allclose(warp_coord(mids, K), mids, atol=1e-9)

    # 3. in a cell interior (away from the Gibbs edge) the warp converges to the staircase
    #    tread: the deviation shrinks as K grows.
    xi = np.linspace(2.15, 2.45, 200)        # interior of the [2,3] cell, clear of the edges
    devs = [float(np.max(np.abs(warp_coord(xi, K) - staircase(xi)))) for K in (2, 8, 32, 128)]
    assert devs[0] > devs[1] > devs[2] > devs[3], devs
    print("warp_coordinate validation OK:")
    print("  numpy phi_K == warp_bridge.warp_phi (<1e-9); midpoints fixed; interior dev "
          + " -> ".join("{:.3f}".format(d) for d in devs) + " as K=2,8,32,128.")


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
def _figure():
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    Ks = [1, 2, 4, 8, 16]
    colors = plt.cm.viridis(np.linspace(0.12, 0.82, len(Ks)))

    # ---- panel (a): the warp coordinate over several cells: linear -> staircase ----
    x = np.linspace(0.0, 4.0, 6000)
    ax[0].plot(x, x, ":", color="0.55", lw=1.6, label=r"$K=0$  (linear, $n^*=x$)")
    for K, c in zip(Ks, colors):
        ax[0].plot(x, warp_coord(x, K), color=c, lw=1.0, label="$K={}$".format(K))
    ax[0].plot(x, staircase(x), "--", color="k", lw=1.8,
               label=r"$K\to\infty$  (midpoint staircase)")
    ax[0].plot([0.5, 1.5, 2.5, 3.5], [0.5, 1.5, 2.5, 3.5], "o", color="k", ms=4, zorder=5)
    ax[0].set_title(r"warp coordinate $n^*=x+\phi_K(x)$: linear $\to$ midpoint staircase")
    ax[0].set_xlabel(r"$x$"); ax[0].set_ylabel(r"$n^*(x)$")
    ax[0].set_xlim(0, 4); ax[0].set_ylim(0, 4)
    ax[0].legend(fontsize=7.5, loc="upper left")

    # ---- panel (b): zoom on one cell edge -- the Gibbs overshoot ("the little jumps") ----
    xz = np.linspace(1.55, 2.45, 5000)
    Kz = [4, 8, 16, 64]
    cz = plt.cm.plasma(np.linspace(0.15, 0.8, len(Kz)))
    for K, c in zip(Kz, cz):
        ax[1].plot(xz, warp_coord(xz, K), color=c, lw=1.1, label="$K={}$".format(K))
    ax[1].plot(xz, staircase(xz), "--", color="k", lw=1.8, label=r"$K\to\infty$")
    ax[1].axvline(2.0, color="0.7", lw=0.8, ls=":")
    ax[1].set_title(r"zoom on a step edge ($x=2$): the Gibbs overshoot")
    ax[1].set_xlabel(r"$x$"); ax[1].set_ylabel(r"$n^*(x)$")
    ax[1].legend(fontsize=8, loc="upper left")

    # ---- panel (c): the driving Fourier sum phi_K -- the sawtooth with its wobble ----
    xs = np.linspace(0.0, 2.0, 6000)
    for K, c in zip(Ks, colors):
        ax[2].plot(xs, phi_K(xs, K), color=c, lw=1.0, label="$K={}$".format(K))
    ax[2].plot(xs, ideal_sawtooth(xs), "--", color="k", lw=1.8,
               label=r"$\frac{1}{2}-\{x\}$  (limit)")
    ax[2].set_title(r"the driving sum $\phi_K(x)=\sum_{k=1}^K \frac{\sin(2\pi k x)}{\pi k}$")
    ax[2].set_xlabel(r"$x$"); ax[2].set_ylabel(r"$\phi_K(x)$")
    ax[2].legend(fontsize=8, loc="upper right")

    fig.suptitle(r"Restoring discreteness bends the coordinate: the $n^*$-warp from linear "
                 r"$x$ to the cell-midpoint staircase", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    figdir = Path(__file__).resolve().parent / "figures"
    figdir.mkdir(exist_ok=True)
    out = figdir / "warp_coordinate.png"
    fig.savefig(out, dpi=150)
    print("wrote {}".format(out))


if __name__ == "__main__":
    _validate()
    _figure()
