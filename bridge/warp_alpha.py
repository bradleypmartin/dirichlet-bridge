r"""General-phase n+alpha warp -> Hurwitz zeta(s, alpha): why only alpha in {0, 1/2, 1}
is clean -- the Davenport-Heilbronn / Saias-Weingartner dichotomy.

The general-phase follow-up to the midpoint warp of the discrete <-> continuous bridge. The
midpoint warp (warp_bridge.py) pinned the n*-warp to cell *midpoints* n+1/2 and found the clean
half-shifted target zeta(s, 1/2) = (2^s - 1) zeta(s), zeros on sigma = 1/2 UNION sigma = 0.
But 1/2 was *chosen*. Here we pin to a **general phase** n+alpha, alpha in (0,1), watch
where the zeros go as a function of alpha, and explain why 1/2 (and 0, 1) are privileged.

The general-phase warp
----------------------
The midpoint warp x + phi_K(x) collapses each cell [m, m+1] onto its midpoint m + 1/2,
because phi_K(x) -> 1/2 - {x}. To pin instead to m + alpha we need the warp to send
x + phi_K^(alpha)(x) -> floor(x) + alpha, i.e. phi_K^(alpha)(x) -> alpha - {x}. Since
alpha - {x} = (1/2 - {x}) + (alpha - 1/2) and 1/2 - {x} = sum_k sin(2 pi k x)/(pi k),

    phi_K^(alpha)(x) = (alpha - 1/2) + phi_K(x)            (`warp_phi_alpha`)

-- the same K-harmonic sawtooth plus the **DC constant alpha - 1/2** (the k=0 Fourier
coefficient of the shifted sawtooth). So "change the phase" = "add a constant offset";
phi_K (the midpoint warp) is the alpha = 1/2 special case (constant 0).

The honest int_1^inf, collapsed period-by-period to the Hurwitz form (valid all sigma):

    warp_alpha(s, K, alpha) = int_0^1 zeta(s, (1/2 + alpha) + u + phi_K(u)) du
                            -> zeta(s, 1 + alpha)            as K -> infinity

(using sum_{m>=1}(m + a)^{-s} = zeta(s, 1+a); the +(1/2+alpha) is 1 + (alpha-1/2)). The
two endpoints:

  * K = 0 (phi_0 = 0, but the DC offset is *always on*):
        warp_alpha(s, 0, alpha) = int_1^inf (x + alpha - 1/2)^{-s} dx
                                = (alpha + 1/2)^{1-s} / (s - 1)
    -- the bland 1/(s-1) ONLY at alpha = 1/2; the phase offset deforms even the
    endpoint (verified exactly).
  * K -> infinity: warp_alpha -> zeta(s, 1 + alpha), the cells 1+alpha, 2+alpha, ...

The completion -- the alpha^{-s} half-cell (the load-bearing term, general-alpha form)
------------------------------------------------------------------------------------
int_1^inf omits the m = 0 cell at alpha, value alpha^{-s}. Restoring it completes the
honest zeta(s, 1+alpha) = sum_{m>=1}(m+alpha)^{-s} to the full Hurwitz zeta:

    warp_complete_alpha(s, K, alpha) = warp_alpha(s, K, alpha) + alpha^{-s}
                                     -> zeta(s, alpha) = sum_{m>=0}(m + alpha)^{-s}.

This is the general-alpha form of the midpoint warp's load-bearing +2^s: at alpha = 1/2 the
half-cell is (1/2)^{-s} = 2^s and zeta(s, 1/2) = (2^s - 1) zeta(s) (so `warp_complete_alpha(.,.,1/2)`
IS the midpoint warp's `warp_half_K`); at alpha = 1 it is 1^{-s} = 1 and zeta(s, 1) = zeta(s).

THE PAYOFF -- clean vertical strings only at alpha in {0, 1/2, 1} (Davenport-Heilbronn)
--------------------------------------------------------------------------------------
The migration target is the completed Hurwitz zeta(s, alpha). Its zeros lie on clean
vertical lines **iff** zeta(s, alpha) factors as P(2^s, 3^s, ...)*L(s, chi) (a polynomial
in prime-powers times a single Dirichlet L). By Saias-Weingartner (Acta Arith. 140, 2009)
a periodic-coefficient Dirichlet series NOT of that form has ~T zeros in *every* strip up
past sigma = 1 -- scattered, off any line. zeta(s, alpha) is a finite combination of
Dirichlet L-functions for rational alpha = p/q; it degenerates to a single P*L only at

  * alpha = 1   -> zeta(s, 1)   = zeta(s)              -> zeros on sigma = 1/2 (plain zeta);
  * alpha = 1/2 -> zeta(s, 1/2) = (2^s - 1) zeta(s)    -> sigma = 1/2 UNION sigma = 0
                                                          (the 2^s - 1 companion);
  * alpha = 0   -> the completion alpha^{-s} diverges, but the *honest* warp
                   zeta(s, 1+0) = zeta(s, 1) = zeta(s) is already plain zeta
                   -> sigma = 1/2 (the comb's plain-zeta target). The clean alpha = 0
                   phase lives in the honest (uncompleted) endpoint, not the completion.

Every *generic* alpha gives a genuine multi-L sum -- the **Davenport-Heilbronn phenomenon**
(Davenport-Heilbronn 1936 / Cassels 1961; computed by Spira 1976): zeros off sigma = 1/2,
spread across the strip and even into sigma > 1. So sweeping alpha through (0,1) is a
numerical demonstration of the Saias-Weingartner P*L classification, with the zero
trajectories rho(alpha) the Garunkstis-Steuding (Math. Comp. 76, 2007) Hurwitz-zero tool.

What this module provides
-------------------------
  * warp_phi_alpha / warp_alpha / warp_alpha_hurwitz / warp_complete_alpha -- the
    general-phase warp (fast grid evaluator + transparent Hurwitz cross-check), reducing
    to warp_bridge's alpha = 1/2 objects;
  * find_zeros -- a seed-grid + findroot + dedupe box zero-finder;
  * clean_score / sweep_alpha -- the dichotomy measures (distance of the zero locus from
    the clean lines; count of sigma > 1 zeros) swept over alpha;
  * reuse of harmonic_bridge.migrate for the warp zero trajectories.

Run directly to validate + plot: writes bridge/figures/warp_alpha.png.
"""
import sys
from pathlib import Path
from typing import List

import mpmath as mp

# This module sits in a flat source dir alongside its bare cross-imports; put that
# directory on sys.path so a direct ``python warp_alpha.py`` run resolves them
# (pytest gets the same path from the root conftest.py).
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import warp_bridge as wb       # noqa: E402  (warp_phi, the shared GL grid + moment machinery, alpha=1/2 objects)
import harmonic_bridge as hb   # noqa: E402  (migrate -- the backward-continuation zero tracker)

mp.mp.dps = 30

PI = mp.pi
LN2 = mp.log(2)

# the privileged phases (Saias-Weingartner P*L degeneracies) and a couple generic ones
CLEAN_ALPHAS = [0.5, 1.0]                 # 0 lives in the honest endpoint (see module docstring)
GENERIC_ALPHAS = [0.3, 0.7]


# --------------------------------------------------------------------------
# the general-phase warp:  phi_K^(alpha) = (alpha - 1/2) + phi_K
# --------------------------------------------------------------------------
def warp_phi_alpha(u, K, alpha):
    """phi_K^(alpha)(u) = (alpha - 1/2) + phi_K(u): the K-harmonic sawtooth, DC-shifted.

    As K -> infinity, -> alpha - {u}, so the warp x + phi_K^(alpha)(x) pins each cell
    [m, m+1] onto m + alpha. alpha = 1/2 (constant 0) is the midpoint warp.
    """
    return (mp.mpf(alpha) - mp.mpf("0.5")) + wb.warp_phi(u, K)


# ---- fast evaluator: the warp_bridge grid+moment expansion, re-keyed on (K, alpha) ----
# Identical machinery to warp_bridge.warp_K (see the block comment there): fix the GL grid
# once, precompute psi = u + (alpha-1/2) + phi_K(u) and per-(K,alpha) logs/moments, then
# evaluate any s by the binomial-moment expansion of int_1^inf = sum_{m>=1} int_0^1. Reuses
# warp_bridge's shared helpers (_gl_nodes / _moment_logs_mus / _moment_series) and its |Im s|-
# scaled working precision / moment-term count / GL degree, so this inherits the high-tau fix.
_GRID = {}            # (K, round(alpha,12), degree, prec) -> (logs[m-1][i] = log(m+psi_i), mus[j])


def _grid(K, alpha, degree, Jmax=60):
    """Precompute (once per (K, alpha, degree, working precision)) the per-cell logs and moments
    mu_j up to Jmax (rebuilt with more terms when a higher |Im s| needs a larger Jmax)."""
    key = (K, round(float(alpha), 12), degree, mp.mp.prec)
    cached = _GRID.get(key)
    if cached is None or len(cached[1]) < Jmax + 1:
        us, ws = wb._gl_nodes(degree)
        c = mp.mpf(alpha) - mp.mpf("0.5")
        psis = [u + c + wb.warp_phi(u, K) for u in us]
        _GRID[key] = wb._moment_logs_mus(us, ws, psis, Jmax)
    return _GRID[key]


def warp_alpha(s, K, alpha):
    """Honest warp int_1^inf (x + phi_K^(alpha))^{-s} dx -> zeta(s, 1 + alpha).

    The fast grid+moment evaluator (warp_bridge.warp_K re-keyed on alpha). K = 0 gives the
    phase-deformed endpoint (alpha + 1/2)^{1-s}/(s-1); valid for all sigma. alpha = 1/2
    reproduces warp_bridge.warp_K (-> zeta(s, 3/2)). Like warp_K it scales its working
    precision / moment-term count / GL degree with |Im s|, so it stays accurate at high tau.
    """
    s = mp.mpc(s)
    base_dps = mp.mp.dps
    with mp.workdps(wb._moment_dps(s)):
        degree = wb._gl_degree(s)
        logs, mus = _grid(K, alpha, degree, wb._moment_jmax(s))
        result = wb._moment_series(s, logs, mus, degree, base_dps)
    return +result                                                 # round back to ambient dps


def warp_alpha_hurwitz(s, K, alpha):
    """Transparent (slow) evaluator: int_0^1 zeta(s, (1/2 + alpha) + u + phi_K(u)) du.

    The period-by-period collapse of int_1^inf; a Hurwitz eval per quad node. Kept as the
    reference the fast `warp_alpha` is validated against (they agree to ~1e-6).
    """
    s = mp.mpc(s)
    c = mp.mpf(alpha) - mp.mpf("0.5")
    return mp.quad(lambda u: mp.zeta(s, 1 + u + c + wb.warp_phi(u, K)), [0, 1])


def warp_complete_alpha(s, K, alpha):
    """Endpoint-completed warp: warp_alpha + alpha^{-s} -> zeta(s, alpha).

    alpha^{-s} = (m=0 cell at alpha) is the load-bearing half-cell, the general-alpha form
    of the midpoint warp's +2^s (= (1/2)^{-s}). The migration object: its limit zeta(s, alpha) is clean
    only at the Saias-Weingartner P*L phases alpha in {1/2, 1}.
    """
    s = mp.mpc(s)
    return warp_alpha(s, K, alpha) + mp.power(mp.mpf(alpha), -s)


def target(s, alpha):
    """The completed warp's limit zeta(s, alpha) (= warp_complete_alpha as K -> infinity)."""
    return mp.zeta(mp.mpc(s), mp.mpf(alpha))


# --------------------------------------------------------------------------
# box zero-finder: seed grid -> findroot -> dedupe
# --------------------------------------------------------------------------
def find_zeros(fn, sig_range, t_range, dsig=0.30, dt=0.6, workdps=18,
               tol=1e-8, dedupe=1e-3, pad=0.35):
    """Zeros of fn(s) in the box sig_range x t_range, by local-minimum-seeded findroot.

    A blind findroot from every grid seed is a trap: a bad seed sends the secant iterates
    flying to extreme s where the Hurwitz zeta is very expensive, so each runaway seed costs
    seconds. Instead we first sample |fn| cheaply on the grid (one eval per node) and run
    findroot ONLY from local minima of |fn| -- each such seed sits near an actual zero, so
    the solve converges in a few steps without escaping. Roots are polished at reduced
    precision `workdps` (zeros want location, not 30 digits), kept if inside the padded box
    with |fn| < tol, and deduped within `dedupe`. Returns the list sorted by (Im, Re).
    """
    (smin, smax), (tmin, tmax) = sig_range, t_range
    zeros = []  # type: List[mp.mpc]
    with mp.workdps(workdps):
        ftol = mp.mpf(10) ** (-(workdps - 4))   # reachable at this precision
        sig_vals = []  # type: List[float]
        s = smin
        while s <= smax + 1e-9:
            sig_vals.append(s)
            s += dsig
        t_vals = []  # type: List[float]
        t = tmin
        while t <= tmax + 1e-9:
            t_vals.append(t)
            t += dt
        grid = [[abs(fn(mp.mpc(sv, tv))) for tv in t_vals] for sv in sig_vals]
        ni, nj = len(sig_vals), len(t_vals)
        for i in range(ni):
            for j in range(nj):
                nbrs = [grid[i2][j2] for i2 in (i - 1, i, i + 1) for j2 in (j - 1, j, j + 1)
                        if 0 <= i2 < ni and 0 <= j2 < nj and (i2, j2) != (i, j)]
                if grid[i][j] > min(nbrs):                       # not a local minimum
                    continue
                try:
                    z = mp.findroot(fn, mp.mpc(sig_vals[i], t_vals[j]), tol=ftol, maxsteps=20)
                except Exception:
                    continue
                if (smin - pad <= z.real <= smax + pad and tmin - pad <= z.imag <= tmax + pad
                        and abs(fn(z)) < tol and all(abs(z - w) > dedupe for w in zeros)):
                    zeros.append(mp.mpc(z))
    zeros.sort(key=lambda z: (float(z.imag), float(z.real)))
    return zeros


def clean_score(zeros, lines=(0.0, 0.5)):
    """How far the zero locus sits from the clean vertical lines `lines` (default {0, 1/2}).

    Returns (max_dev, n_off, n_gt1): the worst-case sigma distance to the nearest clean
    line over all zeros, the count of zeros further than 0.02 from every line, and the count
    with sigma > 1.02 (the Davenport-Heilbronn sigma > 1 signature). Clean phases score ~0.
    """
    if not zeros:
        return (0.0, 0, 0)
    devs = [min(abs(float(z.real) - L) for L in lines) for z in zeros]
    n_off = sum(1 for d in devs if d > 0.02)
    n_gt1 = sum(1 for z in zeros if z.real > 1.02)
    return (max(devs), n_off, n_gt1)


def sweep_alpha(alphas, sig_range=(-0.25, 1.7), t_range=(2.0, 22.0), lines=(0.0, 0.5),
                **kw):
    """For each alpha, the completed-target zeros zeta(s, alpha) in the box + clean_score.

    Returns a list of (alpha, zeros, (max_dev, n_off, n_gt1)). The clean phases alpha in
    {1/2, 1} land all zeros on `lines`; the generic ones scatter (n_off, n_gt1 > 0).
    """
    out = []
    for a in alphas:
        zs = find_zeros(lambda s, a=a: target(s, a), sig_range, t_range, **kw)
        out.append((a, zs, clean_score(zs, lines)))
    return out


# --------------------------------------------------------------------------
# self-validating driver
# --------------------------------------------------------------------------
def _print_identities():
    print("== general-phase warp identities ==")
    print("   honest warp_alpha -> zeta(s, 1+alpha);  +alpha^{-s} -> zeta(s, alpha)")
    s = mp.mpc(2, 1)
    for a in (0.5, 1.0, 0.3, 0.7):
        eh = float(abs(warp_alpha(s, 45, a) - mp.zeta(s, 1 + mp.mpf(a))))
        ec = float(abs(warp_complete_alpha(s, 45, a) - mp.zeta(s, mp.mpf(a))))
        print(f"   alpha={a:>4}:  |warp-z(s,1+a)|={eh:.2e}   |warp+a^-s - z(s,a)|={ec:.2e}")
    # the alpha=1/2 warp IS the midpoint warp's warp_half_K; the endpoint is phase-deformed
    s2 = mp.mpc(1.3, 7)
    print(f"   alpha=1/2 completion == warp_bridge.warp_half_K? "
          f"{float(abs(warp_complete_alpha(s2, 7, 0.5) - wb.warp_half_K(s2, 7))):.1e}")
    print(f"   K=0 endpoint (alpha+1/2)^(1-s)/(s-1): "
          f"alpha=0.3 -> {float(abs(warp_alpha(s2, 0, 0.3) - mp.power(mp.mpf('0.8'), 1-s2)/(s2-1))):.1e}")


def _print_dichotomy(sweep):
    print("\n== the clean-vs-scattered dichotomy (zeros of zeta(s, alpha) in the box) ==")
    print("   alpha   #zeros   max dev from {0,1/2}   #off-line   #sigma>1   verdict")
    for a, zs, (dev, noff, ngt1) in sweep:
        verdict = "CLEAN" if dev < 0.02 else "scattered (Davenport-Heilbronn)"
        print(f"   {a:>5.3f}   {len(zs):>4d}     {dev:>8.4f}            {noff:>4d}      {ngt1:>4d}    {verdict}")


def _main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figstyle
    figstyle.enlarge()

    _print_identities()

    # ---- the alpha sweep (Panel B + the dichotomy table) ----
    sweep_as = [0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 1.00]
    sweep = sweep_alpha(sweep_as)
    _print_dichotomy(sweep)

    # ---- the fixed-alpha scatters (Panel A: two clean vs two generic), a taller box ----
    # generics first, clean (alpha=1/2, then 1) drawn on top -- alpha=1's zeta zeros sit on
    # sigma=1/2 exactly where alpha=1/2's do, so it needs a larger marker not to vanish.
    scatter_as = [(0.3, "C3", "^", 26), (0.7, "C1", "v", 26), (0.5, "C0", "s", 30),
                  (1.0, "C2", "o", 70)]
    scatter = {}
    for a, _c, _m, _sz in scatter_as:
        scatter[a] = find_zeros(lambda s, a=a: target(s, a), (-0.4, 2.3), (1.0, 38.0),
                                dsig=0.33, dt=0.7)

    # ---- the warp anchor (Panel C): warp_complete_alpha -> zeta(s, alpha), O(1/K) ----
    s0 = mp.mpc(0.6, 12.0)
    Ks = [2, 4, 8, 16, 32, 45]
    conv = {a: [float(abs(warp_complete_alpha(s0, K, a) - target(s0, a))) for K in Ks]
            for a in (0.5, 0.3)}

    # =============================== figure ===============================
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 5.2))

    # panel 0: fixed-alpha zero scatter -- clean vertical strings vs scattered cloud
    for a, col, mk, sz in scatter_as:
        zs = scatter[a]
        xs = [float(z.real) for z in zs]
        ys = [float(z.imag) for z in zs]
        tag = "clean" if a in (0.5, 1.0) else "scattered"
        mfc = "none" if a == 1.0 else col
        ax[0].scatter(xs, ys, s=sz, marker=mk, label=rf"$\alpha={a}$ ({tag})",
                      facecolors=mfc, edgecolors=("C2" if a == 1.0 else "k"),
                      linewidths=(1.4 if a == 1.0 else 0.3), zorder=(4 if a in (0.5, 1.0) else 3))
    ax[0].axvline(0.5, ls="-", lw=1, color="0.5", zorder=1)
    ax[0].axvline(0.0, ls=":", lw=1, color="0.5", zorder=1)
    ax[0].axvline(1.0, ls="--", lw=1, color="0.7", zorder=1, label=r"$\sigma>1$ edge")
    ax[0].set_xlabel(r"$\sigma=\mathrm{Re}\,s$")
    ax[0].set_ylabel(r"$t=\mathrm{Im}\,s$")
    ax[0].set_title(r"zeros of $\zeta(s,\alpha)$: clean lines vs scatter")
    ax[0].legend(loc="upper right")
    ax[0].set_xlim(-0.5, 2.4)

    # panel 1: the sweep -- (alpha, sigma) of every zero; clean verticals at 1/2 and 1
    for a, zs, _sc in sweep:
        for z in zs:
            ax[1].plot(a, float(z.real), ".", ms=6, color="C0", alpha=0.7)
    for xa in (0.5, 1.0):
        ax[1].axvline(xa, ls="--", lw=1, color="C2")
    ax[1].axhline(0.5, ls="-", lw=0.8, color="0.6")
    ax[1].axhline(0.0, ls=":", lw=0.8, color="0.6")
    ax[1].axhline(1.0, ls="--", lw=0.8, color="0.75", label=r"$\sigma=1$ (S-W edge)")
    ax[1].text(0.5, 1.05, r"clean", ha="center", fontsize=12, color="C2")
    ax[1].text(1.0, 1.05, r"clean", ha="center", fontsize=12, color="C2")
    ax[1].set_xlabel(r"phase $\alpha$")
    ax[1].set_ylabel(r"$\sigma$ of zeros ($2<t<22$)")
    ax[1].set_title(r"sweep: $\sigma$-locus vs $\alpha$ (clean only at $\frac{1}{2},1$)")
    ax[1].set_ylim(-0.45, 1.18)
    ax[1].legend(loc="lower right")

    # panel 2: the warp anchor -- warp_complete_alpha -> zeta(s, alpha), both O(1/K)
    ax[2].loglog(Ks, conv[0.5], "-o", color="C0", label=r"$\alpha=\frac{1}{2}$ (clean target)")
    ax[2].loglog(Ks, conv[0.3], "--s", color="C3", mfc="none",
                 label=r"$\alpha=0.3$ (scattered target)")
    ax[2].loglog(Ks, [conv[0.5][0] * Ks[0] / K for K in Ks], ":", color="0.5",
                 label=r"$\propto 1/K$")
    ax[2].set_xlabel("K")
    ax[2].set_ylabel(r"$|\mathrm{warp}_\alpha^{(K)}+\alpha^{-s}-\zeta(s,\alpha)|$")
    ax[2].set_title(r"warp anchors the target, any $\alpha$ ($O(1/K)$)")
    ax[2].legend()

    fig.suptitle(r"General-phase $n+\alpha$ warp $\to\ \zeta(s,\alpha)$: clean strings only at "
                 r"$\alpha\in\{0,\frac{1}{2},1\}$ (Davenport-Heilbronn / Saias-Weingartner)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    figdir = Path(__file__).resolve().parent / "figures"
    figdir.mkdir(exist_ok=True)
    out = figdir / "warp_alpha.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    _main()
