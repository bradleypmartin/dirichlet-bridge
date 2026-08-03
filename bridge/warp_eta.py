r"""Warping the eta integral: the midpoint warp dies, alpha=1 reproduces the eta split.

A supplement to the nonlinear warp (warp_bridge.py / warp_alpha.py). Those modules warped
only the *bland* zeta-side integrand x^{-s} -> (x + phi_K(x))^{-s}, landing the headline
alpha=1/2 midpoint warp on the half-shifted zeta(s, 1/2) = (2^s - 1) zeta(s). The continuous-
eta integrand of the bridge's other endpoint,

    F(s) = int_1^inf cos(pi x) x^{-s} dx        (cont_eta.py, the structured endpoint),

never got the warp. This module warps it -- the WHOLE integrand, both the cos(pi x) sign-
carrier and the power -- and records the result, which is a clean illustration of *why*
alpha=1/2 is a zeta-only luxury.

The warped eta integrand
------------------------
Warp the variable as before, n*(x) = x + (alpha - 1/2) + phi_K(x), and warp the whole
integrand g(y) = cos(pi y) y^{-s} through it:

    warp_eta_alpha(s, K, alpha) = int_1^inf cos(pi n*) (n*)^{-s} dx.

Because phi_K has period 1, write x = m + u and collapse period-by-period. With
g(u) = u + (alpha - 1/2) + phi_K(u) and cos(pi(m + g)) = (-1)^m cos(pi g):

    warp_eta_alpha(s, K, alpha) = sum_{m>=1} (-1)^m int_0^1 cos(pi g(u)) (m + g(u))^{-s} du.

As K -> infinity, n* -> floor(x) + alpha, so each cell collapses onto cos(pi(m+alpha))
(m+alpha)^{-s}. THIS is where the two phases part ways:

  * alpha = 1 (counting numbers): cos(pi(m+1)) = (-1)^{m+1}, so the honest sum -> the
    alternating Dirichlet sum  sum_{n>=2} (-1)^n n^{-s} = -eta(s) + 1. Restoring the omitted
    m=0 (n=1) cell  cos(pi)*1^{-s} = -1  completes it to -eta(s). So the migration object

        warp_eta_K(s, K) = -warp_eta_alpha(s, K, 1) + 1   ->   eta(s) = (1 - 2^{1-s}) zeta(s),

    the same +1 = -(n=1 cell) being the LOAD-BEARING lower-endpoint term that the zeta warp
    needed as +2^s (the half-weight, warp_bridge.warp_half_K), now in eta form.

  * alpha = 1/2 (midpoints, the zeta headline): cos(pi(m + 1/2)) = 0 IDENTICALLY. The midpoint
    lattice {m + 1/2} is exactly the **zero set of the eta sign-carrier cos(pi x)**, so the
    honest warp collapses to 0 (numerically, O(1/K) -> 0; the finite-K residual is the Gibbs
    smoothing of the cos zeros). The elegant half-shift is structurally ANNIHILATED on the eta
    side -- there is no nonzero clean object to land on. (Even keeping cos at the original x and
    warping only the power is degenerate: a cell-constant power times the zero-mean cos
    integrates to 0 over every unit cell, for any alpha. The construction must warp both terms,
    and then alpha=1/2 dies.)

So the eta integral admits a warp route too, but ONLY at alpha = 1, and it lands on
eta = (1 - 2^{1-s}) zeta(s) -- whose zeros are the genuine zeta zeros on sigma = 1/2 UNION the
1 - 2^{1-s} prefactor comb on sigma = 1 (t = 2 pi k/ln2). That is the *same* split the additive
comb produced (harmonic_bridge.eta_K), now reached by the independent warp mechanism: a second
route to the eta split, mirroring how warp_bridge and the comb both reach the zeta zeros on
sigma = 1/2. The convergence is the same O(1/K) (the shared sawtooth tail).

Why alpha = 1/2 is a zeta-only luxury (the takeaway)
---------------------------------------------------
The zeta warp could take alpha = 1/2 -- keeping the pristine zeroless endpoint 1/(s-1) AND
landing on a clean line -- because its integrand x^{-s} has no zeros to hit on the half-integer
lattice. The eta warp cannot: cos(pi x) vanishes there. So eta is forced onto alpha = 1 (the
counting-number phase of warp_alpha / warp_phase_compare), paying the DC-offset endpoint the
zeta side avoided. This sharpens the bridge's phase-comparison story (warp_phase_compare.py):
alpha = 1/2 is special for zeta precisely *because* the discreteness it samples carries no
alternating sign.

Evaluators (mirroring warp_bridge: fast primary + transparent cross-checks)
--------------------------------------------------------------------------
  * warp_eta_alpha -- the fast grid + cos-weighted binomial-moment evaluator (the
    warp_bridge.warp_K machinery with the cell sign (-1)^m, the moments weighted by cos(pi g),
    and the ALTERNATING tail -eta(s+j) in place of zeta(s+j)). Valid for all sigma; reuses
    warp_bridge's |Im s|-scaled working precision / moment count / GL degree -- and so
    inherits its CAUTION too: the guard constants presume warp_bridge's ambient dps = 30.
  * warp_eta_lerch -- the transparent reference int_0^1 cos(pi g) (-Phi(-1, s, 1+g)) du via the
    Lerch transcendent (mp.lerchphi at z = -1, the alternating Hurwitz zeta); agrees with the
    fast form to ~1e-16.
  * warp_eta_raw -- the literal oscillatory integral truncated at int_1^M (M=80), integer
    breakpoints (sigma > 1 sanity), the eta analog of warp_bridge.warp_raw.

Run directly to validate + plot: writes bridge/figures/warp_eta.png.
"""
import sys
from pathlib import Path
from typing import List

import mpmath as mp

# This module sits in a flat source dir alongside its bare cross-imports; put that
# directory on sys.path so a direct ``python warp_eta.py`` run resolves them
# (pytest gets the same path from the root conftest.py).
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import warp_bridge as wb       # noqa: E402  (warp_phi, the shared GL grid + moment machinery, |Im s| scalers)
import harmonic_bridge as hb   # noqa: E402  (migrate -- the backward-continuation zero tracker; eta_K compare)

mp.mp.dps = 30

PI = mp.pi
LN2 = mp.log(2)

_M = 14               # direct cells m=1..M; the analytically-continued alternating tail covers m>M


# --------------------------------------------------------------------------
# the warped eta integrand: int_1^inf cos(pi n*) (n*)^{-s} dx
# --------------------------------------------------------------------------
# Fast evaluator -- the warp_bridge.warp_K grid+moment expansion (see the block comment there),
# adapted to the eta integrand g(y) = cos(pi y) y^{-s}. Period-collapse gives
#   int = sum_{m>=1} (-1)^m int_0^1 cos(pi g) (m+g)^{-s} du,  g = u + (alpha-1/2) + phi_K(u),
# and the m>M tail is continued by the binomial-moment series with cos-weighted moments
# nu_j = int_0^1 cos(pi g) g^j du and the ALTERNATING tail sum_{m>M}(-1)^m m^{-s-j} =
# -eta(s+j) - sum_{m<=M}(-1)^m m^{-s-j} (eta = mp.altzeta is entire, carrying the continuation).
# The alternating tail decays like (M+1)^{-j}, so M=14 converges geometrically (ratio ~ 0.1).
_GRID = {}            # (K, round(alpha,12), degree, prec) -> (logs[m-1][i], cgw[i], nus[j])


def _eta_grid(K, alpha, degree, Jmax=60):
    """Precompute (once per (K, alpha, degree, working precision)): per-cell logs log(m+g_i),
    the cos-folded weights cgw_i = w_i cos(pi g_i), and the cos-weighted moments nu_j = sum_i
    cgw_i g_i^j (rebuilt with more terms when a higher |Im s| needs a larger Jmax)."""
    key = (K, round(float(alpha), 12), degree, mp.mp.prec)
    cached = _GRID.get(key)
    if cached is None or len(cached[2]) < Jmax + 1:
        us, ws = wb._gl_nodes(degree)
        c = mp.mpf(alpha) - mp.mpf("0.5")
        gs = [u + c + wb.warp_phi(u, K) for u in us]
        cgw = [ws[i] * mp.cos(PI * gs[i]) for i in range(len(us))]
        logs = [[mp.log(m + gs[i]) for i in range(len(us))] for m in range(1, _M + 1)]
        nus = [mp.fsum(cgw[i] * gs[i] ** j for i in range(len(us))) for j in range(Jmax + 1)]
        _GRID[key] = (logs, cgw, nus)
    return _GRID[key]


def _eta_moment_series(s, logs, cgw, nus, base_dps):
    """The binomial-moment sum for int_1^inf cos(pi n*) (n*)^{-s} dx from a precomputed grid.

    The caller must already be inside mp.workdps(wb._moment_dps(s)); the loop breaks at the
    ambient base_dps so the binomial factor never amplifies floored-tail garbage (exactly as in
    warp_bridge._moment_series, here with the (-1)^m cell sign and the alternating -eta tail)."""
    n = len(cgw)
    eps = mp.mpf(10) ** (-(base_dps - 1))
    tot = mp.fsum((-1) ** m * mp.fsum(cgw[i] * mp.exp(-s * logs[m - 1][i]) for i in range(n))
                  for m in range(1, _M + 1))                       # sum_{m<=M} (-1)^m int cos(pi g)(m+g)^{-s}
    c = mp.mpf(1)                                                  # binom(-s, 0)
    mpows = [(-1) ** m * mp.power(m, -s) for m in range(1, _M + 1)]  # (-1)^m m^{-(s+j)}, stepped /m per j
    for j in range(len(nus)):
        altHM = mp.fsum(mpows)
        term = c * nus[j] * (-mp.altzeta(s + j) - altHM)          # m>M alternating tail, continued
        tot += term
        if j > 4 and abs(term) < eps * (abs(tot) + 1):
            break
        c = c * (-s - j) / (j + 1)                                # binom(-s, j+1)
        for m in range(2, _M + 1):
            mpows[m - 1] /= m                                      # (-1)^m m^{-(s+j)} -> (-1)^m m^{-(s+j+1)}
    return tot


def warp_eta_alpha(s, K, alpha):
    """Honest warped eta integrand  int_1^inf cos(pi n*) (n*)^{-s} dx,  n* = x+(alpha-1/2)+phi_K.

    The fast grid+moment evaluator. alpha = 1 -> -eta(s) + 1 (the n=1 cell omitted by int_1^inf);
    alpha = 1/2 -> 0 (cos vanishes on the half-integer lattice). Valid for all sigma; like
    warp_bridge.warp_K it scales working precision / moment count / GL degree with |Im s|.
    """
    s = mp.mpc(s)
    base_dps = mp.mp.dps
    with mp.workdps(wb._moment_dps(s)):
        degree = wb._gl_degree(s)
        logs, cgw, nus = _eta_grid(K, alpha, degree, wb._moment_jmax(s))
        result = _eta_moment_series(s, logs, cgw, nus, base_dps)
    return +result                                                # round back to ambient dps


def warp_eta_K(s, K):
    """The completed alpha=1 migration object  -warp_eta_alpha(s,K,1) + 1  ->  eta(s).

    The +1 = -(omitted n=1 cell cos(pi)*1^{-s} = -1) is the LOAD-BEARING lower-endpoint term --
    the eta-side analog of the zeta warp's +2^s (warp_bridge.warp_half_K). Its zeros are eta's:
    the genuine zeta zeros on sigma = 1/2 and the 1 - 2^{1-s} prefactor comb on sigma = 1.
    """
    s = mp.mpc(s)
    return -warp_eta_alpha(s, K, 1.0) + 1


def warp_eta_lerch(s, K, alpha):
    """Transparent reference: int_0^1 cos(pi g) (-Phi(-1, s, 1+g)) du, g = u+(alpha-1/2)+phi_K.

    Uses the Lerch transcendent Phi(-1, s, a) = sum_{m>=0} (-1)^m (m+a)^{-s} (mp.lerchphi at
    z = -1) for the alternating cell sum sum_{m>=1}(-1)^m (m+g)^{-s} = -Phi(-1, s, 1+g). A
    Lerch eval per quad node -- slow, kept as the cross-check the fast warp_eta_alpha matches to
    ~1e-16. (Avoid sigma exactly = 1, where the alternating Lerch series crawls.)
    """
    s = mp.mpc(s)
    c = mp.mpf(alpha) - mp.mpf("0.5")
    return mp.quad(lambda u: (lambda g: mp.cos(PI * g) * (-mp.lerchphi(-1, s, 1 + g)))
                   (u + c + wb.warp_phi(u, K)), [0, 1])


def warp_eta_raw(s, K, alpha, M=80):
    """The literal oscillatory int_1^M cos(pi n*) (n*)^{-s} dx, integer breakpoints (sigma > 1).

    An independent coarse confirmation (the eta analog of warp_bridge.warp_raw): the warp's
    kinks sit at the integers, so integrating panel-by-panel over [m, m+1] resolves them.
    """
    s = mp.mpc(s)
    c = mp.mpf(alpha) - mp.mpf("0.5")
    return mp.quad(lambda x: (lambda n: mp.cos(PI * n) * n ** (-s))(x + c + wb.warp_phi(x, K)),
                   [mp.mpf(m) for m in range(1, M + 1)])


# --------------------------------------------------------------------------
# the K -> infinity target and the sigma=1 prefactor comb
# --------------------------------------------------------------------------
def target_eta(s):
    """The K -> infinity warp_eta_K limit  eta(s) = (1 - 2^{1-s}) zeta(s) (= mp.altzeta).

    Its zeros are the two families warp_eta_K migrates onto: the genuine zeta zeros on
    sigma = 1/2, and the 1 - 2^{1-s} prefactor comb on sigma = 1 (t = 2 pi k/ln2). Same split
    the additive comb (harmonic_bridge.eta_K) produces, here by the warp route.
    """
    return mp.altzeta(mp.mpc(s))


def comb_heights(t_max):
    """Heights t = 2 pi k/ln2 of the sigma=1 prefactor zeros 1 - 2^{1-s} = 0 (the eta comb).

    The same lattice as warp_bridge.pref_heights (the sigma=0 companion sits at the FE-mirror
    height); here they are the eta-warp's own sigma=1 family.
    """
    out = []  # type: List[float]
    k = 1
    while 2 * float(PI) * k / float(LN2) <= t_max:
        out.append(2 * float(PI) * k / float(LN2))
        k += 1
    return out


# K schedule for the migration (warp_eta_alpha is the fast evaluator; K=45 lands the zeros
# within ~2e-2 of their lines, like warp_bridge). A hand-kept literal copy of
# warp_bridge.K_SCHEDULE -- edits there must be mirrored here.
K_SCHEDULE = [1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 27, 35, 45]


# --------------------------------------------------------------------------
# self-validating driver
# --------------------------------------------------------------------------
def _print_convergence():
    print("== warp_eta_K -> eta(s) (alpha=1, O(1/K));  alpha=1/2 honest -> 0 (cos killed) ==")
    print("   s                      K    |warp_eta_K - eta|   drop     |alpha=1/2 honest| -> 0")
    for s in [mp.mpc(2, 0), mp.mpc(0.7, 9), mp.mpc(0.5, 14.134725)]:
        eta = mp.altzeta(s)
        prev = None
        for K in (5, 15, 45):
            err = float(abs(warp_eta_K(s, K) - eta))
            half = float(abs(warp_eta_alpha(s, K, 0.5)))
            drop = "  -- " if prev is None else f"{prev/err:4.2f}x"
            prev = err
            print(f"   {complex(s)!s:>18}  {K:3d}   {err:.3e}          {drop}    {half:.3e}")
    # the load-bearing +1, and the lerch / raw cross-checks
    s0 = mp.mpc(2, 1)
    print(f"\n   fast vs transparent lerch  (s={complex(s0)}, alpha=1, K=12): "
          f"{float(abs(warp_eta_alpha(s0, 12, 1.0) - warp_eta_lerch(s0, 12, 1.0))):.2e}")
    sr = mp.mpc(2.5, 1)
    print(f"   fast vs raw oscillatory     (s={complex(sr)}, alpha=1, K=8):  "
          f"{float(abs(warp_eta_alpha(sr, 8, 1.0) - warp_eta_raw(sr, 8, 1.0))):.2e}")


def _print_migration(title, targets, schedule=None):
    print(f"\n== {title} ==")
    hdrK = [1, 5, 16, 27, 45]
    print("   target zero            " + "".join(f"K={k:<6d}" for k in hdrK))
    trajs = []
    for label, zt in targets:
        traj = hb.migrate(warp_eta_K, zt, schedule=schedule if schedule is not None else K_SCHEDULE)
        d = {K: zz for K, zz in traj if zz is not None}
        cells = "".join(f"{float(d[k].real):<7.3f}" if k in d else "  --   " for k in hdrK)
        print(f"   {label:<20}   {cells}")
        trajs.append((label, zt, traj))
    return trajs


def _main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figstyle
    figstyle.enlarge()
    # three wide panels with long titles, plus a long suptitle: trim the title/legend sizes a
    # touch (still much larger than the matplotlib default) so the panel titles do not overrun.
    plt.rcParams.update({"axes.titlesize": 13, "legend.fontsize": 11, "figure.titlesize": 15})

    _print_convergence()

    zeta_targets = [(f"1/2+{g:.3f}i", mp.mpc(0.5, g))
                    for g in (14.134725, 21.022040, 25.010858)]
    comb_targets = [(f"1+{t:.3f}i", mp.mpc(1.0, t)) for t in comb_heights(30.0)]
    cz = _print_migration("warp_eta_K: genuine zeta zeros climb onto sigma=1/2", zeta_targets)
    cc = _print_migration("warp_eta_K: prefactor comb climbs onto sigma=1 (t=2 pi k/ln2)",
                          comb_targets)

    # =============================== figure ===============================
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 5.2))

    # panel 0: the obstruction -- half-integers ARE the zeros of the eta sign-carrier cos(pi x)
    yy = [0.4 + 0.01 * k for k in range(int((5.3 - 0.4) / 0.01) + 1)]
    ax[0].plot(yy, [float(mp.cos(PI * y)) for y in yy], "-", color="0.5", lw=1.4,
               label=r"$\cos(\pi y)$ (the $\eta$ sign)")
    halves = [m + 0.5 for m in range(0, 5)]
    ints = [n for n in range(1, 6)]
    ax[0].plot(halves, [0] * len(halves), "X", color="C3", ms=13, mec="k", mew=0.5,
               label=r"$\alpha=\frac{1}{2}$ samples (midpoints)")
    ax[0].plot(ints, [float(mp.cos(PI * n)) for n in ints], "o", color="C2", ms=11, mec="k",
               mew=0.5, label=r"$\alpha=1$ samples (integers)")
    ax[0].axhline(0.0, ls=":", lw=1, color="0.7")
    ax[0].set_xlabel(r"cell target $y$")
    ax[0].set_ylabel(r"$\cos(\pi y)$")
    ax[0].set_title(r"why $\alpha=\frac{1}{2}$ dies: midpoints are $\cos$'s zeros")
    ax[0].legend(loc="lower right")
    ax[0].set_ylim(-1.5, 1.5)

    # panel 1: convergence -- alpha=1 reconstructs eta (O(1/K)); alpha=1/2 collapses to 0
    s0 = mp.mpc(0.7, 9.0)
    eta0 = mp.altzeta(s0)
    Ks = [2, 4, 8, 16, 32, 45]
    e1 = [float(abs(warp_eta_K(s0, K) - eta0)) for K in Ks]
    e_half = [float(abs(warp_eta_alpha(s0, K, 0.5))) for K in Ks]
    ax[1].loglog(Ks, e1, "-o", color="C0", label=r"$\alpha=1$: $|\mathrm{warp}_\eta^{(K)}-\eta|$")
    ax[1].loglog(Ks, e_half, "--s", color="C3", mfc="none",
                 label=r"$\alpha=\frac{1}{2}$: $|\mathrm{warp}_\eta^{(K)}|\to0$")
    ax[1].loglog(Ks, [e1[0] * Ks[0] / K for K in Ks], ":", color="0.5", label=r"$\propto 1/K$")
    ax[1].set_xlabel("K")
    ax[1].set_ylabel(r"error at $s=0.7+9i$")
    ax[1].set_title(r"$\alpha=1$ rebuilds $\eta$; $\alpha=\frac{1}{2}$ vanishes (both $O(1/K)$)")
    ax[1].legend(loc="lower left", fontsize=11)

    # panel 2: the migration -- the SAME sigma=1/2 U sigma=1 split as the additive comb
    def _xy(traj):
        pts = [(K, zz) for K, zz in traj if zz is not None]
        return ([float(zz.real) for _, zz in pts], [float(zz.imag) for _, zz in pts])

    for label, _zt, traj in cz:
        xs, ys = _xy(traj)
        if xs:
            ax[2].plot(xs, ys, "-o", ms=3, lw=1, color="C0")
            ax[2].plot(xs[-1], ys[-1], "k.", ms=9)
    for label, _zt, traj in cc:
        xs, ys = _xy(traj)
        if xs:
            ax[2].plot(xs, ys, "-o", ms=3, lw=1, color="C1")
            ax[2].plot(xs[-1], ys[-1], "k.", ms=9)
    ax[2].plot([], [], "-o", color="C0", label=r"genuine $\zeta$ zeros")
    ax[2].plot([], [], "-o", color="C1", label=r"prefactor comb")
    ax[2].axvline(0.5, ls="-", lw=1, color="C2", label=r"critical line $\frac{1}{2}$")
    ax[2].axvline(1.0, ls="--", lw=1, color="0.5", label=r"$\sigma=1$ ($1-2^{1-s}$)")
    ax[2].set_xlabel(r"$\sigma=\mathrm{Re}\,s$")
    ax[2].set_ylabel(r"$t=\mathrm{Im}\,s$")
    ax[2].set_title(r"$\alpha=1$ $\eta$-warp: the $\sigma=\frac{1}{2}\cup\sigma=1$ split")
    ax[2].set_xlim(0.2, 1.4)
    ax[2].legend(loc="upper right", fontsize=11)

    fig.suptitle(r"Warping the $\eta$ integral: the midpoint $\alpha=\frac{1}{2}$ warp is annihilated "
                 r"by $\cos(\pi x)$; only $\alpha=1$ survives, reproducing the $\eta$ split")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    figdir = Path(__file__).resolve().parent / "figures"
    figdir.mkdir(exist_ok=True)
    out = figdir / "warp_eta.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    _main()
