r"""Harmonic interpolation  zeta^{(K)} / eta^{(K)}: the zero-migration map.

The additive-comb stage of the discrete <-> continuous bridge. The continuous
endpoint (cont_eta.py) built the *continuous* eta integral F(s) = int_1^inf cos(pi x) x^{-s} dx and
showed it has off-critical zeros (Re -> 3/2) obeying a Riemann-von Mangoldt law.
Here we restore discreteness *one Fourier harmonic at a time* and watch the zeros
move: build interpolants zeta^{(K)} / eta^{(K)} running from a bland/ground
endpoint (K = 0) to the genuine Dirichlet object (K -> infinity), and track each
zero as a curve parameterized by K -- the zero-migration map.

The two routes
--------------
(a) Euler-Maclaurin with the periodic-Bernoulli sawtooth  B~_1(x) = {x} - 1/2 =
    -sum_k sin(2 pi k x)/(pi k).  For g(x) = x^{-s},

        zeta(s) = 1/(s-1) + 1/2 + int_1^inf ({x}-1/2)(-s) x^{-s-1} dx,

    so truncating the sine series at K harmonics gives `zeta_K` (closed form via the
    incomplete-gamma "moments" S(omega,a) = int_1^inf sin(omega x) x^{-a} dx). This
    is the literal realization of the "Fourier series of 1/2 - n on [0,1]".

(b) Abel-Plana with the Bose kernel  1/(e^{2 pi x}-1) = sum_{k>=1} e^{-2 pi k x},
    truncated at K, gives `zeta_K_bose` (a clean, exponentially damped quadrature --
    the rigorous backbone).

**A unification this module makes explicit.** For zeta the two routes are not just
co-convergent, they are *identical term by term* (`test_routes_identical`): the
k-th sawtooth mode  s/(pi k) S(2 pi k, s+1)  equals the k-th Bose term
i int_0^inf [(1+ix)^{-s} - (1-ix)^{-s}] e^{-2 pi k x} dx. The Euler-Maclaurin comb
and the Abel-Plana comb are one harmonic decomposition in two integral guises, so
`zeta_K` (fast closed form) and `zeta_K_bose` (independent quadrature) compute the
same object; we keep the latter as a cross-check.

The eta side, and how it continues the continuous endpoint
----------------------------------------------------------
Euler-Maclaurin on the alternating sum eta(s) = -sum cos(pi n) n^{-s} (with
h(x) = cos(pi x) x^{-s}, so int_1^inf h dx = F(s)) gives

        eta_K(s) = -F(s) + 1/2 + sum_{k=1}^K (1/(pi k)) D_k(s),

the sawtooth modes D_k living at the *odd* multiples (2k +/- 1) pi -- the eta comb's
fundamental is the cos(pi x) already inside F. The **ground state K = 0 is
-F(s) + 1/2**. Note the +1/2: it is the n=1 half-weight, and it is *load-bearing* --
it moves the ground-state zeros off the F-zero string (where F = 0, Re ~ 1.8)
down to Re ~ 0.7 (where F = 1/2) *before any harmonic is switched on*. So the
continuous endpoint's zeros are the zeros of the integrand F, an upstream landmark;
the bridge's own K=0 zeros are a different, already-closer string. (The half-weight
is a known watch-out, and this is exactly where it bites.)

(`eta_K_bose` is the alternating Abel-Plana, expanding 1/(2 sinh(pi x)) =
sum_{k>=0} e^{-(2k+1) pi x} -- an *odd*-pi Bose comb, a co-convergent twin; both
-> eta. Its K=0 anchor differs from -F+1/2, the two routes parting at finite K but
sharing the limit.)

The deliverable -- the zero-migration map (`migrate`, the _main figure/table)
----------------------------------------------------------------------------
`migrate` follows a *true* zero of zeta/eta back down in K by continuation (seed at
large K = the true zero, where the interpolant ~ the real object, then step K
downward). The findings:

  * zeta^{(K)}: the endpoint 1/(s-1)+1/2 is zeroless, so the zeros are *born* -- the
    very first harmonic (K=1) creates them out near Re ~ 1-1.35; further harmonics
    slide them onto Re = 1/2. Approach is O(1/K) and mostly monotone, but some zeros
    overshoot below 1/2 at small K and climb back (e.g. the gamma=21.022 zero).

  * eta^{(K)}: from the K=0 ground string (Re ~ 0.7) the zeros SPLIT as K grows --
    half onto Re = 1/2 (the genuine zeta zeros) and half onto Re = 1 (the zeros of
    the prefactor 1 - 2^{1-s}, at t = 2 pi m / ln 2). No net birth/death: eta's two
    zero densities (t/2pi)ln(t/2pi e) [critical line] + (t ln2/2pi) [prefactor] sum
    to (t/2pi)ln(t/pi e) -- exactly the Riemann-von Mangoldt density of the
    ground string, so the migration is a (near-)bijection of ground zeros onto the
    two target lines (the counting check in _main witnesses this).

Run directly to validate + plot: writes bridge/figures/harmonic_bridge.png.
"""
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import mpmath as mp

# This module sits in a flat source dir alongside its bare cross-imports; put that
# directory on sys.path so a direct ``python harmonic_bridge.py`` run resolves them
# (pytest gets the same path from the root conftest.py).
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cont_eta as ce  # noqa: E402  (F_closed, the K=0 eta anchor; sigma_law)

mp.mp.dps = 30

_J = mp.j
PI = mp.pi


# --------------------------------------------------------------------------
# incomplete-gamma "moments"  int_1^inf e^{+- i omega x} x^{-a} dx
# --------------------------------------------------------------------------
def _Emom(omega, a):
    """int_1^inf e^{i omega x} x^{-a} dx = (-i omega)^{a-1} Gamma(1-a, -i omega)."""
    p = -_J * omega
    return p ** (a - 1) * mp.gammainc(1 - a, p)


def Cmom(omega, a):
    """int_1^inf cos(omega x) x^{-a} dx.  (Cmom(pi, s) == F(s), the continuous-eta object.)"""
    return (_Emom(omega, a) + _Emom(-omega, a)) / 2


def Smom(omega, a):
    """int_1^inf sin(omega x) x^{-a} dx."""
    return (_Emom(omega, a) - _Emom(-omega, a)) / (2 * _J)


# --------------------------------------------------------------------------
# zeta side -- route (a) closed form  ==  route (b) Bose quadrature
# --------------------------------------------------------------------------
def zeta_term(s, k):
    """The k-th harmonic of the zeta comb (sawtooth form):  s/(pi k) S(2 pi k, s+1)."""
    return s / (PI * k) * Smom(2 * PI * k, s + 1)


def zeta_K(s, K):
    """zeta^{(K)}(s) via the Euler-Maclaurin sawtooth (route a). K=0: 1/(s-1)+1/2."""
    s = mp.mpc(s)
    tot = 1 / (s - 1) + mp.mpf("0.5")
    for k in range(1, K + 1):
        tot += zeta_term(s, k)
    return tot


def zeta_term_bose(s, k):
    """The k-th harmonic in Abel-Plana/Bose form -- equals zeta_term(s, k) exactly."""
    s = mp.mpc(s)
    integ = lambda x: ((1 + _J * x) ** (-s) - (1 - _J * x) ** (-s)) * mp.exp(-2 * PI * k * x)
    return _J * mp.quad(integ, [0, mp.inf])


def zeta_K_bose(s, K):
    """zeta^{(K)}(s) via Abel-Plana + Bose kernel (route b) -- the rigorous twin."""
    s = mp.mpc(s)
    tot = 1 / (s - 1) + mp.mpf("0.5")
    for k in range(1, K + 1):
        tot += zeta_term_bose(s, k)
    return tot


# --------------------------------------------------------------------------
# eta side -- route (a) anchored on F, plus the odd-pi Bose twin
# --------------------------------------------------------------------------
def _Dk(s, k):
    """int_1^inf sin(2 pi k x) h'(x) dx, h = cos(pi x) x^{-s} (the eta sawtooth mode).

    h'(x) = -pi sin(pi x) x^{-s} - s cos(pi x) x^{-s-1}; the products collapse (by
    product-to-sum) to cosine/sine moments at the odd frequencies (2k +/- 1) pi.
    """
    return (-PI / 2 * (Cmom((2 * k - 1) * PI, s) - Cmom((2 * k + 1) * PI, s))
            - s / 2 * (Smom((2 * k + 1) * PI, s + 1) + Smom((2 * k - 1) * PI, s + 1)))


def eta_K(s, K):
    """eta^{(K)}(s) via Euler-Maclaurin on -sum cos(pi n) n^{-s} (route a).

    K = 0 is exactly  -F(s) + 1/2  (the continuous-eta integral + the load-bearing n=1
    half-weight); K -> infinity is eta(s).
    """
    s = mp.mpc(s)
    tot = -ce.F_closed(s) + mp.mpf("0.5")
    for k in range(1, K + 1):
        tot += _Dk(s, k) / (PI * k)
    return tot


def eta_K_bose(s, K):
    """eta^{(K)}(s) via the alternating Abel-Plana, odd-pi Bose comb (route b).

    1/(2 sinh(pi x)) = sum_{k>=0} e^{-(2k+1) pi x}; K counts harmonics k = 0..K, so
    K=0 keeps the lone odd-pi fundamental (a co-convergent anchor distinct from
    -F+1/2).
    """
    s = mp.mpc(s)
    tot = mp.mpf("0.5")
    for k in range(0, K + 1):
        integ = lambda x: ((1 + _J * x) ** (-s) - (1 - _J * x) ** (-s)) * mp.exp(-(2 * k + 1) * PI * x)
        tot += _J * mp.quad(integ, [0, mp.inf])
    return tot


# --------------------------------------------------------------------------
# the zero-migration map -- backward continuation in K from a true zero
# --------------------------------------------------------------------------
# K schedule, dense where the motion is fast (small K), sparse where it crawls.
K_SCHEDULE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 22, 27, 33, 40, 50, 65,
              85, 110, 150, 200]


def migrate(fn, z_true, schedule=None, max_jump=6.0):
    """Trajectory of the zero of fn(., K) limiting to z_true, as K decreases.

    Seeds findroot at the largest K with z_true (there fn ~ the true object, so its
    zero is within O(1/K) of z_true), then walks K downward, each solve seeded by the
    previous root. Returns a list of (K, s) in *increasing* K; s is None where the
    continuation lost the zero (jumped > max_jump, escaped to Im <= 1, or no root --
    e.g. the zeta endpoint K=0 is zeroless, so a zeta trajectory terminates at K=1).
    """
    sched = sorted(schedule if schedule is not None else K_SCHEDULE)
    z = mp.mpc(z_true)
    out = []  # type: List[Tuple[int, Optional[mp.mpc]]]
    for K in reversed(sched):
        try:
            zz = mp.findroot(lambda s: fn(s, K), z)
        except (ValueError, ZeroDivisionError):
            out.append((K, None))
            break
        if abs(zz - z) > max_jump or zz.imag <= 1:
            out.append((K, None))
            break
        z = zz
        out.append((K, zz))
    out.reverse()
    return out


# --------------------------------------------------------------------------
# self-validating driver
# --------------------------------------------------------------------------
def _print_convergence():
    print("== zeta^{(K)} / eta^{(K)} -> zeta / eta  (route a closed form; bose twin) ==")
    print("   s                      K   |zeta_K - z|  |bose - z|  |eta_K - e|  |bose - e|")
    for s in [mp.mpc(2, 0), mp.mpc(0.5, 14.134725), mp.mpc(1.3, 7)]:
        z, e = mp.zeta(s), mp.altzeta(s)
        for K in [5, 40]:                      # bose (quadrature) capped for speed
            ez = float(abs(zeta_K(s, K) - z))
            eb = float(abs(zeta_K_bose(s, K) - z))
            te = float(abs(eta_K(s, K) - e))
            tb = float(abs(eta_K_bose(s, K) - e))
            print(f"   {complex(s)!s:>20}  {K:3d}   {ez:.2e}     {eb:.2e}    "
                  f"{te:.2e}    {tb:.2e}")


def _print_migration(title, fn, targets, schedule=None):
    print(f"\n== {title} ==")
    hdrK = [0, 1, 5, 22, 65, 200]          # all on K_SCHEDULE (0 = eta ground)
    print("   target zero          " + "".join(f"K={k:<5d}" for k in hdrK))
    trajs = []
    for label, zt in targets:
        traj = migrate(fn, zt, schedule=schedule)
        d = {K: zz for K, zz in traj if zz is not None}
        cells = "".join(f"{float(d[k].real):<6.3f}" if k in d else "  --  " for k in hdrK)
        print(f"   {label:<18}   {cells}")
        trajs.append((label, zt, traj))
    return trajs


def _print_counting(T=40.0):
    """Witness the density bijection: #eta-zeros(<T) ~ Riemann-von Mangoldt N(T)."""
    ln2 = float(mp.log(2))
    crit = []                                 # all zeta zeros below T, however many
    while True:
        g = float(mp.zetazero(len(crit) + 1).imag)
        if g >= T:
            break
        crit.append(g)
    pref = [2 * float(PI) * m / ln2 for m in range(1, int(T * ln2 / (2 * float(PI))) + 2)]
    pref = [t for t in pref if t < T]
    rvm = float(ce.counting_law(T))           # (T/2pi)ln(T/pi e) + 5/8
    print(f"\n== density bijection check (0 < t < {T:.0f}) ==")
    print(f"   eta zeros: {len(crit)} on Re=1/2 (zeta) + {len(pref)} on Re=1 (prefactor)"
          f" = {len(crit) + len(pref)}")
    print(f"   Riemann-von Mangoldt law N(T) = {rvm:.2f}   "
          f"(ln(t/2pi e) + ln 2 = ln(t/pi e): prefactor supplies the doubling)")


def _main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figstyle
    figstyle.enlarge()

    _print_convergence()

    zeta_targets = [(f"1/2+{g:.3f}i", mp.mpc(0.5, g))
                    for g in (14.134725, 21.022040, 25.010858)]
    ln2 = float(mp.log(2))
    eta_pref = [(f"1+{2*float(PI)*m/ln2:.3f}i", mp.mpc(1.0, 2 * float(PI) * m / ln2))
                for m in (1, 2, 3)]
    eta_crit = [(f"1/2+{g:.3f}i", mp.mpc(0.5, g))
                for g in (14.134725, 21.022040, 25.010858)]
    sched0 = [0] + K_SCHEDULE               # eta reaches the K=0 ground string

    zt = _print_migration("zeta^{(K)}: zeros born onto Re=1/2", zeta_K, zeta_targets)
    ep = _print_migration("eta^{(K)}: ground string -> Re=1 (prefactor zeros)",
                          eta_K, eta_pref, schedule=sched0)
    ec = _print_migration("eta^{(K)}: ground string -> Re=1/2 (zeta zeros)",
                          eta_K, eta_crit, schedule=sched0)
    _print_counting()

    # ---- figure: three migration panels ----
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    def _panel(a, trajs, title, lines, mark_ground=False):
        for label, _zt, traj in trajs:
            pts = [(K, zz) for K, zz in traj if zz is not None]
            if not pts:
                continue
            res = [float(zz.real) for _, zz in pts]
            ims = [float(zz.imag) for _, zz in pts]
            a.plot(res, ims, "-o", ms=3, lw=1)
            a.plot(res[-1], ims[-1], "k.", ms=9)          # true zero (large-K end)
            if mark_ground and pts[0][0] == 0:
                a.plot(res[0], ims[0], "s", ms=7, mfc="none", mec="0.3")  # K=0 ground
        for xv, lab, col, ls in lines:
            a.axvline(xv, ls=ls, lw=1, color=col, label=lab)
        a.set_xlabel(r"$\sigma=\mathrm{Re}\,s$")
        a.set_ylabel(r"$t=\mathrm{Im}\,s$")
        a.set_title(title)
        a.set_xlim(0.2, 2.1)
        a.legend(loc="upper right")

    s_fzero = ce.sigma_law(25)
    _panel(ax[0], zt, r"$\zeta^{(K)}$: zeros born (K=1) $\to\ \sigma=1/2$",
           [(0.5, r"critical line $1/2$", "C2", "-")])
    _panel(ax[1], ep, r"$\eta^{(K)}$: ground $\to\ \sigma=1$ (prefactor)",
           [(1.0, r"$\sigma=1$ ($1-2^{1-s}$)", "C1", "-"),
            (s_fzero, r"continuous-$\eta$ $F$-zeros", "C3", "--")], mark_ground=True)
    _panel(ax[2], ec, r"$\eta^{(K)}$: ground $\to\ \sigma=1/2$ (zeta)",
           [(0.5, r"critical line $1/2$", "C2", "-"),
            (s_fzero, r"continuous-$\eta$ $F$-zeros", "C3", "--")], mark_ground=True)

    fig.suptitle(r"Zero-migration map: restoring discreteness one Fourier harmonic "
                 r"at a time  ($K=0\to\infty$)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    figdir = Path(__file__).resolve().parent / "figures"
    figdir.mkdir(exist_ok=True)
    out = figdir / "harmonic_bridge.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    _main()
