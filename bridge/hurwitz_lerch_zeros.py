r"""Hurwitz / Lerch zero trajectories -- reproducing Garunkstis-Steuding (2007),
the closest *published* deform-and-track analogue of the discrete <-> continuous bridge.

Where our K-knob truncates harmonics from a *zeroless* endpoint, the published siblings
vary a continuous *parameter* between two zero-rich configurations and watch the zeros move.
This driver reproduces the two closest such studies and shows they are two edges of ONE family:

  * Garunkstis-Steuding (2007), Math. Comp. 76(257), 323-337 -- track the zeros of the
    **Hurwitz zeta** zeta(s, alpha) as the shift alpha sweeps from 1 (=zeta) to 1/2 (the
    half-shift (2^s-1)zeta). They call a zero **"stable" iff its trajectory starts AND ends
    on Re s = 1/2.** We reproduce their computer plots and the stable/unstable dichotomy.
  * Garunkstis-Tamosiunas (2017), Lith. Math. J. 57(4) -- the **Lerch** zeta L(lambda, alpha, s)
    with *equal parameters* lambda = alpha, whose nontrivial zeros stay (almost) symmetric about
    Re s = 1/2. We sweep the diagonal lambda = alpha: 1 -> 1/2 and land on 2^s L(s, chi_4).

The Lerch family square -- three ancestors, one picture
-------------------------------------------------------
L(lambda, alpha, s) = sum_{m>=0} e^{2 pi i lambda m} (m + alpha)^{-s}  (= mpmath.lerchphi(e^{2 pi i lambda}, s, alpha))
contains the bridge's four objects at the corners of the unit (lambda, alpha) square, and the
three deformation paths between them are exactly the three published deform-and-track programs:

    corner (lambda, alpha)      L(lambda, alpha, s)              zeros of the endpoint
    ------------------------     ---------------------            ---------------------
    (1, 1)                       zeta(s)                          sigma = 1/2            (RH)
    (1, 1/2)                     zeta(s, 1/2) = (2^s-1)zeta        sigma = 1/2 U sigma = 0
    (1/2, 1)                     eta(s) = (1-2^{1-s})zeta          sigma = 1/2 U sigma = 1
    (1/2, 1/2)                   2^s L(s, chi_4)                   sigma = 1/2  (single line)

  * the  lambda = 1  edge (alpha: 1 -> 1/2) is the **Garunkstis-Steuding Hurwitz shift**;
    its "unstable" zeros land on the  sigma = 0  companion comb  s = 2 pi i m / log 2  (the same
    comb warp_bridge.py's half-shift target carries).
  * the  alpha = 1  edge (lambda: 1 -> 1/2) is the **Fornberg-Kolbig polylog argument knob**
    (jonquiere_zeros.py) in disguise -- z = e^{2 pi i lambda} sweeps the unit circle from 1 to -1,
    and F(-1, s) = -eta; its companion comb sits on  sigma = 1  (the eta prefactor 1 - 2^{1-s}).
  * the diagonal  lambda = alpha (1 -> 1/2) is **Garunkstis-Tamosiunas**; it lands on the primitive
    L(s, chi_4) (issue #20, lfunction_bridge.py), which is PRIMITIVE, so there is **no companion**
    -- the zeros stay glued to sigma = 1/2 the whole way.

So the same family knits together all three published ancestors and the bridge's own objects, with
the companion line (sigma = 0 / sigma = 1 / none) reading off which edge you took.

Why it sits in this repo (the honest-framing payoff)
----------------------------------------------------
The alpha-sweep is the cleanest *contrast* for our K-knob. G-S deform between two configurations
that BOTH already carry infinitely many zeros -- no zero is *born*; a "stable" zero merely leaves
sigma = 1/2 and returns. Our K-knob instead starts from a *zeroless / structured* endpoint, the
zeros are **born**, and they migrate **onto** sigma = 1/2 and stay (converging at O(1/K), rate_law.py).
The taxonomy (LITERATURE.md sec4.0): the alpha-sweep's arrow points *on and off* 1/2 between two
zero-rich ends; ours points *onto* 1/2 from nothing. Panel (1,2) draws both trajectories together.

What this driver reproduces (all self-validated in __main__)
-----------------------------------------------------------
  * Evaluators: zeta(s, alpha) = mpmath.zeta(s, alpha) and L(lambda, alpha, s) = mpmath.lerchphi(
    e^{2 pi i lambda}, s, alpha), each checked against its direct sum; the z ~ 1 hazard (integer
    lambda rounds e^{2 pi i lam} to 1 + O(1e-33)j, where lerchphi silently returns garbage for
    sigma <= 1) handled by splicing in the Hurwitz zeta (lerchphi(1, s, a) = zeta(s, a)).
  * The four unification identities at the corners of the (lambda, alpha) square (to ~1e-25).
  * The Hurwitz alpha-trajectories rho(alpha), alpha: 1 -> 1/2, seeded by the first Riemann zeros,
    Newton-continued in alpha; the G-S stable/unstable split (ends on sigma = 1/2 vs the sigma = 0
    comb), and the extension below alpha = 1/2 into the Davenport-Heilbronn off-line wandering.
  * The Lerch diagonal lambda = alpha: 1 -> 1/2 gluing zeta's zeros to 2^s L(s, chi_4)'s on sigma = 1/2,
    and the alpha = 1 polylog edge lambda: 1 -> 1/2 carrying them to eta's sigma = 1 comb.
  * The contrast with the bridge's K-knob (harmonic_bridge.zeta_K): zeroless-ground -> born -> onto 1/2.

Run directly to validate + plot: writes bridge/figures/hurwitz_lerch_zeros.png.
"""
import sys
from pathlib import Path
from typing import List, Tuple

import mpmath as mp

# This module sits in a flat source dir alongside its bare cross-imports; put that
# directory on sys.path so a direct ``python hurwitz_lerch_zeros.py`` run resolves them
# (pytest gets the same path from the root conftest.py).
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import harmonic_bridge as hb   # noqa: E402  (zeta_K + migrate -- the K-knob contrast trajectory)

mp.mp.dps = 30

PI = mp.pi
LOG2 = mp.log(2)
TWOPI_OVER_LOG2 = 2 * PI / LOG2          # 9.0647202836...: the sigma=0 (half-shift) AND
#                                          sigma=1 (eta) companion-comb spacing 2 pi m / log 2

# First nontrivial zeros of L(s, chi_4) on sigma=1/2 (heights) -- the Lerch-diagonal targets
# (shared with lfunction_bridge.py, issue #20).
L_CHI4_HEIGHTS = [6.020948, 10.243770, 12.988098, 16.342607, 18.291993]


# --------------------------------------------------------------------------
# 1. evaluators: the Hurwitz zeta and the Lerch transcendent
# --------------------------------------------------------------------------
def zeta_hurwitz(s, alpha):
    """zeta(s, alpha) = sum_{m>=0} (m + alpha)^{-s}  (mpmath.zeta; analytic continuation, pole at s=1)."""
    return mp.zeta(mp.mpc(s), mp.mpf(alpha))


def zeta_hurwitz_sum(s, alpha, terms=50000):
    """zeta(s, alpha) = sum_{m>=0} (m + alpha)^{-s} by direct summation -- the sigma>1 reference."""
    s = mp.mpc(s)
    a = mp.mpf(alpha)
    return mp.fsum((m + a) ** (-s) for m in range(terms))


def lerch(lam, alpha, s):
    """L(lambda, alpha, s) = sum_{m>=0} e^{2 pi i lambda m} (m + alpha)^{-s}  (mpmath.lerchphi).

    z = e^{2 pi i lambda}. GOTCHA: at *exact* z = 1 mpmath special-cases lerchphi(1, s, a) =
    zeta(s, a) correctly -- but e^{2 pi i lam} at integer lam never rounds to exactly 1 (it is
    1 + O(1e-33)j at dps=30), and for z that close to 1 (but != 1) lerchphi silently returns
    wildly wrong values for sigma <= 1, with no exception. So splice in the Hurwitz zeta
    whenever z is numerically 1. The Lerch analog of lfunction_bridge's L(1, chi_4) = pi/4
    splice, but sharper: the failure mode is silent garbage, not an inf.
    """
    lam = mp.mpf(lam)
    alpha = mp.mpf(alpha)
    s = mp.mpc(s)
    z = mp.exp(2j * PI * lam)
    if abs(z - 1) < mp.mpf("1e-18"):
        return mp.zeta(s, alpha)                 # lerchphi(1,s,a) = zeta(s,a); z~1 (!=1) is the garbage regime
    return mp.lerchphi(z, s, alpha)


def lerch_sum(lam, alpha, s, terms=20000):
    """L(lambda, alpha, s) by direct summation -- the reference for lerch (|z|=1 needs sigma>1)."""
    lam = mp.mpf(lam)
    alpha = mp.mpf(alpha)
    s = mp.mpc(s)
    z = mp.exp(2j * PI * lam)
    return mp.fsum(z ** m * (m + alpha) ** (-s) for m in range(terms))


# --------------------------------------------------------------------------
# 2. the four unification targets at the corners of the (lambda, alpha) square
# --------------------------------------------------------------------------
def L_chi4(s):
    """L(s, chi_4) via the entire Hurwitz combination 4^{-s}[zeta(s,1/4) - zeta(s,3/4)].

    The individual Hurwitz zetas carry the pole 1/(s-1), cancelling in the difference (L is entire),
    but mpmath still returns inf at s=1 exactly, so splice the removable L(1, chi_4) = pi/4 (Leibniz).
    (Same evaluator as lfunction_bridge.L_chi4, kept local so this driver is standalone.)
    """
    s = mp.mpc(s)
    if abs(s - 1) < mp.mpf("1e-12"):
        return mp.power(4, -s) * (mp.digamma(mp.mpf(3) / 4) - mp.digamma(mp.mpf(1) / 4))
    return mp.power(4, -s) * (mp.zeta(s, mp.mpf(1) / 4) - mp.zeta(s, mp.mpf(3) / 4))


# (lambda, alpha, name, endpoint evaluator) at the corners; the target the corresponding
# deformation path lands on. The evaluators are INDEPENDENT of `lerch` (zeta / half-shift /
# eta / chi_4), so matching them validates the Lerch corner values.
UNIFICATION = [
    (mp.mpf(1),          mp.mpf(1),          "zeta(s)",            lambda s: mp.zeta(s)),
    (mp.mpf(1),          mp.mpf(1) / 2,      "(2^s-1) zeta(s)",    lambda s: (2 ** mp.mpc(s) - 1) * mp.zeta(s)),
    (mp.mpf(1) / 2,      mp.mpf(1),          "eta(s)",             lambda s: (1 - 2 ** (1 - mp.mpc(s))) * mp.zeta(s)),
    (mp.mpf(1) / 2,      mp.mpf(1) / 2,      "2^s L(s, chi_4)",    lambda s: 2 ** mp.mpc(s) * L_chi4(s)),
]


# --------------------------------------------------------------------------
# 3. zero-trajectory tracing (Newton continuation in a parameter)
# --------------------------------------------------------------------------
# Roots good to this tolerance are ample for tracing/plotting; mpmath's default (eps*2^10
# ~ 2e-28 at dps=30) is tighter than the secant reaches on these flat trajectories (cf
# jonquiere_zeros).
_TRACE_TOL = mp.mpf("1e-16")


def trace_zero(make_fn, params, s0, tol=_TRACE_TOL, max_jump=3.0):
    """Newton-continue a zero across a parameter list. `make_fn(p)` returns the s-function F_p.

    Seeded at s0 for params[0], each subsequent solve reuses the previous root. Returns
    [(p, s), ...]; stops (returning what it has) if a step fails to converge or the root jumps
    by more than `max_jump` (a branch hop -- gotcha C.3 / findroot raising on flat trajectories).
    """
    s = mp.mpc(s0)
    out = []  # type: List[Tuple[object, mp.mpc]]
    for p in params:
        try:
            snew = mp.findroot(make_fn(p), s, tol=tol)
        except (ValueError, ZeroDivisionError):
            break
        if abs(snew - s) > max_jump:
            break
        s = snew
        out.append((p, s))
    return out


def _linspace(a, b, n):
    """n params from a to b inclusive (mpf), dense enough that continuation never hops branches."""
    return [mp.mpf(a) + (mp.mpf(b) - mp.mpf(a)) * mp.mpf(i) / (n - 1) for i in range(n)]


def track_hurwitz(gamma, a_hi=1.0, a_lo=0.5, n=26):
    """rho(alpha) for the zeta(s, alpha) zero seeded at 1/2 + i*gamma (a Riemann zero at alpha=1).

    Newton-continued as alpha runs a_hi -> a_lo. Returns [(alpha, s), ...]. Extending below
    alpha = 1/2 exhibits the Davenport-Heilbronn off-line wandering.
    """
    return trace_zero(lambda a: (lambda s: mp.zeta(s, a)), _linspace(a_hi, a_lo, n),
                      mp.mpc(0.5, gamma))


def track_lerch_path(params, gamma):
    """rho along a (lambda, alpha) path, seeded at 1/2 + i*gamma. `params` is a list of (lam, alpha)."""
    return trace_zero(lambda p: (lambda s: lerch(p[0], p[1], s)), params, mp.mpc(0.5, gamma))


def classify(traj, tol=0.06):
    """G-S label for a Hurwitz trajectory by where it ENDS: 'stable' (Re -> 1/2), 'sigma=0' comb, or 'off'."""
    if not traj:
        return "empty"
    re = float(traj[-1][1].real)
    if abs(re - 0.5) < tol:
        return "stable"
    if abs(re) < tol:
        return "sigma=0"
    return "off"


def _nearest_target(t, heights, tol=0.25):
    """The entry of `heights` nearest to t (for labelling which comb/zeta zero a trajectory lands on)."""
    best = min(heights, key=lambda h: abs(h - t))
    return best if abs(best - t) < tol else None


# --------------------------------------------------------------------------
# 4. the K-knob contrast trajectory (harmonic_bridge.zeta_K): zeroless -> born -> onto 1/2
# --------------------------------------------------------------------------
_K_CONTRAST = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]


def kknob_trajectory(gamma):
    """The bridge's OWN motion for the Riemann zero 1/2 + i*gamma: harmonic_bridge.zeta_K's zero
    as K grows. K=0 is zeroless (the trajectory begins at K=1, off the line) and climbs ONTO
    sigma=1/2 as K -> infinity -- the contrast to the alpha-sweep. Returns [(K, s), ...]."""
    traj = hb.migrate(hb.zeta_K, mp.mpc(0.5, gamma), schedule=_K_CONTRAST)
    return [(K, zz) for K, zz in traj if zz is not None]


# --------------------------------------------------------------------------
# self-validating driver
# --------------------------------------------------------------------------
def _print_evaluators():
    print("== evaluators: Hurwitz zeta(s, alpha) and Lerch L(lambda, alpha, s) ==")
    worst_h = 0.0
    for s, a in [(mp.mpc(3, 1), 0.3), (mp.mpc(4, 0), 0.7), (mp.mpc(3, 2), 0.25)]:
        worst_h = max(worst_h, float(abs(zeta_hurwitz(s, a) - zeta_hurwitz_sum(s, a))))
    print(f"   zeta(s, alpha) vs direct sum (sigma>1):  worst |diff| = {worst_h:.2e}")
    worst_l = 0.0
    for lam, a in [(0.5, 1.0), (0.5, 0.5), (0.25, 0.7), (0.75, 0.4)]:
        for s in [mp.mpc(3, 1), mp.mpc(4, 0.5)]:            # |z|=1 sum converges for sigma>1
            worst_l = max(worst_l, float(abs(lerch(lam, a, s) - lerch_sum(lam, a, s))))
    print(f"   L(lambda, alpha, s) vs direct sum (sigma>1):  worst |diff| = {worst_l:.2e}")
    # the z ~ 1 (integer lambda) silent-garbage regime handled by the Hurwitz splice
    s = mp.mpc(0.7, 12.0)
    fb = float(abs(lerch(1, 0.5, s) - mp.zeta(s, mp.mpf(1) / 2)))
    print(f"   z~1 splice  lerch(1, 1/2, s) == zeta(s, 1/2):  |diff| = {fb:.2e}  "
          f"(lerchphi near-but-not-at z=1 silently corrupts)")


def _print_unification():
    print("\n== the four unification identities at the (lambda, alpha) corners ==")
    print("   (lambda, alpha)   L(lambda, alpha, s)      |Lerch - independent evaluator|")
    worst = 0.0
    for s in [mp.mpc(0.7, 12.0), mp.mpc(2, 1)]:
        for lam, a, name, tgt in UNIFICATION:
            err = float(abs(lerch(lam, a, s) - tgt(s)))
            worst = max(worst, err)
            if s.imag == 12.0:
                print(f"   ({float(lam):g}, {float(a):g})           {name:<20} {err:.2e}")
    print(f"   worst over the corners (two heights) = {worst:.2e}")


def _print_hurwitz(trajs):
    print("\n== Hurwitz alpha-trajectories  rho(alpha),  alpha: 1 -> 1/2  (Garunkstis-Steuding) ==")
    print("   a zero is STABLE iff its trajectory starts AND ends on Re s = 1/2 (G-S definition):")
    print("   seed (zeta zero)     rho(1/2) = (sigma, t)      lands on           class")
    comb0 = [float(TWOPI_OVER_LOG2 * m) for m in range(1, 8)]        # sigma=0 comb heights
    zzeros = [float(mp.zetazero(n).imag) for n in range(1, 12)]      # sigma=1/2 zeta heights
    n_stable = n_comb = 0
    for n, traj in trajs:
        end = traj[-1][1]
        cls = classify(traj)
        te = float(end.imag)
        if cls == "stable":
            n_stable += 1
            tgt = _nearest_target(te, zzeros)
            land = f"zeta zero {te:7.4f}i" if tgt else f"~{te:.3f}i"
        elif cls == "sigma=0":
            n_comb += 1
            m = _nearest_target(te, comb0)
            land = f"sigma=0 comb {te:6.3f}i" if m else f"~{te:.3f}i"
        else:
            land = f"off-line {float(end.real):+.3f}"
        print(f"   1/2 + {float(mp.zetazero(n).imag):8.4f}i    "
              f"({float(end.real):+.4f}, {te:8.4f})    {land:<20} {cls}")
    print(f"   -> {n_stable} stable (return to sigma=1/2), {n_comb} unstable (land on the sigma=0 comb")
    print(f"      s = 2 pi i m / log 2 = {float(TWOPI_OVER_LOG2):.4f} m i -- the half-shift's companion).")


def _print_below_half(wander):
    print("\n== below alpha = 1/2: Davenport-Heilbronn off-line wandering (zeros leave 1/2) ==")
    for n, traj in wander:
        devs = [abs(float(s.real) - 0.5) for _a, s in traj]
        a_at_max = traj[devs.index(max(devs))][0]
        print(f"   seed 1/2+{float(mp.zetazero(n).imag):.3f}i:  max |Re-1/2| = {max(devs):.3f} "
              f"at alpha = {float(a_at_max):.3f}   (generic alpha -> zeros scattered, cf warp_alpha.py)")


def _print_lerch_paths(diag, poly):
    print("\n== the Lerch family square: three ancestor-deformations, three companion lines ==")
    print("   diagonal  lambda = alpha: 1 -> 1/2   (Garunkstis-Tamosiunas: zeta -> 2^s L(s, chi_4)):")
    for g, traj in diag:
        end = traj[-1][1]
        tgt = _nearest_target(float(end.imag), L_CHI4_HEIGHTS, tol=0.3)
        land = f"L(chi_4) zero {float(end.imag):.4f}i" if tgt else f"{float(end.imag):.4f}i"
        print(f"     seed 1/2+{g:7.4f}i -> ({float(end.real):+.4f}, {float(end.imag):7.4f})  "
              f"{land}   (stays on sigma=1/2; primitive L, NO companion)")
    print("   edge  alpha = 1, lambda: 1 -> 1/2   (Fornberg-Kolbig polylog knob: zeta -> eta):")
    comb1 = [float(TWOPI_OVER_LOG2 * m) for m in range(1, 8)]
    for g, traj in poly:
        end = traj[-1][1]
        m = _nearest_target(float(end.imag), comb1, tol=0.3)
        land = f"sigma=1 comb {float(end.imag):.3f}i" if m and abs(float(end.real) - 1) < 0.1 \
            else f"({float(end.real):+.3f}, {float(end.imag):.3f})"
        print(f"     seed 1/2+{g:7.4f}i -> ({float(end.real):+.4f}, {float(end.imag):7.4f})  "
              f"{land}   (eta's companion comb sits on sigma=1)")


def _print_contrast(alpha_traj, k_traj):
    print("\n== the contrast with the bridge's K-knob (the taxonomy, LITERATURE.md sec4.0) ==")
    a_end = [float(alpha_traj[0][1].real), float(alpha_traj[-1][1].real)]
    print(f"   alpha-sweep (Hurwitz): Re(rho) starts {a_end[0]:.3f} (alpha=1) ends {a_end[1]:.3f} "
          f"(alpha=1/2) -- both zero-rich ends ON 1/2, wanders OFF between (stable zero).")
    print(f"   K-knob (zeta_K):       Re(rho) starts {float(k_traj[0][1].real):.3f} (K={k_traj[0][0]}, "
          f"off-line ground) ends {float(k_traj[-1][1].real):.3f} (K={k_traj[-1][0]}) -- born from a")
    print("   zeroless endpoint (K=0), climbs ONTO 1/2 and stays (O(1/K)).  Arrow: onto vs on/off.")


def _traj_xy(traj):
    return ([float(s.real) for _p, s in traj], [float(s.imag) for _p, s in traj])


def _main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figstyle
    figstyle.enlarge()
    # dense 6-panel figure with long titles: trim title/legend sizes locally (still >> default).
    plt.rcParams.update({"axes.titlesize": 12.5, "legend.fontsize": 10, "figure.titlesize": 16})

    _print_evaluators()
    _print_unification()

    # ---- the Hurwitz alpha-trajectories (lambda=1 edge), first 8 Riemann zeros ----
    n_track = 8
    hur = [(n, track_hurwitz(float(mp.zetazero(n).imag))) for n in range(1, n_track + 1)]
    _print_hurwitz(hur)

    # ---- extend two stable ones below alpha=1/2 into the DH wandering ----
    wander = [(n, track_hurwitz(float(mp.zetazero(n).imag), a_hi=1.0, a_lo=0.2, n=41))
              for n in (2, 4)]
    _print_below_half(wander)

    # ---- the Lerch diagonal (lambda=alpha) and the alpha=1 polylog edge ----
    diag_params = [(v, v) for v in _linspace(1.0, 0.5, 26)]
    poly_params = [(v, mp.mpf(1)) for v in _linspace(1.0, 0.5, 26)]
    diag = [(float(mp.zetazero(n).imag), track_lerch_path(diag_params, float(mp.zetazero(n).imag)))
            for n in (1, 2, 3)]
    poly = [(float(mp.zetazero(n).imag), track_lerch_path(poly_params, float(mp.zetazero(n).imag)))
            for n in (1, 2, 3)]
    _print_lerch_paths(diag, poly)

    # ---- the K-knob contrast trajectory ----
    alpha_contrast = next(t for n, t in hur if n == 4)          # a stable one (bulges then returns)
    k_contrast = kknob_trajectory(float(mp.zetazero(3).imag))
    _print_contrast(alpha_contrast, k_contrast)

    # ============================== figure ==============================
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10))

    # (0,0) Hurwitz alpha-trajectories in the s-plane (the G-S headline)
    ax = axes[0, 0]
    for n, traj in hur:
        xs, ys = _traj_xy(traj)
        cls = classify(traj)
        col = "C2" if cls == "stable" else ("C3" if cls == "sigma=0" else "C1")
        ax.plot(xs, ys, "-", color=col, lw=1.3)
        ax.plot(xs[0], ys[0], "o", color="k", ms=5, zorder=4)          # alpha=1 seed (zeta zero)
        ax.plot(xs[-1], ys[-1], "s" if cls == "sigma=0" else "^", color=col, ms=7, zorder=4)
    ax.axvline(0.5, ls=":", color="0.4", lw=1.2)
    ax.axvline(0.0, ls="--", color="C3", lw=1)
    for m in range(1, 4):
        ax.plot(0.0, float(TWOPI_OVER_LOG2 * m), "x", color="C3", ms=8)
    ax.plot([], [], "-", color="C2", label="stable (ends on $\\frac{1}{2}$)")
    ax.plot([], [], "-", color="C3", label=r"unstable $\to\ \sigma=0$ comb")
    ax.plot([], [], "o", color="k", ms=5, label=r"$\alpha=1$ seed ($\zeta$ zero)")
    ax.set_xlim(-0.35, 1.15)
    ax.set_ylim(7, 34)
    ax.set_xlabel(r"$\sigma=\mathrm{Re}\,s$")
    ax.set_ylabel(r"$t=\mathrm{Im}\,s$")
    ax.set_title(r"Hurwitz $\zeta(s,\alpha)$ zero paths, $\alpha:1\to\frac{1}{2}$"
                 "\n(Garunkstis-Steuding 2007)")
    ax.legend(loc="upper right")

    # (0,1) the "stable zero" diagnostic: Re(rho) vs alpha
    ax = axes[0, 1]
    for n, traj in hur:
        alphas = [float(p) for p, _s in traj]
        res = [float(s.real) for _p, s in traj]
        cls = classify(traj)
        col = "C2" if cls == "stable" else ("C3" if cls == "sigma=0" else "C1")
        ax.plot(alphas, res, "-", color=col, lw=1.3)
    ax.axhline(0.5, ls=":", color="0.4", lw=1.2, label=r"critical line $\sigma=\frac{1}{2}$")
    ax.axhline(0.0, ls="--", color="C3", lw=1, label=r"$\sigma=0$ (companion)")
    ax.axvline(0.5, ls="-", color="0.8", lw=1)
    ax.axvline(1.0, ls="-", color="0.8", lw=1)
    ax.text(1.0, 0.60, r"$\zeta$", ha="center", fontsize=13)
    ax.text(0.5, 0.60, r"$(2^s\!-\!1)\zeta$", ha="center", fontsize=11)
    ax.set_xlabel(r"shift $\alpha$")
    ax.set_ylabel(r"$\mathrm{Re}\,\rho(\alpha)$")
    ax.set_title(r"the 'stable zero' test: $\mathrm{Re}\,\rho$ vs $\alpha$"
                 "\n" r"(stable $\Leftrightarrow$ both ends on $\frac{1}{2}$)")
    ax.set_xlim(0.48, 1.02)
    ax.legend(loc="lower left")

    # (0,2) below alpha=1/2: Davenport-Heilbronn off-line wandering
    ax = axes[0, 2]
    for n, traj in wander:
        alphas = [float(p) for p, _s in traj]
        res = [float(s.real) for _p, s in traj]
        ax.plot(alphas, res, "-o", ms=2.5, lw=1.2, label=rf"seed $\frac{{1}}{{2}}+{float(mp.zetazero(n).imag):.1f}i$")
    ax.axhline(0.5, ls=":", color="0.4", lw=1.2)
    ax.axhline(1.0, ls="--", color="0.7", lw=1)
    ax.axvspan(0.2, 0.5, color="C3", alpha=0.06)
    ax.axvline(0.5, ls="-", color="0.8", lw=1)
    ax.text(0.35, 0.80, "generic $\\alpha$:\noff-line\n(Davenport-\nHeilbronn)", ha="center",
            va="top", fontsize=9.5, color="C3")
    ax.text(0.5, -0.28, r"$\frac{1}{2}$", ha="center", fontsize=12)
    ax.set_xlabel(r"shift $\alpha$")
    ax.set_ylabel(r"$\mathrm{Re}\,\rho(\alpha)$")
    ax.set_title(r"extend below $\frac{1}{2}$: zeros wander off"
                 "\n(the generic-$\\alpha$ scatter)")
    ax.set_xlim(0.18, 1.02)
    ax.legend(loc="upper right")

    # (1,0) the (lambda, alpha) family square: four corners, three ancestor paths
    ax = axes[1, 0]
    ax.set_xlim(0.35, 1.15)
    ax.set_ylim(0.35, 1.15)
    # the three paths
    ax.annotate("", xy=(1.0, 0.5), xytext=(1.0, 1.0),
                arrowprops=dict(arrowstyle="->", color="C3", lw=2))
    ax.annotate("", xy=(0.5, 1.0), xytext=(1.0, 1.0),
                arrowprops=dict(arrowstyle="->", color="C1", lw=2))
    ax.annotate("", xy=(0.5, 0.5), xytext=(1.0, 1.0),
                arrowprops=dict(arrowstyle="->", color="C2", lw=2))
    corners = {(1.0, 1.0): r"$\zeta(s)$", (1.0, 0.5): r"$(2^s\!-\!1)\zeta$",
               (0.5, 1.0): r"$\eta(s)$", (0.5, 0.5): r"$2^sL(s,\chi_4)$"}
    for (lm, al), lab in corners.items():
        ax.plot(lm, al, "o", color="k", ms=9, zorder=5)
        ax.annotate(lab, (lm, al), textcoords="offset points",
                    xytext=(0, 14 if al > 0.7 else -20), ha="center", fontsize=12)
    ax.text(1.02, 0.75, "G-S\n$\\sigma\\!=\\!0$", color="C3", fontsize=9.5, va="center")
    ax.text(0.75, 1.03, "F-K  $\\sigma\\!=\\!1$", color="C1", fontsize=9.5, ha="center")
    ax.text(0.70, 0.66, "G-T\nno comp.", color="C2", fontsize=9.5, ha="center")
    ax.set_xlabel(r"$\lambda$  (phase $e^{2\pi i\lambda}$)")
    ax.set_ylabel(r"$\alpha$  (shift)")
    ax.set_title(r"the Lerch square $L(\lambda,\alpha,s)$:"
                 "\nthree ancestors, one family")
    ax.set_xticks([0.5, 1.0])
    ax.set_yticks([0.5, 1.0])

    # (1,1) Lerch diagonal + polylog edge trajectories in the s-plane
    ax = axes[1, 1]
    for g, traj in diag:
        xs, ys = _traj_xy(traj)
        ax.plot(xs, ys, "-", color="C2", lw=1.4)
        ax.plot(xs[-1], ys[-1], "^", color="C2", ms=8, zorder=4)
    for g, traj in poly:
        xs, ys = _traj_xy(traj)
        ax.plot(xs, ys, "-", color="C1", lw=1.4)
        ax.plot(xs[-1], ys[-1], "s", color="C1", ms=7, zorder=4)
    for n in (1, 2, 3):
        ax.plot(0.5, float(mp.zetazero(n).imag), "o", color="k", ms=5, zorder=5)
    ax.axvline(0.5, ls=":", color="0.4", lw=1.2)
    ax.axvline(1.0, ls="--", color="C1", lw=1)
    ax.plot([], [], "-", color="C2", label=r"diagonal $\to 2^sL(\chi_4)$ (on $\frac{1}{2}$)")
    ax.plot([], [], "-", color="C1", label=r"$\alpha\!=\!1$ edge $\to\eta$ ($\sigma=1$ comb)")
    ax.plot([], [], "o", color="k", ms=5, label=r"$\zeta$ zeros (seeds)")
    ax.set_xlim(0.2, 1.15)
    ax.set_ylim(4, 27)
    ax.set_xlabel(r"$\sigma=\mathrm{Re}\,s$")
    ax.set_ylabel(r"$t=\mathrm{Im}\,s$")
    ax.set_title(r"Lerch paths: diagonal glues $\zeta\!\to\!L(\chi_4)$"
                 "\non $\\frac{1}{2}$; polylog edge $\\to\\sigma=1$")
    ax.legend(loc="upper right")

    # (1,2) the contrast with the K-knob (the taxonomy)
    ax = axes[1, 2]
    a_p = [i / (len(alpha_contrast) - 1) for i in range(len(alpha_contrast))]
    a_re = [float(s.real) for _p, s in alpha_contrast]
    ax.plot(a_p, a_re, "-o", ms=3, color="C0",
            label=r"$\alpha$-sweep: $\zeta\!\leftrightarrow\!$half-shift")
    k_p = [i / (len(k_contrast) - 1) for i in range(len(k_contrast))]
    k_re = [float(s.real) for _K, s in k_contrast]
    ax.plot(k_p, k_re, "-s", ms=4, color="C3", mfc="none",
            label=r"$K$-knob (this work): born $\to$ onto $\frac{1}{2}$")
    ax.axhline(0.5, ls=":", color="0.4", lw=1.2)
    ax.annotate("both ends\non $\\frac{1}{2}$", (0.0, a_re[0]), textcoords="offset points",
                xytext=(6, 10), fontsize=9.5, color="C0")
    ax.annotate("off-line\nground", (0.0, k_re[0]), textcoords="offset points",
                xytext=(6, -2), fontsize=9.5, color="C3")
    ax.set_xlabel("deformation progress  ($\\alpha:1\\to\\frac{1}{2}$  |  $K:1\\to\\infty$)")
    ax.set_ylabel(r"$\mathrm{Re}\,\rho$")
    ax.set_title(r"taxonomy: wander $\it{on/off}$ $\frac{1}{2}$ (sweep)"
                 "\nvs born $\\it{onto}$ $\\frac{1}{2}$ (K-knob)")
    ax.legend(loc="center right")

    fig.suptitle(r"Hurwitz / Lerch zero trajectories (Garunkstis-Steuding 2007): the published "
                 r"$\it{parameter}$-sweep sibling of the $K$-knob")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    figdir = Path(__file__).resolve().parent / "figures"
    figdir.mkdir(exist_ok=True)
    out = figdir / "hurwitz_lerch_zeros.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    _main()
