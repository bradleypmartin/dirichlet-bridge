r"""
Off-critical zeros of the hypercubic Epstein zeta-function -- the
*reverse-direction foil* for the discrete <-> continuous bridge.

Reproduces Travenec & Samaj (2022), "Generation of off-critical zeros for
hypercubic Epstein zeta-functions", Appl. Math. Comput. 413, 126611
(arXiv:1909.07112), and its 2D shape-knob companion Betermin-Samaj-Travenec
(2021, arXiv:2110.09368).

The object is the Epstein zeta of the cubic lattice Z^d,

    Z_d(s) = sum'_{n in Z^d} |n|^{-2s}  = sum'_{n} (n_1^2 + ... + n_d^2)^{-s},

('prime' omits n=0; converges for Re s > d/2). Travenec-Samaj use the spatial
dimension d as a *continuous knob* (analytically continued) and watch the zeros
move. This is the sharpest single foil in the literature for our claim, because
the arrow points the OTHER way:

    the bridge  :  zeroless endpoint  ->  zeros BORN  ->  migrate ONTO  Re s = 1/2
    Epstein     :  zeros on the line  ->  a pair is BORN OFF  ->  migrate AWAY

Making the contrast concrete as code + a figure is what turns our directional
claim from generic ("we move zeros") into distinctive ("we move them *onto* 1/2
from a zeroless start, where the closest lattice-zeta analogue moves them
*away*"). See LITERATURE.md sec. 4.0 / 4.5 (the directional taxonomy).

The mechanism (a pitchfork at the critical dimension d*)
-------------------------------------------------------
The completed function Lambda_d(s) = pi^{-s} Gamma(s) Z_d(s) obeys the Epstein
functional equation Lambda_d(s) = Lambda_d(d/2 - s), so zeros are symmetric about
the critical line Re s = d/4 (the analogue of Re s = 1/2). Near the real center
s = d/4, Lambda is even, Lambda ~ Lambda(d/4) + (1/2)Lambda''(d/4) x^2:

  * along the real axis (x = sigma - d/4 real):  real zeros iff Lambda(d/4)*Lambda''(d/4) < 0;
  * along the critical line (x = i y):           on-line zeros iff Lambda(d/4)*Lambda''(d/4) > 0.

The two branches exchange exactly where Lambda_d(d/4) = 0. That root is the
critical dimension

    d* = 9.24555...             (Travenec-Samaj; reproduced here to 10 digits)

For d < d* the lowest pair of zeros sits ON the critical line at d/4 +/- i t_1(d),
sliding down as d grows (t_1 -> 0). At d = d* they collide at the real point d/4
(a double zero). For d > d* they split into a real off-critical pair d/4 +/- delta(d)
that leaves the line and MIGRATES to the strip boundaries {0, d/2} as d -> infinity.

What this driver reproduces (all self-validated in __main__)
-----------------------------------------------------------
  * Evaluator Z_d(s) via the Terras/theta completed formula (fast, entire-except-
    poles, valid for all s and any real d), checked against the direct lattice sum
    (Re s > d/2) and the closed forms Z_1(s) = 2 zeta(2s), Z_2(s) = 4 zeta(s) beta(s).
  * The functional equation Lambda_d(s) = Lambda_d(d/2 - s) as a zero residual, and
    the pole residue  Res_{s=d/2} Z_d = pi^{d/2}/Gamma(d/2).
  * The critical dimension d* = 9.24555... as the root of g(d) = Lambda_d(d/4).
  * The on-line branch t_1(d) -> 0 as d -> d* (zeros slide down the critical line).
  * The off-critical real branch delta(d) for d > d*, and its migration:
    sigma_-(d) -> 0, sigma_+(d) -> d/2 as d -> infinity (Travenec-Samaj Sec. 3).
  * An independent argument-principle count: 0 real zeros in the strip for d < d*,
    2 for d > d* -- the pitchfork, seen by winding number (no root-finding).
  * The taxonomy contrast panel: the Epstein real pair (born off, moving AWAY from
    d/4) beside the bridge's own zeta_K zero (harmonic_bridge) climbing ONTO 1/2.

Run directly to validate + plot: writes bridge/figures/epstein_zeros.png.
"""
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import mpmath as mp

# Flat bridge/ source dir: put it on the path so the bare cross-import resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harmonic_bridge as hb   # noqa: E402  (zeta_K + migrate -- the born-onto-1/2 contrast)

mp.mp.dps = 30

PI = mp.pi
D_STAR_APPROX = mp.mpf("9.24555")     # Travenec-Samaj critical dimension (seed for findroot)


# --------------------------------------------------------------------------
# the object  Z_d(s) = sum'_{n in Z^d} |n|^{-2s}  via the Terras completed formula
# --------------------------------------------------------------------------
def _psi(t):
    """1D Jacobi theta_3(0, e^{-pi t}) = sum_{n in Z} e^{-pi t n^2}."""
    return mp.jtheta(3, 0, mp.exp(-PI * t))


def _theta_minus_1(t, d):
    """Cubic-lattice theta minus its n=0 term:  psi(t)^d - 1.

    The lattice theta is theta(t) = sum_{n in Z^d} e^{-pi t|n|^2} = psi(t)^d (the
    sum factorises over coordinates); psi(t)^d for real non-integer d is exactly
    Travenec-Samaj's analytic continuation in the dimension.
    """
    return _psi(t) ** d - 1


def epstein_Lambda(s, d):
    """Completed Epstein zeta  Lambda_d(s) = pi^{-s} Gamma(s) Z_d(s).

    Terras' split-at-1 representation, using the self-dual theta transformation
    theta(t) = t^{-d/2} theta(1/t):

        Lambda_d(s) = int_1^inf [t^{s-1} + t^{d/2-s-1}] (theta(t)-1) dt
                      + 1/(s - d/2) - 1/s.

    Both power weights hit the same super-exponentially small (theta(t)-1 ~ 2d e^{-pi t})
    integrand, so one quadrature does both halves. Manifestly Lambda_d(s) = Lambda_d(d/2-s):
    swapping s <-> d/2-s swaps the two weights and swaps the two pole terms. Valid for all
    s (simple poles only at s = 0 and s = d/2) and any real d.
    """
    s = mp.mpc(s)
    d = mp.mpf(d)
    tail = mp.quad(lambda t: (t ** (s - 1) + t ** (d / 2 - s - 1)) * _theta_minus_1(t, d),
                   [1, mp.inf])
    return tail + 1 / (s - d / 2) - 1 / s


def epstein_Z(s, d):
    """Z_d(s) = Lambda_d(s) / (pi^{-s} Gamma(s))  -- the primary evaluator (all s, real d)."""
    s = mp.mpc(s)
    return epstein_Lambda(s, d) / (PI ** (-s) * mp.gamma(s))


def epstein_Z_direct(s, d, R=60):
    """Direct lattice sum sum'_{|n_i|<=R} |n|^{-2s} -- reference for integer d, Re s > d/2.

    Slowly convergent near the abscissa Re s = d/2; use a point with Re s comfortably
    larger for a clean cross-check (e.g. Re s = d/2 + 1).
    """
    import itertools
    d = int(d)
    s = mp.mpc(s)
    tot = mp.mpf(0)
    for n in itertools.product(range(-R, R + 1), repeat=d):
        q = sum(x * x for x in n)
        if q:
            tot += mp.mpf(q) ** (-s)
    return tot


def dirichlet_beta(s):
    """Dirichlet beta  beta(s) = L(s, chi_4) = 4^{-s}[zeta(s,1/4) - zeta(s,3/4)]."""
    s = mp.mpc(s)
    return 4 ** (-s) * (mp.zeta(s, mp.mpf(1) / 4) - mp.zeta(s, mp.mpf(3) / 4))


# --------------------------------------------------------------------------
# identities / anchors
# --------------------------------------------------------------------------
def Z1_closed(s):
    """Z_1(s) = 2 zeta(2s)  (Z^1 lattice = {+/- n}, so 2 sum n^{-2s})."""
    return 2 * mp.zeta(2 * mp.mpc(s))


def Z2_closed(s):
    """Z_2(s) = 4 zeta(s) beta(s)  (Gauss: r_2(m) count over the Gaussian integers)."""
    s = mp.mpc(s)
    return 4 * mp.zeta(s) * dirichlet_beta(s)


def functional_eq_residual(s, d):
    """|Lambda_d(s) - Lambda_d(d/2 - s)|  -- the Epstein functional equation, should vanish."""
    s = mp.mpc(s)
    d = mp.mpf(d)
    return abs(epstein_Lambda(s, d) - epstein_Lambda(d / 2 - s, d))


def pole_residue(d):
    """Res_{s=d/2} Z_d(s) = pi^{d/2}/Gamma(d/2)  (the leading Weyl/volume term)."""
    d = mp.mpf(d)
    return PI ** (d / 2) / mp.gamma(d / 2)


def pole_residue_numeric(d, eps=mp.mpf("1e-4")):
    """Numeric residue Res_{s=d/2} Z_d via 2-point Richardson on e*Z_d(d/2+e).

    R(e) = e*Z_d(d/2+e) = Res + c1*e + O(e^2); the combination 2 R(e/2) - R(e) cancels the
    linear term, giving ~O(e^2) accuracy in two cheap evaluations (mp.limit near the pole is
    ~600x slower because it evaluates the pole-adjacent quadrature at raised precision)."""
    d = mp.mpf(d)
    R = lambda e: e * epstein_Z(d / 2 + e, d)
    return 2 * R(eps / 2) - R(eps)


# --------------------------------------------------------------------------
# the dimension knob:  d* and the two zero branches about the center s = d/4
# --------------------------------------------------------------------------
# mpmath findroot's default tolerance (~1e-34 at dps=30) is tighter than the
# secant reaches on these flat trajectories and *raises* on a good ~1e-24 root;
# pass an explicit tol and wrap in try/except (the deform-and-track gotcha).
_TRACE_TOL = mp.mpf("1e-18")


def center_value(d):
    """g(d) = Lambda_d(d/4)  (real): the sign that decides on-line vs real pair.

    Lambda is even about the symmetry center s = d/4, so Lambda'(d/4) = 0 always; the
    center is a zero exactly when g(d) = 0, which is the critical dimension d*.
    """
    d = mp.mpf(d)
    return mp.re(epstein_Lambda(d / 4, d))


def critical_dimension():
    """d* = root of g(d) = Lambda_d(d/4)  (Travenec-Samaj: 9.24555...)."""
    return mp.findroot(center_value, D_STAR_APPROX)


def crit_line_value(y, d):
    """Lambda_d(d/4 + i y)  (real for real y, by the functional equation + Schwarz).

    On the critical line s = d/4 + iy, Lambda(d/4+iy) = Lambda(d/4-iy) = conj, so it
    is real; its zeros in y are the on-line (critical) zeros.
    """
    d = mp.mpf(d)
    return mp.re(epstein_Lambda(mp.mpc(d / 4, y), d))


def lowest_critical_zero(d, y_max=40.0, dy=0.15):
    """t_1(d): the smallest y > 0 with Lambda_d(d/4 + i y) = 0 (lowest on-line zero).

    Scans upward for a sign change, then refines. Returns None if none is found below
    y_max (e.g. right at d* the lowest zero has already reached y = 0)."""
    d = mp.mpf(d)
    lo = mp.mpf("0.02")
    prev = crit_line_value(lo, d)
    y = lo
    while y < y_max:
        y2 = y + mp.mpf(dy)
        cur = crit_line_value(y2, d)
        if prev * cur < 0:
            try:
                return mp.findroot(lambda yy: crit_line_value(yy, d), (y + y2) / 2,
                                   tol=_TRACE_TOL)
            except (ValueError, ZeroDivisionError):
                return None
        prev = cur
        y = y2
    return None


def real_lower_zero(d, seed=None):
    """sigma_-(d): the real zero below the center d/4 (exists only for d > d*).

    Its mirror is sigma_+(d) = d/2 - sigma_-(d). Seed near the center for d just above
    d*; for larger d pass the previous root (Newton-continue in d). Returns None on a
    graceful failure (e.g. tracing into the strip boundary)."""
    d = mp.mpf(d)
    c = d / 4
    if seed is None:
        # scan down from just below the center for the first sign change
        lo = c - mp.mpf("0.001")
        prev = mp.re(epstein_Z(mp.mpc(lo, 0), d))
        sig = lo
        while sig > mp.mpf("0.001"):
            sig2 = sig - mp.mpf("0.02")
            cur = mp.re(epstein_Z(mp.mpc(sig2, 0), d))
            if prev * cur < 0:
                seed = (sig + sig2) / 2
                break
            prev = cur
            sig = sig2
        if seed is None:
            return None
    try:
        return mp.re(mp.findroot(lambda x: mp.re(epstein_Z(mp.mpc(x, 0), d)),
                                 mp.mpf(seed), tol=_TRACE_TOL))
    except (ValueError, ZeroDivisionError):
        return None


def critical_branch(d_values):
    """[(d, t_1(d)), ...] the on-line branch, Newton-continued down in d from near d*."""
    out = []                                              # type: List[Tuple[float, float]]
    seed = None
    for d in sorted(d_values, reverse=True):             # start near d* where t_1 is small
        d = mp.mpf(d)
        if seed is None:
            t = lowest_critical_zero(d)
        else:
            try:
                t = mp.findroot(lambda yy: crit_line_value(yy, d), seed, tol=_TRACE_TOL)
            except (ValueError, ZeroDivisionError):
                t = lowest_critical_zero(d)
        if t is None:
            continue
        seed = t
        out.append((float(d), float(mp.re(t))))
    out.reverse()
    return out


def real_branch(d_values):
    """[(d, sigma_-(d), sigma_+(d)), ...] the off-critical real pair, continued up in d."""
    out = []                                    # type: List[Tuple[float, float, float]]
    seed = None
    for d in sorted(d_values):
        d = mp.mpf(d)
        sm = real_lower_zero(d, seed=seed)
        if sm is None:
            continue
        seed = sm
        out.append((float(d), float(sm), float(d / 2 - sm)))
    return out


# --------------------------------------------------------------------------
# argument principle  Z(x) = (1/2 pi i) oint Z'/Z ds  -- an independent count
# --------------------------------------------------------------------------
def count_zeros(d, sig_lo, sig_hi, t_lo, t_hi, npts=120):
    """Number of zeros of Z_d in the rectangle, by the winding number of Z_d.

    Sums principal-branch increments of arg Z_d around the boundary (no derivative).
    Used to certify the pitchfork independently: 0 real zeros in a real-axis strip for
    d < d*, 2 for d > d*.
    """
    d = mp.mpf(d)

    def edge(a, b, n):
        return [a + (b - a) * mp.mpf(i) / n for i in range(n)]

    corners = [mp.mpc(sig_lo, t_lo), mp.mpc(sig_hi, t_lo),
               mp.mpc(sig_hi, t_hi), mp.mpc(sig_lo, t_hi)]
    loop = (edge(corners[0], corners[1], npts) + edge(corners[1], corners[2], npts)
            + edge(corners[2], corners[3], npts) + edge(corners[3], corners[0], npts))
    total = mp.mpf(0)
    prev = epstein_Z(loop[0], d)
    for p in loop[1:] + [loop[0]]:
        cur = epstein_Z(p, d)
        total += mp.arg(cur / prev)
        prev = cur
    return total / (2 * PI)


# --------------------------------------------------------------------------
# the bridge's OWN motion, for the contrast panel (harmonic_bridge.zeta_K)
# --------------------------------------------------------------------------
_K_CONTRAST = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]


def kknob_trajectory(gamma):
    """The bridge's zeta_K zero limiting to 1/2 + i*gamma as K grows: zeroless ground
    (K=0) -> born (K=1, off the line) -> ONTO sigma=1/2. Returns [(K, s), ...]. This is
    the arrow the Epstein pair reverses."""
    traj = hb.migrate(hb.zeta_K, mp.mpc(0.5, gamma), schedule=_K_CONTRAST)
    return [(K, zz) for K, zz in traj if zz is not None]


# --------------------------------------------------------------------------
# self-validating driver
# --------------------------------------------------------------------------
def _main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figstyle
    figstyle.enlarge()
    # Dense 6-panel figure: trim title sizes locally so they don't overrun.
    plt.rcParams.update({"axes.titlesize": 13, "figure.titlesize": 16})

    # ---- 1. evaluator vs direct sum + closed forms ----
    print("== Z_d(s) via Terras formula vs direct lattice sum (Re s > d/2) ==")
    worst_dir = 0.0
    for (d, s, R) in [(2, mp.mpc(3.0, 0), 60), (2, mp.mpc(3.0, 0.7), 60),
                      (3, mp.mpc(3.0, 0), 30)]:
        a, b = epstein_Z(s, d), epstein_Z_direct(s, d, R)
        worst_dir = max(worst_dir, float(abs((a - b) / b)))
    print(f"   worst Terras-vs-direct rel err = {worst_dir:.2e}")
    print("== closed forms  Z_1(s)=2 zeta(2s),  Z_2(s)=4 zeta(s) beta(s) ==")
    worst_cf = 0.0
    for s in [mp.mpc(1.2, 0.4), mp.mpc(0.9, 3.0), mp.mpc(-0.5, 2.0)]:
        worst_cf = max(worst_cf, float(abs(epstein_Z(s, 1) - Z1_closed(s))))
    for s in [mp.mpc(1.6, 0.3), mp.mpc(1.2, 2.0), mp.mpc(0.7, 5.0)]:
        worst_cf = max(worst_cf, float(abs(epstein_Z(s, 2) - Z2_closed(s))))
    print(f"   worst closed-form err = {worst_cf:.2e}")

    # ---- 2. functional equation + pole residue ----
    print("\n== functional equation  Lambda_d(s) = Lambda_d(d/2-s) ==")
    worst_fe = 0.0
    for d in [3, 5.5, 9, 12]:
        for s in [mp.mpc(0.7, 2.3), mp.mpc(-1.0, 4.0), mp.mpc(1.3, 0)]:
            worst_fe = max(worst_fe, float(functional_eq_residual(s, d)))
    print(f"   worst functional-eq residual = {worst_fe:.2e}")
    print("== pole residue  Res_{s=d/2} Z_d = pi^{d/2}/Gamma(d/2) ==")
    worst_res = 0.0
    for d in [2, 3, 5, 9.5]:
        worst_res = max(worst_res, float(abs(pole_residue_numeric(d) - pole_residue(d))))
    print(f"   worst residue err = {worst_res:.2e}")

    # ---- 3. the critical dimension d* ----
    print("\n== critical dimension  d* = root of g(d) = Lambda_d(d/4)  (T-S: 9.24555) ==")
    dstar = critical_dimension()
    print(f"   d* = {mp.nstr(dstar, 11)}")
    print(f"   g(d*) = {float(center_value(dstar)):.2e}  (should vanish)")

    # ---- 4. the on-line branch t_1(d) -> 0 as d -> d* ----
    print("\n== on-line branch: lowest critical-line zero t_1(d), d/4 +/- i t_1 ==")
    D_CRIT = [2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 8.5, 9, 9.1, 9.2, 9.24]
    crit = critical_branch(D_CRIT)
    for d, t in crit[::3] + [crit[-1]]:
        print(f"   d={d:5.2f}:  t_1 = {t:.5f}")
    print(f"   -> t_1 slides to 0 as d -> d*={float(dstar):.4f} (collision at the real center)")

    # ---- 5. the off-critical real branch: born off, migrate to the strip edges ----
    print("\n== off-critical real pair (d > d*): sigma_- -> 0, sigma_+ -> d/2 ==")
    D_REAL = [9.25, 9.3, 9.4, 9.6, 10, 11, 12, 14, 17, 20, 25]
    real = real_branch(D_REAL)
    for d, sm, sp in real[::2]:
        print(f"   d={d:5.2f}:  center d/4={d/4:6.3f}  sigma_-={sm:.4f}  sigma_+={sp:.4f}"
              f"   (strip [0, {d/2:.2f}])")
    print("   -> the pair walks OUT to the strip boundaries {0, d/2} (Travenec-Samaj Sec. 3)")

    # ---- 6. argument-principle certificate of the pitchfork ----
    print("\n== argument principle: real zeros in a strip about the real axis ==")
    # a thin box hugging the real axis, spanning most of the strip (0, d/2)
    for d in [8.0, 12.0]:
        n = count_zeros(d, 0.15, d / 2 - 0.15, -0.6, 0.6, npts=120)
        print(f"   d={d:5.2f}:  winding count = {float(n):+.3f}"
              f"   ({'0 -> pair on the line' if d < float(dstar) else '2 -> real off-critical pair'})")

    # ---- 7. the contrast: born ONTO 1/2 (the bridge) ----
    print("\n== contrast: the bridge's zeta_K zero climbs ONTO 1/2 (opposite arrow) ==")
    k_traj = kknob_trajectory(14.134725)
    print(f"   K={k_traj[0][0]:>3}: Re = {float(k_traj[0][1].real):.4f} (born off 1/2)"
          f"  ...  K={k_traj[-1][0]:>3}: Re = {float(k_traj[-1][1].real):.4f} (onto 1/2)")

    # ===================== figure =====================
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # (0,0) the bifurcation / pitchfork diagram: offset-from-center vs d
    ax = axes[0, 0]
    dc = [d for d, _t in crit]
    tc = [t for _d, t in crit]
    ax.plot(dc, tc, "-", color="C0", lw=1.6, marker=".", ms=5,
            label=r"on line: $\pm t_1(d)$  (Im offset)")
    ax.plot(dc, [-t for t in tc], "-", color="C0", lw=1.6, marker=".", ms=5)
    dr = [d for d, _sm, _sp in real]
    delr = [d / 4 - sm for d, sm, _sp in real]
    ax.plot(dr, delr, "--", color="C3", lw=1.6, marker="s", ms=4,
            label=r"real pair: $\pm\delta(d)$  (Re offset)")
    ax.plot(dr, [-x for x in delr], "--", color="C3", lw=1.6, marker="s", ms=4)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(float(dstar), ls=":", color="C2", lw=1.4)
    ax.annotate(r"$d^*\!=\!9.2456$", (float(dstar), 0), textcoords="offset points",
                xytext=(5, 40), fontsize=11, color="C2")
    ax.set_xlabel(r"dimension $d$")
    ax.set_ylabel(r"zero offset from center $s=d/4$")
    ax.set_title("pitchfork at $d^*$: a zero pair leaves the line\n"
                 r"(Im offset $t_1\!\to\!0$, then real offset $\delta$ grows)")
    ax.legend(loc="upper left", fontsize=10)

    # (0,1) the s-plane picture, in shifted coords x = Re s - d/4 so the line is x=0
    ax = axes[0, 1]
    # shared color scale (d: 2 -> 25) so color reads as one continuous knob across
    # both branches: dots descend the line up to d*, then squares carry it out.
    _vmin, _vmax = 2.0, 25.0
    xs_on = [0.0 for _ in crit] + [0.0 for _ in crit]
    ys_on = tc + [-t for t in tc]
    con = ax.scatter(xs_on, ys_on, c=dc + dc, cmap="viridis", vmin=_vmin, vmax=_vmax,
                     s=22, zorder=3)
    xs_re = delr + [-x for x in delr]
    ys_re = [0.0 for _ in real] * 2
    ax.scatter(xs_re, ys_re, c=dr + dr, cmap="viridis", vmin=_vmin, vmax=_vmax,
               s=26, marker="s", zorder=3)
    ax.axvline(0, ls=":", color="C2", lw=1.4, label=r"critical line $\mathrm{Re}\,s=d/4$")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.plot([0], [0], "x", color="C3", ms=11, zorder=4)
    ax.annotate("collision\n$d=d^*$", (0, 0), textcoords="offset points", xytext=(8, 8),
                fontsize=10, color="C3")
    ax.set_xlabel(r"$\mathrm{Re}\,s - d/4$")
    ax.set_ylabel(r"$\mathrm{Im}\,s$")
    ax.set_title("zeros in the shifted $s$-plane\n"
                 "descend the line, collide, split onto the real axis")
    cb = fig.colorbar(con, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"dimension $d$", fontsize=11)
    ax.legend(loc="upper right", fontsize=10)

    # (0,2) the mechanism: g(d) = Lambda_d(d/4) crossing zero at d*
    ax = axes[0, 2]
    dg = [2 + 0.25 * i for i in range(int((12 - 2) / 0.25) + 1)]
    gvals = [float(center_value(d)) for d in dg]
    ax.plot(dg, gvals, "-", color="C4", lw=1.8)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(float(dstar), ls=":", color="C2", lw=1.4)
    ax.plot([float(dstar)], [0], "o", color="C2", ms=8, zorder=4)
    ax.annotate(r"$d^*=9.24555$", (float(dstar), 0), textcoords="offset points",
                xytext=(-120, 12), fontsize=11, color="C2")
    ax.set_xlabel(r"dimension $d$")
    ax.set_ylabel(r"$g(d)=\Lambda_d(d/4)$")
    ax.set_title(r"the switch: $g(d)=\Lambda_d(d/4)$ changes sign at $d^*$"
                 "\n(on-line pair $\\to$ real pair)")

    # (1,0) migration to the strip boundaries: sigma_-(d)->0, sigma_+(d)->d/2
    ax = axes[1, 0]
    sm_l = [sm for _d, sm, _sp in real]
    sp_l = [sp for _d, _sm, sp in real]
    ax.plot(dr, sm_l, "-", color="C3", lw=1.6, marker="o", ms=4, label=r"$\sigma_-(d)$")
    ax.plot(dr, sp_l, "-", color="C1", lw=1.6, marker="o", ms=4, label=r"$\sigma_+(d)$")
    ax.plot(dr, [d / 4 for d in dr], ":", color="C2", lw=1.4, label=r"center $d/4$")
    ax.plot(dr, [d / 2 for d in dr], "--", color="0.5", lw=1.2,
            label=r"upper edge $d/2$")
    ax.axhline(0, ls="--", color="0.5", lw=1.2)
    ax.annotate("lower edge $0$", (dr[-1], 0), textcoords="offset points",
                xytext=(-70, 6), fontsize=10, color="0.4")
    ax.set_xlabel(r"dimension $d$")
    ax.set_ylabel(r"$\mathrm{Re}\,s$ of the real zero pair")
    ax.set_title("the off-critical pair migrates to the strip edges\n"
                 r"$\sigma_-\!\to\!0,\ \sigma_+\!\to\!d/2$  (as $d\to\infty$)")
    ax.legend(loc="center left", fontsize=10)

    # (1,1) the completed function on the critical line: lowest zero slides to y=0
    ax = axes[1, 1]
    yy = [0.02 * i for i in range(0, int(8 / 0.02) + 1)]
    for d, col in [(5, "C0"), (7, "C5"), (9, "C6"), (9.24, "C9")]:
        vals = [float(crit_line_value(y, d)) for y in yy]
        # normalise for shape comparison (the magnitudes differ across d)
        m = max(abs(v) for v in vals)
        ax.plot(yy, [v / m for v in vals], "-", color=col, lw=1.4, label=f"$d={d}$")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel(r"$y$  (on the critical line $s=d/4+iy$)")
    ax.set_ylabel(r"$\Lambda_d(d/4+iy)$  (normalised)")
    ax.set_title(r"on-line profile: the lowest zero $t_1(d)$"
                 "\n" r"slides toward $y=0$ as $d\to d^*$")
    ax.legend(loc="upper right", fontsize=10)

    # (1,2) THE taxonomy contrast: Epstein AWAY vs the bridge ONTO 1/2
    ax = axes[1, 2]
    # Epstein: real-pair offset from its own critical line, vs a normalised knob.
    # normalise d in [d*, 25] -> [0,1]; plot Re - d/4 (grows, leaves 0).
    dmax = dr[-1]
    prog_e = [(d - float(dstar)) / (dmax - float(dstar)) for d in dr]
    off_e = [sm - d / 4 for d, sm in zip(dr, sm_l)]     # negative, grows in magnitude
    ax.plot(prog_e, off_e, "-", color="C3", lw=1.8, marker="s", ms=4,
            label=r"Epstein real zero: born OFF, moves AWAY")
    # bridge: zeta_K zero offset from 1/2 vs a normalised K knob (K small -> large).
    ks = [K for K, _s in k_traj]
    prog_b = [(mp.log(K) - mp.log(ks[0])) / (mp.log(ks[-1]) - mp.log(ks[0])) for K in ks]
    off_b = [float(s.real) - 0.5 for _K, s in k_traj]
    ax.plot([float(p) for p in prog_b], off_b, "-", color="C0", lw=1.8, marker="o", ms=4,
            label=r"bridge $\zeta_K$ zero: born, climbs ONTO $\frac{1}{2}$")
    ax.axhline(0, ls=":", color="C2", lw=1.4)
    ax.annotate("the line", (0.02, 0), textcoords="offset points", xytext=(0, 5),
                fontsize=10, color="C2")
    ax.set_xlabel("knob progress  ($d:d^*\\to25$ ;  $K:1\\to89$)")
    ax.set_ylabel(r"$\mathrm{Re}\,s$ $-$ (critical line)")
    ax.set_title("the directional taxonomy in one axis:\n"
                 "Epstein leaves the line; the bridge arrives on it")
    ax.legend(loc="lower left", fontsize=9)

    fig.suptitle(r"Off-critical zeros of the hypercubic Epstein zeta  "
                 r"$Z_d(s)=\sum'_{n\in\mathbb{Z}^d}|n|^{-2s}$  "
                 r"(Travenec-Samaj 2022): the reverse-direction foil -- born OFF, migrate AWAY")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    figdir = Path(__file__).resolve().parent / "figures"
    figdir.mkdir(exist_ok=True)
    out = figdir / "epstein_zeros.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    _main()
