r"""Where the trivial zeros come from: birth on the negative real axis (issue #10).

The bridge has tracked, at length, how the *non-trivial* zeros appear and migrate
onto Re s = 1/2 (and, for the prefactor families, onto Re s = 1 and Re s = 0). But
zeta also has the **trivial** zeros at s = -2, -4, -6, ..., and the bland continuous
endpoint 1/(s-1)+1/2 has none of them (it has a single real zero, at s = -1, and
that is all). So somewhere in the restoration of discreteness the trivial zeros, too,
must be *born*. Since stability.py established that zeta^{(K)} -> zeta with no
abscissa -- the same O(1/K) convergence at sigma = -2 as at sigma = +2 -- we can
follow the negative real axis numerically and watch it happen.

The mechanism: a rising tide on the negative real axis
------------------------------------------------------
On the real axis every harmonic is real, so zeta^{(K)} is a real analytic function of
real s, and its trivial zeros are honest sign changes. The leading truncation error
(rate_law) is  zeta(s) - zeta^{(K)}(s) ~ s/(2 pi^2 K), so

        zeta^{(K)}(s)  ~  zeta(s) + |s|/(2 pi^2 K)      (s real < 0),

a POSITIVE upward shift that grows leftward like |s| and decays like 1/K. zeta on the
negative axis oscillates about 0 with the trivial zeros as its crossings; raising the
whole curve by |s|/(2 pi^2 K) fills in the shallow dips where zeta < 0 and ERASES the
two crossings that bound them. The dips (zeta < 0) sit on the intervals
(-2,0), (-6,-4), (-10,-8), (-14,-12), ... -- so the erased crossings come in the pairs

        {-4,-6},  {-8,-10},  {-12,-14},  {-16,-18},  ...

each straddling an odd-integer midpoint -5, -9, -13, -17 where the dip is deepest. As K
grows the tide |s|/(2 pi^2 K) recedes and each pair is *uncovered* -- born.

Two things make the order counterintuitive (and pretty):

  * **The first trivial zero -2 is INHERITED, not born.** The dip on (-2,0) is bounded
    on the right not by a zero but by the pole side; raising the endpoint's lone zero
    (s=-1 of 1/(s-1)+1/2) just slides it leftward, and it limits onto -2 as
    -2 + |a_1|/K with a_1 = -2/(2 pi^2 zeta'(-2)) = 3.328 (`inherited_branch`,
    `disp_coeff_trivial`). The bland integral's single zero seeds the -2 branch; every
    later trivial zero arrives by its own conjugate-pair pinch (next bullet).

  * **Depth beats shallowness: the deep pairs are born FIRST.** A pair is uncovered when
    the tide drops below the dip depth |zeta(midpoint)|, i.e. at

        K_pinch  ~  |s_mid| / (2 pi^2 |zeta(s_mid)|),   s_mid = -(4j+1)   (`pinch_K_law`)

    and |zeta(-(2m+1))| = |B_{2m+2}|/(2m+2) grows *factorially*, so the deep dips are
    bottomless and their pairs arrive early -- by K ~ 8 for the {-12,-14} pair (j=3)
    and essentially at K = 1 from j >= 4 on -- while the SHALLOWEST dip --
    the first one, |zeta(-5)| ~ 0.004 -- is the last to clear: the {-4,-6} pair is not
    born until K ~ 62. Born from both ends, the negative axis fills in toward the
    middle; the hardest trivial zeros to resolve are -4 and -6.

How a pair is born: a conjugate pair pinches onto the axis
----------------------------------------------------------
Below its birth-K a pair is not absent from the plane -- it is a complex-conjugate pair
sitting OFF the real axis, just above and below the midpoint. As K increases the pair
descends (Im -> 0), pinches onto the axis at s ~ s_mid (a double real zero), and splits,
one zero sliding right to -2(2j) and one left to -2(2j+1) (`pinch_fork`). This is the
trivial-zero mirror of the non-trivial story: there a zero is born off its line and
migrates onto Re=1/2 (a horizontal approach to a vertical locus); here a conjugate pair
is born off ITS line and migrates onto Im=0 (a vertical approach to the horizontal
locus), then slides along it to the even integers. Same slogan -- *born off the locus,
migrate onto it* -- rotated ninety degrees, and in pairs.

The rate law already covers it
------------------------------
None of this needs new constants. The unified displacement law of rate_law -- a zero
s_K of object^{(K)} near a zero rho of the target obeys s_K - rho ~ a_1/K with
a_1 = -Delta_1(rho)/target'(rho) -- applies verbatim with target = zeta and rho = -2n:

        disp_coeff_trivial(n)  ==  rate_law.disp_coeff("comb_zeta", -2n)
                               =   (-2n) / (2 pi^2 zeta'(-2n)).

a_1 is real here (everything is real on the axis) and its magnitude peaks in the middle
(|a_1| at -2,-4,-6,-8,-10 is 3.3, 25, 52, 49, 27): a "hardest in the middle" that
rhymes with the pinch picture, though the two rankings differ -- |a_1| peaks at -6/-8
(displacement), while the pinch/birth-K law makes {-4,-6} the last-born -- overlapping
at -6, read off the single displacement coefficient the rest of the bridge already
uses. What is genuinely new on this axis is the *birth* event -- the conjugate-pair
pinch and its valley-depth birth-K law -- which the non-trivial zeta family never
undergoes (born at K=1, it only migrates; eta's ground string is present already
at K=0).

Run directly to validate + plot: writes bridge/figures/trivial_zeros.png.
"""
import sys
from pathlib import Path
from typing import List, Tuple

import mpmath as mp

# This module sits in a flat source dir alongside its bare cross-imports; put that
# directory on sys.path so a direct ``python trivial_zeros.py`` run resolves them
# (pytest gets the same path from the root conftest.py).
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import harmonic_bridge as hb  # noqa: E402  (zeta_K -- the additive-comb interpolant)
import rate_law as rl         # noqa: E402  (disp_coeff/rate_comb -- the unified displacement law)

mp.mp.dps = 30

PI = mp.pi


# --------------------------------------------------------------------------
# the interpolant on the negative real axis, and the leading predictions
# --------------------------------------------------------------------------
def endpoint_zero():
    """The bland K=0 endpoint 1/(s-1)+1/2 has a single real zero, at s = -1.

    Solve 1/(s-1) + 1/2 = 0  ->  s - 1 = -2  ->  s = -1. This lone zero is the seed of
    the entire trivial-zero string: `inherited_branch` continues it leftward onto -2.
    """
    return mp.mpf(-1)


def trivial_value(n, K):
    """zeta^{(K)}(-2n) at the exact n-th trivial zero -2n; -> 0 as K grows."""
    return hb.zeta_K(mp.mpc(-2 * n, 0), K)


def predict_value(n, K):
    """Leading prediction for zeta^{(K)}(-2n):  |2n|/(2 pi^2 K)  = |rate_comb(-2n)|/K.

    The truncation error zeta(s)-zeta^{(K)}(s) ~ s/(2 pi^2 K) evaluated at s=-2n (where
    zeta=0) leaves zeta^{(K)}(-2n) ~ -s/(2 pi^2 K) = 2n/(2 pi^2 K): small, positive, 1/K.
    """
    return float(abs(rl.rate_comb(mp.mpc(-2 * n, 0)))) / K


def disp_coeff_trivial(n):
    """a_1 in s_K - (-2n) ~ a_1/K -- the rate_law displacement coefficient at rho = -2n.

    Identically rate_law.disp_coeff("comb_zeta", -2n) = (-2n)/(2 pi^2 zeta'(-2n)); real
    on the axis. Its sign is the approach side (a_1>0: zero limits onto -2n from the
    right/above in s; a_1<0: from the left), and |a_1| -- largest for the middle zeros --
    measures how many harmonics the zero needs to settle.
    """
    return rl.disp_coeff("comb_zeta", mp.mpc(-2 * n, 0))


def predict_zero(n, K):
    """Leading prediction for the real zero of zeta^{(K)} near -2n:  -2n + Re(a_1)/K."""
    return -2 * n + float(disp_coeff_trivial(n).real) / K


# --------------------------------------------------------------------------
# real zeros on the negative axis (sign-change scan + refine)
# --------------------------------------------------------------------------
def real_zeros(K, lo=-13.0, hi=-0.35, n_grid=260, tol_imag=1e-6):
    """Real zeros of zeta^{(K)} on (lo, hi), found by scanning for sign changes.

    Each bracketed crossing is refined with findroot and kept only if it lands on the
    real axis (|Im| < tol_imag). Returns an increasing list of floats. The count grows
    with K as successive conjugate pairs pinch onto the axis (`pinch_fork`).
    """
    xs = [lo + (hi - lo) * i / n_grid for i in range(n_grid + 1)]
    vals = [float(hb.zeta_K(x, K).real) for x in xs]
    out = []  # type: List[float]
    for i in range(n_grid):
        if vals[i] == 0.0:
            cand = xs[i]
        elif vals[i] * vals[i + 1] < 0:
            cand = None
            try:
                z = mp.findroot(lambda s: hb.zeta_K(s, K), (xs[i] + xs[i + 1]) / 2)
                if abs(z.imag) < tol_imag:
                    cand = float(z.real)
            except (ValueError, ZeroDivisionError):
                cand = None
        else:
            continue
        if cand is not None and (not out or abs(cand - out[-1]) > 1e-3):
            out.append(cand)
    return out


# --------------------------------------------------------------------------
# the inherited -2 zero: continuation of the endpoint zero upward in K
# --------------------------------------------------------------------------
_INHERIT_SCHEDULE = [0, 1, 2, 3, 4, 6, 8, 12, 18, 27, 40, 60, 90, 130]


def inherited_branch(schedule=None):
    """Trajectory of the FIRST trivial zero as the continuation of the endpoint zero.

    Seeds at K=0 with `endpoint_zero` (s=-1) and walks K upward, each solve seeded by the
    previous root. The branch is real throughout and limits onto -2 as -2 + |a_1|/K
    (a_1 = disp_coeff_trivial(1) = 3.328). Returns a list of (K, s) in increasing K.
    """
    sched = sorted(schedule if schedule is not None else _INHERIT_SCHEDULE)
    z = mp.mpc(endpoint_zero(), 0)
    out = []  # type: List[Tuple[int, mp.mpc]]
    for K in sched:
        z = mp.findroot(lambda s: hb.zeta_K(s, K), z)
        out.append((K, z))
    return out


# --------------------------------------------------------------------------
# the paired trivial zeros: midpoints, valley depths, and the birth-K law
# --------------------------------------------------------------------------
def pair_zeros(j):
    """The j-th annihilation pair of trivial zeros: (-4j, -(4j+2))  (j >= 1)."""
    return (-4 * j, -(4 * j + 2))


def pair_mid(j):
    """The odd-integer midpoint of the j-th pair:  s_mid = -(4j+1)."""
    return -(4 * j + 1)


def valley_depth(j):
    """Dip depth |zeta(s_mid)| at the j-th midpoint -- the height of the tide a pair needs.

    |zeta(-(2m+1))| = |B_{2m+2}|/(2m+2) grows factorially in the depth, so deep dips are
    bottomless: their pairs survive any tide and are born at K ~ 1. The first dip,
    |zeta(-5)| ~ 0.004, is the shallowest, so {-4,-6} is the last pair born.
    """
    return abs(mp.zeta(pair_mid(j)))


def pinch_K_law(j):
    """Birth-K law for the j-th pair:  |s_mid| / (2 pi^2 |zeta(s_mid)|).

    The pair is uncovered once the tide |s_mid|/(2 pi^2 K) drops below the dip depth
    |zeta(s_mid)|. Decreasing in j (deeper = earlier); largest at j=1 ({-4,-6}, ~64).
    """
    return float(abs(pair_mid(j))) / (2 * float(PI) ** 2 * float(valley_depth(j)))


# --------------------------------------------------------------------------
# how a pair is born: a conjugate pair descends and pinches onto the axis
# --------------------------------------------------------------------------
def _continue(seed, schedule, max_jump=3.0):
    """Continue a zero of zeta^{(K)} across `schedule` (in its given order), seeded along.

    Returns a list of (K, s); stops if a step jumps more than max_jump (lost the branch).
    """
    z = mp.mpc(seed)
    out = []  # type: List[Tuple[int, mp.mpc]]
    for K in schedule:
        try:
            zz = mp.findroot(lambda s: hb.zeta_K(s, K), z)
        except (ValueError, ZeroDivisionError):
            break
        if abs(zz - z) > max_jump and out:
            break
        z = zz
        out.append((K, zz))
    return out


def pinch_fork(j, span=70, n_steps=16, seed_imag=0.55):
    """The 'tuning-fork' birth of the j-th pair: a conjugate pair pinches onto the axis.

    Returns dict(complex_branch, prong_left, prong_right):

      * complex_branch -- the upper (Im > 0) member of the conjugate pair, continued
        DOWNWARD in K from just below the pinch (it climbs off the axis as K shrinks);
      * prong_right / prong_left -- the two real zeros above the pinch, continued downward
        from large K toward the pinch (right -> -4j, left -> -(4j+2)).

    The two prongs meet the descending conjugate pair at s ~ s_mid: born off the axis,
    pinched onto it, split toward the even integers. (This function draws the fork;
    the *measured* pinch-K compared against `pinch_K_law` is `pinch_K_measured`.)
    """
    mid = pair_mid(j)
    zr, zl = pair_zeros(j)                      # right (-4j), left (-(4j+2))
    kp = max(2.0, pinch_K_law(j))
    k_hi = int(round(kp + span))

    # real prongs: from large K (two simple real zeros) down toward the pinch
    sched_down = sorted({int(round(kp + span * i / n_steps)) for i in range(n_steps + 1)},
                        reverse=True)
    sched_down = [K for K in sched_down if K >= 2]
    prong_right = _continue(mp.mpc(predict_zero(2 * j, k_hi), 0), sched_down)
    prong_left = _continue(mp.mpc(predict_zero(2 * j + 1, k_hi), 0), sched_down)

    # conjugate pair: from just below the pinch downward to small K (Im grows)
    k_start = max(2, int(round(kp * 0.9)))
    sched_cx = sorted({max(2, int(round(k_start * (1 - 0.85 * i / n_steps))))
                       for i in range(n_steps + 1)}, reverse=True)
    complex_branch = _continue(mp.mpc(mid, seed_imag), sched_cx, max_jump=2.5)
    complex_branch = [(K, z) for (K, z) in complex_branch if z.imag > 1e-4]
    complex_branch.sort(key=lambda kz: kz[0])   # K only: mpc's don't order

    return {
        "complex_branch": complex_branch,
        "prong_right": prong_right,
        "prong_left": prong_left,
        "mid": mid, "zr": zr, "zl": zl,
    }


def _sign_changes(K, lo, hi, n_grid=120):
    """Count sign changes of real zeta^{(K)} on (lo, hi) -- a findroot-free zero count."""
    prev = None
    count = 0
    for i in range(n_grid + 1):
        x = lo + (hi - lo) * i / n_grid
        v = float(hb.zeta_K(x, K).real)
        if prev is not None and prev * v < 0:
            count += 1
        prev = v
    return count


def pinch_K_measured(j, k_lo=2, k_hi=None):
    """Smallest K at which both real zeros of the j-th pair are present (born) -- by bisection.

    Below its birth-K the pair sits OFF the axis (0 real zeros in the midpoint window
    s_mid +/- 1.6); above, it has split into two simple real zeros (2 sign changes). That
    step is monotone in K, so we bisect for the transition rather than scanning every K --
    a bounded, coarse witness for `pinch_K_law`, not a high-precision bifurcation solve.
    """
    mid = pair_mid(j)
    lo, hi = mid - 1.6, mid + 1.6
    if k_hi is None:
        k_hi = int(round(pinch_K_law(j))) + 8
    born = lambda K: _sign_changes(K, lo, hi) >= 2
    if born(k_lo):                      # already born at the floor (deep pairs, K_pinch < 2)
        return k_lo
    if not born(k_hi):                  # not yet born by k_hi -- widen once
        k_hi *= 2
        if not born(k_hi):
            return None
    a, b = k_lo, k_hi                   # a: not born, b: born
    while b - a > 1:
        m = (a + b) // 2
        if born(m):
            b = m
        else:
            a = m
    return b


# --------------------------------------------------------------------------
# self-validating driver
# --------------------------------------------------------------------------
def _print_inherited():
    print("== the FIRST trivial zero -2 is inherited: continue the endpoint zero (-1) up in K ==")
    print("   K      s_K (real)     -2 + a_1/K (a_1 = 3.328)")
    a1 = float(disp_coeff_trivial(1).real)
    for K, z in inherited_branch():
        pred = "   --   " if K == 0 else f"{-2 + a1 / K:+8.4f}"
        print(f"   {K:4d}   {float(z.real):+9.5f}      {pred}")
    print("   (the bland 1/(s-1)+1/2 has ONE real zero, at -1; it slides onto -2, never 'born'.)")


def _print_convergence():
    print("\n== zeta^{(K)} at the exact trivial zeros -2n  ->  0  ~  |2n|/(2 pi^2 K) ==")
    print("   -2n     K     zeta^{(K)}(-2n)   |2n|/(2 pi^2 K)   ratio")
    for n in (1, 2, 3, 4, 6, 8):
        for K in (40, 160):
            v = float(trivial_value(n, K).real)
            p = predict_value(n, K)
            print(f"   {-2*n:4d}   {K:4d}   {v:+.6e}    {p:.6e}    {v/p:5.3f}")


def _print_displacement():
    print("\n== displacement a_1 (s_K - (-2n) ~ a_1/K) -- rate_law.disp_coeff at rho=-2n ==")
    print("   -2n    a_1 = disp_coeff_trivial    a_1/160     measured (K=160)")
    for n in (1, 2, 3, 4, 5, 6):
        a1 = float(disp_coeff_trivial(n).real)
        pred = a1 / 160
        try:
            zm = mp.findroot(lambda s: hb.zeta_K(s, 160), mp.mpc(-2 * n + pred, 0))
            meas = float((zm + 2 * n).real)
        except (ValueError, ZeroDivisionError):
            meas = float("nan")
        rl_check = float(rl.disp_coeff("comb_zeta", mp.mpc(-2 * n, 0)).real)
        tag = "ok" if abs(a1 - rl_check) < 1e-9 else "MISMATCH"
        print(f"   {-2*n:4d}   {a1:+10.4f}  [{tag}]        {pred:+.5f}    {meas:+.5f}")
    print("   (|a_1| peaks in the middle: -6,-8 are the slowest to settle -- hardest in the middle.)")


def _print_birth(measured):
    print("\n== birth-K law: a pair clears the tide at K ~ |s_mid|/(2 pi^2 |zeta(s_mid)|) ==")
    print("   pair        s_mid   |zeta(s_mid)|   K_law    K_measured")
    for j in (1, 2, 3):
        zr, zl = pair_zeros(j)
        print(f"   {{{zr},{zl}}}{'':<4}  {pair_mid(j):5d}   {float(valley_depth(j)):.4e}   "
              f"{pinch_K_law(j):6.1f}    {measured[j]}")
    print("   (deep dips are factorially bottomless -> born first; the shallow {-4,-6} is born last.)")


def _validate():
    # convergence to 0 at the trivial zeros, clean O(1/K) and on the |2n|/(2pi^2 K) law
    v40, v160 = float(trivial_value(2, 40).real), float(trivial_value(2, 160).real)
    assert v40 > 0 and v160 > 0, "zeta^{(K)} sits just ABOVE 0 at -2n (positive tide)"
    assert 3.5 < v40 / v160 < 4.5, "4x K -> ~4x smaller at -4 (O(1/K))"
    assert abs(v160 / predict_value(2, 160) - 1) < 0.02, "value tracks |2n|/(2 pi^2 K)"
    # rate_law reuse: disp_coeff_trivial IS the comb_zeta displacement at -2n
    for n in (1, 2, 3):
        a = disp_coeff_trivial(n)
        b = rl.disp_coeff("comb_zeta", mp.mpc(-2 * n, 0))
        assert abs(a - b) < mp.mpf("1e-25"), "disp_coeff_trivial == rate_law at -2n"
    # the inherited branch is real, monotone, and limits onto -2 (closer than -1.95 by K=130)
    br = inherited_branch()
    reals = [float(z.real) for _, z in br]
    assert reals[0] == -1.0, "endpoint zero is -1"
    assert all(b < a for a, b in zip(reals, reals[1:])), "inherited zero moves monotonically left"
    assert -2.0 < reals[-1] < -1.95, "inherited zero approaches -2 from the right"
    # birth order: the shallow {-4,-6} pair is born LATER than the deeper {-12,-14}
    assert pinch_K_law(1) > pinch_K_law(3), "deep pairs are born first (smaller K)"
    print("\n[validate] convergence + rate_law-reuse + inherited-branch + birth-order checks passed.")


def _main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figstyle
    figstyle.enlarge()
    # three panels with longish titles; trim titles a touch like the other dense drivers.
    plt.rcParams.update({"axes.titlesize": 13, "legend.fontsize": 11})

    measured = {j: pinch_K_measured(j) for j in (1, 2, 3)}   # the js _print_birth/panel 2 use

    _print_inherited()
    _print_convergence()
    _print_displacement()
    _print_birth(measured)
    _validate()

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))

    # ---- panel 0: the rising tide on the negative axis carves the trivial zeros ----
    xs = [(-9.4) + (9.0) * i / 300 for i in range(301)]                 # -9.4 .. -0.4
    for K, col in [(0, "0.6"), (64, "C0"), (120, "C3")]:
        ys = [float(hb.zeta_K(x, K).real) for x in xs]
        ax[0].plot(xs, ys, "-", color=col, lw=1.6, label=rf"$K={K}$")
        for z in real_zeros(K, lo=-9.4, hi=-0.4, n_grid=240):
            ax[0].plot(z, 0, "o", color=col, ms=5)
    ax[0].axhline(0.0, ls="-", lw=1, color="0.3")
    for m in (-2, -4, -6, -8):
        ax[0].axvline(m, ls=":", lw=1, color="0.55")
    ax[0].annotate("inherited\nendpoint zero ($K{=}0$)", (-1.0, 0), textcoords="offset points",
                   xytext=(6, 40), fontsize=10, color="0.3",
                   arrowprops=dict(arrowstyle="->", color="0.4"))
    ax[0].annotate(r"$\{-4,-6\}$ just born", (-5.0, -0.02), textcoords="offset points",
                   xytext=(-2, -34), fontsize=10, color="C0", ha="center",
                   arrowprops=dict(arrowstyle="->", color="C0"))
    ax[0].set_ylim(-0.33, 0.42)
    ax[0].set_xlabel(r"$s$ (real, negative)")
    ax[0].set_ylabel(r"$\zeta^{(K)}(s)$")
    ax[0].set_title(r"rising tide $+|s|/2\pi^2K$ carves zeros onto $-2n$")
    ax[0].legend(loc="upper right")

    # ---- panel 1: the conjugate-pair pinch -- born off-axis, descend onto Im=0, split ----
    pinch_pts = []
    for j, col in [(1, "C3"), (2, "C0")]:
        fk = pinch_fork(j)
        cb = fk["complex_branch"]
        if cb:
            re = [float(z.real) for _, z in cb]
            im = [float(z.imag) for _, z in cb]
            ax[1].plot(re, im, "-o", color=col, ms=3, lw=1)
            ax[1].plot(re, [-y for y in im], "-o", color=col, ms=3, lw=1)  # conjugate
            ax[1].plot(re[0], im[0], "*", color=col, ms=12)                # off-axis birth (small K)
        for key in ("prong_right", "prong_left"):
            pr = fk[key]
            if pr:
                ax[1].plot([float(z.real) for _, z in pr],
                           [float(z.imag) for _, z in pr], "-", color=col, lw=1.4)
        zr, zl = fk["zr"], fk["zl"]
        ax[1].plot([zr, zl], [0, 0], "x", color=col, ms=8, mew=2)
        pinch_pts.append((fk["mid"], col, zr, zl))
    ax[1].axhline(0.0, ls="-", lw=1, color="0.3")
    for m in (-4, -6, -8, -10):
        ax[1].axvline(m, ls=":", lw=1, color="0.55")
    ax[1].annotate(r"born off-axis", (pinch_pts[0][0], 0.78), fontsize=10, color="C3",
                   ha="center")
    ax[1].annotate(r"pinch onto $\mathrm{Im}\,s=0$, split", (-7.0, 0.07), fontsize=10,
                   color="0.25", ha="center")
    ax[1].set_xlabel(r"$\mathrm{Re}\,s$")
    ax[1].set_ylabel(r"$\mathrm{Im}\,s$")
    ax[1].set_title(r"pairs born off-axis, pinch onto $\mathrm{Im}=0$ ($K\nearrow$)")
    ax[1].set_xlim(-11, -3)
    ax[1].set_ylim(-1.0, 1.0)

    # ---- panel 2: birth-K law -- deep pairs first, shallow {-4,-6} last ----
    js = [1, 2, 3, 4]
    klaw = [pinch_K_law(j) for j in js]
    mids = [abs(pair_mid(j)) for j in js]
    ax[2].semilogy(mids, klaw, "-", color="C0", lw=1.5,
                   label=r"law $|s_{\rm mid}|/2\pi^2|\zeta(s_{\rm mid})|$")
    mj = [(abs(pair_mid(j)), measured[j]) for j in (1, 2, 3) if measured[j]]
    ax[2].semilogy([m for m, _ in mj], [k for _, k in mj], "o", color="C3", ms=9,
                   label="measured pinch-$K$ (born)")
    # stagger the pair labels so they don't collide near the top-left
    offsets = {1: (8, -16), 2: (8, 8), 3: (8, 6), 4: (2, -15)}
    for j, m, kl in zip(js, mids, klaw):
        zr, zl = pair_zeros(j)
        ax[2].annotate(rf"$\{{{zr},{zl}\}}$", (m, kl), textcoords="offset points",
                       xytext=offsets[j], fontsize=10, color="0.3")
    ax[2].set_ylim(0.2, 130)
    ax[2].set_xlabel(r"$|s_{\rm mid}|$ (depth of the pair)")
    ax[2].set_ylabel(r"birth-$K$ (pinch onto axis)")
    ax[2].set_title(r"deep pairs born first; shallow $\{-4,-6\}$ last")
    ax[2].legend(loc="lower left")

    fig.suptitle(r"Where the trivial zeros come from: a rising tide $+|s|/2\pi^2K$ carves "
                 r"$-2,-4,-6,\dots$ onto the negative real axis ($K=0\to\infty$)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    figdir = Path(__file__).resolve().parent / "figures"
    figdir.mkdir(exist_ok=True)
    out = figdir / "trivial_zeros.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    _main()
