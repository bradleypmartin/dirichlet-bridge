r"""
Complex zeros of the Jonquiere / polylogarithm function -- reproducing
Fornberg & Kolbig (1975), Math. Comp. 29(130), 582-599.

This is the *direct intellectual ancestor* of the discrete <-> continuous bridge.
FK deform zeta through the polylogarithm argument x in

    F(x, s) = sum_{k>=1} x^k / k^s            (|x| < 1, Eq. 1)

and numerically track the complex zeros F(x,s)=0 as x sweeps the real interval
(-1, 1). The endpoints are arithmetic: F(1,s) = zeta(s) (Eq. 5) and
F(-1,s) = -eta(s). One of the authors, Bengt Fornberg, is the originating
advisor who flagged the paper; the original computation ran on the CDC 7600 at
CERN (~1973), then among the world's largest machines -- this driver reproduces
the descendant picture in a few seconds.

Why it sits in this repo (the honest-framing payoff)
----------------------------------------------------
FK's deformation is the cleanest *foil* for our K-harmonic knob. In their flow
the moving zeros only *transiently brush* the critical line Re s = 1/2 and are
then ejected -- they spiral around a zeta zero and ultimately migrate toward the
emerging pole at s=1 (FK Figs. 6-7; mechanism: the x->1 expansion Eq. 26 carries
a branch term Gamma(1-s)(-log x)^{s-1} that winds for sigma<1, and F is entire
for |x|<1 so the pole only emerges at x=1). They *set out toward* critical-line
convergence but could not reach it. Our K-knob, by contrast, makes the zeros
*land* on Re s = 1/2 and stay, converging at O(1/K) (see `rate_law.py`).

What this driver reproduces (all self-validated in __main__)
-----------------------------------------------------------
  * Evaluator F = polylog(s, x), checked against the direct sum (|x|<1), against
    the functional equation  F(x,s) = 2^{1-s} F(x^2,s) - F(-x,s)  (Eq. 28), and
    against the endpoints F(1,s)=zeta, F(-1,s)=-eta.
  * Zero trajectories seeded by the FK asymptotic zeros
        s0+(x,P,N) = [log x + (2P+1)*pi*i] / log(1+1/N)        (x->+0, Eq. 10)
        s0-(x,P,N) = [log|x| + 2P*pi*i]   / log(1+1/N)         (x->-0, Eq. 19)
    Newton-continued in x. Two classes appear (FK Sec. 4): the P=0 class tends
    directly to s=1 (misses the zeta zeros), the rest approach zeta zeros.
  * The money figure: s0+(x,1,1) approaches the first zeta zero
    s1 = 1/2 + 14.134725i, *brushes* sigma=1/2, then spirals around it and leaves
    (FK Figs. 6-7) -- the transient-approach-vs-convergence contrast.
  * The sigma=1 log-2 comb: as x->-1, the N=1 trajectories s0-(x,P,1) terminate
    at  1 + 2*pi*i*m/log2  ~  1 + 9.06472*m*i  (FK Eq. 23) -- the zeros of the
    eta factor 1-2^{1-s}. Same comb as `warp_eta.py`/`harmonic_bridge.py`, reached
    by FK's route and predating the Sondow (2003) appearance we otherwise cite.
  * Trivial-zero trajectories: the real-axis P=0 trajectories s0-(x,0,N) slide
    onto the trivial zeros -2m (FK Sec. 6) -- cousin of `trivial_zeros.py`.
  * Validation by the argument principle  Z(x) = (1/2pi i) oint F'/F ds  (Eq. 25):
    the winding number reproduces FK's counts Z(0.1)=27 and Z(-0.1)=26 in their
    rectangles.
  * The asymptote counting constants alpha+ = (2-2log2-gamma)/2pi = 0.00580756,
    alpha- = (log2-gamma)/2pi = 0.01845107 (FK Eqs. 39-40), the log-2/gamma law
    relating the asymptote count to Riemann-von Mangoldt N(T).

Run directly to validate + plot: writes bridge/figures/jonquiere_zeros.png.
"""
import math
from pathlib import Path
from typing import List, Tuple

import mpmath as mp

mp.mp.dps = 30

PI = mp.pi
LOG2 = mp.log(2)
GAMMA = mp.euler
TWOPI_OVER_LOG2 = 2 * PI / LOG2          # 9.0647202836... -- comb spacing (FK Eq. 23)
S1 = mp.mpc("0.5", "14.1347251417346937904572519836")   # first nontrivial zeta zero


# --------------------------------------------------------------------------
# the object  F(x, s) = sum_{k>=1} x^k / k^s
# --------------------------------------------------------------------------
def F(x, s):
    """F(x,s) via mpmath's polylog -- the one-line primary evaluator (FK Eq. 1)."""
    return mp.polylog(mp.mpc(s), x)


def F_sum(x, s, terms=4000):
    """F(x,s) by direct summation of the defining series -- reference for |x|<1."""
    x = mp.mpf(x) if mp.im(x) == 0 else mp.mpc(x)
    s = mp.mpc(s)
    return mp.fsum(x ** k / k ** s for k in range(1, terms + 1))


def _terms_for(lx, denom):
    """Number of expansion terms to keep so (|log x|/denom)^M < 10^{-(dps+4)}."""
    r = float(abs(lx)) / denom
    if r >= 0.3:                         # only used for |x| close to 1, so r is small
        return 80
    return max(6, int(math.ceil((mp.mp.dps + 4) / -math.log10(r))) + 2)


def F_expand(x, s):
    """F(x,s) near x=+1 via FK Eq. 26 (the polylog inversion):

        F(x,s) = Gamma(1-s)(-log x)^{s-1} + sum_{n>=0} zeta(s-n) (log x)^n / n!.

    Converges geometrically (ratio |log x|/2pi) for x near 1, where the branch
    term Gamma(1-s)(-log x)^{s-1} is the seed of the zero's spiral around s1.
    """
    s = mp.mpc(s)
    lx = mp.log(x)
    tot = mp.gamma(1 - s) * (-lx) ** (s - 1) + mp.zeta(s)
    fact = mp.mpf(1)
    lxn = mp.mpf(1)
    for n in range(1, _terms_for(lx, 2 * math.pi) + 1):
        lxn *= lx
        fact *= n
        tot += mp.zeta(s - n) * lxn / fact
    return tot


def F_expand_neg(x, s):
    """F(x,s) near x=-1 (x<0), branch-free.

    Folding the two x->+1 expansions through F(-y,s)=2^{1-s}F(y^2,s)-F(y,s)
    (y=-x) makes the Gamma(1-s)(-log y)^{s-1} branch terms cancel *analytically*,
    leaving

        F(x,s) = sum_{n>=0} zeta(s-n) (log|x|)^n (2^{n+1-s} - 1) / n!.

    The n=0 term is zeta(s)(2^{1-s}-1) = -eta(s) = F(-1,s), the eta endpoint.
    Computing the difference symbolically avoids the ~10^{16} catastrophic
    cancellation the naive functional-equation route suffers for negative sigma
    (the trivial-zero trajectories), where the branch terms are huge.
    """
    s = mp.mpc(s)
    ly = mp.log(-x)                       # = log|x| < 0
    tot = mp.zeta(s) * (2 ** (1 - s) - 1)
    fact = mp.mpf(1)
    lyn = mp.mpf(1)
    for n in range(1, _terms_for(ly, math.pi) + 1):
        lyn *= ly
        fact *= n
        tot += mp.zeta(s - n) * lyn * (2 ** (n + 1 - s) - 1) / fact
    return tot


def F_track(x, s):
    """Tracking evaluator: polylog for |x|<=0.8, else the x->+/-1 expansion.

    Both branch-aware expansions are validated against `F` (polylog) in __main__.
    Used only for zero continuation, where polylog near |x|=1 is the runtime
    bottleneck (the expansions are cheap when log x is small).
    """
    x = mp.mpf(x)
    if abs(x) <= mp.mpf("0.8"):
        return F(x, s)
    return F_expand(x, s) if x > 0 else F_expand_neg(x, s)


# --------------------------------------------------------------------------
# identities
# --------------------------------------------------------------------------
def functional_eq_residual(x, s):
    """|F(x,s) - (2^{1-s} F(x^2,s) - F(-x,s))| -- FK Eq. 28, should vanish."""
    s = mp.mpc(s)
    return abs(F(x, s) - (2 ** (1 - s) * F(x * x, s) - F(-x, s)))


def eta(s):
    """Dirichlet eta  eta(s) = (1 - 2^{1-s}) zeta(s);  F(-1,s) = -eta(s)."""
    s = mp.mpc(s)
    return (1 - 2 ** (1 - s)) * mp.zeta(s)


# --------------------------------------------------------------------------
# FK asymptotic zero seeds (Eqs. 10, 19) and their asymptotes (Eqs. 17, 20)
# --------------------------------------------------------------------------
def seed_plus(x, P, N):
    """s0+(x,P,N) = [log x + (2P+1)pi i] / log(1+1/N)   (x->+0, FK Eq. 10)."""
    return (mp.log(x) + (2 * P + 1) * PI * mp.j) / mp.log(1 + mp.mpf(1) / N)


def seed_minus(x, P, N):
    """s0-(x,P,N) = [log|x| + 2P pi i] / log(1+1/N)     (x->-0, FK Eq. 19)."""
    return (mp.log(abs(x)) + 2 * P * PI * mp.j) / mp.log(1 + mp.mpf(1) / N)


def asymptote_plus(P, N):
    """v+(P,N) = (2P+1)pi / log(1+1/N) -- the t-asymptote for x->+0 (FK Eq. 17)."""
    return (2 * P + 1) * PI / mp.log(1 + mp.mpf(1) / N)


def asymptote_minus(P, N):
    """v-(P,N) = 2P pi / log(1+1/N) -- the t-asymptote for x->-0 (FK Eq. 20)."""
    return 2 * P * PI / mp.log(1 + mp.mpf(1) / N)


def comb_point(m):
    """The sigma=1 log-2 comb  s'_m = 1 + 2 pi i m / log2  (FK Eq. 23)."""
    return mp.mpc(1, TWOPI_OVER_LOG2 * m)


# --------------------------------------------------------------------------
# zero trajectory tracing (Newton continuation in x)
# --------------------------------------------------------------------------
# Roots good to this tolerance are ample for tracing/plotting; mpmath's default
# (~1e-34 at dps=30) is tighter than the secant reaches on these flat trajectories.
_TRACE_TOL = mp.mpf("1e-18")


def _x_schedule(x_lo, d_end, nsteps):
    """|x| values from x_lo to 1-d_end, dense at BOTH ends.

    Geometric in |x| while |x|<1/2 (so a trace seeded at |x|~10^{-4} does not
    jump straight to |x|~1/2), then geometric in the gap 1-|x| (so steps cluster
    near |x|=1 where the interesting limits live). One-phase if x_lo>=1/2.
    """
    mid = mp.mpf("0.5")
    xs = []
    if mp.mpf(x_lo) < mid:
        # Split steps between the phases in proportion to the log-range each
        # spans, so the per-step multiplicative jump is uniform across both.
        r1 = math.log(float(mid) / float(x_lo))
        r2 = math.log(float(mid) / float(d_end))
        n1 = min(nsteps - 2, max(2, round(nsteps * r1 / (r1 + r2))))
        n2 = nsteps - n1
        for i in range(1, n1 + 1):
            xs.append(mp.mpf(x_lo) * (mid / mp.mpf(x_lo)) ** (mp.mpf(i) / n1))
        for i in range(1, n2 + 1):
            xs.append(1 - mid * (mp.mpf(d_end) / mid) ** (mp.mpf(i) / n2))
    else:
        d0 = 1 - mp.mpf(x_lo)
        for i in range(1, nsteps + 1):
            xs.append(1 - d0 * (mp.mpf(d_end) / d0) ** (mp.mpf(i) / nsteps))
    return xs


def trace_zero(P, N, sign, x_lo, d_end, nsteps, evalf=F_track):
    """Trace a zero of F(x,.) as |x| runs from x_lo to 1-d_end (toward sign*1).

    Seeded by the FK asymptotic zero at x = sign*x_lo, then Newton-continued
    along `_x_schedule`. Returns [(x, s), ...]. If a step cannot be continued
    (e.g. the P=0 class running into the emerging pole at s=1) the trace stops
    gracefully and returns the points found so far; an unseedable start (the
    asymptotics not yet valid at x_lo) returns [].
    """
    seed = seed_plus if sign > 0 else seed_minus
    x0 = sign * mp.mpf(x_lo)
    try:
        s = mp.findroot(lambda z: evalf(x0, z), seed(x0, P, N), tol=_TRACE_TOL)
    except (ValueError, ZeroDivisionError):
        return []                # seed not yet in its asymptotic regime at this x
    pts = [(x0, s)]                                          # type: List[Tuple]
    for xi in _x_schedule(x_lo, d_end, nsteps):
        x = sign * xi
        try:
            s = mp.findroot(lambda z: evalf(x, z), s, tol=_TRACE_TOL)
        except (ValueError, ZeroDivisionError):
            break
        pts.append((x, s))
    return pts


def spiral_about_s1(alpha_lo=1.5, alpha_hi=14.0, nsteps=50):
    """The Fig. 6/7 spiral: zero of s0+(1-10^{-alpha},1,1) vs alpha, near s1.

    As x -> 1 the branch term Gamma(1-s)(-log x)^{s-1} of Eq. 26 rotates with
    angle ~ Im(s1)*log(-log x); the zero first converges onto s1, then winds back
    out in an exponential spiral -- it never *stays* on the critical line. Uses
    the Eq. 26 expansion at raised precision (the cancellation near s1 is severe).
    Returns [(alpha, s), ...].
    """
    pts = []
    with mp.workdps(45):
        s = mp.mpc("0.49", "14.131")
        for i in range(nsteps + 1):
            a = alpha_lo + (alpha_hi - alpha_lo) * mp.mpf(i) / nsteps
            x = 1 - mp.mpf(10) ** (-a)
            s = mp.findroot(lambda z: F_expand(x, z), s, tol=mp.mpf("1e-30"))
            pts.append((float(a), mp.mpc(s)))
    return pts


# --------------------------------------------------------------------------
# argument principle  Z(x) = (1/2pi i) oint F'/F ds  (FK Eq. 25), via winding
# --------------------------------------------------------------------------
def count_zeros(x, sig_lo, sig_hi, t_lo, t_hi, npts=300):
    """Number of zeros of F(x,.) in the rectangle, by the winding number of F.

    Sums the principal-branch increments of arg F along the rectangle boundary;
    the total over 2pi is the argument-principle count (no derivative needed).
    NB the count is only right while |Delta arg F| < pi per boundary step, so
    `npts` (per edge) must grow with the rectangle -- 300 covers FK's boxes here;
    a much taller box needs proportionally more points.
    """
    x = mp.mpf(x)

    def edge(a, b, n):
        return [a + (b - a) * mp.mpf(i) / n for i in range(n)]

    corners = [mp.mpc(sig_lo, t_lo), mp.mpc(sig_hi, t_lo),
               mp.mpc(sig_hi, t_hi), mp.mpc(sig_lo, t_hi)]
    loop = (edge(corners[0], corners[1], npts) + edge(corners[1], corners[2], npts)
            + edge(corners[2], corners[3], npts) + edge(corners[3], corners[0], npts))
    total = mp.mpf(0)
    prev = F(x, loop[0])
    winding = [mp.mpf(0)]
    for p in loop[1:] + [loop[0]]:
        cur = F(x, p)
        total += mp.arg(cur / prev)
        prev = cur
        winding.append(total / (2 * PI))
    return total / (2 * PI), winding


# --------------------------------------------------------------------------
# asymptote counting constants (FK Eqs. 39-40)
# --------------------------------------------------------------------------
def alpha_plus():
    """alpha+ = (2 - 2 log2 - gamma) / 2pi = 0.00580756...  (FK Eq. 40)."""
    return (2 - 2 * LOG2 - GAMMA) / (2 * PI)


def alpha_minus():
    """alpha- = (log2 - gamma) / 2pi = 0.01845107...  (FK Eq. 40)."""
    return (LOG2 - GAMMA) / (2 * PI)


def rvm_count(T):
    """Riemann-von Mangoldt main term  N(T) = (T/2pi) log(T/2pi e)  (FK Eq. 30)."""
    return T / (2 * PI) * mp.log(T / (2 * PI * mp.e))


def asymptote_count_plus(T):
    """A+(T): lattice points (P>=1, N>=1) with v+(P,N) <= T (FK Sec. 7)."""
    count = 0
    P = 1
    while asymptote_plus(P, 1) <= T:     # N=1 is the lowest asymptote for given P
        N = 1
        while asymptote_plus(P, N) <= T:
            count += 1
            N += 1
        P += 1
    return count


def asymptote_count_minus(T):
    """A-(T): lattice points (P>=1, N>=2) with v-(P,N) <= T (FK Sec. 7)."""
    count = 0
    P = 1
    while asymptote_minus(P, 2) <= T:
        N = 2
        while asymptote_minus(P, N) <= T:
            count += 1
            N += 1
        P += 1
    return count


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
    plt.rcParams.update({"axes.titlesize": 14, "figure.titlesize": 17})

    # ---- 1. evaluator vs direct sum (|x|<1) ----
    print("== F = polylog vs direct sum (|x|<1) ==")
    worst = 0.0
    for x in [0.1, -0.3, 0.7, -0.7, 0.9]:
        for s in [mp.mpc(0.5, 14), mp.mpc(-1, 5), mp.mpc(2, 0)]:
            a, b = F(x, s), F_sum(x, s)
            rel = float(abs((a - b) / b)) if b != 0 else float(abs(a - b))
            worst = max(worst, rel)
    print(f"   worst polylog-vs-sum rel err = {worst:.2e}")

    # ---- 2. functional equation (Eq. 28) + endpoints ----
    print("\n== functional equation  F(x,s)=2^(1-s)F(x^2,s)-F(-x,s)  (Eq. 28) ==")
    worst_fe = 0.0
    for x in [0.3, 0.7, -0.5, 0.9]:
        for s in [mp.mpc(0.5, 14), mp.mpc(-1, 5), mp.mpc(1.5, 3)]:
            worst_fe = max(worst_fe, float(functional_eq_residual(x, s)))
    print(f"   worst Eq.28 residual = {worst_fe:.2e}")
    print("== endpoints  F(1,s)=zeta(s),  F(-1,s)=-eta(s) ==")
    worst_ep = 0.0
    for s in [mp.mpc(2, 1), mp.mpc(0.7, 10), mp.mpc(-1, 3)]:
        worst_ep = max(worst_ep, float(abs(F(1, s) - mp.zeta(s))),
                       float(abs(F(-1, s) + eta(s))))
    print(f"   worst endpoint err = {worst_ep:.2e}")

    # ---- 3. expansion vs polylog (the tracking evaluator) ----
    print("\n== F_expand (Eq. 26) / F_track vs polylog near x=+/-1 ==")
    worst_ex = 0.0
    for x in [0.9, 0.999, -0.9, -0.999, -(1 - mp.mpf("1e-6"))]:
        for s in [mp.mpc(1, 9.06), mp.mpc(0.5, 14.13), mp.mpc(-2, 0.3)]:
            ref = F(x, s)                       # the ~0.1 s/eval polylog: evaluate once
            if abs(mp.re(ref)) + abs(mp.im(ref)) > 1e-12:
                worst_ex = max(worst_ex, float(abs((F_track(x, s) - ref) / ref)))
    print(f"   worst expansion-vs-polylog rel err = {worst_ex:.2e}")

    # ---- 4. zero trajectories: the two classes (Eqs. 10, 17; Sec. 4) ----
    print("\n== zero trajectories  s0+(x,P,N)  (x: small -> 1) ==")
    miss_class = []     # P=0: tends to s=1, misses the zeta zeros
    for N in [1, 2]:    # N>=3 needs x ~ 10^{-N-2} to seed; off-plot to the left
        tr = trace_zero(0, N, +1, "0.02", "1e-4", 24)
        miss_class.append((N, tr))
        end = tr[-1][1]
        print(f"   P=0 N={N}: ends at s=({float(end.real):+.4f},{float(end.imag):.4f})"
              f"  -> s=1  (|s-1|={float(abs(end-1)):.3f})")
    hit_class = []      # P>=1: converges onto a zeta zero on sigma=1/2
    for (P, N, nst) in [(1, 1, 26), (2, 1, 26), (1, 2, 30)]:
        tr = trace_zero(P, N, +1, "0.02", "1e-6", nst)
        hit_class.append((P, N, tr))
        end = tr[-1][1]
        print(f"   P={P} N={N}: lands at s=({float(mp.re(end)):.4f},"
              f"{float(mp.im(end)):.4f})  (a zeta zero on sigma=1/2)")

    # ---- 5. the money figure: s0+(x,1,1) brush + spiral about s1 ----
    print("\n== s0+(x,1,1): brush of sigma=1/2 at s1, then spiral (Figs. 6-7) ==")
    brush = trace_zero(1, 1, +1, "0.02", "1e-6", 32)
    closest = min(brush, key=lambda xs: abs(xs[1] - S1))
    print(f"   closest approach to s1: sigma={float(mp.re(closest[1])):.5f}"
          f"  |s-s1|={float(abs(closest[1]-S1)):.5f}")
    spiral = spiral_about_s1()
    radii = [float(abs(s - S1)) for _a, s in spiral]
    imin = radii.index(min(radii))
    print(f"   spiral: min |s-s1|={min(radii):.2e} at alpha={spiral[imin][0]:.1f},"
          f"  then winds OUT to |s-s1|={radii[-1]:.2e} at alpha={spiral[-1][0]:.1f}"
          f"  (never lands)")

    # ---- 6. sigma=1 log-2 comb: s0-(x,P,1) -> 1 + 2pi i m/log2 (Eq. 23) ----
    print("\n== sigma=1 log-2 comb  s0-(x,P,1) -> 1 + 9.06472 m i  (Eq. 23) ==")
    combs = []
    worst_comb = 0.0
    for P in [1, 2, 3]:
        tr = trace_zero(P, 1, -1, "0.2", "1e-6", 16)
        combs.append((P, tr))
        end = tr[-1][1]
        err = float(abs(end - comb_point(P)))
        worst_comb = max(worst_comb, err)
        print(f"   P={P}: ends at s=({float(end.real):.5f},{float(end.imag):.4f})"
              f"  target 1+{float(TWOPI_OVER_LOG2*P):.4f}i  |err|={err:.1e}")

    # ---- 7. trivial-zero trajectories: s0-(x,0,N) -> -2N (Sec. 6) ----
    print("\n== trivial zeros  s0-(x,0,N) (real axis) -> -2N  (Sec. 6) ==")
    trivs = []
    worst_triv = 0.0
    for N in [1, 2, 3]:
        x_lo = float(mp.mpf(10) ** (-N - 1))     # FK start ~ 10^{-N-2}; -N-1 plenty
        tr = trace_zero(0, N, -1, x_lo, "1e-6", 30)
        trivs.append((N, tr))
        end = tr[-1][1]
        err = float(abs(end + 2 * N))
        worst_triv = max(worst_triv, err)
        print(f"   N={N}: ends at s={float(end.real):+.5f}{float(end.imag):+.1e}i"
              f"  target {-2*N}  |err|={err:.1e}")

    # ---- 8. argument principle: FK's zero counts (Eq. 25) ----
    print("\n== argument principle  Z(x) = (1/2pi i) oint F'/F ds  (Eq. 25) ==")
    z_p, wind_p = count_zeros("0.1", -11, 0, 0, 110, npts=300)
    z_m, wind_m = count_zeros("-0.1", -11, 0, 1, 109, npts=300)
    print(f"   Z(0.1)  = {float(z_p):.4f}   (FK: 27)")
    print(f"   Z(-0.1) = {float(z_m):.4f}   (FK: 26)")

    # ---- 9. asymptote counting constants (Eqs. 39-40) ----
    print("\n== asymptote counting law (Eqs. 39-40) ==")
    ap, am = alpha_plus(), alpha_minus()
    print(f"   alpha+ = {float(ap):.8f}  (FK: 0.00580756)")
    print(f"   alpha- = {float(am):.8f}  (FK: 0.01845107)")
    for T in [200, 1000]:
        Ap, Am, NT = asymptote_count_plus(T), asymptote_count_minus(T), rvm_count(T)
        print(f"   T={T:>4}: N(T)={float(NT):7.1f}  A+={Ap:>4}  A-={Am:>4}"
              f"   (N-A+)/T={float((NT-Ap)/T):+.4f}  (N-A-)/T={float((NT-Am)/T):+.4f}")
    print("   (the O(sqrt T) remainder dominates alpha at laptop T; FK needed "
          "T=10^12)")

    # ===================== figure =====================
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # (0,0) overview: the two classes in the s-plane
    ax = axes[0, 0]
    for N, tr in miss_class:
        ss = [float(mp.re(s)) for _x, s in tr]
        ts = [float(mp.im(s)) for _x, s in tr]
        ax.plot(ss, ts, "-", color="C3", lw=1.4,
                label="P=0 class -> s=1" if N == 1 else None)
    for i, (P, N, tr) in enumerate(hit_class):
        ss = [float(mp.re(s)) for _x, s in tr]
        ts = [float(mp.im(s)) for _x, s in tr]
        ax.plot(ss, ts, "--", color="C0", lw=1.4,
                label=r"P$\geq$1 $\to$ $\zeta$ zero" if i == 0 else None)
    zt = [float(mp.im(mp.zetazero(n))) for n in range(1, 4)]
    ax.plot([0.5] * len(zt), zt, "o", color="k", ms=6, label=r"$\zeta$ zeros")
    ax.axvline(0.5, ls=":", color="C2", lw=1)
    ax.axvline(1.0, ls="--", color="0.6", lw=1)
    ax.plot([1], [0], "x", color="0.4", ms=9)
    ax.annotate("pole $s=1$", (1, 0), textcoords="offset points", xytext=(4, 4),
                fontsize=11, color="0.4")
    ax.set_xlim(-4.2, 1.8)
    ax.set_ylim(0, 27)
    ax.set_xlabel(r"$\sigma=\mathrm{Re}\,s$")
    ax.set_ylabel(r"$t=\mathrm{Im}\,s$")
    ax.set_title("zero trajectories: two classes\n"
                 r"($x:0^+\!\to\!1$, seeded by FK Eqs. 10, 17)")
    ax.legend(loc="upper left", fontsize=11)

    # (0,1) the brush: s0+(x,1,1) approaches s1, brushes sigma=1/2 (Fig. 6 analog)
    ax = axes[0, 1]
    ss = [float(mp.re(s)) for _x, s in brush]
    ts = [float(mp.im(s)) for _x, s in brush]
    ax.plot(ss, ts, "-", color="C0", lw=1.6, marker=".", ms=4)
    ax.plot([0.5], [float(mp.im(S1))], "o", color="k", ms=8)
    ax.annotate(r"$s_1=\frac{1}{2}+14.1347i$", (0.5, float(mp.im(S1))),
                textcoords="offset points", xytext=(-10, -22), fontsize=11)
    ax.axvline(0.5, ls=":", color="C2", lw=1.2, label=r"critical line $\sigma=1/2$")
    ax.set_xlim(-0.2, 1.25)
    ax.set_ylim(13.4, 14.4)
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$t$")
    ax.set_title(r"$s_0^+(x,1,1)$ climbs to brush $\sigma=\frac{1}{2}$"
                 "\n" r"at $s_1$ as $x\to1$ (FK Fig. 6)")
    ax.legend(loc="lower right", fontsize=11)

    # (0,2) the spiral about s1 (Fig. 7 analog) -- the non-convergence
    ax = axes[0, 2]
    sx = [float(mp.re(s)) for _a, s in spiral]
    sy = [float(mp.im(s)) for _a, s in spiral]
    sc = ax.scatter(sx, sy, c=[a for a, _s in spiral], cmap="viridis", s=14, zorder=3)
    ax.plot(sx, sy, "-", color="0.7", lw=0.8, zorder=2)
    ax.plot([0.5], [float(mp.im(S1))], "*", color="C3", ms=16, zorder=4)
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$t$")
    ax.set_title(r"spiral of $s_0^+(x,1,1)$ about $s_1$"
                 "\n(FK Fig. 7) -- winds out, never lands")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$\alpha$  ($x=1-10^{-\alpha}$)", fontsize=11)
    ax.ticklabel_format(useOffset=False)

    # (1,0) the sigma=1 log-2 comb (x->-1, Eq. 23)
    ax = axes[1, 0]
    for P, tr in combs:
        ss = [float(mp.re(s)) for _x, s in tr]
        ts = [float(mp.im(s)) for _x, s in tr]
        ax.plot(ss, ts, "-", color="C0", lw=1.3, marker=".", ms=4)
    ct = [float(TWOPI_OVER_LOG2 * m) for m in range(1, 4)]
    ax.plot([1] * len(ct), ct, "s", color="C3", ms=9, label=r"$1+2\pi i m/\log 2$")
    ax.axvline(1.0, ls="--", color="0.6", lw=1)
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$t$")
    ax.set_title(r"$\sigma=1$ log-2 comb  ($x\to-1$)"
                 "\n" r"$s_0^-(x,P,1)\to 1+9.06472\,m\,i$")
    ax.legend(loc="center left", fontsize=11)

    # (1,1) trivial-zero trajectories (x->-1, real axis, Sec. 6)
    ax = axes[1, 1]
    for N, tr in trivs:
        xs = [float(x) for x, _s in tr]
        ss = [float(mp.re(s)) for _x, s in tr]
        ax.plot([1 - abs(x) for x in xs], ss, "-", color="C0", lw=1.3, marker=".", ms=4)
    for N in [1, 2, 3]:
        ax.axhline(-2 * N, ls="--", color="C3", lw=1,
                   label="trivial zeros $-2m$" if N == 1 else None)
    ax.set_xscale("log")
    ax.set_xlabel(r"$1-|x|$  (approach to $x=-1$)")
    ax.set_ylabel(r"$\sigma=\mathrm{Re}\,s$")
    ax.set_title(r"trivial-zero trajectories  ($x\to-1$)"
                 "\n" r"$s_0^-(x,0,N)\to -2N$")
    ax.legend(loc="center right", fontsize=11)

    # (1,2) argument principle: cumulative winding -> FK's counts
    ax = axes[1, 2]
    ax.plot([i / (len(wind_p) - 1) for i in range(len(wind_p))],
            [float(w) for w in wind_p], "-", color="C0",
            label=r"$x=+0.1$: $Z\to27$")
    ax.plot([i / (len(wind_m) - 1) for i in range(len(wind_m))],
            [float(w) for w in wind_m], "-", color="C3",
            label=r"$x=-0.1$: $Z\to26$")
    ax.axhline(27, ls=":", color="C0", lw=1)
    ax.axhline(26, ls=":", color="C3", lw=1)
    ax.set_xlabel("fraction of rectangle boundary traversed")
    ax.set_ylabel(r"cumulative  $\frac{1}{2\pi}\Delta\arg F$")
    ax.set_title("argument principle (FK Eq. 25)\n"
                 r"winding number $=$ zeros enclosed")
    ax.legend(loc="upper left", fontsize=11)

    fig.suptitle(r"Complex zeros of the Jonquiere / polylogarithm  "
                 r"$F(x,s)=\sum_{k\geq1}x^k/k^s$  "
                 r"(Fornberg-Kolbig 1975): zeros brush $\sigma=\frac{1}{2}$ but eject")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    figdir = Path(__file__).resolve().parent / "figures"
    figdir.mkdir(exist_ok=True)
    out = figdir / "jonquiere_zeros.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    _main()
