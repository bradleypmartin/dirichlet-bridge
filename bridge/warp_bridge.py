r"""The nonlinear n*-warp  vs the harmonic comb: the 2^s - 1 half-shift payoff.

The nonlinear midpoint-warp stage of the discrete <-> continuous bridge. The additive comb
(harmonic_bridge.py) restored discreteness *additively* -- it added Fourier harmonics of the
discreteness comb to the integrand and watched the zeros migrate. Here we do the bridge's
*literal* construction, the nonlinear cousin: **warp the integration variable** so the
measure is pulled toward integer plateaus, and study int (n*)^{-s} dn.

The warp
--------
n* = x + phi_K(x),  phi_K(x) = sum_{k=1}^K sin(2 pi k x)/(pi k)   (`warp_phi`).

phi_K is the K-harmonic partial sum of the sawtooth 1/2 - {x} (= the Fourier series
of 1 - x on [0,1], the bridge's kernel). As K -> infinity, phi_K(x) -> 1/2 - {x}, so
x + phi_K(x) -> floor(x) + 1/2: the warp collapses each unit cell [m, m+1] onto its
**midpoint m + 1/2**. That half-integer pinning is the whole story: it is why the
warp's natural target is a *half-shifted* zeta.

    warp_K(s, K) = int_1^inf (x + phi_K(x))^{-s} dx.

Because phi_K has period 1, the period-by-period decomposition collapses to a single
proper integral of the Hurwitz zeta (valid for all sigma incl. sigma <= 0 since Hurwitz
zeta is entire in s):

    warp_K(s, K) = int_0^1 zeta(s, 1 + u + phi_K(u)) du            (`warp_hurwitz`)

(using sum_{m>=1}(m + a)^{-s} = zeta(s, 1+a)). That transparent form is `warp_hurwitz`;
`warp_K` itself is a ~20x-faster grid+moment evaluator agreeing with it to ~1e-9 (a fixed
Gauss-Legendre grid with phi_K precomputed per K, then a Hurwitz-free moment expansion --
see the block comment at `warp_K`), which is what makes the migration tractable. `warp_raw`
is the genuine oscillatory int_1^inf in its original guise -- a coarse sigma>1 cross-check.

The two endpoints, and the LOAD-BEARING 2^s
-------------------------------------------
  * K = 0 (phi_0 = 0): warp_K(s, 0) = int_0^1 zeta(s, 1+u) du = int_1^inf x^{-s} dx =
    1/(s-1) -- exactly the bland zeta-side endpoint of the bridge. (`warp_K(.,0)` is *exact*.)

  * K -> infinity: warp_K -> int_0^1 zeta(s, 3/2) du = zeta(s, 3/2) = sum_{m>=1}(m+1/2)^{-s}
    -- the cell midpoints 3/2, 5/2, ... This is the *honest* limit of int_1^inf, and it
    is the **ugly** Hurwitz zeta: its zeros are scattered, NOT on a clean line.

The clean half-shifted object  zeta(s, 1/2) = sum_{m>=0}(m+1/2)^{-s} = (2^s - 1) zeta(s)
is reached only by restoring the m=0 midpoint cell 1/2, whose value is (1/2)^{-s} = 2^s:

    warp_half_K(s, K) = warp_K(s, K) + 2^s   ->   zeta(s, 1/2) = (2^s - 1) zeta(s).

That added 2^s is the **lower-endpoint / half-weight term, and it is load-bearing** --
exactly the additive comb's lesson in warp form. Without it the limit is zeta(s, 3/2) and the
sigma=0 companion below is *absent* (e.g. |zeta(s,3/2)| = 1 at s = 2 pi i/ln2, where
|zeta(s,1/2)| = 0). With it, the (2^s - 1) factor that the bridge set aside re-emerges *on
the nose* as the lower-endpoint cell -- the half-shift payoff is literally the
half-weight.

The migration -- sigma=1/2 UNION sigma=0  (`warp_half_K` + the reused hb.migrate)
--------------------------------------------------------------------------------
Tracking zeros of warp_half_K by backward continuation in K (the same machinery as
the additive comb, imported from harmonic_bridge):

  * the genuine zeta zeros (zeta(s)=0) migrate onto **sigma = 1/2** -- but, unlike the
    comb's zeros which are *born high* (sigma ~ 1.2) and *descend*, the warp's are born
    *low* (sigma ~ 0 at K=1) and *climb* onto 1/2 from below;

  * a companion family, the zeros of the prefactor 2^s - 1 at t = 2 pi k/ln2, migrates
    onto **sigma = 0** -- the mirror image, across sigma = 1/2 by the functional
    equation, of the additive comb's eta companion on sigma = 1, at the *same* heights. (`PREF_HEIGHTS`.)

So the warp is **not** the same bridge as the comb: it secretly targets the half-shifted
Hurwitz zeta(s, 1/2), carrying the sigma=0 prefactor companion the additive comb lacks.
The convergence rate is the same O(1/K) (the first sawtooth-tail term governs both).

Run directly to validate + plot: writes bridge/figures/warp_bridge.png.
"""
import math
import sys
from pathlib import Path
from typing import List, Tuple

import mpmath as mp

# This module sits in a flat source dir alongside its bare cross-imports; put that
# directory on sys.path so a direct ``python warp_bridge.py`` run resolves them
# (pytest gets the same path from the root conftest.py).
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import harmonic_bridge as hb  # noqa: E402  (reuse `migrate`, the comb `zeta_K`)
import cont_eta as ce         # noqa: E402  (the continuous-eta Riemann-von Mangoldt counting_law)

mp.mp.dps = 30

PI = mp.pi
LN2 = mp.log(2)


# --------------------------------------------------------------------------
# the warp and its interpolants
# --------------------------------------------------------------------------
def warp_phi(u, K):
    """phi_K(u) = sum_{k=1}^K sin(2 pi k u)/(pi k): the K-harmonic sawtooth 1/2 - {u}.

    K = 0 is identically 0 (no warp). As K -> infinity, phi_K(u) -> 1/2 - {u}, so the
    warp x + phi_K(x) pins each cell [m, m+1] onto its midpoint m + 1/2.
    """
    if K <= 0:
        return mp.mpf(0)
    return mp.fsum(mp.sin(2 * PI * k * u) / (PI * k) for k in range(1, K + 1))


# ---- fast evaluator: fixed Gauss-Legendre grid + Hurwitz-free moment expansion ----
# A naive int_0^1 zeta(s, 1+u+phi_K) quadrature is correct (see `warp_hurwitz`) but slow:
# phi_K has ~K oscillations, so each call re-evaluates a K-term sine sum at every node and
# a Hurwitz zeta there too, and the migration needs hundreds of calls. Instead fix a GL grid
# once, precompute phi_K (hence psi = u + phi_K) and the logs/moments per K, then evaluate any
# s cheaply via the binomial moment expansion of int_1^inf = sum_{m>=1} int_0^1 (m+psi)^{-s}du:
#   warp_K = sum_{m=1}^M int_0^1 (m+psi)^{-s} du
#            + sum_{j>=0} binom(-s,j) mu_j (zeta(s+j) - sum_{m=1}^M m^{-s-j}),
# mu_j = int_0^1 psi^j du, valid for all sigma (the zeta(s+j) carry the continuation; the
# m>M tail series converges like (max|psi|/M)^j). ~20x faster than the Hurwitz form, agrees
# with it to ~1e-9, exact at K=0 (-> 1/(s-1) to ~1e-29).
#
# High-tau stability. The moment series has three |Im s|-dependent failure modes that
# wreck it above |Im s| ~ 110 at the base dps=30 / fixed grid -- all three fixed by scaling
# the corresponding knob with |Im s| (`_moment_dps` / `_moment_jmax` / `_gl_degree`):
#   * binomial amplification. binom(-s,j) ~ 10^{0.4|s|} grows enormous and multiplies the m>M
#     tail (zeta(s+j)-H_M), which decays past mpmath's *absolute* error floor ~10^{-dps} -- so
#     the floored tail's garbage is amplified into spurious O(10^6) terms. Cured by raising the
#     working precision (`_moment_dps`: ~0.45 guard digit per unit |Im s|).
#   * series truncation. The series peaks at j ~ |Im s|/(M+1) and must run well past the peak
#     to converge, but a fixed cap truncates it. Cured by growing the term count (`_moment_jmax`).
#   * moment-grid resolution. mu_j = int psi^j concentrates near psi_max in a width-~1/K Gibbs
#     spike; high j (weighted more as |Im s| grows) needs a finer grid or mu_j is biased and the
#     sum drifts (~5e-3 at |Im s|=300 with the base 384 nodes -- not a blow-up, but wrong).
#     Cured by raising the GL degree (`_gl_degree`: 8 -> 9 -> ... doubling the nodes each step).
# With all three, warp_K matches warp_hurwitz to its native ~1e-9 (or better) through
# |Im s| ~ 300 and stays accurate (degrading gracefully, no blow-up) well beyond -- slower
# there (~2 s/eval at |Im s|=300) but correct. Below |Im s| = 40 nothing changes (wp=dps,
# Jmax=60, degree=8); the worst case is just under a degree threshold (~2e-8 near |Im s|=150).
_GL_DEGREE = 8        # base GaussLegendre degree: nodes ~ 3*2^(degree-1) = 384 (>= ~8/oscillation to K~45)
_M = 14               # direct cells m=1..M; the moment tail (ratio ~ (1/M)^j) covers m>M
_GUARD_ONSET = 40.0   # |Im s| below which the base dps / Jmax=60 / degree=8 already converge
_GUARD_SLOPE = 0.45   # extra working-precision digits per unit |Im s| (binomial amplification)
_JMAX_SLOPE = 0.15    # extra moment terms per unit |Im s| (the j ~ |Im s|/(M+1) series peak)
_JMAX_BASE = 50       # moment terms beyond the peak for the tail to decay below the break tol
_GL_DEGREE_STEPS = (150.0, 350.0, 700.0)  # |Im s| thresholds bumping the GL degree 8->9->10->11

_GL_CACHE = {}        # (degree, prec) -> (nodes, weights) on [0, 1]
_GRID = {}            # (K, degree, prec) -> (logs[m-1][i] = log(m+psi_i),  mus[j] = sum_i w_i psi_i^j)


def _gl_nodes(degree=_GL_DEGREE):
    """Fixed Gauss-Legendre nodes/weights on [0, 1], cached per (degree, working precision)."""
    key = (degree, mp.mp.prec)
    if key not in _GL_CACHE:
        from mpmath.calculus.quadrature import GaussLegendre
        nw = GaussLegendre(mp.mp).calc_nodes(degree, mp.mp.prec)   # (x, w) on [-1, 1]
        _GL_CACHE[key] = ([(x + 1) / 2 for x, _w in nw], [w / 2 for _x, w in nw])
    return _GL_CACHE[key]


def _moment_dps(s):
    """Working precision for the moment series: base dps + guard digits that grow with |Im s|
    to absorb the binomial-moment cancellation (see the block comment above)."""
    tau = float(abs(mp.im(s)))
    return mp.mp.dps + max(0, int(math.ceil(_GUARD_SLOPE * (tau - _GUARD_ONSET))))


def _moment_jmax(s):
    """Number of binomial-moment terms: the series peaks at j ~ |Im s|/(M+1), so the term
    count must grow with |Im s| or a fixed cap truncates it before it converges."""
    tau = float(abs(mp.im(s)))
    return max(60, int(math.ceil(_JMAX_SLOPE * tau)) + _JMAX_BASE)


def _gl_degree(s):
    """GL quadrature degree: the high-j moments mu_j = int psi^j (psi peaks in a width-~1/K
    Gibbs spike) need a finer grid as |Im s| weights higher j, or the moments bias the sum.
    Degree 8 (~384 nodes) holds to |Im s|~150; each +1 doubles the nodes."""
    tau = float(abs(mp.im(s)))
    return _GL_DEGREE + sum(1 for t in _GL_DEGREE_STEPS if tau > t)


def _moment_logs_mus(us, ws, psis, Jmax):
    """From the GL nodes/weights and the warp's psi node-values (all at the current precision):
    the per-cell logs log(m+psi_i) and the moments mu_j = sum_i w_i psi_i^j, j=0..Jmax. Shared
    by warp_K (psi = u + phi_K) and warp_alpha (psi = u + (alpha-1/2) + phi_K)."""
    logs = [[mp.log(m + p) for p in psis] for m in range(1, _M + 1)]
    mus = [mp.fsum(ws[i] * psis[i] ** j for i in range(len(us))) for j in range(Jmax + 1)]
    return logs, mus


def _moment_series(s, logs, mus, degree, base_dps):
    """The binomial-moment sum int_1^inf (x+phi)^{-s} dx from a precomputed (logs, mus) grid
    built on the GL nodes of `degree`.

    The caller must already be inside `mp.workdps(_moment_dps(s))`. The loop breaks at the
    *ambient* accuracy base_dps (not the inflated working precision): that keeps the
    truncation comfortably above the working-precision floor, so the binomial factor never
    amplifies floored-tail garbage. Shared by warp_K and warp_alpha.warp_alpha.
    """
    us, ws = _gl_nodes(degree)
    n = len(us)
    eps = mp.mpf(10) ** (-(base_dps - 1))
    tot = mp.fsum(ws[i] * mp.e ** (-s * logs[m - 1][i])
                  for m in range(1, _M + 1) for i in range(n))      # sum_{m<=M} int (m+psi)^{-s}
    c = mp.mpf(1)                                                   # binom(-s, 0)
    for j in range(len(mus)):
        HM = mp.fsum(mp.power(m, -(s + j)) for m in range(1, _M + 1))
        term = c * mus[j] * (mp.zeta(s + j) - HM)                  # m>M tail, analytically continued
        tot += term
        if j > 4 and abs(term) < eps * (abs(tot) + 1):
            break
        c = c * (-s - j) / (j + 1)                                 # binom(-s, j+1)
    return tot


def _warp_grid(K, degree, Jmax=60):
    """Precompute (once per (K, degree, working precision)) the per-cell logs and moments mu_j
    up to Jmax, rebuilding with more terms when a higher |Im s| needs a larger Jmax."""
    key = (K, degree, mp.mp.prec)
    cached = _GRID.get(key)
    if cached is None or len(cached[1]) < Jmax + 1:
        us, ws = _gl_nodes(degree)
        psis = [u + warp_phi(u, K) for u in us]
        _GRID[key] = _moment_logs_mus(us, ws, psis, Jmax)
    return _GRID[key]


def warp_K(s, K):
    """warp_K(s, K) = int_1^inf (x + phi_K)^{-s} dx, the fast grid+moment evaluator.

    K = 0 gives 1/(s-1) exactly; K -> infinity gives zeta(s, 3/2) (the cell midpoints
    3/2, 5/2, ... -- the *honest* int_1^inf limit). Valid for all sigma, and for all
    heights: the working precision, moment-term count and grid degree scale with |Im s| so the
    binomial cancellation / truncation / moment-grid bias that broke it above |Im s| ~ 110 no
    longer bite. See the block comment above; `warp_hurwitz` is the transparent cross-check.
    """
    s = mp.mpc(s)
    base_dps = mp.mp.dps
    with mp.workdps(_moment_dps(s)):
        degree = _gl_degree(s)
        logs, mus = _warp_grid(K, degree, _moment_jmax(s))
        result = _moment_series(s, logs, mus, degree, base_dps)
    return +result                                                 # round back to ambient dps


def warp_hurwitz(s, K):
    """The transparent (slow) evaluator: int_0^1 zeta(s, 1+u+phi_K(u)) du.

    The period-by-period collapse of int_1^inf (using sum_{m>=1}(m+a)^{-s} = zeta(s,1+a)).
    Obviously correct but a Hurwitz-zeta eval per quad node; kept as the reference that the
    fast `warp_K` is validated against (they agree to ~1e-9).
    """
    s = mp.mpc(s)
    return mp.quad(lambda u: mp.zeta(s, 1 + u + warp_phi(u, K)), [0, 1])


def warp_half_K(s, K):
    """The endpoint-completed warp:  warp_K(s, K) + 2^s  ->  zeta(s, 1/2) = (2^s-1) zeta(s).

    The +2^s = (1/2)^{-s} restores the dropped m=0 midpoint cell (the half-integer 1/2
    that int_1^inf, starting at x=1, omits). It is the load-bearing half-weight: it turns
    the ugly limit zeta(s, 3/2) into the clean half-shifted zeta(s, 1/2), and only then
    does the sigma=0 prefactor companion (2^s - 1 = 0) exist. This is the migration object.
    """
    s = mp.mpc(s)
    return warp_K(s, K) + mp.power(2, s)


def warp_raw(s, K):
    """warp_K in its original oscillatory guise int_1^inf (x + phi_K)^{-s} dx (sigma>1).

    A coarse *independent* confirmation of the identity (quadosc on a conditionally
    convergent integral -- only ~1e-4-1e-5, vs the grid evaluator's ~1e-9).
    """
    s = mp.mpc(s)
    return mp.quadosc(lambda x: (x + warp_phi(x, K)) ** (-s), [1, mp.inf], period=1)


# --------------------------------------------------------------------------
# the sigma=0 prefactor companion  2^s - 1 = 0  at  s = 2 pi i k / ln 2
# --------------------------------------------------------------------------
def pref_heights(t_max):
    """Heights t = 2 pi k / ln 2 of the prefactor zeros 2^s - 1 = 0 (on sigma = 0)."""
    out = []  # type: List[float]
    k = 1
    while 2 * float(PI) * k / float(LN2) <= t_max:
        out.append(2 * float(PI) * k / float(LN2))
        k += 1
    return out


# K schedule for the migration. Affordable now that `warp_K` is the fast grid evaluator,
# so we run to K=45 (O(1/K) has the trajectories within ~2e-2 of the limit there); capped
# below the additive comb's 200 only because the picture is fully settled by then.
K_SCHEDULE = [1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 27, 35, 45]


# --------------------------------------------------------------------------
# the sigma=1/2 U sigma=0 density bijection -- MEASURED
# --------------------------------------------------------------------------
# The additive comb *counted* the eta split (harmonic_bridge._print_counting): #eta-zeros(<T) = the
# sigma=1/2 zeta zeros + the sigma=1 prefactor zeros = the Riemann-von Mangoldt N(T).
# This is the warp mirror, across sigma=1/2: the warp's target zeta(s,1/2)=(2^s-1)zeta(s)
# carries the genuine zeta zeros on sigma=1/2 and the 2^s-1 companions on sigma=0, and we
# now *find* both families and bin them (rather than asserting the companion line "by the
# functional equation"). The densities sum exactly to the ground density
#   (1/2pi)ln(t/2pi) [zeta, sigma=1/2] + (ln2/2pi) [companion, sigma=0] = (1/2pi)ln(t/pi),
# so N_1/2(T) + N_0(T) reproduces the RvM total to leading order. (The integrated counts
# carry an O(1) offset of 1/4: zeta's RvM constant is 7/8, the continuous-eta counting_law has 5/8 -- a
# constant, not a density, mismatch; the eta side of the additive comb has the identical 1/4 offset.)
def target_half(s):
    """The K -> infinity warp target  zeta(s, 1/2) = (2^s - 1) zeta(s)  (the limit of warp_half_K).

    Its zeros are exactly the bijection's two families: the genuine zeta zeros on sigma = 1/2,
    and the prefactor companions 2^s - 1 = 0 on sigma = 0 (t = 2 pi k/ln2). The cheap, exact
    object the measured count classifies (warp_half_K at finite K lands on it to O(1/K)).
    """
    s = mp.mpc(s)
    return (mp.power(2, s) - 1) * mp.zeta(s)


def count_on_lines(fn, t_max, sig_box=(-0.35, 1.0), t_min=2.0, lines=(0.0, 0.5),
                   tol_line=0.06, dsig=0.25, dt=0.55, **kw):
    """Find zeros of fn in sig_box x [t_min, t_max] and bin each onto the nearest of `lines`.

    Uses warp_alpha.find_zeros (the local-minimum-seeded box finder; imported lazily because
    warp_alpha imports *this* module at load, so a top-level import here would be circular).
    Returns (counts, off, zeros): counts[L] = #zeros within tol_line of line L; off = the
    zeros further than tol_line from every line; zeros = the full (Im, Re)-sorted locus. This
    is the *measured* side of the bijection -- no family is assumed, each zero is classified by
    its own sigma.
    """
    import warp_alpha as wa  # lazy (wa imports wb at load; top-level here would be circular)
    zeros = wa.find_zeros(fn, sig_box, (t_min, t_max), dsig=dsig, dt=dt, **kw)
    counts = {L: 0 for L in lines}
    off = []  # type: List[mp.mpc]
    for z in zeros:
        nearest = min(lines, key=lambda L: abs(float(z.real) - L))
        if abs(float(z.real) - nearest) <= tol_line:
            counts[nearest] += 1
        else:
            off.append(z)
    return counts, off, zeros


# --------------------------------------------------------------------------
# self-validating driver
# --------------------------------------------------------------------------
def _print_counting(T=40.0):
    """Witness the warp density bijection by MEASUREMENT -- the mirror of the additive comb's eta count.

    Find the zeros of the warp target zeta(s,1/2)=(2^s-1)zeta(s) up to height T, bin them onto
    sigma=1/2 (genuine zeta zeros) and sigma=0 (the 2^s-1 companions), and check N_1/2 + N_0
    against the Riemann-von Mangoldt counting_law -- the same total the eta split reproduced
    on sigma=1/2 U sigma=1.
    """
    counts, off, zeros = count_on_lines(target_half, T)
    nhalf, n0 = counts[0.5], counts[0.0]
    rvm = float(ce.counting_law(T))                       # (T/2pi)ln(T/pi e) + 5/8
    n0_law = float(T * LN2 / (2 * PI))                    # (T ln2)/2pi -- the companion density
    print(f"\n== warp density bijection -- MEASURED zeros of (2^s-1)zeta(s)  (0 < t < {T:.0f}) ==")
    print(f"   found {len(zeros)} zeros: {nhalf} on sigma=1/2 (zeta) + {n0} on sigma=0 "
          f"(2^s-1 companion) + {len(off)} off-line")
    print(f"   N_1/2 + N_0 = {nhalf} + {n0} = {nhalf + n0}   vs   RvM N(T) = {rvm:.3f}   "
          f"(densities (1/2pi)ln(t/2pi)+(ln2/2pi)=(1/2pi)ln(t/pi) match exactly)")
    print(f"   sigma=0 companion count {n0} == floor(T ln2/2pi) = {int(n0_law)}  "
          f"(T ln2/2pi = {n0_law:.4f}) -- on the nose:")
    for z in sorted((zz for zz in zeros if abs(float(zz.real)) <= 0.06),
                    key=lambda zz: float(zz.imag)):
        t = float(z.imag)
        k = round(t * float(LN2) / (2 * float(PI)))
        print(f"      t={t:8.4f}  k={k}  2 pi k/ln2={2*float(PI)*k/float(LN2):8.4f}  "
              f"diff={abs(t - 2*float(PI)*k/float(LN2)):.1e}")


def _print_convergence():
    print("== warp_K -> zeta(s,3/2) (honest int_1^inf);  warp_half_K -> zeta(s,1/2)=(2^s-1)zeta ==")
    print("   s                     K    |warp - z(3/2)|   |warp_half - z(1/2)|   drop / 3x K")
    for s in [mp.mpc(2, 0), mp.mpc(0.5, 14.134725), mp.mpc(1.3, 7)]:
        z32 = mp.zeta(s, mp.mpf(3) / 2)
        z12 = mp.zeta(s, mp.mpf(1) / 2)
        prev = None
        for K in [5, 15, 45]:
            ew = float(abs(warp_K(s, K) - z32))
            eh = float(abs(warp_half_K(s, K) - z12))
            rat = "  --" if prev is None else f"{prev/ew:4.1f}x   (1/K -> 3.0x)"
            prev = ew
            print(f"   {complex(s)!s:>18}  {K:3d}   {ew:.3e}        {eh:.3e}        {rat}")
    # endpoint exactness + the load-bearing 2^s
    print("\n   warp_K(s,0) == 1/(s-1)?  ",
          {f"{complex(s):.0f}": float(abs(warp_K(s, 0) - 1 / (s - 1)))
           for s in [mp.mpc(2, 0), mp.mpc(1.3, 7)]})
    sp = mp.mpc(0, 2 * float(PI) / float(LN2))
    print(f"   load-bearing 2^s at s=2 pi i/ln2 (sigma=0, t={float(sp.imag):.4f}):")
    print(f"      |zeta(s,3/2)| = {float(abs(mp.zeta(sp, mp.mpf(3)/2))):.4f} (no zero)  "
          f"|zeta(s,1/2)| = {float(abs(mp.zeta(sp, mp.mpf(1)/2))):.1e} (zero)")


def _print_migration(title, targets, fn=warp_half_K, schedule=None):
    print(f"\n== {title} ==")
    hdrK = [1, 5, 16, 27, 45]
    print("   target zero            " + "".join(f"K={k:<6d}" for k in hdrK))
    trajs = []
    for label, zt in targets:
        traj = hb.migrate(fn, zt, schedule=schedule if schedule is not None else K_SCHEDULE)
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

    _print_convergence()

    zeta_targets = [(f"1/2+{g:.3f}i", mp.mpc(0.5, g))
                    for g in (14.134725, 21.022040, 25.010858)]
    pref_targets = [(f"0+{2*float(PI)*k/float(LN2):.3f}i",
                     mp.mpc(0, 2 * float(PI) * k / float(LN2))) for k in (1, 2, 3)]

    wz = _print_migration("warp_half_K: zeta zeros climb onto sigma=1/2 (from below)",
                          zeta_targets)
    wp = _print_migration("warp_half_K: prefactor companions climb onto sigma=0 (the eta mirror)",
                          pref_targets)
    # comb comparison: the additive bridge's zeta zeros (born high, descend) -- same
    # destination sigma=1/2, opposite approach, and NO sigma=0 companion. (Same reduced
    # schedule as the warp, for a like-for-like K range in the figure.)
    cz = _print_migration("comb zeta_K: zeta zeros born high, descend onto sigma=1/2",
                          zeta_targets, fn=hb.zeta_K, schedule=K_SCHEDULE)
    _print_counting()      # the measured sigma=1/2 U sigma=0 density bijection

    # ---- figure: three panels ----
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    def _traj_xy(traj):
        pts = [(K, zz) for K, zz in traj if zz is not None]
        return ([float(zz.real) for _, zz in pts], [float(zz.imag) for _, zz in pts])

    # panel 0: warp zeta zeros (climb from below) vs comb (descend from above)
    for label, _zt, traj in wz:
        xs, ys = _traj_xy(traj)
        if xs:
            ax[0].plot(xs, ys, "-o", ms=3, lw=1, color="C0")
            ax[0].plot(xs[-1], ys[-1], "k.", ms=9)
    for label, _zt, traj in cz:
        xs, ys = _traj_xy(traj)
        if xs:
            ax[0].plot(xs, ys, "--s", ms=3, lw=1, color="C1", mfc="none")
    ax[0].plot([], [], "-o", color="C0", label="warp (climbs from below)")
    ax[0].plot([], [], "--s", color="C1", mfc="none", label="comb (descends)")
    ax[0].axvline(0.5, ls="-", lw=1, color="C2", label=r"critical line $1/2$")
    ax[0].set_title(r"$\zeta$ zeros $\to\ \sigma=1/2$: warp vs comb")
    ax[0].set_xlim(-0.1, 1.5)

    # panel 1: prefactor companions -> sigma=0
    for label, _zt, traj in wp:
        xs, ys = _traj_xy(traj)
        if xs:
            ax[1].plot(xs, ys, "-o", ms=3, lw=1, color="C3")
            ax[1].plot(xs[-1], ys[-1], "k.", ms=9)
    ax[1].axvline(0.0, ls="-", lw=1, color="C4", label=r"$\sigma=0$ ($2^s-1$)")
    ax[1].axvline(1.0, ls="--", lw=1, color="C1", label=r"$\sigma=1$ ($\eta$ mirror)")
    ax[1].set_title(r"prefactor companions $\to\ \sigma=0$ (the $\eta$ mirror)")
    ax[1].set_xlim(-0.6, 1.2)

    for a in (ax[0], ax[1]):
        a.set_xlabel(r"$\sigma=\mathrm{Re}\,s$")
        a.set_ylabel(r"$t=\mathrm{Im}\,s$")
        a.legend(loc="upper right")

    # panel 2: convergence O(1/K) -- warp and comb the same rate
    s0 = mp.mpc(0.5, 14.134725)
    z32, zt = mp.zeta(s0, mp.mpf(3)/2), mp.zeta(s0)
    Ks = [2, 4, 8, 16, 32, 45]
    ew = [float(abs(warp_K(s0, K) - z32)) for K in Ks]
    ec = [float(abs(hb.zeta_K(s0, K) - zt)) for K in Ks]
    ax[2].loglog(Ks, ew, "-o", color="C0", label=r"warp $|{-}\zeta(s,3/2)|$")
    ax[2].loglog(Ks, ec, "--s", color="C1", mfc="none", label=r"comb $|{-}\zeta(s)|$")
    ax[2].loglog(Ks, [ew[0] * Ks[0] / K for K in Ks], ":", color="0.5", label=r"$\propto 1/K$")
    ax[2].set_xlabel("K"); ax[2].set_ylabel(r"error at $s=1/2+14.13i$")
    ax[2].set_title(r"convergence: both $O(1/K)$")
    ax[2].legend()

    fig.suptitle(r"$n^*$-warp vs harmonic comb: the half-shifted $\zeta(s,1/2)=(2^s-1)\zeta(s)$ "
                 r"and its $\sigma=0$ companion")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    figdir = Path(__file__).resolve().parent / "figures"
    figdir.mkdir(exist_ok=True)
    out = figdir / "warp_bridge.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    _main()
