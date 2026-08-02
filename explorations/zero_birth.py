r"""Zero birth from the identically-zero endpoint: normalizing the character K-family.

EXPERIMENTAL (issue #39, sub-issue of #37). Lives in `explorations/`, disjoint from
the frozen manuscript arc; run its tests with `pytest explorations/tests`; the driver
is local/manual only (minutes; never in CI). Experimental-mathematics framing;
**no RH/GRH claims**.

The question this module owns (issue #39, question 1)
-----------------------------------------------------
character_bridge.py established the projector: under the DC-in-knob convention every
warp component's K = 0 endpoint is the same bland integral 1/(s-1), and orthogonality
(sum_a chi(a) = 0) kills the combination identically -- a non-principal L-function is
born from the ZERO FUNCTION. But the zero function has no zero set, so "where are the
zeros born?" needs a normalization that makes the K -> 0 limit of the normalized
family well-defined. Which one is right?

The answer (measured here, algebra below):

  1. For K >= 1 the DC-in-knob and DC-always-on families COINCIDE -- the conventions
     only disagree about the K = 0 object -- so the raw K >= 1 family is
     convention-free, and any *scalar* normalizer (per-K L2 norm, amplitude at a
     reference point) is nonvanishing and moves no zeros. The whole question lives at
     the endpoint (the #38-session comment on issue #39, point 1).
  2. The right formalization of "K -> 0" is a knob AMPLITUDE lambda scaling the
     entire discreteness datum -- DC offset (alpha - 1/2), completion cell
     alpha^{-s}, and harmonics phi_K -- at fixed K:

         F_K^lam(s) = q^{-s} sum_a chi(a) [ int_1^inf (x + lam((a/q - 1/2)
                        + phi_K(x)))^{-s} dx + lam (a/q)^{-s} ].

     F_K^0 == 0 for every non-principal chi (the projector), and the normalized
     family F_K^lam / lam has an honest lam -> 0 limit, THE SEED OBJECT:

         D(s) = sum_a chi(a) a^{-s} + q^{-s} L(0, chi),
         L(0, chi) = -(1/q) sum_a chi(a) a       (classical; 0 for even chi).

     Derivation: differentiate at lam = 0. The phi_K harmonics and the -1/2 of the
     DC offset enter the derivative alpha-INDEPENDENTLY, so orthogonality kills
     them; the surviving linear content is the completion cells (-> the one-period
     section sum chi(a) a^{-s}) plus the DC ramp (-> q^{-s} L(0, chi)). Every K
     gives the SAME tangent: the K-families all sprout from one seed set. By
     Hurwitz's theorem (the complex-analysis one) the zeros of F_K^lam converge to
     the zeros of D as lam -> 0, so the normalized zero set is well-defined and
     K-independent at birth.
  3. The amplitude embedding is canonical, not just convenient: the natural rival
     (morph the completion cell's LOCATION from x = 1 to x = alpha instead of its
     amplitude) has tangent (1+s) q^{-s} L(0, chi) -- identically ZERO for even chi
     and zero-free-but-for s = -1 for odd chi. A degenerate seed either way. The
     amplitude path is the one with a non-degenerate, K-independent tangent for
     every non-principal chi.

Seed geometry: a wobbling sigma = 0 comb (and the section skeleton B)
---------------------------------------------------------------------
Write B(s) = sum_a chi(a) a^{-s} (the one-period section). For the two-unit moduli
(phi(q) = 2: q = 3, 4, 6, units {1, p} with p = q - 1) B = 1 - p^{-s}, whose zeros
are EXACTLY the sigma = 0 comb t_m = 2 pi m / ln p. The seed D = B + q^{-s} L(0,chi)
perturbs it: to first order in the wobble parameter L(0,chi)/ln p,

    seed_m  ~  i t_m  -  q^{-i t_m} L(0, chi) / ln p,

a comb displaced by a CONSTANT-modulus, equidistributing-phase wobble (radius
L0/ln p ~ 0.48, 0.46, 0.41 for q = 3, 4, 6; phases 2 pi m ln q / ln p mod 2 pi,
equidistributed since ln q / ln p is irrational). So for a non-principal character
the zeros are seeded in a radius-~0.45 annulus AROUND SIGMA = 0 -- where zeta's comb
zeros are born near sigma ~ 1 (harmonic_bridge: K = 1 creates them at Re ~ 1-1.35
from the pole side). The left-edge-vs-right-edge contrast is issue #37 prediction
3's "qualitatively different birth story", made concrete.

For ODD chi there is a second, sparser seed string (`conductor_string`): D is an
exponential polynomial with frequencies {ln a} + {ln q}, so its mean-motion zero
density is ln q / 2 pi -- MORE than the main comb's ln p / 2 pi -- and the surplus
ln(q/p) / 2 pi lives where the two fastest terms balance, sigma* = ln L0 / ln(q/p)
(q = 3: sigma* = -2.71, spacing 2 pi/ln(3/2) = 15.5; measured -2.87 + 15.21i, the
wobble from the neglected constant term). The conductor frequency enters D ONLY
through the DC-ramp term q^{-s} L(0, chi), so the far-left string is the
conductor's own signature in the seeds; even chi (L0 = 0) have no such string.

The section family ties the module together: the comb ground state is
E_chi = B(s-1)/(q(s-1)) + B(s)/2 (character_bridge.comb_endpoint), the warp seed is
D = B(s) + q^{-s} L(0,chi), and the rate constant is (sq/2 pi^2) B(s+1)
(rate_comb_chi) -- the K-knob's ground-and-rate data is the one-period section at
three consecutive arguments. At large t the pole-weighted term of E_chi decays like
p/(q|s|), so E_chi -> B/2 and the COMB ground string collapses onto B's zero set
(rate ~ 1/t), while the WARP seeds orbit it at constant radius: both routes' ground
strings live on the same section skeleton, decorated differently.

Birth bookkeeping without normalization (issue #39, question 2)
---------------------------------------------------------------
The #38 walk validated the smooth count N(T) = theta_chi(T)/pi + S(T) for entire L
(NO +1 -- zeta's +1 is the pole's argument-principle term). Every non-principal
finite-K interpolant is entire too (the Hurwitz poles cancel by orthogonality), so
birth-K is bookkept as a box census N_K(T) computed by the argument principle
(adaptive phase-winding around the box -- `winding_count`), against the
K = infinity target. Structurally, the K = 0 / lam -> 0 ground objects are
length-q Dirichlet POLYNOMIALS with BOUNDED zero density (ln(q-1)/2 pi, resp.
ln q/2 pi with the conductor string), while the target needs
(T/2 pi) ln(qT/2 pi e): a ln T surplus separates ground from target. The census
MEASURES where that surplus appears, and the answer is: instantly. In the
standard box (t < 30.29) the chi_3 census reads ground 3 -> N_K = 8 = target at
EVERY K >= 1, both routes -- the first harmonic already carries the full
Riemann-von Mangoldt count (its incomplete-gamma moments are not almost-periodic,
so the bounded-density argument stops applying the moment K = 1), and the K-knob
from there on is pure O(1/K) displacement, zero births none. GEOMETRIC BIRTH IS A
lam PHENOMENON AT K = 1, resolvable only by the amplitude knob -- which is what
makes the lam-flow above the birth instrument, not a convenience. Contrast zeta
(ground = pole, no zeros; its comb census reads 4 -> 3 = target: one transient
extra zero DIES at K = 2, the rest of the count is instant too) and eta (ground
density already matches: pure bijection). The three ground stories differ --
pole / zero function / matched string -- but the K-knob's census behavior is
shared: the count completes at K ~ 1 and the knob spends the rest of its range
moving zeros, not making them.

The displacement law survives chi-weighting: at a simple zero rho of L(s, chi),

    s_K - rho  ~  a_1 / K,   a_1 = +rate_comb_chi(rho, chi) / L'(rho, chi)

(`disp_coeff_chi`, the rate_law.disp_coeff machinery: a_1 = -Delta_1/L' with
Delta_1 = lim K (L_comb_K - L) = -rate_comb_chi, the chi comb constant being
quoted target-minus-object), verified against measured comb roots at K = 25..45.

Ground-string geometry (issue #39, question 3)
----------------------------------------------
For phi(q) = 2 the E_chi zero condition collapses to p^{-s} = Moebius(s) (unimodular
precisely on sigma = 0; character_bridge docstring), which on the axis is ONE real
phase equation: t ln p + pi - 2 arctan(qt/(q-2)) = 2 pi m (`axis_roots`, solved by
bisection -- the whole ground string in closed form up to 1-D root-finding). The
argument-principle census in a box around the strip then CERTIFIES the claim "the
full zero set is on the axis": box count == axis-root count (`certify_axis`).
For phi(q) > 2 the reflection is gone and the ground zeros scatter off the axis --
but they stay on the section skeleton: E_chi zeros approach B's zeros at rate ~ 1/t,
with the scatter width set by the section, not by q (measured across
q = 5, 7, 9 in the driver).

chi_6, the imprimitive case: does the lock pre-exist? (#38-session comment, point 4)
------------------------------------------------------------------------------------
chi_6's ground string sits exactly on sigma = 0 -- which is also where its target
Euler comb (1 + 2^{-s}, teeth at (2m+1) pi / ln 2) lives, and #38 measured the
finished comb parked at the primitive zeros' density minima (the anti-phase lock).
The migration (K = 0 ground -> sigma = 1/2 union comb) answers whether the
comb-bound ground zeros already sit at the locked phase at K = 0 or drift there.
Measured answer: THE LOCK IS NOT PRE-EXISTING. The comb-bound ground zeros sit
at offsets -0.85, +1.97, +0.73 from their final teeth (vs the 3.90 axis-root
spacing) -- large fractions of a spacing, in both directions: the anti-phase
lock develops ALONG the migration, it is not inherited from the ground geometry.
The walk also witnesses the census section's arithmetic: 7 targets below t ~ 24
reach only 6 axis roots (two walks double-land on t = 15.563) -- the
bounded-density ground under-supplies the union by exactly the born surplus, so
the backward continuation cannot be injective (eta's bijection has no chi_6
analog; near-collisions at small K let trajectories merge).

Honest framing
--------------
Experimental mathematics; no RH/GRH claims. L(0, chi) = -(1/q) sum chi(a) a is
classical (finite-sum special values), Hurwitz's zero-convergence theorem is
textbook complex analysis, and Dirichlet-polynomial zero strips are classical
(Bohr almost-periodicity). What has no published analog we know of (per the #37
adversarial lit pass) is the finite-K deformation itself -- here specifically: the
identically-zero endpoint's tangent-seed normalization, the K-independent seed set,
and the census-measured migrated-backbone + born-surplus decomposition.

Run directly to validate + plot: writes explorations/figures/zero_birth.png.
Runtime ~10 minutes (the lam-flow and the census dominate); local/manual only
(#37's CI-economy note).
"""
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mpmath as mp

# Path shim: explorations/ module cross-importing the flat bridge/ sources plus its
# sibling character_bridge (explorations/conftest.py does the same for pytest).
_HERE = Path(__file__).resolve().parent
_BRIDGE = _HERE.parent / "bridge"
for _p in (str(_HERE), str(_BRIDGE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import character_bridge as cb  # noqa: E402  (characters, L_chi, combs, endpoints)
import harmonic_bridge as hb   # noqa: E402  (zeta_K, migrate)
import warp_alpha as wa        # noqa: E402  (find_zeros)
import warp_bridge as wb       # noqa: E402  (the shared GL grid + moment machinery)

mp.mp.dps = 30

_J = mp.j
PI = mp.pi


# --------------------------------------------------------------------------
# 1. the section family and the seed object
# --------------------------------------------------------------------------
def section_B(s, chi):
    """B(s) = sum_a chi(a) a^{-s}: the one-period section, the ground skeleton.

    A length-q Dirichlet polynomial; for phi(q) = 2 it is 1 - p^{-s} (p = q - 1),
    zeros exactly the sigma = 0 comb 2 pi m / ln p. Appears at three consecutive
    arguments across the knob's ground data: E_chi = B(s-1)/(q(s-1)) + B(s)/2,
    D = B(s) + q^{-s} L(0,chi), rate_comb_chi = (sq/2 pi^2) B(s+1).
    """
    s = mp.mpc(s)
    return mp.fsum(chi(a) * mp.power(a, -s) for a in cb.units(chi.q))


def L0_chi(chi):
    """L(0, chi) = -(1/q) sum_a chi(a) a (classical finite-sum special value).

    Zero for even non-principal chi (the s = 0 trivial zero); the odd-chi values
    feed the seed's wobble: L0/ln(q-1) is the seed comb's displacement radius.
    """
    q = chi.q
    return -mp.fsum(chi(a) * a for a in cb.units(q)) / q


def seed_D(s, chi):
    """The seed object D(s) = B(s) + q^{-s} L(0, chi): the lam -> 0 tangent of the
    amplitude-normalized DC-in-knob family, INDEPENDENT of K (orthogonality kills
    the alpha-independent first-order content: harmonics and the -1/2). The zeros
    of D are where the non-principal K-family's zeros are born."""
    s = mp.mpc(s)
    return section_B(s, chi) + mp.power(chi.q, -s) * L0_chi(chi)


def seed_prediction(chi, m):
    """First-order seed position at tooth m (phi(q) = 2 moduli only):

        seed_m ~ i t_m - q^{-i t_m} L0 / ln p,   t_m = 2 pi m / ln p

    -- the sigma = 0 comb displaced by a constant-modulus (L0/ln p), rotating-phase
    wobble. Accurate to O((L0/ln p)^2); the driver measures the residual.
    """
    p = chi.q - 1
    lnp = mp.log(p)
    tm = 2 * PI * m / lnp
    return mp.mpc(0, tm) - mp.power(chi.q, -mp.mpc(0, tm)) * L0_chi(chi) / lnp


def conductor_string(chi):
    """(sigma*, spacing) of the CONDUCTOR seed string (odd phi(q) = 2 chi).

    D is a length-3 exponential polynomial with frequencies {0, ln p, ln q}: by
    the classical mean-motion count its total zero density is ln q / 2 pi -- MORE
    than the main wobbling comb's ln p / 2 pi. The surplus ln(q/p) / 2 pi lives in
    a sparse second string where the two fastest terms balance,
    |chi(p) p^{-s}| = |q^{-s} L0|:

        sigma* = ln L0 / ln(q/p),      spacing 2 pi / ln(q/p)

    (q = 3: sigma* = -2.71, spacing 15.5) -- the conductor frequency ln q, present
    only through the DC-ramp term q^{-s} L(0, chi), carries its own far-left seed
    string. For even chi L0 = 0: no conductor string, density ln(q-1)/2 pi.
    """
    p = chi.q - 1
    lnr = mp.log(mp.mpf(chi.q) / p)
    return float(mp.log(L0_chi(chi)) / lnr), float(2 * PI / lnr)


def rival_tangent(s, chi):
    """The location-morphing rival's tangent (1+s) q^{-s} L(0, chi): the degenerate
    seed -- identically 0 for even chi, zero-free but for s = -1 (the bland
    endpoint's lone zero) for odd chi. Why amplitude scaling is the right
    normalization: it is the embedding whose tangent carries a genuine zero string
    for EVERY non-principal chi."""
    s = mp.mpc(s)
    return (1 + s) * mp.power(chi.q, -s) * L0_chi(chi)


# --------------------------------------------------------------------------
# 2. the amplitude family F_K^lam -- the fast warp evaluator, lambda-scaled
# --------------------------------------------------------------------------
# warp_bridge's grid+moment machinery with psi = u + lam*((alpha - 1/2) + phi_K(u)):
# identical to warp_alpha._grid but keyed on lam too. lam = 1 IS the raw warp family
# (character_bridge.L_warp_K); lam -> 0 is the projector limit.
_GRID_LAM = {}   # (K, lam, alpha, degree, prec) -> (logs, mus)


def _grid_lam(K, lam, alpha, degree, Jmax=60):
    key = (K, round(float(lam), 12), round(float(alpha), 12), degree, mp.mp.prec)
    cached = _GRID_LAM.get(key)
    if cached is None or len(cached[1]) < Jmax + 1:
        us, ws = wb._gl_nodes(degree)
        c = mp.mpf(alpha) - mp.mpf("0.5")
        lam = mp.mpf(lam)
        psis = [u + lam * (c + wb.warp_phi(u, K)) for u in us]
        _GRID_LAM[key] = wb._moment_logs_mus(us, ws, psis, Jmax)
    return _GRID_LAM[key]


def warp_component_lam(s, K, lam, alpha):
    """int_1^inf (x + lam((alpha - 1/2) + phi_K(x)))^{-s} dx, all sigma, via the
    binomial-moment expansion (warp_bridge machinery; inherits the |Im s|-scaled
    precision knobs). lam = 1 equals warp_alpha.warp_alpha; lam = 0 is 1/(s-1)."""
    s = mp.mpc(s)
    base_dps = mp.mp.dps
    with mp.workdps(wb._moment_dps(s)):
        degree = wb._gl_degree(s)
        logs, mus = _grid_lam(K, lam, alpha, degree, wb._moment_jmax(s))
        result = wb._moment_series(s, logs, mus, degree, base_dps)
    return +result


def F_lam(s, K, lam, chi):
    """The amplitude family F_K^lam = q^{-s} sum_a chi(a) [component + lam alpha^{-s}].

    lam = 1: the raw (convention-free) warp family == character_bridge.L_warp_K.
    lam = 0: identically 0 for non-principal chi (the projector). The completion
    cell rides the SAME amplitude as the rest of the discreteness datum.
    """
    s = mp.mpc(s)
    q = chi.q
    lam = mp.mpf(lam)
    return mp.power(q, -s) * mp.fsum(
        chi(a) * (warp_component_lam(s, K, lam, mp.mpf(a) / q)
                  + lam * mp.power(mp.mpf(a) / q, -s)) for a in cb.units(q))


def G_lam(s, K, lam, chi):
    """The normalized family F_K^lam / lam (== seed_D at lam = 0). Scalar
    normalization: same zeros as F for lam > 0; its whole content is the limit."""
    if lam == 0:
        return seed_D(s, chi)
    return F_lam(s, K, lam, chi) / mp.mpf(lam)


# lam schedule for the birth flow: dense near 0 where the motion is fast.
LAM_SCHEDULE = [1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05, 0.03,
                0.02, 0.012]
_RESID_OK = 1e-6      # acceptance residual: the fast evaluator's dps-15 noise
#                       floor is ~5e-9, and |G'| = O(0.1..1) localizes the zero
#                       to ~1e-5 from a 1e-6 residual -- ample for tracking


def _clamped_root(fn, z, max_jump, ftol):
    """findroot with an escape clamp: a diverging secant iterate out at large
    |Im s| is EXPENSIVE (the moment machinery inflates its precision knobs with
    |Im s|), so any iterate leaving the 3*max_jump neighborhood aborts the solve
    immediately via ValueError instead of wandering. Returns the root or None."""
    lim = 3 * max_jump

    def g(s):
        if abs(s - z) > lim:
            raise ValueError("iterate escaped")
        return fn(s)

    try:
        zz = mp.findroot(g, z, tol=ftol, maxsteps=25)
    except (ValueError, ZeroDivisionError):
        return None
    if abs(zz - z) > max_jump or abs(fn(zz)) > _RESID_OK:
        return None
    return mp.mpc(zz)


def _flow_step(z, K, chi, lam_from, lam_to, max_jump, ftol, depth=3):
    """One lam step of the flow, with adaptive substepping: on failure (jump,
    escape, or non-convergence -- e.g. through a near-collision where the motion
    outruns the schedule) insert the geometric-mean lam and recurse."""
    fn = (lambda s: G_lam(s, K, lam_to, chi)) if lam_to > 0 \
        else (lambda s: seed_D(s, chi))
    zz = _clamped_root(fn, z, max_jump, ftol)
    if zz is not None or depth == 0:
        return zz
    lam_mid = math.sqrt(lam_from * lam_to) if lam_to > 0 else lam_from / 3
    zm = _flow_step(z, K, chi, lam_from, lam_mid, max_jump, ftol, depth - 1)
    if zm is None:
        return None
    return _flow_step(zm, K, chi, lam_mid, lam_to, max_jump, ftol, depth - 1)


def flow_lambda(z0, K, chi, schedule=None, max_jump=2.0, workdps=15):
    """Trajectory of a zero of the raw K-family as the knob amplitude lam -> 0.

    Seeds findroot at lam = 1 from z0 (a zero of the raw family), then walks lam
    down the schedule, each solve seeded by the previous root -- harmonic_bridge's
    `migrate`, with lam for K, plus escape clamps and adaptive lam-substepping.
    Returns [(lam, s or None)] in DEscending lam; ends with (0.0, seed) resolved
    on the seed object D itself. A None entry ends the trajectory (honest loss).
    """
    sched = sorted(schedule if schedule is not None else LAM_SCHEDULE, reverse=True)
    z = mp.mpc(z0)
    out = []  # type: List[Tuple[float, Optional[mp.mpc]]]
    with mp.workdps(workdps):
        ftol = mp.mpf(10) ** (-(workdps - 4))
        lam_prev = None
        for lam in sched + [0.0]:
            if lam_prev is None:        # first entry: a direct polish from z0
                zz = _clamped_root(lambda s: G_lam(s, K, lam, chi), z,
                                   max_jump, ftol)
            else:
                zz = _flow_step(z, K, chi, lam_prev, lam, max_jump, ftol)
            out.append((lam, zz))
            if zz is None:
                break
            z, lam_prev = zz, lam
    return out


# --------------------------------------------------------------------------
# 3. the census instrument: argument-principle box counts by phase winding
# --------------------------------------------------------------------------
def winding_count(fn, sig_range, t_range, dsig=0.35, dt=0.5, workdps=12,
                  refine=1.0, max_depth=16):
    """Zeros of fn (analytic, pole-free in the box) inside sig_range x t_range, by
    the argument principle: walk the rectangle boundary, accumulate the unwrapped
    phase, adaptively bisecting any segment whose phase step exceeds `refine`
    radians (up to max_depth splits) so no winding is missed. Returns the integer
    count; raises if the total fails to close to an integer multiple of 2 pi
    (a zero ON the contour, or undersampling)."""
    (slo, shi), (tlo, thi) = sig_range, t_range

    def _edge_points(za, zb):
        n = max(2, int(mp.ceil(abs(zb - za) / (dsig if za.imag == zb.imag else dt))))
        return [za + (zb - za) * i / n for i in range(n)]

    corners = [mp.mpc(slo, tlo), mp.mpc(shi, tlo), mp.mpc(shi, thi), mp.mpc(slo, thi)]
    with mp.workdps(workdps):
        pts = []
        for i in range(4):
            pts.extend(_edge_points(corners[i], corners[(i + 1) % 4]))
        vals = [fn(z) for z in pts]
        total = mp.mpf(0)
        npts = len(pts)
        for i in range(npts):
            za, zb = pts[i], pts[(i + 1) % npts]
            fa, fb = vals[i], vals[(i + 1) % npts]
            stack = [(za, zb, fa, fb, 0)]
            while stack:
                a, b, fa_, fb_, depth = stack.pop()
                dph = mp.arg(fb_ / fa_)
                if abs(dph) <= refine or depth >= max_depth:
                    total += dph
                else:
                    m = (a + b) / 2
                    fm = fn(m)
                    stack.append((m, b, fm, fb_, depth + 1))
                    stack.append((a, m, fa_, fm, depth + 1))
    n = float(total / (2 * PI))
    if abs(n - round(n)) > 0.15:
        raise RuntimeError(f"winding failed to close: {n:.3f} (zero on contour?)")
    return int(round(n))


def theta_chi(t, chi):
    """theta_chi(t) = Im log Gamma((1/2 + kappa + it)/2) + (t/2) ln(q/pi): the
    completed-L phase for primitive chi. Smooth zero count for ENTIRE L:
    N(T) ~ theta_chi(T)/pi + S(T), with NO +1 (zeta's +1 is the pole's
    argument-principle term -- the #38 counting-convention lesson)."""
    k = chi.parity
    t = mp.mpf(t)
    return (mp.im(mp.loggamma((mp.mpf("0.5") + k + _J * t) / 2))
            + t / 2 * mp.log(mp.mpf(chi.q) / PI))


# --------------------------------------------------------------------------
# 4. the displacement law, chi-weighted
# --------------------------------------------------------------------------
def disp_coeff_chi(rho, chi):
    """a_1 = -Delta_1/L' = +rate_comb_chi(rho, chi) / L'(rho, chi): the comb-route
    zero-displacement coefficient at a simple zero rho of L(s, chi) --
    rate_law.disp_coeff with Delta_1 = lim K (L_comb_K - L) = -(sq/2 pi^2)
    B(rho+1) (rate_comb_chi is quoted target-minus-object). s_K - rho ~ a_1/K."""
    rho = mp.mpc(rho)
    Lp = mp.diff(lambda z: cb.L_chi(z, chi), rho)
    return cb.rate_comb_chi(rho, chi) / Lp


def measured_disp(rho, chi, K, workdps=18):
    """K (s_K - rho) for the comb family: findroot L_comb_K near rho. The measured
    side of the displacement law; -> disp_coeff_chi as K grows."""
    with mp.workdps(workdps):
        sK = mp.findroot(lambda s: cb.L_comb_K(s, K, chi), mp.mpc(rho))
    return K * (sK - mp.mpc(rho))


# --------------------------------------------------------------------------
# 5. ground-string geometry: exact axis roots (phi(q)=2) + certification
# --------------------------------------------------------------------------
def axis_roots(q, t_max, t_min=0.0):
    """The phi(q) = 2 ground string, in closed form up to 1-D bisection.

    E_chi = 0 collapses to p^{-s} = (qs - (q-2))/(qs + (q-2)) (character_bridge
    docstring); on s = it both sides are unimodular and the condition is the ONE
    real phase equation

        u(t) = t ln p + pi - 2 arctan(qt/(q-2)) = 2 pi m,  m = 1, 2, ...

    (u(0) = pi, u increasing for t past its early dip, one root per m). Returns
    the roots in (t_min, t_max].
    """
    p = q - 1
    lnp = mp.log(p)

    def u(t):
        return t * lnp + PI - 2 * mp.atan(q * t / (q - 2))

    roots = []
    m = 1
    t_hi_cap = mp.mpf(t_max) + 3
    while True:
        target = 2 * PI * m
        lo, hi = mp.mpf(0), mp.mpf(1)
        while u(hi) < target:
            hi *= 2
            if hi > t_hi_cap * 4:
                break
        r = mp.findroot(lambda t: u(t) - target, (lo + hi) / 2, solver="secant")
        if r > t_max:
            break
        if r > t_min:
            roots.append(r)
        m += 1
    return roots


def certify_axis(chi, t_range, sig_half_width=2.0, **kw):
    """Certify 'the FULL ground zero set is on sigma = 0' in a box: the argument-
    principle count of E_chi over [-w, w] x t_range must equal the number of axis
    roots there. Returns (box_count, axis_count)."""
    n_box = winding_count(lambda s: cb.comb_endpoint(s, chi),
                          (-sig_half_width, sig_half_width), t_range, **kw)
    n_axis = len(axis_roots(chi.q, t_range[1], t_range[0]))
    return n_box, n_axis


def ground_scatter(chi, sig_range=(-2.5, 2.5), t_range=(0.5, 40.0), **kw):
    """E_chi zeros in a box (any q) + the section-skeleton match: for each ground
    zero, the distance to the nearest zero of B(s). B is searched on a WIDER
    sigma window (its far-left strings sit beyond E's box for larger phi(q), and
    a truncated skeleton fakes huge distances). Returns (zeros, skeleton_zeros,
    dists) -- the driver reports the scatter stats and the ~1/t skeleton collapse."""
    ez = wa.find_zeros(lambda s: cb.comb_endpoint(s, chi), sig_range, t_range, **kw)
    b_sig = (sig_range[0] - 3.5, sig_range[1] + 1.0)
    b_t = (max(0.5, t_range[0] - 2.0), t_range[1] + 2.0)
    bz = wa.find_zeros(lambda s: section_B(s, chi), b_sig, b_t, **kw)
    dists = [min(abs(z - w) for w in bz) if bz else mp.inf for z in ez]
    return ez, bz, dists


def migrate_robust(fn, z_true, schedule, max_jump=6.0, workdps=20):
    """harmonic_bridge.migrate with a robust root step, for the chi_6 lock walk.

    mpmath's plain-secant findroot defaults its second iterate to x0 + 1/4 --
    a quarter-strip hop that can stall between the closely packed near-axis
    zeros here (measured: the K = 45 -> 35 tooth step dies while the same solve
    from a local seed triple converges). Muller with a tight local triple, at
    reduced working precision, with an explicit residual acceptance. Same return
    convention as hb.migrate: [(K, s or None)] ascending, None ends the walk.
    """
    sched = sorted(schedule)
    z = mp.mpc(z_true)
    out = []  # type: List[Tuple[int, Optional[mp.mpc]]]
    h = mp.mpc("1e-3", "1e-3")
    with mp.workdps(workdps):
        ftol = mp.mpf(10) ** (-(workdps - 6))
        for K in reversed(sched):
            try:
                zz = mp.findroot(lambda s: fn(s, K), (z, z + h, z - mp.conj(h)),
                                 solver="muller", maxsteps=30)
            except (ValueError, ZeroDivisionError):
                out.append((K, None))
                break
            if (abs(zz - z) > max_jump or zz.imag <= 1
                    or abs(fn(zz, K)) > ftol * 1000):
                out.append((K, None))
                break
            z = zz
            out.append((K, mp.mpc(zz)))
    out.reverse()
    return out


# --------------------------------------------------------------------------
# 6. driver
# --------------------------------------------------------------------------
CENSUS_BOX = ((-0.97, 2.03), (0.53, 30.29))    # off-round edges dodge contour zeros
CENSUS_KS = [1, 2, 3, 5, 9, 15, 25, 45]


def _print_seed_tangent():
    print("== the seed object: F_K^lam / lam -> D(s), K-independently ==")
    s = mp.mpc(0.5, 6)
    for chi in (cb.CHI3, cb.CHI5_I):
        Dv = seed_D(s, chi)
        print(f"   {chi.label:<14} D({complex(s):.1f}) = {complex(Dv):.6f}")
        for K in (1, 3):
            errs = [float(abs(G_lam(s, K, lam, chi) - Dv))
                    for lam in (mp.mpf("0.01"), mp.mpf("0.001"))]
            print(f"      K={K}:  |F/lam - D| at lam=1e-2, 1e-3:  "
                  f"{errs[0]:.2e}, {errs[1]:.2e}   (ratio {errs[0]/errs[1]:.1f} "
                  f"~ 10: linear in lam)")
    # lam = 1 is the raw warp family (convention-free for K >= 1)
    s2 = mp.mpc(2, 1)
    worst = max(float(abs(F_lam(s2, K, 1, cb.CHI3) - cb.L_warp_K(s2, K, cb.CHI3)))
                for K in (1, 4))
    print(f"   lam=1 == character_bridge.L_warp_K (raw family): worst |diff| = "
          f"{worst:.1e}")
    # the rival tangent is degenerate
    print(f"   rival (cell-location) tangent: even chi_5 quad L0 = "
          f"{float(abs(L0_chi(cb.CHI5_QUAD))):.1e} (tangent == 0); "
          f"odd chi_3 tangent zero-free but s=-1")


def _print_seed_geometry():
    print("\n== seed geometry: wobbling sigma = 0 comb + conductor string "
          "(phi(q) = 2) ==")
    print("   q   tooth m   seed (measured)        1st-order prediction   |resid|")
    seeds = {}
    for chi in (cb.CHI3, cb.CHI4):
        zs = wa.find_zeros(lambda s, c=chi: seed_D(s, c), (-4.5, 1.5), (2.0, 40.0),
                           dsig=0.25, dt=0.5)
        main = [z for z in zs if float(z.real) > -1.5]
        cond = [z for z in zs if float(z.real) <= -1.5]
        seeds[chi.label] = (main, cond)
        for m, z in enumerate(main, start=1):
            pred = seed_prediction(chi, m)
            print(f"   {chi.q}   {m:>5}     {complex(z):.3f}    "
                  f"{complex(pred):.3f}   {float(abs(z - pred)):.3f}")
        sig_c, spac = conductor_string(chi)
        print(f"   q={chi.q} conductor string (pred sigma* = {sig_c:.2f}, spacing "
              f"{spac:.1f}): measured "
              + (", ".join(f"{complex(z):.3f}" for z in cond) or "(none in box)"))
    r3 = float(L0_chi(cb.CHI3) / mp.log(2))
    print(f"   wobble radius L0/ln p: q=3 -> {r3:.3f}; measured max |Re seed| = "
          f"{max(abs(float(z.real)) for z in seeds[cb.CHI3.label][0]):.3f}")
    print("   density bookkeeping: main ln(q-1)/2pi + conductor ln(q/(q-1))/2pi "
          "= ln q/2pi (mean motion)")
    return seeds


def _print_birth_flow():
    print("\n== the birth flow: raw K=1 zeros -> lam -> 0 -> the seed set ==")
    # raw K=1 warp-family zeros of chi_3 (convention-free objects); the box
    # reaches left to sigma ~ -3.2 to keep conductor-string flows in view
    raw = wa.find_zeros(lambda s: F_lam(s, 1, 1, cb.CHI3), (-3.2, 2.0), (2.0, 30.0),
                        dsig=0.3, dt=0.6)
    print(f"   raw K=1 family (chi_3): {len(raw)} zeros in the box: "
          + ", ".join(f"{complex(z):.2f}" for z in raw))
    flows = []
    for z0 in raw:
        traj = flow_lambda(z0, 1, cb.CHI3)
        flows.append((z0, traj))
        end = traj[-1]
        if end[0] == 0.0 and end[1] is not None:
            status = f"-> seed {complex(end[1]):.3f}"
        else:
            tracked = [(lm, zz) for lm, zz in traj if zz is not None]
            status = (f"exits the compact (last tracked lam={tracked[-1][0]} at "
                      f"{complex(tracked[-1][1]):.3f})" if tracked
                      else "polish failed at lam=1")
        print(f"      {complex(z0):.3f}  {status}")
    n_seeds = 5      # 4 main-string + 1 conductor-string seeds in this box
    print(f"   (Hurwitz bookkeeping: {len(raw)} raw zeros vs ~{n_seeds} box seeds"
          " -- the surplus must leave every compact as lam -> 0)")
    # K-independence of the seeds: flow K=3's zeros too, expect the SAME endpoints
    raw3 = wa.find_zeros(lambda s: F_lam(s, 3, 1, cb.CHI3), (-3.2, 2.0), (2.0, 20.0),
                         dsig=0.3, dt=0.6)
    ends3 = []
    for z0 in raw3:
        traj = flow_lambda(z0, 3, cb.CHI3)
        if traj[-1][0] == 0.0 and traj[-1][1] is not None:
            ends3.append(traj[-1][1])
    print(f"   raw K=3 family: {len(raw3)} zeros (t < 20) -> flow endpoints: "
          + ", ".join(f"{complex(z):.3f}" for z in ends3))
    return flows


def _print_census():
    print("\n== the census: N_K(T) in the box vs the entire-L target (no +1) ==")
    (sr, tr) = CENSUS_BOX
    n_L = winding_count(lambda s: cb.L_chi(s, cb.CHI3), sr, tr)
    n_z = winding_count(mp.zeta, sr, tr)
    th = float(theta_chi(tr[1], cb.CHI3) / PI)
    print(f"   box {sr} x {tr}")
    print(f"   targets: L(chi_3) count = {n_L} (theta_chi/pi = {th:.2f}); "
          f"zeta count = {n_z}")
    rows = []
    for K in CENSUS_KS:
        t0 = time.time()
        nc = winding_count(lambda s: cb.L_comb_K(s, K, cb.CHI3), sr, tr)
        nw = winding_count(lambda s: F_lam(s, K, 1, cb.CHI3), sr, tr)
        nz = winding_count(lambda s: hb.zeta_K(s, K), sr, tr)
        rows.append((K, nc, nw, nz))
        print(f"   K={K:>2}:  chi_3 comb {nc:>2}  chi_3 warp {nw:>2}  zeta comb "
              f"{nz:>2}   ({time.time() - t0:.0f} s)")
    n_ground = winding_count(lambda s: cb.comb_endpoint(s, cb.CHI3), sr, tr)
    dens = float(tr[1] - tr[0]) * float(mp.log(2)) / (2 * float(PI))
    print(f"   K=0 comb ground (E_chi) count = {n_ground} (bounded density "
          f"ln(q-1)/2pi x T = {dens:.1f}); target needs the ln T surplus BORN")
    return rows, (n_L, n_z, n_ground)


def _print_displacement():
    print("\n== the displacement law: K(s_K - rho) -> a_1 = -rate_comb_chi/L' ==")
    print("   zero                     a_1 (predicted)      K(s_K-rho), K=25"
          "      K=45")
    out = []
    for chi, ts in ((cb.CHI3, cb.CHI3_ZEROS[:3]), (cb.CHI5_I, cb.CHI5_I_ZEROS[:2])):
        for t in ts:
            rho = cb.polish_zero(chi, t)
            a1 = disp_coeff_chi(rho, chi)
            m25 = measured_disp(rho, chi, 25)
            m45 = measured_disp(rho, chi, 45)
            out.append((chi.label, rho, a1, m25, m45))
            print(f"   {chi.label} 1/2+{t:7.3f}i   {complex(a1):.4f}   "
                  f"{complex(m25):.4f}   {complex(m45):.4f}")
    worst = max(float(abs(m45 - a1) / abs(a1)) for _, _, a1, _, m45 in out)
    print(f"   worst |K=45 measured - a_1|/|a_1| = {worst:.2f} "
          f"(the O(1/K^2) remainder)")
    return out


def _print_ground_geometry():
    print("\n== ground-string geometry: certified axis (phi(q)=2), section skeleton "
          "(all q) ==")
    for chi in (cb.CHI3, cb.CHI4, cb.CHI6):
        nb, na = certify_axis(chi, (0.53, 40.29))
        verdict = ("ALL box zeros on sigma = 0" if nb == na else "MISMATCH")
        print(f"   {chi.label:<14} box count {nb} {'==' if nb == na else '!='} "
              f"axis-root count {na}   ({verdict})")
    print("   phi(q) > 2: scatter off the axis, but ON the section skeleton "
          "(dist to nearest B-zero ~ 1/t):")
    ch7 = cb.characters(7)
    ch9 = cb.characters(9)
    sweep = [(cb.CHI3, "phi=2"), (cb.CHI4, "phi=2"), (cb.CHI5_I, "phi=4"),
             (cb.CHI5_QUAD, "phi=4"), (ch7[1], "phi=6"), (ch7[3], "phi=6"),
             (ch9[1], "phi=6")]
    geo = {}
    for chi, tag in sweep:
        ez, bz, dists = ground_scatter(chi)
        n = len(ez)
        dens = n / 39.5
        pred_dens = float(mp.log(chi.q - 1)) / (2 * float(PI))
        sig = [float(z.real) for z in ez]
        # skeleton collapse: t * dist should stay bounded as t grows
        tds = sorted((float(z.imag), float(d)) for z, d in zip(ez, dists))
        hi_t = sorted(t * d for t, d in tds if t > 15)
        med = hi_t[len(hi_t) // 2] if hi_t else float("nan")
        geo[chi.label] = (ez, bz, dists)
        print(f"   {chi.label:<16} ({tag}): {n:>2} zeros, density {dens:.3f} "
              f"(ln(q-1)/2pi = {pred_dens:.3f}), max|sigma| = "
              f"{max(abs(x) for x in sig):.2f}, t*dist(B) t>15 med/max: "
              f"{med:.2f}/{max(hi_t) if hi_t else float('nan'):.2f}")
    return geo


def _print_chi6_lock():
    print("\n== chi_6: is the Euler-comb lock already in the ground string? ==")
    LN2 = mp.log(2)
    teeth = [(2 * m + 1) * PI / LN2 for m in range(3)]        # 4.53, 13.60, 22.66
    targets = ([("comb", mp.mpc(0, t)) for t in teeth]
               + [("half", mp.mpc(0.5, t)) for t in cb.CHI3_ZEROS])
    sched0 = [0] + cb.K_SCHEDULE
    ax = axis_roots(6, 27.0)
    print("   K=0 axis roots (2pi/ln5 spacing): "
          + ", ".join(f"{float(t):.3f}" for t in ax))
    rows = []
    for kind, zt in targets:
        traj = migrate_robust(lambda ss, K: cb.L_comb_K(ss, K, cb.CHI6), zt,
                              schedule=sched0)
        d = {K: z for K, z in traj if z is not None}
        g = d.get(0)
        rows.append((kind, zt, g, traj))
        gtxt = f"{complex(g):.3f}" if g is not None else "lost"
        print(f"   {kind}  target {complex(zt):.3f}  ->  K=0 ground {gtxt}")
    print("   comb-bound ground offsets from their final teeth:")
    for kind, zt, g, _ in rows:
        if kind == "comb" and g is not None:
            dt_off = float(g.imag - zt.imag)
            print(f"      tooth t={float(zt.imag):6.3f}: ground t="
                  f"{float(g.imag):6.3f}, offset {dt_off:+.3f} "
                  f"(spacing 2pi/ln5 = {float(2 * PI / mp.log(5)):.3f})")
    return rows, ax


def _traj_xy(traj):
    pts = [(k, z) for k, z in traj if z is not None]
    return ([float(z.real) for _, z in pts], [float(z.imag) for _, z in pts],
            [k for k, _ in pts])


def _main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figstyle
    figstyle.enlarge()
    plt.rcParams.update({"axes.titlesize": 12.5, "legend.fontsize": 10,
                         "figure.titlesize": 15.5})
    t_start = time.time()

    _print_seed_tangent()
    seeds = _print_seed_geometry()
    flows = _print_birth_flow()
    census_rows, census_targets = _print_census()
    disp = _print_displacement()
    geo = _print_ground_geometry()
    lock_rows, ax6 = _print_chi6_lock()

    # ============================== figure ==============================
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10))

    # (0,0) the birth flow: raw K=1 zeros -> seeds as lam -> 0
    ax = axes[0, 0]
    main3, cond3 = seeds[cb.CHI3.label]
    Dz = main3 + cond3
    for z0, traj in flows:
        xs, ys, _ = _traj_xy([(l, z) for l, z in traj])
        if xs:
            ax.plot(xs, ys, "-o", ms=2.5, lw=1, color="C0", alpha=0.85)
            ax.plot(xs[0], ys[0], "k.", ms=8)
    ax.plot([], [], "-o", color="C0", ms=2.5, label=r"zero flow, $\lambda: 1\to 0$")
    ax.plot([float(z.real) for z in Dz], [float(z.imag) for z in Dz], "*",
            color="C3", ms=13, mec="k", mew=0.5, label=r"seeds: zeros of $D(s)$",
            zorder=5)
    ax.axvline(0, ls=":", lw=1.2, color="0.5")
    ax.axvline(0.5, ls="-", lw=1, color="C2", alpha=0.6)
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$t$")
    ax.set_title(r"birth flow: raw $K{=}1$ zeros $\to$ the seed set"
                 "\n" r"$F^\lambda_K/\lambda \to D = B + q^{-s}L(0,\chi)$")
    ax.legend(loc="upper right")

    # (0,1) seed geometry: wobbling comb + conductor string, q=3 and q=4
    ax = axes[0, 1]
    for chi, col in ((cb.CHI3, "C0"), (cb.CHI4, "C1")):
        main, cond = seeds[chi.label]
        lnp = float(mp.log(chi.q - 1))
        r = float(L0_chi(chi) / mp.log(chi.q - 1))
        for m, z in enumerate(main, start=1):
            tm = 2 * float(PI) * m / lnp
            th = [i * 2 * float(PI) / 60 for i in range(61)]
            ax.plot([r * math.cos(a) for a in th],
                    [tm + r * math.sin(a) for a in th], "-", lw=0.7, color=col,
                    alpha=0.45)
            ax.plot(0, tm, "+", color=col, ms=8, mew=1.5)
        ax.plot([float(z.real) for z in main], [float(z.imag) for z in main], "o",
                color=col, ms=6, label=rf"$q={chi.q}$ seeds")
        sig_c, _spac = conductor_string(chi)
        ax.axvline(sig_c, ls="--", lw=1, color=col, alpha=0.6)
        if cond:
            ax.plot([float(z.real) for z in cond], [float(z.imag) for z in cond],
                    "D", color=col, ms=6, mfc="none")
    ax.axvline(0, ls=":", lw=1.2, color="0.5")
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$t$")
    ax.set_xlim(-4.4, 1.3)
    ax.set_title(r"seeds: $\sigma=0$ comb wobbled on radius-$L_0/\ln p$ circles"
                 "\n" r"+ conductor string at $\ln L_0/\ln\frac{q}{p}$ "
                 r"(diamonds)")
    ax.legend(loc="upper left")

    # (0,2) the census
    ax = axes[0, 2]
    Ks = [r[0] for r in census_rows]
    n_L, n_z, n_ground = census_targets
    ax.plot(Ks, [r[1] for r in census_rows], "-o", color="C0",
            label=r"$\chi_3$ comb $N_K$")
    ax.plot(Ks, [r[2] for r in census_rows], "-s", ms=5, color="C3", mfc="none",
            label=r"$\chi_3$ warp $N_K$")
    ax.plot(Ks, [r[3] for r in census_rows], "-^", ms=5, color="0.4",
            label=r"$\zeta$ comb $N_K$")
    ax.axhline(n_L, ls="--", lw=1.2, color="C0", alpha=0.7,
               label=rf"$L(\chi_3)$ target ({n_L})")
    ax.axhline(n_z, ls="--", lw=1.2, color="0.4", alpha=0.7,
               label=rf"$\zeta$ target ({n_z})")
    ax.axhline(n_ground, ls=":", lw=1.4, color="C1",
               label=rf"$E_\chi$ ground ({n_ground})")
    ax.set_xscale("log")
    ax.set_xlabel("K")
    ax.set_ylabel(rf"$N_K$ in box, $t<{CENSUS_BOX[1][1]:.0f}$")
    ax.set_title("the census: bounded-density ground backbone\n+ born surplus "
                 r"($\zeta$: all born; entire $L$: no $+1$)")
    ax.legend(loc="center right", fontsize=8.5)

    # (1,0) displacement law
    ax = axes[1, 0]
    labels = []
    for i, (lab, rho, a1, m25, m45) in enumerate(disp):
        col = "C0" if lab == cb.CHI3.label else "C3"
        ax.plot([i - 0.12, i + 0.12], [abs(complex(m25)), abs(complex(m45))],
                "-o", ms=5, color=col)
        ax.plot(i, abs(complex(a1)), "*", ms=14, color=col, mec="k", mew=0.5)
        labels.append(f"{float(mp.im(rho)):.1f}")
    ax.set_xticks(range(len(disp)))
    ax.set_xticklabels(labels)
    ax.plot([], [], "-o", color="0.5", label=r"$|K(s_K-\rho)|$, $K=25\to45$")
    ax.plot([], [], "*", ms=12, color="0.5", mec="k", label=r"$|a_1|$ predicted")
    ax.set_xlabel(r"zero height $\gamma$  (blue $\chi_3$, red $\chi_5$)")
    ax.set_ylabel(r"$|K(s_K-\rho)|$")
    ax.set_title(r"displacement law: $s_K-\rho\sim a_1/K$,"
                 "\n" r"$a_1=\frac{sq}{2\pi^2}B(\rho{+}1)/L'(\rho)$")
    ax.legend(loc="upper left")

    # (1,1) ground scatter + skeleton
    ax = axes[1, 1]
    cols = {"phi=2": "C0", "phi=4": "C3", "phi=6": "C1"}
    seen = set()
    for lab, (ez, bz, dists) in geo.items():
        phi_tag = ("phi=2" if lab in (cb.CHI3.label, cb.CHI4.label)
                   else ("phi=4" if "5" in lab else "phi=6"))
        col = cols[phi_tag]
        kw = {}
        if phi_tag not in seen:
            kw["label"] = rf"$\varphi(q)={phi_tag[-1]}$"
            seen.add(phi_tag)
        ax.plot([float(z.real) for z in ez], [float(z.imag) for z in ez], "o",
                ms=4, color=col, alpha=0.7, **kw)
    ax.axvline(0, ls=":", lw=1.2, color="0.5")
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$t$")
    ax.set_title(r"ground strings $E_\chi$: exact axis at $\varphi(q)=2$,"
                 "\nsection-skeleton scatter beyond")
    ax.legend(loc="upper right")

    # (1,2) chi_6 lock
    ax = axes[1, 2]
    for kind, zt, g, traj in lock_rows:
        xs, ys, ks = _traj_xy(traj)
        if not xs:
            continue
        col = "C1" if kind == "comb" else "C0"
        ax.plot(xs, ys, "-o", ms=2.5, lw=1, color=col, alpha=0.8)
        ax.plot(xs[-1], ys[-1], "k.", ms=8)
        if ks[0] == 0:
            ax.plot(xs[0], ys[0], "s", ms=7, mfc="none", mec="0.3")
    for t in ax6:
        ax.plot(-0.55, float(t), ">", color="0.6", ms=6)
    LN2f = float(mp.log(2))
    for m in range(3):
        ax.plot(0, (2 * m + 1) * float(PI) / LN2f, "x", color="C1", ms=11, mew=2)
    ax.plot([], [], "-o", color="C1", label=r"$\to\sigma=0$ comb")
    ax.plot([], [], "-o", color="C0", label=r"$\to\sigma=\frac{1}{2}$")
    ax.plot([], [], ">", color="0.6", label="K=0 axis roots")
    ax.axvline(0, ls="--", lw=1, color="C1", alpha=0.6)
    ax.axvline(0.5, ls="-", lw=1, color="C2", alpha=0.6)
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$t$")
    ax.set_title(r"$\chi_6$: ground $\to$ $\frac{1}{2}\,\cup$ comb;"
                 "\ndoes the lock pre-exist at $K=0$?")
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(r"Zero birth from the identically-zero endpoint: "
                 r"$F_K^\lambda/\lambda\to D(s)$ seeds, census, and ground "
                 r"geometry (issue #39)")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    figdir = _HERE / "figures"
    figdir.mkdir(exist_ok=True)
    out = figdir / "zero_birth.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}   ({time.time() - t_start:.0f} s total)")


if __name__ == "__main__":
    _main()
