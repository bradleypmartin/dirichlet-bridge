r"""Unified O(1/K) rate / birth-K law across the three bridge routes.

Follow-up across the whole bridge. The earlier stages (cont_eta, harmonic_bridge,
warp_bridge, warp_alpha) each found the zeros converging onto their lines at
O(1/K) but described the *approach* only qualitatively -- comb zeros "born high and
descend (some overshoot)", warp zeros "born low and climb from below". This module makes
that one analysis: a single leading-rate constant and a single zero-displacement
coefficient, route by route, and the sub-leading correction that explains the
comb-overshoot / warp-no-overshoot asymmetry. Everything below is verified against the
real evaluators (zeta_K/eta_K/warp_K) and the hb.migrate trajectories.

The one mechanism: the first sawtooth tail  Sum_{k>K} 1/(pi k)^2 ~ 1/(pi^2 K)
--------------------------------------------------------------------------------
Every route truncates the SAME object -- the periodic-Bernoulli sawtooth
B~_1(x) = {x}-1/2 = -Sum_k sin(2 pi k x)/(pi k) -- at K harmonics, so every route's
leftover is that sawtooth's tail, whose L^2 power is Sum_{k>K} 1/(pi k)^2 ~ 1/(pi^2 K).
That is why ALL routes are O(1/K). What differs is HOW the route sees the tail:

  * COMB (additive): the tail enters the integrand *linearly*. The k-th sawtooth
    mode is  zeta_term(s,k) = (s/pi k) Int_1^inf sin(2 pi k x) x^{-s-1} dx, and endpoint
    integration by parts at the integer frequency (cos 2 pi k = 1, sin 2 pi k = 0) gives
    zeta_term(s,k) = s/(2 pi^2 k^2) + O(1/k^4). Summing the tail:

        zeta(s) - zeta^{(K)}(s)  =  Sum_{k>K} zeta_term(s,k)
                                 =  (s/2 pi^2)(1/K - 1/(2K^2) + ...).        (`err_comb`)

    A CLEAN closed constant  s/(2 pi^2), and -- because the x=1 endpoint of the eta comb
    has the same |cos pi x|=1 -- *identical* for eta (`test_*`). (`rate_comb`)

  * WARP (nonlinear): the tail enters through the *argument*
    a_K(u) = 3/2 - r_K(u),  r_K(u) = Sum_{k>K} sin(2 pi k u)/(pi k). The sawtooth's zero
    mean kills the linear (first-order) response -- Int_0^1 r_K = 0 -- so the bulk rate is
    *second* order, 1/2 zeta_aa''(s,3/2) Int_0^1 r_K^2 = s(s+1)zeta(s+2,3/2)/(4 pi^2 K)
    (`rate_warp_bulk`). BUT the warp integrand's endpoints reach a=1 and a=2 (not 3/2)
    across O(1/K)-wide **Gibbs boundary layers** (where r_K ~ O(1), profile
    (1/pi)(pi/2 - Si(2 pi v))), and those contribute at the SAME 1/K order. So the warp
    constant is bulk + a boundary-layer (sine-integral) integral with NO elementary form;
    we take it numerically, lim K (warp_K - zeta(s,3/2)) (`rate_warp`). The nonlinearity
    is exactly what turns the comb's clean first-order  s/2 pi^2  into a messy
    boundary-layer-laden second-order constant -- a finding, not a nuisance.

The unified zero-displacement law  s_K - rho ~ -Delta_1(rho)/(target'(rho) K)
----------------------------------------------------------------------------
Write any interpolant as  object^{(K)} = target + Delta_K,  Delta_K -> 0 the convergence
error above (Delta_K = object - target, so Delta_1 = lim K Delta_K is -s/2 pi^2 for the
comb, +C_warp for the warp). A zero s_K of object^{(K)} near a zero rho of `target`
(target(rho)=0) satisfies  0 = target'(rho)(s_K-rho) + Delta_K(rho) + ..., so

        s_K - rho  ~  -Delta_K(rho)/target'(rho)  =  a_1(rho)/K,
        a_1(rho)   =  -Delta_1(rho)/target'(rho).                        (`disp_coeff`)

This ONE formula covers all five zero families (`ROUTES`), each just a different
(Delta_1, target'):

  family           target              target'(rho)                Delta_1
  comb_zeta  (s=1/2) zeta                zeta'(rho)                  -rho/2 pi^2
  comb_eta   (s=1/2) eta=(1-2^{1-s})zeta (1-2^{1-rho}) zeta'(rho)    -rho/2 pi^2
  eta_pref   (s=1)   eta                 ln2 zeta(rho)               -rho/2 pi^2
  warp_zeta  (s=1/2) (2^s-1)zeta         (2^rho-1) zeta'(rho)        +C_warp(rho)
  warp_pref  (s=0)   (2^s-1)zeta         ln2 zeta(rho)               +C_warp(rho)

The SIGN of Re a_1 is the approach side (Re a_1 > 0: from above; < 0: from below). For
the comb it ALTERNATES with the zero (the phase of 1/zeta'(rho) wanders), so some comb
zeros descend onto 1/2 from above and some from below; for the warp it is uniformly < 0
(from below). That is the "comb descends / warp climbs" contrast, sharpened: not a
blanket direction but the sign of a computable coefficient.

Overshoot, the sub-leading correction (`disp_coeff2`)
----------------------------------------------------
Carrying Delta_K = b_1/K + b_2/K^2 (b_2 = -b_1/2, the Euler-Maclaurin tail of Sum 1/k^2)
through gives the comb zero to O(1/K^2) (`disp_coeff2`, matches hb.migrate to an extra
digit by K~20). The OVERSHOOT then has a clean criterion. Empirically every comb zero is
*born above* 1/2 (sigma at K=1 is 1.19 -> 0.88, decreasing with height but always > 1/2),
and ends on the side sign(Re a_1); so

    comb overshoots (crosses 1/2)  <=>  Re a_1(rho) < 0   (`overshoots`)

-- born above, must dive below to reach its from-below limit. The WARP zeros are born
*below* 1/2 (sigma at K=1 is 0.03 -> 0.22) AND end below (Re a_1 < 0 uniformly), so they
never cross: warp NEVER overshoots. That is the comb-overshoot / warp-no-overshoot
asymmetry, resolved.

Birth-K law (`catch_K`)
-----------------------
"Which K first catches gamma_n" (lands its zero within eps of the limiting line):
|Re a_1|/K < eps, so  catch_K(rho, eps) = |Re a_1(rho)|/eps. Since |a_1| ~
|rho|/(2 pi^2 |target'(rho)|) and |target'| grows only polylogarithmically, catch_K grows
(on average -- it fluctuates with |zeta'(rho)|) with the height gamma: higher zeros, being
denser by the Riemann-von Mangoldt law (local spacing 2 pi/ln(t/pi)), need more
harmonics to resolve. (`catch_K`)

Vertical migration: the Gram phase  arg(a_1) ~ pi delta_n
---------------------------------------------------------
a_1 is COMPLEX; its imaginary part is the vertical (Im s) migration Im(s_K - rho) ~
Im(a_1)/K (`vertical_side`). For the comb, the Hardy Z-function collapses the whole
displacement angle to the zero's position in its Gram interval. With
zeta'(rho) = -i e^{-i theta(gamma)} Z'(gamma) and sign Z'(gamma_n) = (-1)^{n-1},

    Re a_1 ~ cos(pi delta_n),   Im a_1 ~ sin(pi delta_n),
    delta_n = theta(gamma_n)/pi - (n-2)   (Gram-interval position, `gram_position`).

A Gram-REGULAR zero has delta_n in (0,1), so sin(pi delta_n) > 0 ALWAYS -> Im a_1 > 0
(vertical pinned UP), while cos(pi delta_n) changes sign at delta_n = 1/2 -> Re a_1
ALTERNATES (the 'comb approach side alternates', now explained: the zero sits in the
early/late half of its Gram interval). This is the **axis-duality** sharpened: the comb
pins the VERTICAL axis (Im a_1 > 0) and lets the horizontal wander; the warp pins the
HORIZONTAL (Re a_1 < 0 uniformly) and lets the vertical oscillate (`vertical_side`
over warp_zeta takes both signs -- the warp inherits the wandering phase of its non-closed-
form C_warp, so there is no comb-style arithmetic rule for it).

The headline: comb Im a_1 > 0 is **NOT universal** -- it is exactly **Gram's law**. Im a_1
flips negative precisely when Gram's law fails (delta_n leaves (0,1)); the first three are
the 127th, 136th, 196th zeros (gamma ~ 282.47, 295.57, 391.46), n=127 being the classically
first Gram's-law failure. There |Im a_1| ~ |sin(pi delta_n)| is tiny (the soft sign, the
zero nearly ON a Gram point), so those zeros are the slowest to settle their vertical
approach. The convention-free criterion sign(Im a_1) = sign((-1)^n sin theta(gamma_n))
matches over n=1..200 with zero exceptions. So the bridge's vertical migration is a
**Gram's-law detector**. (Needs zeros up to gamma ~ 400 to see -- comb-only, so the
high-tau warp caveats do not bite the comb result.)

The continuous-eta sigma_law sub-leading (an open item)
-------------------------------------------------------
Separately, the continuous-eta off-line string sigma(t) -> 3/2: its closed-form `cont_eta.sigma_law`
already realizes the F_asym magnitude balance to O(1/t^2) -- solving that balance
*exactly* (`sigma_law_exact`) leaves the SAME residual to the true zeros. So the leftover
drift is not a Stirling/|s-1| approximation but the F_asym *truncation* (the dropped
higher incomplete-gamma terms of F); it has no clean elementary closed form. A negative
result that closes the open item: refine F, not the balance.

Three rate-law loose ends
-------------------------
1. A CLOSED FORM for the warp constant C_warp (`c_warp_closed`). The 1/K warp constant is
   rate_warp_bulk (the global quadratic) + a Gibbs BOUNDARY LAYER. Near each cell edge the
   sawtooth tail r_K does not shrink; it relaxes through the Gibbs profile
   P(v) = (1/pi)(pi/2 - Si(2 pi v)), v = K u  (P(0)=1/2 -> argument 1; P(inf)=0 -> 3/2), so
   each O(1/K)-wide edge contributes at the same 1/K order. Combining both edges (the linear
   +-P counterterms cancel):

       C_warp(s) = rate_warp_bulk(s)
         + Int_0^inf [zeta(s,3/2-P) + zeta(s,3/2+P) - 2 zeta(s,3/2) - P^2 s(s+1)zeta(s+2,3/2)] dv.

   The -P^2 zeta_aa'' is the QUADRATIC the bulk already counts: subtracting it both avoids
   double-counting AND makes the integrand O(P^4) ~ 1/v^4 (converged by v ~ 20). The naive
   linear-counterterm-only guess double-counts and overshoots ~2-3x. C_warp is a prime-FREE,
   purely analytic object -- the SAME Gibbs layer that under-resolves the moment grid at
   high tau, and (item 3) the term that breaks the sigma=1/sigma=0 FE mirror.

2. catch_K <-> |zeta'(rho)|, the Montgomery a-values (`montgomery_a`, `catch_K(mode=...)`,
   `catch_K_normalized`). a_1 = rho/(2 pi^2 zeta'(rho)) carries 1/|zeta'(rho)|, and by the
   Hardy-Z reduction |zeta'(rho)| = |Z'(gamma)| EXACTLY. So the height-normalized
   catch_K(mode='abs') = |a_1|/eps is EXACTLY 1/|zeta'(rho)| -- the Montgomery a-value
   reciprocal: which zeros are hardest to resolve is set by |zeta'|, and the small-|zeta'|
   (near-degenerate) zeros are slowest (corr(|zeta'|, relative spacing) ~ 0.74 over the first 80
   zeros: small |zeta'| <=> closely-spaced pair). The DEFAULT mode='re' (the birth-K) carries
   a SECOND fluctuation: |Re a_1| = |rho|/(2 pi^2 |Z'|) |cos pi delta_n|, the Gram factor
   (|cos arg a_1| ~ |cos pi delta_n| to < 0.01 by gamma ~ 60). So 'abs' isolates the Montgomery
   a-values; 're' adds the Gram modulation -- the two pick DIFFERENT hardest zeros (small-|zeta'|
   vs small-|zeta'| AND mid-Gram-interval delta_n ~ 1/2).

3. The FE-mirror companions only SHARE HEIGHTS (`fe_mirror`). The sigma=1 eta prefactor
   (1-2^{1-s}=0) and sigma=0 warp companion (2^s-1=0) are exact s<->1-s images as zero SETS
   (heights +-2 pi k/ln2, the p=2 comb) -- but their migration dynamics are NOT mirrors.
   In a_1 = -Delta_1/target': target' = ln2 zeta(rho) has the same form on both lines (at the
   FE-conjugate arguments 1+it_k vs it_k), but Delta_1 differs in KIND -- eta's is the closed
   additive-comb constant -rho/2 pi^2 (Re == -1/2 pi^2, a rigid line), warp's is the non-closed
   Gibbs constant C_warp(rho) (Re wanders 0.40 -> 1.64). The mirror breaks entirely in Delta_1,
   for an analytic (boundary-layer = C_warp) not arithmetic reason -- dovetailing the Gram
   phase (warp Im a_1 has no arithmetic rule) and the two-component spectrum (mirror-exact
   zero sets, broken dynamics).

Run directly to validate + plot: writes bridge/figures/rate_law.png + rate_law_loose_ends.png.
"""
import sys
from pathlib import Path
import mpmath as mp

# This module sits in a flat source dir alongside its bare cross-imports; put that
# directory on sys.path so a direct ``python rate_law.py`` run resolves them
# (pytest gets the same path from the root conftest.py).
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cont_eta as ce         # noqa: E402  (sigma_law, F_asym -- the continuous-eta string)
import harmonic_bridge as hb  # noqa: E402  (zeta_K/eta_K, migrate -- the comb routes)
import warp_bridge as wb      # noqa: E402  (warp_K, target_half -- the warp route)

mp.mp.dps = 30

PI = mp.pi
LN2 = mp.log(2)


# --------------------------------------------------------------------------
# leading convergence-rate constant  Delta_1(s) = lim_K K (object^{(K)} - target)
# --------------------------------------------------------------------------
def rate_comb(s):
    """Comb leading constant: zeta(s) - zeta^{(K)}(s) ~ rate_comb/K  (and eta, same endpoint).

    The closed form s/(2 pi^2): each sawtooth mode is s/(2 pi^2 k^2)+O(1/k^4), summed over
    the tail Sum_{k>K} 1/k^2 ~ 1/K. Identical for the eta comb (its x=1 endpoint has the
    same |cos pi x| = 1). NB this is target - object; Delta_1 = -rate_comb (object - target).
    """
    return mp.mpc(s) / (2 * PI**2)


def err_comb(s, K):
    """The 2-term model of the comb error  zeta(s) - zeta^{(K)}(s) ~ (s/2 pi^2)(1/K - 1/2K^2).

    The 1/(2K^2) is the Euler-Maclaurin tail of Sum_{k>K} 1/k^2 (so the sub-leading
    coefficient is exactly -1/2 of the leading); higher modes (1/k^4) enter only at O(1/K^3).
    """
    s = mp.mpc(s)
    return s / (2 * PI**2) * (mp.mpf(1) / K - mp.mpf(1) / (2 * K**2))


def rate_warp_bulk(s):
    """Warp BULK constant (no boundary layer): 1/2 zeta_aa''(s,3/2) * (1/2 pi^2).

    The second-order response 1/2 zeta_aa''(s,3/2) Int_0^1 r_K^2, with zeta_aa''(s,a) =
    s(s+1) zeta(s+2,a) and Int_0^1 r_K^2 = (1/2 pi^2) Sum_{k>K}1/k^2 ~ 1/(2 pi^2 K). This is
    only PART of rate_warp: the Gibbs boundary layers add a comparable, non-elementary piece.
    """
    s = mp.mpc(s)
    return s * (s + 1) * mp.zeta(s + 2, mp.mpf(3) / 2) / (4 * PI**2)


_RATE_WARP_CACHE = {}   # (s, K, richardson, closed, Vmax, prec) -> constant


def rate_warp(s, K=40, richardson=False, closed=False, Vmax=30):
    """Warp full constant  lim_K K (warp_K(s,K) - zeta(s,3/2))  -- bulk + Gibbs boundary layer.

    Three ways to get it:
      * default: the fast grid warp_K at K=40 (~2-3 digits, enough for the sign and magnitude
        of the displacement coefficient);
      * richardson=True: the slow exact warp_hurwitz at K=80,160 extrapolated (~5 digits);
      * closed=True: the CLOSED FORM `c_warp_closed` (the Si-Gibbs boundary-layer integral)
        -- exact, the cleanest. delta1/disp_coeff/catch_K forward this via **kw.

    Memoized on (s, mode, precision): the driver re-asks the same few heights across its
    print sections and figure panels, and each warp evaluation costs seconds.
    """
    s = mp.mpc(s)
    key = (s, K, richardson, closed, Vmax, mp.mp.prec)
    if key in _RATE_WARP_CACHE:
        return _RATE_WARP_CACHE[key]
    if closed:
        out = c_warp_closed(s, Vmax=Vmax)
    elif richardson:
        z32 = mp.zeta(s, mp.mpf(3) / 2)
        v80 = 80 * (wb.warp_hurwitz(s, 80) - z32)
        v160 = 160 * (wb.warp_hurwitz(s, 160) - z32)
        out = 2 * v160 - v80                        # kill the 1/K term
    else:
        out = K * (wb.warp_K(s, K) - mp.zeta(s, mp.mpf(3) / 2))
    _RATE_WARP_CACHE[key] = out
    return out


def gibbs_profile(v):
    """The cell-edge Gibbs relaxation profile  P(v) = (1/pi)(pi/2 - Si(2 pi v)).

    The K -> infinity limit of the sawtooth tail r_K in the inner variable v = K u: near a cell
    edge r_K(v/K) -> P(v), a Riemann sum of int_1^inf sin(2 pi v w)/(pi w) dw = (1/pi)(pi/2 -
    Si(2 pi v)). P(0) = 1/2 (the warp argument ramps to 1 at the left edge, 2 at the right),
    P(v) -> 0 in the bulk (~ cos(2 pi v)/(2 pi^2 v)), so the layer is O(1/K) wide. This is the
    SAME profile the moment grid mu_j = int psi^j under-resolves at high tau -- the
    high-tau resolution difficulty and the warp constant C_warp are two faces of one Gibbs layer.
    """
    return (PI / 2 - mp.si(2 * PI * v)) / PI


def c_warp_closed(s, Vmax=30):
    r"""Closed form for the warp constant  C_warp(s) = lim_K K (warp_K(s,K) - zeta(s,3/2)).

    warp_K(s,K) = int_0^1 zeta(s, 3/2 - r_K(u)) du with the sawtooth tail r_K(u) =
    sum_{k>K} sin(2 pi k u)/(pi k). The 1/K constant splits into the second-order BULK response
    (rate_warp_bulk, the global quadratic 1/2 zeta_aa'' int r_K^2) and a Gibbs BOUNDARY LAYER:
    near each cell edge r_K does not shrink, it relaxes through P(v) = gibbs_profile(v) over an
    O(1/K)-wide layer (v = K u), so the argument ramps from 3/2 to 1 (left edge, r_K = +P) and
    to 2 (right edge, r_K = -P). Combining the two edges, the linear +-P zeta_a' counterterms
    cancel, leaving the symmetric

        C_warp(s) = rate_warp_bulk(s)
          + Int_0^inf [zeta(s,3/2-P) + zeta(s,3/2+P) - 2 zeta(s,3/2) - P^2 s(s+1)zeta(s+2,3/2)] dv.

    The -P^2 s(s+1)zeta(s+2,3/2) is the QUADRATIC the bulk already counts; subtracting it both
    avoids double-counting AND makes the integrand O(P^4) ~ 1/v^4 (converged by v ~ 20, so the
    Vmax default 30 is ample for moderate |Im s|; raise it for very large |Im s|). NB the naive
    linear-counterterm-only form (no -P^2 term) double-counts the boundary quadratic and
    overshoots C_warp by ~2-3x -- the quadratic subtraction is required. Matches the numerical
    rate_warp(richardson=True) to that truth's own ~5-digit accuracy (the closed form is exact;
    the residual is the Richardson truncation, not this integral).

    A prime-FREE, purely analytic (boundary-layer) object -- the term that breaks the
    sigma=1/sigma=0 FE mirror (`fe_mirror`) and drives the warp's wandering Im a_1, and
    the same Gibbs layer the moment grid under-resolves at high tau.
    """
    s = mp.mpc(s)
    z32 = mp.zeta(s, mp.mpf(3) / 2)
    D2 = s * (s + 1) * mp.zeta(s + 2, mp.mpf(3) / 2)   # zeta_aa''(s,3/2)

    def integrand(v):
        p = gibbs_profile(v)
        return (mp.zeta(s, mp.mpf(3) / 2 - p) + mp.zeta(s, mp.mpf(3) / 2 + p)
                - 2 * z32 - p**2 * D2)

    boundary = mp.quad(integrand, [mp.mpf(j) for j in range(Vmax + 1)])
    return rate_warp_bulk(s) + boundary


# --------------------------------------------------------------------------
# the five zero families: (target', Delta_1) per route  ->  a_1 = -Delta_1/target'
# --------------------------------------------------------------------------
# Each entry maps a route name to callables (target_deriv(rho), delta1(rho)). target' is
# evaluated at the relevant zero (a zeta-zero, an eta-prefactor zero, or a 2^s-1 companion),
# using target(rho)=0 to drop the vanishing product-rule term.
ROUTES = ("comb_zeta", "comb_eta", "eta_pref", "warp_zeta", "warp_pref")


def target_deriv(route, rho):
    """target'(rho) for the named route's zero family (the migration-target derivative)."""
    rho = mp.mpc(rho)
    zp = mp.zeta(rho, derivative=1)
    if route == "comb_zeta":
        return zp                                            # target = zeta
    if route == "comb_eta":
        return (1 - mp.power(2, 1 - rho)) * zp               # eta', at a zeta-zero
    if route == "eta_pref":
        return LN2 * mp.zeta(rho)                            # eta', at 1-2^{1-rho}=0
    if route == "warp_zeta":
        return (mp.power(2, rho) - 1) * zp                   # ((2^s-1)zeta)', at a zeta-zero
    if route == "warp_pref":
        return LN2 * mp.zeta(rho)                            # ((2^s-1)zeta)', at 2^rho=1
    raise ValueError(route)


def delta1(route, rho, **kw):
    """Delta_1(rho) = lim K (object^{(K)} - target) for the route (comb: -rho/2 pi^2; warp: C_warp)."""
    rho = mp.mpc(rho)
    if route in ("comb_zeta", "comb_eta", "eta_pref"):
        return -rho / (2 * PI**2)
    if route in ("warp_zeta", "warp_pref"):
        return rate_warp(rho, **kw)
    raise ValueError(route)


def disp_coeff(route, rho, **kw):
    """Leading zero-displacement coefficient  a_1: s_K - rho ~ a_1/K  =  -Delta_1/target'."""
    return -delta1(route, rho, **kw) / target_deriv(route, rho)


def disp_coeff2(rho):
    """Sub-leading (1/K^2) displacement for the COMB zeta family: s_K - rho ~ a_1/K + a_2/K^2.

    From Delta_K = b_1/K + b_2/K^2 (b_1 = -rho/2 pi^2, b_2 = +rho/4 pi^2 = -b_1/2):
       a_2 = -[1/2 zeta''(rho) a_1^2 + b_1'(rho) a_1 + b_2(rho)]/zeta'(rho),
    with b_1'(s) = -1/2 pi^2. (Matches hb.migrate to an extra digit by K~20.)
    """
    rho = mp.mpc(rho)
    zp = mp.zeta(rho, derivative=1)
    zpp = mp.zeta(rho, derivative=2)
    a1 = rho / (2 * PI**2 * zp)
    b1p = -1 / (2 * PI**2)
    b2 = rho / (4 * PI**2)
    return -(mp.mpf("0.5") * zpp * a1**2 + b1p * a1 + b2) / zp


def approach_side(route, rho, **kw):
    """'above' or 'below' -- the side from which s_K reaches its line, = sign(Re a_1)."""
    return "above" if disp_coeff(route, rho, **kw).real > 0 else "below"


def vertical_side(route, rho, **kw):
    """'up' or 'down' -- the VERTICAL (Im s) approach = sign(Im a_1).

    The imaginary part of the same displacement coefficient gives the vertical migration
    Im(s_K - rho) ~ Im(a_1)/K. 'up' (Im a_1 > 0): the zero descends onto its height gamma
    from ABOVE; 'down' (Im a_1 < 0): it climbs from below. This is the vertical analog of
    approach_side, and the two together make the displacement law a 2-D statement.
    """
    return "up" if disp_coeff(route, rho, **kw).imag > 0 else "down"


def gram_position(g, n):
    """Gram-interval position delta_n = theta(gamma)/pi - (n-2) of the n-th zeta zero.

    The Hardy Z-function collapses the comb displacement angle to  arg(a_1) ~ pi*delta_n.
    From a_1 = rho/(2 pi^2 zeta'(rho)) and zeta'(rho) = -i e^{-i theta(gamma)} Z'(gamma)
    (Z the Hardy function) with sign Z'(gamma_n) = (-1)^{n-1} (simple real zeros, Z(0)<0):

        Re a_1 ~ cos(pi delta_n),    Im a_1 ~ sin(pi delta_n).

    A Gram-REGULAR zero sits one-per-Gram-interval, delta_n in (0,1): then sin(pi delta_n) > 0
    ALWAYS, so Im a_1 > 0 (vertical pinned UP), while cos(pi delta_n) flips sign at delta_n = 1/2,
    so Re a_1 ALTERNATES (the 'comb approach side alternates', now read off the zero's
    position within its Gram interval). Im a_1 <= 0 happens EXACTLY when Gram's law fails
    (delta_n leaves (0,1)) -- first at the 127th zero (gamma ~ 282.47, delta ~ 1.006), so the
    comb vertical-up uniformity asserted at low gamma is NOT universal: it IS Gram's law. The
    exact convention-free criterion is sign(Im a_1) = sign((-1)^n sin theta(gamma_n)).
    """
    return float(mp.siegeltheta(g) / PI) - (n - 2)


def overshoots(rho, **kw):
    """Does the COMB zeta zero cross 1/2 (overshoot)?  Born above 1/2 and ends below <=> Re a_1 < 0.

    Every comb zero is born above 1/2 (sigma at K=1 in 1.19..0.88), so it overshoots exactly
    when its from-below limit forces a crossing. (The warp zeros are born below and stay below
    -- they never overshoot -- so this predicate is the comb's alone.)
    """
    return disp_coeff("comb_zeta", rho, **kw).real < 0


def montgomery_a(rho):
    """|zeta'(rho)| at a zero rho = 1/2 + i gamma -- the Montgomery a-value (loose end 2).

    By the Hardy-Z reduction zeta'(rho) = -i e^{-i theta(gamma)} Z'(gamma), so
    |zeta'(rho)| = |Z'(gamma)| EXACTLY: the Montgomery a-values ARE the Hardy-Z slopes at its
    zeros. a_1 = rho/(2 pi^2 zeta'(rho)) carries 1/|zeta'(rho)|, so this single number sets how
    hard the bridge works to resolve gamma (catch_K below). The small-|zeta'| (near-degenerate,
    closely-spaced) zeros are the slowest to catch.
    """
    return abs(mp.zeta(mp.mpc(rho), derivative=1))


def catch_K(route, rho, eps=0.05, mode="re", **kw):
    """Birth-K: smallest K landing the zero within eps of its line, ~ |a_1 component|/eps.

    mode='re' (default, the birth-K): |Re a_1|/eps, the HORIZONTAL catch. It carries TWO
    fluctuations -- the Montgomery 1/|zeta'(rho)| AND the Gram factor |cos pi delta_n| (the Gram
    reduction |Re a_1| = |rho|/(2 pi^2 |Z'|) |cos pi delta_n|).
    mode='abs': |a_1|/eps -- carries ONLY 1/|zeta'(rho)| (the Montgomery a-value), so it ranks
    zeros purely by |zeta'| (small-|zeta'| = hardest). The two modes pick DIFFERENT 'hard'
    populations: small-|zeta'| (mode='abs') vs small-|zeta'| AND mid-Gram-interval delta_n ~ 1/2
    (mode='re'). Both grow on average with height (denser zeros, spacing 2 pi/ln(t/pi)).
    """
    a1 = disp_coeff(route, rho, **kw)
    mag = abs(a1) if mode == "abs" else abs(a1.real)
    return abs(float(mag)) / eps


def catch_K_normalized(route, rho, eps=0.05, mode="abs", **kw):
    """catch_K with the smooth |rho|/(2 pi^2 eps) height-growth divided out (loose end 2).

    For the comb_zeta family with mode='abs' this is EXACTLY 1/|zeta'(rho)| -- the Montgomery
    a-value reciprocal, the pure zero-to-zero fluctuation with the trivial height growth removed.
    (catch_K(abs) = |a_1|/eps = |rho|/(2 pi^2 |zeta'(rho)|)/eps, and the base is |rho|/(2 pi^2 eps).)
    """
    base = float(abs(mp.mpc(rho))) / (2 * float(PI)**2 * eps)
    return catch_K(route, rho, eps=eps, mode=mode, **kw) / base


# --------------------------------------------------------------------------
# the sigma=1 / sigma=0 FE-mirror companions: same heights, broken dynamics (loose end 3)
# --------------------------------------------------------------------------
def fe_mirror(k, **kw):
    """Compare the FE-mirror prefactor companions at height t_k = 2 pi k/ln2 (loose end 3).

    The sigma=1 eta prefactor zero (1 - 2^{1-s} = 0) and the sigma=0 warp companion
    (2^s - 1 = 0) are exact s <-> 1-s images as zero SETS -- 1 - 2^{1-(1-s)} = -(2^s - 1), so
    the two prefactor zero sets coincide under s <-> 1-s, at the IDENTICAL heights +-2 pi k/ln2
    (the p=2 comb). This returns their migration data to test whether the DYNAMICS mirror too.

    They do not. In a_1 = -Delta_1/target':
      * target' = ln2 zeta(rho) has the same functional form on both lines, but at the
        FE-conjugate arguments rho = 1 + i t_k (eta) vs rho = i t_k (warp) -- so it differs only
        by the FE chi-factor, an analytic (arithmetic-free) difference;
      * Delta_1 differs in KIND: eta migrates by the ADDITIVE comb, Delta_1 = -rho/2 pi^2
        (Re == -1/2 pi^2, a rigid line); warp migrates by the NONLINEAR warp,
        Delta_1 = C_warp(rho), the non-closed Gibbs constant whose Re wanders.
    So the mirror is exact in the spectrum and broken in the dynamics, and the break is entirely
    Delta_1 = C_warp -- an analytic (boundary-layer) not arithmetic reason (cf the warp's Im a_1
    with no arithmetic rule). Pass closed=True to take the warp Delta_1 from the item-1 closed form.

    Returns a dict: k, t, rho_eta, rho_warp, a1_eta, a1_warp, d1_eta, d1_warp, tprime_eta,
    tprime_warp.
    """
    t = 2 * float(PI) * k / float(LN2)
    rho_e, rho_w = mp.mpc(1, t), mp.mpc(0, t)
    return {
        "k": k, "t": t, "rho_eta": rho_e, "rho_warp": rho_w,
        "a1_eta": disp_coeff("eta_pref", rho_e),
        "a1_warp": disp_coeff("warp_pref", rho_w, **kw),
        "d1_eta": delta1("eta_pref", rho_e),
        "d1_warp": delta1("warp_pref", rho_w, **kw),
        "tprime_eta": target_deriv("eta_pref", rho_e),
        "tprime_warp": target_deriv("warp_pref", rho_w),
    }


# --------------------------------------------------------------------------
# the continuous-eta sigma_law sub-leading: the residual is F_asym truncation, not the balance
# --------------------------------------------------------------------------
def sigma_law_exact(t):
    """Re of the continuous-eta zero at height t by solving the F_asym magnitude balance EXACTLY.

    cont_eta.sigma_law uses leading Stirling + |s-1| ~ t; here we solve
       pi^{sigma-1} e^{pi t/2} |Gamma(1-s)| |s-1| / 2 = 1
    with the exact Gamma and |s-1|. It lands within O(1/t^2) of sigma_law -- i.e. the
    closed form already captures the balance; the residual to the TRUE zeros is the
    F_asym truncation, not this refinement (see test_sigma_law_residual_is_truncation).
    """
    t = mp.mpf(t)

    def bal(sig):
        s = mp.mpc(sig, t)
        return mp.log(PI**(s - 1) * mp.e**(PI * t / 2) * abs(mp.gamma(1 - s)) * abs(s - 1) / 2).real

    return float(mp.findroot(bal, ce.sigma_law(float(t))))


# --------------------------------------------------------------------------
# self-validating driver
# --------------------------------------------------------------------------
# a representative point and the first few zeros / companions used throughout the driver
_S0 = mp.mpc(0.5, 14.134725142)
_ZGAM = [14.134725142, 21.022039639, 25.010857580, 30.424876126, 32.935061588, 37.586178159]
_PREF = [2 * float(PI) * k / float(LN2) for k in (1, 2, 3)]
# a short K schedule for the (continuation-heavy) driver tables/figure
_SCHED = [1, 2, 3, 5, 8, 13, 21, 34, 55]


def _print_rate_constants():
    print("== leading O(1/K) rate: K (target - object^{(K)}) -> rate constant ==")
    print("   route        s                     K     K*err           predicted const")
    for s in [mp.mpc(2, 0), _S0, mp.mpc(1.3, 7)]:
        for K in (40, 160):
            ez = float(abs(K * (mp.zeta(s) - hb.zeta_K(s, K))))
            print(f"   comb_zeta    {complex(s)!s:>18}  {K:4d}  {ez:.6f}      "
                  f"{float(abs(rate_comb(s))):.6f}")
        ew = float(abs(rate_warp(s, richardson=True)))
        ewb = float(abs(rate_warp_bulk(s)))
        print(f"   warp         {complex(s)!s:>18}    *  full={ew:.6f}   bulk={ewb:.6f}  "
              f"(boundary layer = {ew - ewb:+.4f}; non-elementary closed form -> c_warp_closed)")


def _print_displacement():
    print("\n== zero-displacement coefficient  a_1 (s_K - rho ~ a_1/K) and approach side ==")
    print("   route        rho                          Re a_1     Im a_1     side")
    rows = ([("comb_zeta", mp.mpc(0.5, g)) for g in _ZGAM[:3]]
            + [("warp_zeta", mp.mpc(0.5, g)) for g in _ZGAM[:2]]
            + [("warp_pref", mp.mpc(0, t)) for t in _PREF[:1]]
            + [("eta_pref", mp.mpc(1, t)) for t in _PREF[:1]])
    for route, rho in rows:
        a1 = disp_coeff(route, rho)
        print(f"   {route:<11}  {complex(rho)!s:>22}   {float(a1.real):+8.4f}   "
              f"{float(a1.imag):+8.4f}   {approach_side(route, rho)}")
    print("   (comb Re a_1 ALTERNATES sign -> some zeros from above, some below;")
    print("    warp Re a_1 < 0 uniformly -> all from below.  The warp/comb contrast, sharpened.)")


def _print_overshoot():
    print("\n== overshoot: comb born ABOVE 1/2, ends sign(Re a_1); warp born BELOW, ends below ==")
    print("   gamma      comb Re a_1   comb overshoot?   warp Re a_1   warp overshoot?")
    for g in _ZGAM:
        rho = mp.mpc(0.5, g)
        ac = float(disp_coeff("comb_zeta", rho).real)
        aw = float(disp_coeff("warp_zeta", rho).real)
        print(f"   {g:8.4f}   {ac:+8.4f}      {str(overshoots(rho)):<5s}            "
              f"{aw:+8.4f}      False")


def _print_subleading():
    print("\n== sub-leading: 2-term comb displacement a_1/K + a_2/K^2 vs hb.migrate ==")
    for g in (14.134725142, 21.022039639):
        rho = mp.mpc(0.5, g)
        a1, a2 = disp_coeff("comb_zeta", rho), disp_coeff2(rho)
        traj = dict((K, zz) for K, zz in hb.migrate(hb.zeta_K, rho, schedule=_SCHED)
                    if zz is not None)
        print(f"   gamma={g:.3f}:  a_1={float(a1.real):+.4f}  a_2={float(a2.real):+.4f}")
        for K in (8, 21, 55):
            if K in traj:
                actual = float((traj[K] - rho).real)
                pred1 = float((a1 / K).real)
                pred2 = float((a1 / K + a2 / K**2).real)
                print(f"      K={K:3d}  actual={actual:+.5f}   a1/K={pred1:+.5f}   "
                      f"2-term={pred2:+.5f}")


def _print_birth_K():
    print("\n== birth-K law: catch_K(rho, eps=0.05) ~ |Re a_1|/eps grows (on avg) with gamma ==")
    print("   gamma     comb |Re a_1|   catch_K    RvM density (1/2pi)ln(t/2pi)")
    for g in _ZGAM:
        rho = mp.mpc(0.5, g)
        a1 = abs(float(disp_coeff("comb_zeta", rho).real))
        dens = float(mp.log(g / (2 * PI)) / (2 * PI))
        print(f"   {g:8.4f}    {a1:.4f}        {catch_K('comb_zeta', rho):5.1f}      {dens:.4f}")


def _print_axis_duality():
    print("\n== axis-duality: comb pins the VERTICAL (Im a_1>0), warp pins the HORIZONTAL (Re a_1<0) ==")
    print("   gamma      comb Re a_1  comb Im a_1  |  warp Re a_1  warp Im a_1")
    for g in _ZGAM:
        rho = mp.mpc(0.5, g)
        c = disp_coeff("comb_zeta", rho)
        w = disp_coeff("warp_zeta", rho)
        print(f"   {g:8.4f}   {float(c.real):+8.4f}    {float(c.imag):+8.4f}    |  "
              f"{float(w.real):+8.4f}    {float(w.imag):+8.4f}")
    print("   comb: Re a_1 ALTERNATES (cos pi*delta) / Im a_1 uniformly > 0 (sin pi*delta, = Gram-regular).")
    print("   warp: Re a_1 uniformly < 0 / Im a_1 OSCILLATES (inherits the wandering C_warp phase).")
    print("   -> each route is uniform on one axis, alternating on the other, and they SWAP axes.")


def _print_vertical_universality():
    print("\n== vertical universality: comb Im a_1>0 IS Gram's law; first flips at n=127/136/196 ==")
    print("   n     gamma      delta_n    Im a_1        (-1)^n sin th    Im a_1>0?  Gram-regular?")
    for n in (1, 2, 50, 100, 126, 127, 136, 195, 196):
        g = mp.zetazero(n).imag
        a1 = disp_coeff("comb_zeta", mp.mpc(0.5, g))
        d = gram_position(g, n)
        crit = float((-1)**n * mp.sin(mp.siegeltheta(g)))
        reg = 0 < d < 1
        flag = "  <== FLIP" if a1.imag <= 0 else ""
        print(f"   {n:>3}  {float(g):8.3f}  {d:+8.4f}   {float(a1.imag):+.4e}   {crit:+.4e}      "
              f"{str(a1.imag > 0):>5}      {str(reg):>5}{flag}")
    print("   delta_n in (0,1) <=> Im a_1 > 0 (vertical UP).  Flips have |Im a_1| ~ 0 (zero nearly ON a")
    print("   Gram point) -> slowest to settle.  arg(a_1) ~ pi*delta_n: vertical migration = Gram detector.")


def _print_sigma_law():
    print("\n== continuous-eta sigma_law sub-leading: exact balance == closed form -> residual is truncation ==")
    print("   t       sigma_true   sigma_law   sigma_law_exact   res_law    res_exact")
    zeros = ce.find_zeros(60.0)
    for z in zeros[::3]:
        t, st = float(z.imag), float(z.real)
        sl, se = ce.sigma_law(t), sigma_law_exact(t)
        print(f"   {t:7.3f}  {st:.5f}     {sl:.5f}     {se:.5f}        "
              f"{st - sl:+.2e}   {st - se:+.2e}")
    print("   (res_law ~ res_exact: refining the balance doesn't help; the drift is the")
    print("    F_asym truncation -- the dropped higher incomplete-gamma terms of F.)")


def _print_cwarp():
    print("\n== loose end 1: C_warp closed form -- bulk + Si-Gibbs boundary layer ==")
    print("   s                   |bulk|     |closed|   |richardson(num)|   |closed-num|")
    for s in [mp.mpc(2, 0), mp.mpc(1.3, 7)]:
        bulk = abs(rate_warp_bulk(s))
        clo = c_warp_closed(s)
        num = rate_warp(s, richardson=True)
        print(f"   {complex(s)!s:>16}   {float(bulk):.5f}    {float(abs(clo)):.5f}     "
              f"{float(abs(num)):.5f}            {float(abs(clo - num)):.2e}")
    print("   (closed == numerical to the richardson truth's own ~5-digit accuracy; bulk alone")
    print("    misses the Gibbs boundary layer; the naive linear-only form overshoots ~2-3x.)")


def _print_montgomery():
    import numpy as np
    print("\n== loose end 2: catch_K <-> |zeta'(rho)| (Montgomery a-values) ==")
    print("   n   gamma     |zeta'|  1/|zeta'|  catch_abs  catch_re  |cos arg a1|")
    N = 20
    norm, invz = [], []
    for n in range(1, N + 1):
        g = mp.zetazero(n).imag
        rho = mp.mpc(0.5, g)
        a1 = disp_coeff("comb_zeta", rho)
        zp = float(montgomery_a(rho))
        ca = catch_K("comb_zeta", rho, mode="abs")
        cr = catch_K("comb_zeta", rho, mode="re")
        cosg = float(abs(a1.real) / abs(a1))
        norm.append(catch_K_normalized("comb_zeta", rho, mode="abs"))
        invz.append(1.0 / zp)
        if n <= 12:
            print(f"   {n:2d}  {float(g):8.3f}  {zp:.4f}   {1/zp:.4f}    {ca:6.2f}    "
                  f"{cr:6.2f}    {cosg:.4f}")
    c = float(np.corrcoef(norm, invz)[0, 1])
    print(f"   corr(normalized catch_abs, 1/|zeta'|) = {c:.6f}  (== 1: catch_abs IS 1/|zeta'|,")
    print("    the Montgomery a-value). mode='re' adds the Gram factor |cos pi delta_n|.")


def _print_fe_mirror():
    print("\n== loose end 3: FE-mirror companions -- same heights, broken dynamics ==")
    print("   k   t_k       eta a1 (sigma=1)      warp a1 (sigma=0)     d1_eta Re  d1_warp=C_warp")
    for k in (1, 2, 3, 4):
        m = fe_mirror(k)
        ae, aw, de, dw = m["a1_eta"], m["a1_warp"], m["d1_eta"], m["d1_warp"]
        print(f"   {k}  {m['t']:8.3f}  Re{float(ae.real):+.3f} Im{float(ae.imag):+.3f}  "
              f"Re{float(aw.real):+.3f} Im{float(aw.imag):+.3f}  {float(de.real):+.5f}  "
              f"Re{float(dw.real):+.3f} Im{float(dw.imag):+.3f}")
    print(f"   eta d1 Re == -1/2 pi^2 = {-1/(2*float(PI)**2):+.5f} (rigid line); warp d1=C_warp Re wanders.")
    print("   zero SETS are exact s<->1-s mirrors (the p=2 comb); the DYNAMICS are not -- the")
    print("   break is entirely Delta_1 = C_warp (analytic boundary layer, not arithmetic).")


def _main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figstyle
    figstyle.enlarge()
    # these two figures are dense (2x2 + a 3-up) with long titles, so keep titles and
    # legends a touch smaller than the global style while still much larger than before.
    plt.rcParams.update({"axes.titlesize": 13, "legend.fontsize": 11})

    _print_rate_constants()
    _print_displacement()
    _print_overshoot()
    _print_subleading()
    _print_birth_K()
    _print_axis_duality()
    _print_vertical_universality()
    _print_sigma_law()
    _print_cwarp()
    _print_montgomery()
    _print_fe_mirror()

    # ================================ figure ================================
    fig, axg = plt.subplots(2, 2, figsize=(14, 9.5))
    ax = [axg[0, 0], axg[0, 1], axg[1, 0]]      # the three rate-law panels keep their indices

    # panel 0: leading rate -- K|target - object^{(K)}| -> the route constant (flat = O(1/K))
    Ks = [4, 8, 16, 32, 64, 128]
    comb = [float(abs(K * (mp.zeta(_S0) - hb.zeta_K(_S0, K)))) for K in Ks]
    Kw = [4, 8, 16, 24, 32, 40]
    warp = [float(abs(K * (wb.warp_K(_S0, K) - mp.zeta(_S0, mp.mpf(3) / 2)))) for K in Kw]
    ax[0].semilogx(Ks, comb, "-o", color="C0", label=r"comb $K|\zeta-\zeta^{(K)}|$")
    ax[0].axhline(float(abs(rate_comb(_S0))), ls=":", color="C0",
                  label=r"$|s/2\pi^2|$ (closed)")
    ax[0].semilogx(Kw, warp, "-s", color="C3", label=r"warp $K|warp_K-\zeta(s,3/2)|$")
    ax[0].axhline(float(abs(rate_warp(_S0, richardson=True))), ls="--", color="C3",
                  label=r"$|C_{\rm warp}|$ (bulk+Gibbs)")
    ax[0].axhline(float(abs(rate_warp_bulk(_S0))), ls=":", color="C1",
                  label=r"bulk only (misses BL)")
    ax[0].set_xlabel("K"); ax[0].set_ylabel(r"$K\cdot|\mathrm{error}|$ at $s=\frac{1}{2}+14.13i$")
    ax[0].set_title(r"leading rate: $K\cdot$error $\to$ a route constant")
    ax[0].legend(loc="center right")

    # panel 1: displacement Re(s_K - rho) vs 1/K -- comb from above (g=14), comb overshoot
    # (g=21, crosses 0), warp from below (g=14). Lines = predicted slope a_1.
    def _traj_re(fn, rho, sched):
        d = dict((K, zz) for K, zz in hb.migrate(fn, rho, schedule=sched) if zz is not None)
        return [(1.0 / K, float((d[K] - rho).real)) for K in sched if K in d]

    cases = [(hb.zeta_K, mp.mpc(0.5, 14.134725142), "comb $\\gamma$=14 (above)", "C0", "o"),
             (hb.zeta_K, mp.mpc(0.5, 21.022039639), "comb $\\gamma$=21 (overshoot)", "C1", "s"),
             (wb.warp_half_K, mp.mpc(0.5, 14.134725142), "warp $\\gamma$=14 (below)", "C3", "^")]
    for fn, rho, lab, col, mk in cases:
        sched = wb.K_SCHEDULE if fn is wb.warp_half_K else _SCHED
        pts = _traj_re(fn, rho, sched)
        if pts:
            xs, ys = zip(*pts)
            ax[1].plot(xs, ys, mk, color=col, ms=4, label=lab)
            route = "warp_zeta" if fn is wb.warp_half_K else "comb_zeta"
            a1 = float(disp_coeff(route, rho).real)
            xx = [0, max(xs)]
            ax[1].plot(xx, [a1 * x for x in xx], "-", color=col, lw=1)
    ax[1].axhline(0.0, ls="-", lw=1, color="0.6")
    ax[1].set_xlabel(r"$1/K$"); ax[1].set_ylabel(r"$\mathrm{Re}(s_K-\rho)$")
    ax[1].set_title(r"displacement $\to a_1/K$ (slope $=a_1$); sign $=$ approach side")
    ax[1].legend(loc="upper left")

    # panel 2: birth-K law -- catch_K(comb_zeta) grows with gamma; RvM density overlaid
    gs = [float(mp.zetazero(n).imag) for n in range(1, 16)]
    cks = [catch_K("comb_zeta", mp.mpc(0.5, g)) for g in gs]
    ax[2].plot(gs, cks, "o", color="C0", label=r"catch-$K=|\mathrm{Re}\,a_1|/\epsilon$")
    # running max to show the growing envelope through the per-zero fluctuation
    run = []
    m = 0.0
    for c in cks:
        m = max(m, c)
        run.append(m)
    ax[2].plot(gs, run, "-", color="C0", lw=1, alpha=0.5, label="running max (envelope)")
    ax[2].set_xlabel(r"$\gamma$ (zero height)")
    ax[2].set_ylabel(r"catch-$K$ ($\epsilon=0.05$)")
    ax[2].set_title(r"birth-$K$ grows with height (denser zeros, more harmonics)")
    ax[2].legend(loc="upper left")

    # panel 3: vertical migration Im a_1 vs gamma. Comb pinned > 0 (= Gram's law);
    # it flips NEGATIVE exactly where Gram's law fails (delta_n leaves (0,1)) -- first the
    # 127th zero. Warp Im a_1 (low gamma) oscillates -- the axis-duality.
    NMAX = 140
    gv = [float(mp.zetazero(n).imag) for n in range(1, NMAX + 1)]
    ima = [float(disp_coeff("comb_zeta", mp.mpc(0.5, g)).imag) for g in gv]
    reg = [0 < gram_position(g, n) < 1 for n, g in enumerate(gv, start=1)]
    axv = axg[1, 1]
    gr = [(g, y) for g, y, r in zip(gv, ima, reg) if r]
    ir = [(g, y) for g, y, r in zip(gv, ima, reg) if not r]
    if gr:
        axv.plot(*zip(*gr), "o", color="C0", ms=3.5, label=r"Gram-regular ($\mathrm{Im}\,a_1>0$)")
    if ir:
        axv.plot(*zip(*ir), "v", color="C3", ms=8, label="Gram-law failure (flip)")
    # warp Im a_1 (first zeros) -- oscillates around 0 (the swapped axis)
    gw = _ZGAM
    iw = [float(disp_coeff("warp_zeta", mp.mpc(0.5, g)).imag) for g in gw]
    axv.plot(gw, iw, "s", color="C1", ms=4, alpha=0.7, label=r"warp $\mathrm{Im}\,a_1$ (oscillates)")
    axv.axhline(0.0, ls="-", lw=1, color="0.6")
    for n, dx in ((127, (-34, 7)), (136, (6, -14))):
        axv.annotate(f"n={n}", (gv[n - 1], ima[n - 1]), textcoords="offset points",
                     xytext=dx, fontsize=11, color="C3")
    axv.set_xlabel(r"$\gamma$ (zero height)")
    axv.set_ylabel(r"$\mathrm{Im}\,a_1$ (vertical migration)")
    axv.set_title(r"comb $\mathrm{Im}\,a_1>0$ IS Gram's law (flips at $n$=127,136); warp oscillates")
    axv.legend(loc="upper left")

    fig.suptitle(r"Unified $O(1/K)$ rate / birth-$K$ law + vertical axis-duality: "
                 r"$\arg a_1\!\approx\!\pi\delta_n$ -- comb $\mathrm{Im}\,a_1>0$ iff Gram-regular")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    figdir = Path(__file__).resolve().parent / "figures"
    figdir.mkdir(exist_ok=True)
    out = figdir / "rate_law.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")

    # ===================== loose-ends figure (3 panels) =====================
    fig2, bx = plt.subplots(1, 3, figsize=(16, 4.8))

    # panel A (item 1): C_warp closed form is the K->inf limit of K(warp_K - zeta(s,3/2));
    # the bulk alone misses the Gibbs boundary layer.
    KwA = [4, 8, 16, 24, 32, 40]
    z32 = mp.zeta(_S0, mp.mpf(3) / 2)
    warpA = [float(abs(K * (wb.warp_K(_S0, K) - z32))) for K in KwA]
    bx[0].semilogx(KwA, warpA, "-s", color="C3", label=r"$K|warp_K-\zeta(s,\frac{3}{2})|$")
    bx[0].axhline(float(abs(c_warp_closed(_S0))), ls="--", color="C0",
                  label=r"$|C_{\rm warp}|$ closed (Si-Gibbs)")
    bx[0].axhline(float(abs(rate_warp_bulk(_S0))), ls=":", color="C1",
                  label="bulk only (misses BL)")
    bx[0].set_xlabel("K"); bx[0].set_ylabel(r"$K\cdot|\mathrm{error}|$ at $s=\frac{1}{2}+14.13i$")
    bx[0].set_title(r"item 1: $C_{\rm warp}=$ bulk $+$ Si-Gibbs boundary layer")
    bx[0].legend(loc="center right")

    # panel B (item 2): normalized catch_K vs 1/|zeta'|. mode='abs' lies ON the diagonal
    # (== the Montgomery a-value 1/|zeta'|); mode='re' scatters BELOW (the Gram |cos pi delta|<=1).
    gsB = [mp.zetazero(n).imag for n in range(1, 31)]
    invz = [1.0 / float(montgomery_a(mp.mpc(0.5, g))) for g in gsB]
    nabs = [catch_K_normalized("comb_zeta", mp.mpc(0.5, g), mode="abs") for g in gsB]
    nre = [catch_K_normalized("comb_zeta", mp.mpc(0.5, g), mode="re") for g in gsB]
    lim = [0, max(invz) * 1.05]
    bx[1].plot(lim, lim, "-", color="0.6", lw=1, label="diagonal $y=x$")
    bx[1].plot(invz, nabs, "o", color="C0", ms=5, label=r"mode='abs' $=1/|\zeta'|$ (Montgomery)")
    bx[1].plot(invz, nre, "v", color="C3", ms=5, label=r"mode='re' $\times|\cos\pi\delta_n|$ (Gram)")
    bx[1].set_xlabel(r"$1/|\zeta'(\rho)|$ (Montgomery a-value reciprocal)")
    bx[1].set_ylabel(r"normalized catch-$K$")
    bx[1].set_title(r"item 2: catch-$K$(abs)$=1/|\zeta'|$; (re) adds Gram $|\cos\pi\delta_n|$")
    bx[1].legend(loc="upper left")

    # panel C (item 3): Re Delta_1 at the FE-mirror companion heights. eta = -1/2 pi^2 (rigid
    # line, the closed comb constant); warp = C_warp (wanders). Same heights, broken dynamics.
    mirrors = [fe_mirror(k) for k in range(1, 7)]
    ts = [m["t"] for m in mirrors]
    re_eta = [float(m["d1_eta"].real) for m in mirrors]
    re_warp = [float(m["d1_warp"].real) for m in mirrors]
    bx[2].axhline(-1 / (2 * float(PI)**2), ls="--", color="C0",
                  label=r"$\eta$: $\Delta_1=-\rho/2\pi^2$, $\mathrm{Re}=-1/2\pi^2$ (rigid)")
    bx[2].plot(ts, re_eta, "o", color="C0", ms=5)
    bx[2].plot(ts, re_warp, "s-", color="C3", ms=5,
               label=r"warp: $\Delta_1=C_{\rm warp}(\rho)$ (wanders)")
    bx[2].set_xlabel(r"$t_k=2\pi k/\ln2$ (shared companion height)")
    bx[2].set_ylabel(r"$\mathrm{Re}\,\Delta_1$")
    bx[2].set_title(r"item 3: same heights, broken dynamics")
    bx[2].legend(loc="upper left")

    fig2.suptitle(r"Rate-law loose ends: $C_{\rm warp}$ closed form; catch-$K$ vs "
                  r"$|\zeta'|$ (Montgomery a-values); FE-mirror companions' broken dynamics")
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    out2 = figdir / "rate_law_loose_ends.png"
    fig2.savefig(out2, dpi=150)
    print(f"wrote {out2}")


if __name__ == "__main__":
    _main()
