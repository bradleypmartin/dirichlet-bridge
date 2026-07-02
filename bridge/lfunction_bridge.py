r"""The K-knob (+ warp) on a Dirichlet L-function  L(s, chi_4) -- beyond zeta/eta.

The headline fork-2 experiment (issue #20, epic #16). Every earlier stage of the
discrete <-> continuous bridge worked on zeta or eta. Here we take the SAME machinery --
the harmonic-truncation K-knob (the Euler-Maclaurin sawtooth == Abel-Plana comb,
harmonic_bridge.py) and the nonlinear warp (warp_bridge.py / warp_eta.py) -- and apply
it to a genuine primitive Dirichlet L-function, the first `zeroless-endpoint -> born ->
onto sigma=1/2` test for a family beyond zeta/eta (LITERATURE.md sec4).

    L(s, chi_4) = sum_{n>=1} chi_4(n) n^{-s} = 1 - 3^{-s} + 5^{-s} - 7^{-s} + ...

chi_4 is the odd non-principal character mod 4. It has its OWN critical line Re s = 1/2,
its OWN functional equation (odd character, Gauss sum tau(chi_4) = 2i, root number W = 1),
and NO pole (entire). It sits in the Lerch family at  L(1/2, 1/2, s) = 2^s L(s, chi_4)
(GarunkstisTamosiunas2017; LITERATURE.md sec4.3).

The one identity that drives everything
---------------------------------------
chi_4(n) is the value at the integers of a single continuous sign-carrier:

    chi_4(n) = sin(pi n / 2)        (n = 1,2,3,4,...  ->  +1, 0, -1, 0, +1, ...).

So chi_4 is a mean-zero periodic coefficient, exactly like eta's (-1)^{n-1} = -cos(pi n).
That settles the design crux (issue #20): the bland continuous endpoint is NOT the zeroless
1/(s-1) of the zeta side, but the STRUCTURED oscillatory integral

    G(s) = int_1^inf sin(pi x / 2) x^{-s} dx   (= harmonic_bridge.Smom(pi/2, s)),

the chi_4 analog of eta's continuous endpoint F (cont_eta.py). G carries an off-critical
Riemann-von Mangoldt zero string (Re -> 3/2, same laws as F with pi -> pi/2). So the chi_4
bridge is **eta-type** -- zeros migrate from a ground string, not born from nothing --
but with a decisive simplification: L(s, chi_4) is PRIMITIVE (does not factor as
prefactor x zeta), so there is **no companion line**. Where eta split its ground string
onto sigma=1/2 U sigma=1, chi_4's whole string lands on the single line sigma=1/2. The
densities even match on the nose: G's string has density (t/2pi) ln(2t/pi e), and that IS
the L(s, chi_4) zero density (conductor q=4 supplies the doubling eta got from its comb),
so the migration is a 1:1 map of the ground string onto the critical line -- no split.

The K-knob (the additive comb) -- the headline
----------------------------------------------
Euler-Maclaurin on sum_{n>=1} f(n), f(x) = sin(pi x/2) x^{-s}, with the sawtooth
B~_1(x) = {x} - 1/2 = -sum_k sin(2 pi k x)/(pi k) truncated at K harmonics:

    L_comb_K(s) = G(s) + 1/2 - sum_{k=1}^K D_k(s)/(pi k)   ->   L(s, chi_4),

f(1)/2 = sin(pi/2)/2 = 1/2 the load-bearing half-weight (as in eta_K), and D_k the k-th
sawtooth mode of f' living at the ODD-QUARTER frequencies (2k +/- 1/2) pi (chi_4's carrier
sin(pi x/2) has fundamental pi/2, so its comb sits at 2k +/- 1/2, where eta's sat at 2k +/- 1).
The K=0 anchor G + 1/2 is the ground string (Re ~ 0.8-0.96, pulled off G's Re ~ 1.7 string
by the half-weight, exactly as in eta_K). As K grows the zeros migrate onto sigma=1/2 --
ALL of them, one clean line -- at O(1/K) (rate_law.py's constant s/2 pi^2 governs it, the
shared sawtooth tail).

The warp -- the Lerch tie-in, and a modulus surprise
----------------------------------------------------
The nonlinear midpoint warp x -> n* = x + phi_K(x) pins each cell [m, m+1] onto m + 1/2.
Warping a sign-carrier c(y) y^{-s} through it (warp_eta.py's construction), cell m collapses
onto c(m + 1/2)(m + 1/2)^{-s}. Two carriers, two stories:

  * c(y) = sin(pi y)  (the phase-1/2 carrier): sin(pi(m + 1/2)) = (-1)^m, so the warp lands on
        warp_chi4_K(s, K) = warp_chi4_alpha(s, K, 1/2) + 2^s  ->  sum_{m>=0} (-1)^m (m+1/2)^{-s}
                          = L(1/2, 1/2, s) = 2^s L(s, chi_4),
    the Lerch tie-in (issue #20's lead). The +2^s = (1/2)^{-s} is the SAME load-bearing m=0
    midpoint half-weight the zeta warp used (warp_bridge.warp_half_K). Its zeros are L(s, chi_4)'s
    -- on sigma=1/2, with NO companion (2^s never vanishes; contrast the zeta warp's sigma=0
    comb). This is the exact half-cell-shifted DUAL of warp_eta: there cos(pi y) had alpha=1
    (integers) survive and alpha=1/2 (midpoints) die; here sin(pi y) has alpha=1/2 survive and
    alpha=1 die (sin vanishes at the integers). cos and sin, midpoint and integer, swapped.

  * c(y) = sin(pi y / 2)  (the GENUINE chi_4 carrier): its midpoint samples sin(pi(m+1/2)/2) run
    +,+,-,- with period 8, NOT chi_4's period 4. So warping the genuine chi_4 integrand at the
    midpoints lands NOT on chi_4 but on a mod-8 character,
        sum_{m>=0} sin(pi(m+1/2)/2)(m+1/2)^{-s} = 2^s (sqrt2/2) L(s, chi_{-8}),
    chi_{-8} the odd character mod 8 (chi(1,3,5,7) = +,+,-,-). A documented modulus shift:
    sampling a period-q carrier on the half-integer lattice can reveal a period-2q character.
    (Verified as an identity in __main__; the clean chi_4 target is the sin(pi y) route above.)

Both warp targets are clean single lines sigma=1/2 -- the L-function has no prefactor to
spin off a companion.

The fork-2 verdict (LITERATURE.md sec4): the bridge's `born-and-migrate-onto-1/2` behaviour
SURVIVES for a Dirichlet L-function with its own critical line and functional equation -- and
is CLEANER there than for eta (single line, no split). No prior deform-and-track work combines
a zeroless/structured endpoint + zeros born from it + migration ONTO 1/2 for any family != zeta/eta.

Run directly to validate + plot: writes bridge/figures/lfunction_bridge.png.
"""
import math
import sys
from pathlib import Path
from typing import List

import mpmath as mp

# This module sits in a flat source dir alongside its bare cross-imports; put that
# directory on sys.path so a direct ``python lfunction_bridge.py`` run resolves them
# (pytest gets the same path from the root conftest.py).
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import harmonic_bridge as hb   # noqa: E402  (Cmom/Smom moments, migrate -- the comb machinery)
import warp_bridge as wb       # noqa: E402  (warp_phi, the shared GL grid + moment machinery, |Im s| scalers)
import rate_law as rl          # noqa: E402  (rate_comb = s/2 pi^2 -- the shared O(1/K) comb constant)

mp.mp.dps = 30

PI = mp.pi
HALF_FREQ = PI / 2                         # chi_4's carrier sin(pi x/2) has fundamental pi/2
_LN_HALF = float(mp.log(PI / 2))           # ln(pi/2), the pi -> pi/2 shift of cont_eta's laws


# --------------------------------------------------------------------------
# 1. the L-function  L(s, chi_4)  and its anchors
# --------------------------------------------------------------------------
def chi4(n):
    """The odd non-principal character mod 4:  chi_4(n) = sin(pi n / 2) at the integers."""
    n = int(n) % 4
    return {0: 0, 1: 1, 2: 0, 3: -1}[n]


def L_chi4(s):
    """L(s, chi_4) via the Hurwitz combination  4^{-s}[zeta(s,1/4) - zeta(s,3/4)]  (entire).

    = sum_{m>=0}[(4m+1)^{-s} - (4m+3)^{-s}] = 1 - 3^{-s} + 5^{-s} - ... The individual Hurwitz
    zetas each carry the pole 1/(s-1), which cancels in the difference -- L is entire -- but the
    mpmath evaluator still returns inf at s=1 exactly, so we splice in the removable value
    L(1, chi_4) = 4^{-1}(psi(3/4) - psi(1/4)) = pi/4 (Leibniz) there.
    """
    s = mp.mpc(s)
    if abs(s - 1) < mp.mpf("1e-12"):
        return mp.power(4, -s) * (mp.digamma(mp.mpf(3) / 4) - mp.digamma(mp.mpf(1) / 4))
    return mp.power(4, -s) * (mp.zeta(s, mp.mpf(1) / 4) - mp.zeta(s, mp.mpf(3) / 4))


def L_chi4_sum(s, terms=200000):
    """L(s, chi_4) by direct summation of the defining series -- the sigma>1 reference.

    chi_4 vanishes on the evens, so only the odd terms n = 2m+1 (with chi_4 = (-1)^m)
    are computed -- half the mpc powers of the naive sum over every n <= terms.
    """
    s = mp.mpc(s)
    return mp.nsum(lambda m: (-1) ** m * (2 * m + 1) ** (-s), [0, (terms - 1) // 2])


def completed_L(s):
    """The completed L-function  Lambda(s) = (q/pi)^{(s+1)/2} Gamma((s+1)/2) L(s, chi_4), q=4.

    For the odd primitive character chi_4 the functional equation is  Lambda(s) = W Lambda(1-s)
    with root number W = tau(chi_4)/(i sqrt q) = 2i/(2i) = 1. So Lambda is symmetric about
    sigma=1/2 (an anchor: `functional_eq_residual` vanishes).
    """
    s = mp.mpc(s)
    return (mp.mpf(4) / PI) ** ((s + 1) / 2) * mp.gamma((s + 1) / 2) * L_chi4(s)


def functional_eq_residual(s):
    """|Lambda(s) - Lambda(1-s)| -- vanishes (odd-character functional equation, W = 1)."""
    return abs(completed_L(s) - completed_L(1 - s))


# --------------------------------------------------------------------------
# 2. the continuous endpoint  G(s) = int_1^inf sin(pi x/2) x^{-s} dx  (the eta-type string)
# --------------------------------------------------------------------------
def G(s):
    """The bland continuous endpoint  G(s) = int_1^inf sin(pi x/2) x^{-s} dx = Smom(pi/2, s).

    The chi_4 analog of eta's continuous endpoint F (cont_eta.F_closed = Cmom(pi, s)). NOT
    zeroless: it carries an off-critical Riemann-von Mangoldt string, Re -> 3/2, with the
    cont_eta laws at frequency pi/2. This is the honest answer to issue #20's design crux --
    the mean-zero periodic coefficient gives a structured (eta-type) endpoint, not a trivial one.
    """
    return hb.Smom(HALF_FREQ, s)


def G_quad(s):
    """G(s) by direct oscillatory quadrature (period-4 sine) -- the reference for G = Smom."""
    s = mp.mpc(s)
    return mp.quadosc(lambda x: mp.sin(HALF_FREQ * x) * x ** (-s), [1, mp.inf], period=4)


def sigma_law(t):
    """Real part of the G-string zero at height t (magnitude balance) -> 3/2.

    Solves (pi/2)^{sigma-1} sqrt(2 pi) t^{3/2-sigma} = 2 -- cont_eta.sigma_law with pi -> pi/2.
    """
    lt = math.log(t)
    return (1.5 * lt + 0.5 * math.log(2 * math.pi) - math.log(2) - _LN_HALF) / (lt - _LN_HALF)


def spacing_law(t):
    """Gap to the next G-string zero at height t:  2 pi / ln(t/(pi/2))  (cont_eta with pi->pi/2)."""
    return 2 * math.pi / (math.log(t) - _LN_HALF)


def counting_law(t):
    """Number of G-string zeros with 0 < Im <= t (Riemann-von Mangoldt form, freq pi/2).

    (t/2pi) ln(t/((pi/2) e)) + 5/8 = (t/2pi) ln(2t/(pi e)) + 5/8. The leading density
    (t/2pi) ln(2t/pi e) IS the L(s, chi_4) zero density (q=4 doubling) -- witnessed in _main.
    """
    return t / (2 * math.pi) * (math.log(t) - _LN_HALF - 1) + 0.625


def _polish_G(t_seed):
    """findroot on G near (sigma_law(t_seed), t_seed); return the root or None."""
    try:
        z = mp.findroot(G, mp.mpc(sigma_law(t_seed), t_seed))
    except (ValueError, ZeroDivisionError):
        return None
    if abs(G(z)) < mp.mpf(10) ** (-mp.mp.dps + 8) and z.imag > 1:
        return z
    return None


def find_G_zeros(t_max, t_first=7.3):
    """Zeros of G with 0 < Im <= t_max, walked upward via the spacing law (cf cont_eta.find_zeros)."""
    zeros = []  # type: List[mp.mpc]
    z = _polish_G(t_first)
    if z is None:
        for ts in [t_first - 0.5, t_first + 0.5, t_first + 1.5, t_first + 2.5]:
            z = _polish_G(ts)
            if z is not None:
                break
    if z is None:
        raise RuntimeError("could not locate the first G-string zero")
    zeros.append(z)
    t = float(z.imag)
    while True:
        t_next = t + spacing_law(t)
        if t_next > t_max:
            break
        z = _polish_G(t_next)
        if z is None or float(z.imag) <= t + 0.1:
            z = _polish_G(t_next + 0.3)
        if z is None or float(z.imag) <= t + 0.1:
            break
        zeros.append(z)
        t = float(z.imag)
    return zeros


# --------------------------------------------------------------------------
# 3. the additive comb  L_comb_K(s) = G(s) + 1/2 - sum_k D_k(s)/(pi k)  ->  L(s, chi_4)
# --------------------------------------------------------------------------
def _Dk(s, k):
    """int_1^inf sin(2 pi k x) f'(x) dx,  f = sin(pi x/2) x^{-s} (the chi_4 sawtooth mode).

    f'(x) = (pi/2) cos(pi x/2) x^{-s} - s sin(pi x/2) x^{-s-1}; the products collapse (by
    product-to-sum) to sine/cosine moments at the ODD-QUARTER frequencies (2k +/- 1/2) pi -- the
    exact analog of eta's _Dk (sin<->cos swapped, and 2k +/- 1 -> 2k +/- 1/2 for the pi/2 carrier).
    """
    s = mp.mpc(s)
    wm = (2 * k - mp.mpf(1) / 2) * PI
    wp = (2 * k + mp.mpf(1) / 2) * PI
    return (PI / 4 * (hb.Smom(wp, s) + hb.Smom(wm, s))
            - s / 2 * (hb.Cmom(wm, s + 1) - hb.Cmom(wp, s + 1)))


def L_comb_K(s, K):
    """L^{(K)}(s, chi_4) via the Euler-Maclaurin sawtooth. K=0 anchor: G(s) + 1/2.

    Restores discreteness one Fourier harmonic at a time; K -> infinity is L(s, chi_4). The
    K=0 ground state G + 1/2 (the continuous-endpoint integral + the load-bearing n=1 half-weight
    sin(pi/2)/2 = 1/2) carries the ground zero string that migrates onto sigma=1/2.
    """
    s = mp.mpc(s)
    tot = G(s) + mp.mpf("0.5")
    for k in range(1, K + 1):
        tot -= _Dk(s, k) / (PI * k)
    return tot


# --------------------------------------------------------------------------
# 4. the warp  ->  2^s L(s, chi_4) = L(1/2, 1/2, s)  (the phase-1/2 carrier sin(pi y))
# --------------------------------------------------------------------------
# Fast evaluator -- the warp_bridge.warp_K grid+moment expansion (see the block comment there),
# adapted to the phase-1/2 carrier g(y) = sin(pi y) y^{-s}. Period-collapse with
# sin(pi(m+g)) = (-1)^m sin(pi g) gives
#   int_1^inf sin(pi n*)(n*)^{-s} dx = sum_{m>=1} (-1)^m int_0^1 sin(pi g)(m+g)^{-s} du,
# g = u + (alpha-1/2) + phi_K(u), and the m>M tail is continued by the binomial-moment series with
# SIN-weighted moments nu_j = int_0^1 sin(pi g) g^j du and the alternating tail
# sum_{m>M}(-1)^m m^{-s-j} = -eta(s+j) - sum_{m<=M}(-1)^m m^{-s-j} (mp.altzeta is entire). This is
# warp_eta.py's machinery verbatim with cos(pi g) -> sin(pi g); the (-1)^m cell sign and the
# alternating -eta tail are shared.
_M = 14               # direct cells m=1..M; the analytically-continued alternating tail covers m>M
_GRID = {}            # (K, round(alpha,12), degree, prec) -> (logs[m-1][i], sgw[i], nus[j])


def _chi4_grid(K, alpha, degree, Jmax=60):
    """Precompute (once per (K, alpha, degree, working precision)): per-cell logs log(m+g_i),
    the sin-folded weights sgw_i = w_i sin(pi g_i), and the sin-weighted moments nu_j = sum_i
    sgw_i g_i^j (rebuilt with more terms when a higher |Im s| needs a larger Jmax)."""
    key = (K, round(float(alpha), 12), degree, mp.mp.prec)
    cached = _GRID.get(key)
    if cached is None or len(cached[2]) < Jmax + 1:
        us, ws = wb._gl_nodes(degree)
        c = mp.mpf(alpha) - mp.mpf("0.5")
        gs = [u + c + wb.warp_phi(u, K) for u in us]
        sgw = [ws[i] * mp.sin(PI * gs[i]) for i in range(len(us))]
        logs = [[mp.log(m + gs[i]) for i in range(len(us))] for m in range(1, _M + 1)]
        nus = [mp.fsum(sgw[i] * gs[i] ** j for i in range(len(us))) for j in range(Jmax + 1)]
        _GRID[key] = (logs, sgw, nus)
    return _GRID[key]


def _chi4_moment_series(s, logs, sgw, nus, base_dps):
    """The binomial-moment sum for int_1^inf sin(pi n*)(n*)^{-s} dx from a precomputed grid.

    The caller must already be inside mp.workdps(wb._moment_dps(s)); the loop breaks at the
    ambient base_dps so the binomial factor never amplifies floored-tail garbage (exactly as in
    warp_eta._eta_moment_series, here with the sin(pi g) weight)."""
    n = len(sgw)
    eps = mp.mpf(10) ** (-(base_dps - 1))
    tot = mp.fsum((-1) ** m * mp.fsum(sgw[i] * mp.exp(-s * logs[m - 1][i]) for i in range(n))
                  for m in range(1, _M + 1))                        # sum_{m<=M} (-1)^m int sin(pi g)(m+g)^{-s}
    c = mp.mpf(1)                                                   # binom(-s, 0)
    mpows = [(-1) ** m * mp.power(m, -s) for m in range(1, _M + 1)]  # (-1)^m m^{-(s+j)}, stepped /m per j
    for j in range(len(nus)):
        altHM = mp.fsum(mpows)
        term = c * nus[j] * (-mp.altzeta(s + j) - altHM)           # m>M alternating tail, continued
        tot += term
        if j > 4 and abs(term) < eps * (abs(tot) + 1):
            break
        c = c * (-s - j) / (j + 1)                                 # binom(-s, j+1)
        for m in range(2, _M + 1):
            mpows[m - 1] /= m                                       # (-1)^m m^{-(s+j)} -> (-1)^m m^{-(s+j+1)}
    return tot


def warp_chi4_alpha(s, K, alpha):
    """Honest warped phase-1/2 integrand  int_1^inf sin(pi n*)(n*)^{-s} dx,  n* = x+(alpha-1/2)+phi_K.

    The fast grid+moment evaluator. alpha = 1/2 -> sum_{m>=1}(-1)^m (m+1/2)^{-s} (midpoints survive);
    alpha = 1 -> 0 (the integers are sin's zeros -- the dual of warp_eta's alpha=1/2 death). Valid for
    all sigma; like warp_bridge.warp_K it scales working precision / moment count / GL degree with |Im s|.
    """
    s = mp.mpc(s)
    base_dps = mp.mp.dps
    with mp.workdps(wb._moment_dps(s)):
        degree = wb._gl_degree(s)
        logs, sgw, nus = _chi4_grid(K, alpha, degree, wb._moment_jmax(s))
        result = _chi4_moment_series(s, logs, sgw, nus, base_dps)
    return +result                                                 # round back to ambient dps


def warp_chi4_K(s, K):
    """The completed alpha=1/2 migration object  warp_chi4_alpha(s,K,1/2) + 2^s  ->  2^s L(s, chi_4).

    The +2^s = (1/2)^{-s} restores the omitted m=0 midpoint cell sin(pi/2)(1/2)^{-s} = 2^s -- the
    SAME load-bearing half-weight the zeta warp used (warp_bridge.warp_half_K). The limit
    sum_{m>=0}(-1)^m (m+1/2)^{-s} = L(1/2,1/2,s) = 2^s L(s, chi_4); its zeros are L(s, chi_4)'s, on
    the single line sigma=1/2 (2^s never vanishes -- no companion).
    """
    s = mp.mpc(s)
    return warp_chi4_alpha(s, K, 0.5) + mp.power(2, s)


def warp_chi4_lerch(s, K, alpha):
    """Transparent reference: int_0^1 sin(pi g) (-Phi(-1, s, 1+g)) du, g = u+(alpha-1/2)+phi_K.

    Uses the Lerch transcendent Phi(-1, s, a) = sum_{m>=0}(-1)^m (m+a)^{-s} (mp.lerchphi at z=-1)
    for the alternating cell sum sum_{m>=1}(-1)^m (m+g)^{-s} = -Phi(-1, s, 1+g) -- the sin-weighted
    twin of warp_eta.warp_eta_lerch. A Lerch eval per quad node (slow); the cross-check the fast
    warp_chi4_alpha matches to ~1e-16. (Avoid sigma exactly = 1, where the alternating Lerch crawls.)
    """
    s = mp.mpc(s)
    c = mp.mpf(alpha) - mp.mpf("0.5")
    return mp.quad(lambda u: (lambda g: mp.sin(PI * g) * (-mp.lerchphi(-1, s, 1 + g)))
                   (u + c + wb.warp_phi(u, K)), [0, 1])


def warp_chi4_raw(s, K, alpha, M=80):
    """The literal oscillatory int_1^M sin(pi n*)(n*)^{-s} dx, integer breakpoints (sigma > 1).

    An independent coarse confirmation (the sin-carrier analog of warp_bridge.warp_raw): the warp's
    kinks sit at the integers, so integrating panel-by-panel over [m, m+1] resolves them.
    """
    s = mp.mpc(s)
    c = mp.mpf(alpha) - mp.mpf("0.5")
    return mp.quad(lambda x: (lambda n: mp.sin(PI * n) * n ** (-s))(x + c + wb.warp_phi(x, K)),
                   [mp.mpf(m) for m in range(1, M + 1)])


def target_2sL(s):
    """The K -> infinity warp_chi4_K limit  2^s L(s, chi_4) = L(1/2, 1/2, s) = Phi(-1, s, 1/2)."""
    s = mp.mpc(s)
    return mp.power(2, s) * L_chi4(s)


# --------------------------------------------------------------------------
#   the mod-8 aside: warping the GENUINE chi_4 carrier sin(pi y/2) lands on chi_{-8}
# --------------------------------------------------------------------------
def L_chi_m8(s):
    """L(s, chi_{-8}), the odd character mod 8 (chi(1,3,5,7) = +1,+1,-1,-1), via Hurwitz on residues."""
    s = mp.mpc(s)
    return mp.power(8, -s) * (mp.zeta(s, mp.mpf(1) / 8) + mp.zeta(s, mp.mpf(3) / 8)
                              - mp.zeta(s, mp.mpf(5) / 8) - mp.zeta(s, mp.mpf(7) / 8))


def midpoint_chi4_carrier_limit(s):
    """The K->inf limit of warping the GENUINE chi_4 integrand sin(pi y/2) y^{-s} at the midpoints:

        sum_{m>=0} sin(pi(m+1/2)/2)(m+1/2)^{-s}.

    Its half-integer samples sin(pi(m+1/2)/2) = (sqrt2/2)(+,+,-,-) run with period 8, so this is
    2^s (sqrt2/2) L(s, chi_{-8}) -- a mod-8 character, NOT chi_4. `_main` verifies the identity.
    """
    s = mp.mpc(s)
    return mp.nsum(lambda m: mp.sin(PI * (m + mp.mpf(1) / 2) / 2) * (m + mp.mpf(1) / 2) ** (-s),
                   [0, mp.inf])


# --------------------------------------------------------------------------
# zero locations and the K schedule
# --------------------------------------------------------------------------
# First nontrivial zeros of L(s, chi_4) on sigma=1/2 (heights t), the migration targets.
L_ZERO_HEIGHTS = [6.020948, 10.243770, 12.988098, 16.342607, 18.291993]

# K schedule for the migration -- both the comb (closed-form moments) and the fast warp are
# affordable to K=45 (O(1/K) lands the zeros within ~2e-2 of sigma=1/2 by then).
K_SCHEDULE = [1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 27, 35, 45]


# --------------------------------------------------------------------------
# self-validating driver
# --------------------------------------------------------------------------
def _print_evaluator():
    print("== L(s, chi_4) evaluator + anchors ==")
    print("   Hurwitz combination 4^{-s}[zeta(s,1/4)-zeta(s,3/4)] vs direct sum (sigma>1):")
    worst = 0.0
    for s in [mp.mpc(2, 0), mp.mpc(3, 1), mp.mpc(2.5, 4)]:
        rel = float(abs((L_chi4(s) - L_chi4_sum(s)) / L_chi4_sum(s)))
        worst = max(worst, rel)
        print(f"      s={complex(s)!s:>12}  |L_hur - L_sum|/|L| = {rel:.2e}")
    print(f"   worst = {worst:.2e}")
    L1 = L_chi4(mp.mpc(1, 0))
    print(f"   anchor L(1, chi_4) = {float(mp.re(L1)):.12f}   pi/4 = {float(PI / 4):.12f}   "
          f"|diff| = {float(abs(L1 - PI / 4)):.1e}  (Leibniz)")
    print("   anchor functional equation  Lambda(s) = Lambda(1-s)  (odd chi, root number W=1):")
    worst_fe = 0.0
    for s in [mp.mpc(2, 1), mp.mpc(0.5, 6), mp.mpc(-1, 3)]:
        worst_fe = max(worst_fe, float(functional_eq_residual(s)))
    print(f"      worst |Lambda(s) - Lambda(1-s)| = {worst_fe:.2e}")


def _print_endpoint(zeros):
    print("\n== continuous endpoint  G(s) = int_1^inf sin(pi x/2) x^{-s} dx = Smom(pi/2, s) ==")
    worst = 0.0
    for s in [mp.mpc(2, 1), mp.mpc(0.8, 10), mp.mpc(1.5, 25)]:
        worst = max(worst, float(abs((G(s) - G_quad(s)) / G_quad(s))))
    print(f"   G = Smom vs oscillatory quadrature: worst rel err = {worst:.2e}")
    ts = [float(z.imag) for z in zeros]
    ss = [float(z.real) for z in zeros]
    print(f"   {len(zeros)} off-critical zeros with 0 < Im <= {ts[-1]:.0f}  (design-crux answer:")
    print("   the endpoint is STRUCTURED, eta-type -- NOT zeroless):")
    print("     n      t      sigma   sigma_law  |  gap    law     |  N     law")
    max_gap_rel = 0.0
    for i, (t, sg) in enumerate(zip(ts, ss)):
        sl = sigma_law(t)
        if i == 0:
            print(f"    {i+1:>2}  {t:7.3f}  {sg:7.4f}  {sl:7.4f}   |   --      --    "
                  f"|  {i+1:>2}  {counting_law(t):6.3f}")
            continue
        gap = ts[i] - ts[i - 1]
        law = spacing_law(0.5 * (ts[i] + ts[i - 1]))
        max_gap_rel = max(max_gap_rel, abs(gap - law) / gap)
        print(f"    {i+1:>2}  {t:7.3f}  {sg:7.4f}  {sl:7.4f}   |  {gap:5.3f}  {law:5.3f}  "
              f"|  {i+1:>2}  {counting_law(t):6.3f}")
    print(f"   sigma range = [{min(ss):.3f}, {max(ss):.3f}] (-> 3/2);  worst spacing-law rel err "
          f"= {max_gap_rel:.2e}")


def _print_counting(g_zeros, t_max=20.0):
    """Density conservation: G's string density == L(s, chi_4) zero density (the NO-SPLIT check)."""
    Lz = [t for t in L_ZERO_HEIGHTS if t <= t_max]
    Gz = [float(z.imag) for z in g_zeros if float(z.imag) <= t_max]
    print(f"\n== density bijection -- the SINGLE-LINE check (0 < t < {t_max:.0f}) ==")
    print(f"   L(s, chi_4) zeros on sigma=1/2:  {len(Lz)}   (heights {[round(t,2) for t in Lz]})")
    print(f"   G-string zeros (off-critical):   {len(Gz)}   RvM law N({t_max:.0f}) "
          f"= {counting_law(t_max):.2f}")
    print("   G-string density (t/2pi)ln(2t/pi e) == the L(s,chi_4) zero density (q=4 doubling),")
    print("   so the whole ground string maps 1:1 onto sigma=1/2 -- NO split (unlike eta's 1/2 U 1).")


def _print_comb_migration():
    print("\n== additive comb  L_comb_K -> L(s, chi_4): zeros born, migrate onto sigma=1/2 ==")
    hdrK = [0, 1, 5, 21, 45]
    print("   target zero        " + "".join(f"K={k:<6d}" for k in hdrK))
    trajs = []
    sched0 = [0] + K_SCHEDULE
    for g in L_ZERO_HEIGHTS[:4]:
        traj = hb.migrate(L_comb_K, mp.mpc(0.5, g), schedule=sched0)
        d = {K: zz for K, zz in traj if zz is not None}
        cells = "".join(f"{float(d[k].real):<7.3f}" if k in d else "  --   " for k in hdrK)
        print(f"   1/2+{g:8.4f}i   {cells}")
        trajs.append((g, traj))
    print("   (K=0 = the G+1/2 ground string, Re ~ 0.8-1.0;  K=45 -> sigma=1/2, one clean line)")
    return trajs


def _print_rate_constant():
    """The O(1/K) comb rate REUSES rate_law: K(L - L_comb_K) -> rate_law.rate_comb(s) = s/2 pi^2.

    chi_4's comb shares the *identical* leading constant with the zeta and eta combs -- the
    sawtooth tail sum_{k>K} 1/(pi k)^2 ~ 1/(pi^2 K) is weighted by the sign-carrier at the integer
    endpoint x=1, and sin(pi/2) = 1 there, just as zeta has 1 and eta has |cos pi| = 1. So no new
    rate constant is needed; rate_law.rate_comb applies verbatim.
    """
    print("\n== O(1/K) rate -- REUSING rate_law: K(L - L_comb_K) -> rate_comb(s) = s/2 pi^2 ==")
    print("   s                   K     |K(L - L_comb_K)|    |rate_comb| = |s/2 pi^2|")
    for s in [mp.mpc(2, 0), mp.mpc(0.5, 6.020948), mp.mpc(1.3, 7)]:
        L = L_chi4(s)
        rc = float(abs(rl.rate_comb(s)))
        for K in (40, 160):
            kerr = float(abs(K * (L - L_comb_K(s, K))))
            print(f"   {complex(s)!s:>16}  {K:4d}   {kerr:.6f}          {rc:.6f}")
    print("   (chi_4's comb has the SAME constant as zeta/eta: sign-carrier = +1 at the x=1 endpoint)")


def _print_warp():
    print("\n== warp  ->  2^s L(s, chi_4) = L(1/2,1/2,s)  (the phase-1/2 carrier sin(pi y)) ==")
    print("   s                    K    |warp_chi4_K - 2^s L|   drop     |alpha=1 honest| -> 0")
    for s in [mp.mpc(2, 0), mp.mpc(0.5, 6.020948), mp.mpc(1.3, 7)]:
        tgt = target_2sL(s)
        prev = None
        for K in (5, 15, 45):
            err = float(abs(warp_chi4_K(s, K) - tgt))
            dead = float(abs(warp_chi4_alpha(s, K, 1.0)))
            drop = "  -- " if prev is None else f"{prev/err:4.2f}x"
            prev = err
            print(f"   {complex(s)!s:>16}  {K:3d}   {err:.3e}          {drop}    {dead:.3e}")
    # the Lerch identity and the load-bearing +2^s
    s0 = mp.mpc(0.5, 6.020948)
    print(f"   identity 2^s L(s,chi_4) == Phi(-1,s,1/2) [= L(1/2,1/2,s)]: "
          f"{float(abs(target_2sL(s0) - mp.lerchphi(-1, s0, mp.mpf(1)/2))):.1e}")
    print(f"   load-bearing +2^s (m=0 midpoint cell): warp_chi4_K - warp_chi4_alpha(.,.,1/2) - 2^s "
          f"= {float(abs(warp_chi4_K(s0, 7) - warp_chi4_alpha(s0, 7, 0.5) - mp.power(2, s0))):.1e}")
    # fast grid evaluator vs the transparent Lerch / raw oscillatory cross-checks
    sl = mp.mpc(2, 1)
    print(f"   fast vs transparent Lerch  (s={complex(sl)}, alpha=1/2, K=12): "
          f"{float(abs(warp_chi4_alpha(sl, 12, 0.5) - warp_chi4_lerch(sl, 12, 0.5))):.2e}")
    sr = mp.mpc(2.5, 1)
    print(f"   fast vs raw oscillatory     (s={complex(sr)}, alpha=1/2, K=8):  "
          f"{float(abs(warp_chi4_alpha(sr, 8, 0.5) - warp_chi4_raw(sr, 8, 0.5))):.2e}")
    # the mod-8 aside: the GENUINE chi_4 carrier's midpoints give chi_{-8}, not chi_4
    print("   modulus surprise -- warping the GENUINE chi_4 carrier sin(pi y/2) at the midpoints:")
    worst8 = 0.0
    for s in [mp.mpc(2, 0), mp.mpc(2.5, 1), mp.mpc(0.5, 5)]:
        lhs = midpoint_chi4_carrier_limit(s)
        rhs = mp.power(2, s) * (mp.sqrt(2) / 2) * L_chi_m8(s)
        worst8 = max(worst8, float(abs(lhs - rhs)))
    print(f"      sum sin(pi(m+1/2)/2)(m+1/2)^-s == 2^s(sqrt2/2) L(s, chi_{{-8}}):  worst "
          f"{worst8:.1e}  (a mod-8 character!)")


def _print_warp_migration():
    print("\n== warp  2^s L(s, chi_4): zeros climb onto sigma=1/2 (single line, no companion) ==")
    hdrK = [1, 5, 21, 45]
    print("   target zero        " + "".join(f"K={k:<6d}" for k in hdrK))
    trajs = []
    for g in L_ZERO_HEIGHTS[:4]:
        traj = hb.migrate(warp_chi4_K, mp.mpc(0.5, g), schedule=K_SCHEDULE)
        d = {K: zz for K, zz in traj if zz is not None}
        cells = "".join(f"{float(d[k].real):<7.3f}" if k in d else "  --   " for k in hdrK)
        print(f"   1/2+{g:8.4f}i   {cells}")
        trajs.append((g, traj))
    return trajs


def _traj_xy(traj):
    pts = [(K, zz) for K, zz in traj if zz is not None]
    return ([float(zz.real) for _, zz in pts], [float(zz.imag) for _, zz in pts],
            [K for K, _ in pts])


def _main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figstyle
    figstyle.enlarge()
    # dense 6-panel figure with long titles: trim title/legend sizes locally (still >> default).
    plt.rcParams.update({"axes.titlesize": 13, "legend.fontsize": 10.5, "figure.titlesize": 16})

    _print_evaluator()
    g_zeros = find_G_zeros(26.0)
    _print_endpoint(g_zeros)
    _print_counting(g_zeros)
    comb = _print_comb_migration()
    _print_rate_constant()
    _print_warp()
    warp = _print_warp_migration()

    # ============================== figure ==============================
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10))

    # (0,0) the continuous endpoint: off-critical string -> 3/2 (design-crux: structured, eta-type)
    ax = axes[0, 0]
    gss = [float(z.real) for z in g_zeros]
    gts = [float(z.imag) for z in g_zeros]
    tt = [6 + 0.4 * k for k in range(int((26 - 6) / 0.4) + 1)]
    ax.plot(gss, gts, "o", ms=5, color="C0", label=r"zeros of $G(s)$")
    ax.plot([sigma_law(t) for t in tt], tt, "-", color="C3", lw=1.2, label=r"$\sigma$-law")
    ax.axvline(1.5, ls="--", color="0.5", lw=1, label=r"$\sigma=3/2$")
    ax.axvline(0.5, ls=":", color="C2", lw=1.2, label=r"critical line $\frac{1}{2}$")
    ax.set_xlabel(r"$\sigma=\mathrm{Re}\,s$")
    ax.set_ylabel(r"$t=\mathrm{Im}\,s$")
    ax.set_title(r"endpoint $G=\int_1^\infty\!\sin(\frac{\pi x}{2})x^{-s}dx$:"
                 "\nstructured ($\\eta$-type), $\\sigma\\to\\frac{3}{2}$")
    ax.set_xlim(0.3, 2.1)
    ax.legend(loc="lower right")

    # (0,1) the comb migration: G+1/2 ground string -> sigma=1/2, one clean line
    ax = axes[0, 1]
    for g, traj in comb:
        xs, ys, ks = _traj_xy(traj)
        if xs:
            ax.plot(xs, ys, "-o", ms=3, lw=1, color="C0")
            ax.plot(xs[-1], ys[-1], "k.", ms=9)                 # K=45 end (on sigma=1/2)
            if ks[0] == 0:
                ax.plot(xs[0], ys[0], "s", ms=7, mfc="none", mec="0.3")   # K=0 ground
    ax.plot([], [], "-o", color="C0", label=r"$L^{(K)}_{\rm comb}$ zeros")
    ax.plot([], [], "s", mfc="none", mec="0.3", label=r"$K=0$ ground ($G+\frac{1}{2}$)")
    ax.axvline(0.5, ls="-", lw=1.2, color="C2", label=r"critical line $\frac{1}{2}$")
    ax.set_xlabel(r"$\sigma=\mathrm{Re}\,s$")
    ax.set_ylabel(r"$t=\mathrm{Im}\,s$")
    ax.set_title(r"comb $K$-knob: zeros born, migrate onto $\frac{1}{2}$"
                 "\n(single line -- no companion)")
    ax.set_xlim(0.2, 1.15)
    ax.legend(loc="upper right")

    # (0,2) O(1/K) convergence: comb and warp, both ~ 1/K
    ax = axes[0, 2]
    s0 = mp.mpc(0.5, 6.020948)
    Lt = L_chi4(s0)
    z2sL = target_2sL(s0)
    Ks = [2, 4, 8, 16, 32, 45]
    ec = [float(abs(L_comb_K(s0, K) - Lt)) for K in Ks]
    ew = [float(abs(warp_chi4_K(s0, K) - z2sL)) for K in Ks]
    rc = float(abs(rl.rate_comb(s0)))                              # rate_law's s/2 pi^2
    ax.loglog(Ks, ec, "-o", color="C0", label=r"comb $|L^{(K)}-L|$")
    ax.loglog(Ks, ew, "--s", color="C3", mfc="none", label=r"warp $|-2^sL|$")
    ax.loglog(Ks, [rc / K for K in Ks], ":", color="0.5",
              label=r"rate_law $|s/2\pi^2|/K$")
    ax.set_xlabel("K")
    ax.set_ylabel(r"error at $s=\frac{1}{2}+6.02i$")
    ax.set_title(r"both routes $O(1/K)$; comb constant $=$ rate_law")
    ax.legend(loc="lower left")

    # (1,0) the warp duality: sin(pi y) samples -- alpha=1/2 survives, alpha=1 dies (dual of warp_eta)
    ax = axes[1, 0]
    yy = [0.4 + 0.01 * k for k in range(int((5.3 - 0.4) / 0.01) + 1)]
    ax.plot(yy, [float(mp.sin(PI * y)) for y in yy], "-", color="0.5", lw=1.4,
            label=r"$\sin(\pi y)$ (phase-$\frac{1}{2}$ sign)")
    halves = [m + 0.5 for m in range(0, 5)]
    ints = [n for n in range(1, 6)]
    ax.plot(halves, [float(mp.sin(PI * h)) for h in halves], "o", color="C2", ms=11, mec="k",
            mew=0.5, label=r"$\alpha=\frac{1}{2}$ (midpoints): $\pm1$")
    ax.plot(ints, [0] * len(ints), "X", color="C3", ms=13, mec="k", mew=0.5,
            label=r"$\alpha=1$ (integers): dies")
    ax.axhline(0.0, ls=":", lw=1, color="0.7")
    ax.set_xlabel(r"cell target $y$")
    ax.set_ylabel(r"$\sin(\pi y)$")
    ax.set_title(r"warp dual of $\eta$: $\alpha=\frac{1}{2}$ survives"
                 "\n($\\to 2^sL$), $\\alpha=1$ dies")
    ax.set_ylim(-1.5, 1.5)
    ax.legend(loc="lower right")

    # (1,1) the warp migration: 2^s L zeros climb onto sigma=1/2 (single line)
    ax = axes[1, 1]
    for g, traj in warp:
        xs, ys, ks = _traj_xy(traj)
        if xs:
            ax.plot(xs, ys, "-o", ms=3, lw=1, color="C3")
            ax.plot(xs[-1], ys[-1], "k.", ms=9)
    ax.plot([], [], "-o", color="C3", label=r"$2^sL^{(K)}$ zeros (warp)")
    ax.axvline(0.5, ls="-", lw=1.2, color="C2", label=r"critical line $\frac{1}{2}$")
    ax.axvline(0.0, ls=":", lw=1, color="0.6", label=r"$\sigma=0$ (no companion: $2^s\neq0$)")
    ax.set_xlabel(r"$\sigma=\mathrm{Re}\,s$")
    ax.set_ylabel(r"$t=\mathrm{Im}\,s$")
    ax.set_title(r"warp $\to 2^sL(s,\chi_4)$: climb onto $\frac{1}{2}$"
                 "\n(Lerch $L(\\frac{1}{2},\\frac{1}{2},s)$; single line)")
    ax.set_xlim(-0.15, 1.05)
    ax.legend(loc="upper right")

    # (1,2) density conservation / counting: G string count == L zero count (the no-split headline)
    ax = axes[1, 2]
    t_max = 26.0
    gts_sorted = sorted(gts)
    Lz = sorted(t for t in L_ZERO_HEIGHTS if t <= t_max)
    ax.step(gts_sorted, range(1, len(gts_sorted) + 1), where="post", color="C0",
            label=r"$G$-string count (off-line)")
    ax.step(Lz, range(1, len(Lz) + 1), where="post", color="C3",
            label=r"$L(s,\chi_4)$ zeros ($\sigma=\frac{1}{2}$)")
    tt2 = [7 + 0.4 * k for k in range(int((t_max - 7) / 0.4) + 1)]
    ax.plot(tt2, [counting_law(t) for t in tt2], "-", color="0.5", lw=1.2,
            label=r"$\frac{t}{2\pi}\ln\frac{2t}{\pi e}+\frac{5}{8}$")
    ax.set_xlabel(r"$t=\mathrm{Im}\,s$")
    ax.set_ylabel(r"$N(t)$")
    ax.set_title(r"density conserved: string $\to$ one line"
                 "\n(same count -- no $\\eta$-style split)")
    ax.legend(loc="upper left")

    fig.suptitle(r"The $K$-knob $+$ warp on $L(s,\chi_4)=\sum\chi_4(n)n^{-s}$: zeros born from a "
                 r"structured endpoint, migrating onto $\sigma=\frac{1}{2}$ (one clean line)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    figdir = Path(__file__).resolve().parent / "figures"
    figdir.mkdir(exist_ok=True)
    out = figdir / "lfunction_bridge.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    _main()
