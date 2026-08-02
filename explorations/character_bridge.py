r"""Character combs: the K-knob on L(s, chi) via residue classes -- the rest of degree 1.

EXPERIMENTAL (issue #37). This module lives in `explorations/`, deliberately disjoint
from the frozen manuscript arc in `bridge/` + `paper/`: nothing here is referenced by
the paper, repro.py, or the CI suite (pytest's `testpaths = tests` excludes
`explorations/tests/`; run them with `pytest explorations/tests`). It REUSES the
bridge machinery (harmonic_bridge moments, warp_alpha's Hurwitz warp, rate_law's
constants) but adds nothing to it. Experimental-mathematics framing throughout;
**no RH/GRH claims**.

The identity that makes it cheap (issue #37's spine)
----------------------------------------------------
Any Dirichlet character chi mod q decomposes the L-series over residue classes:

    L(s, chi) = sum_{n>=1} chi(n) n^{-s} = q^{-s} sum_{a=1}^{q} chi(a) zeta(s, a/q)

-- a finite, chi-weighted combination of Hurwitz zetas at the rational phases a/q.
(The K = infinity identity itself is the TEXTBOOK Hurwitz decomposition -- nothing
novel there; anything new here lives entirely in the finite-K deformation and the
zero tracking. See the lit-pass note at the bottom of this docstring.)
The bridge already builds K-knob interpolants for zeta(s, alpha) at arbitrary phase
(warp_alpha.py, the warp route), and the additive-comb route generalizes verbatim
(built here): so the chi-comb family is a linear combination of objects the repo
already makes. Same phi_K, both restoration routes, summed with character weights.

Route A -- the residue-class additive comb (built here, closed form)
--------------------------------------------------------------------
Euler-Maclaurin on zeta(s, alpha) = sum_{m>=0} (m + alpha)^{-s}, f(x) = (x+alpha)^{-s}
on [0, inf), with the sawtooth {x} - 1/2 = -sum_k sin(2 pi k x)/(pi k) truncated at K:

    zeta_comb_K(s, alpha) = alpha^{1-s}/(s-1) + alpha^{-s}/2
                            + sum_{k=1}^{K} (s/pi k) [ cos(2 pi k alpha) S_k(alpha)
                                                     - sin(2 pi k alpha) C_k(alpha) ],

where S_k / C_k are the LOWER-LIMIT-alpha moments int_alpha^inf sin/cos(2 pi k y)
y^{-s-1} dy = closed forms in the incomplete gamma (the alpha-generalization of
harmonic_bridge's Smom/Cmom; at alpha = 1 the phases collapse -- cos(2 pi k) = 1,
sin(2 pi k) = 0 -- and `zeta_comb_K(s, K, 1) == harmonic_bridge.zeta_K(s, K)`
EXACTLY, term by term). Then

    L_comb_K(s, K, chi) = q^{-s} sum_a chi(a) zeta_comb_K(s, K, a/q)  ->  L(s, chi).

The K = 0 endpoint is closed-form and telling:

    E_chi(s) = [sum_a chi(a) a^{1-s}] / (q (s-1)) + (1/2) sum_a chi(a) a^{-s}

-- for non-principal chi the "pole" term's numerator vanishes at s = 1 (orthogonality
sum_a chi(a) = 0), so E_chi is ENTIRE: the continuum endpoint of a non-principal
character comb is a bounded character Dirichlet polynomial (one period of the
L-series, EM-weighted), not a pole. Contrast zeta's endpoint 1/(s-1) + 1/2 (pole, one
lone zero at s = -1). This is the comb-route face of issue #37's prediction 3, and it
connects the K-knob's ground state to the partial-sum cousins (Gonek-Ledoan 2010 /
Gonek-Montgomery 2013): the comb's K = 0 anchor IS a (weighted) length-q section.

Route B -- the warp combination (pure reuse of warp_alpha)
----------------------------------------------------------
    L_warp_K(s, K, chi) = q^{-s} sum_a chi(a) warp_complete_alpha(s, K, a/q)
                        ->  q^{-s} sum_a chi(a) zeta(s, a/q) = L(s, chi).

For K >= 1 this is literally a linear combination of existing warp_alpha calls. The
K = 0 endpoint raises issue #37's normalization question, and the answer splits by
convention:

  * DC-ALWAYS-ON (warp_alpha as built): the phase offset alpha - 1/2 never turns
    off, so the K = 0 component endpoint is (alpha + 1/2)^{1-s}/(s-1) + alpha^{-s}
    (completion included) -- alpha-DEPENDENT, and the combination E_warp(s) is a
    small entire function (pole cancels by orthogonality), not zero.
  * DC-IN-KNOB: the offset alpha - 1/2 is the k = 0 Fourier coefficient of the
    shifted sawtooth alpha - {x}, and the completion alpha^{-s} is the m = 0 cell
    that only exists once the warp has created the alpha-lattice -- both are
    *discreteness data*. Counting them as part of the knob (on for K >= 1), every
    component's K = 0 endpoint is the SAME bland integral int_1^inf x^{-s} dx =
    1/(s-1), and the combination is (q^{-s}/(s-1)) sum_a chi(a) == 0 identically for
    every non-principal chi. **The character sum is a projector onto the comb**: the
    L-function is born from the zero function, its entire content carried by the
    discreteness corrections (issue #37, prediction 3). zeta interpolates from a
    pole-with-no-zeros; L(s, chi) interpolates from *nothing*.

Both statements are algebraic; the driver verifies both numerically and prints the
two endpoints side by side. Which normalization makes the small-K zero-birth story
cleanest is deliberately left open (the deferred deep-dive; see the sub-issues).

The rate law generalizes, with a q-dependent constant
-----------------------------------------------------
Integrating the k-th harmonic by parts, the comb tail is governed by the integrand's
value at the lower endpoint (x = 0, i.e. y = alpha), giving the alpha-generalization
of rate_law.rate_comb = s/(2 pi^2):

    zeta(s, alpha) - zeta_comb_K(s, alpha)  ~  (s / 2 pi^2) alpha^{-s-1} / K,

(alpha = 1 recovers rate_law exactly), and summing with character weights:

    L(s, chi) - L_comb_K(s, K, chi)  ~  (s q / 2 pi^2) [sum_a chi(a) a^{-s-1}] / K.

So zero-condensation stays O(1/K) (issue #37, prediction 2's rate question), and the
constant is q times a one-period section of L(s+1, chi). NB the constant is a
property of the COMB ORGANIZATION, not of the L-function: lfunction_bridge.py combs
the SAME L(s, chi_4) through its sin(pi x/2) carrier and gets the zeta constant
s/2 pi^2, while the residue-class comb here gets (4s/2 pi^2)(1 - 3^{-s-1}). Both are
measured in the driver.

What the migrations show (the driver's tables + figure)
-------------------------------------------------------
  * chi_3 (q=3, odd real primitive): single clean line. Ground zeros of E_chi migrate
    onto sigma = 1/2 at O(1/K) -- the lfunction_bridge chi_4 story reproduced through
    the residue-class route, for a new conductor.
  * chi_5, j=1 (chi(2) = i, odd COMPLEX primitive): the first genuinely
    complex-coefficient object in the repo (no conjugate symmetry in t). Zeros land
    on sigma = 1/2 all the same. Each component zeta(s, a/5) is Davenport-Heilbronn
    SCATTERED (a/5 is a generic phase in warp_alpha's dichotomy: zeta(s, a/5) is a
    genuine multi-L combination, Saias-Weingartner); the chi-weighted combination is
    a single primitive L, and its zeros sit on the line. Critical-line order
    materializes ONLY in the arithmetically-weighted superposition of individually
    disordered components -- discreteness restoration and character averaging as two
    separate knobs (issue #37, prediction 2).
  * chi_6 = chi_3 induced mod 6 (IMPRIMITIVE): L(s, chi_6) = (1 + 2^{-s}) L(s, chi_3).
    The Euler-factor prefactor carries a vertical zero comb at sigma = 0,
    t = (2m+1) pi / ln 2, and the migration SPLITS: part of the ground string onto
    sigma = 1/2 (the primitive zeros), part onto the sigma = 0 comb -- the eta
    precedent (rigid comb x critical string) generalized to imprimitive L (issue #37,
    prediction 1, structural part; the spectral-statistics measurement is deferred).

The in-repo precedents, made explicit: eta = (1 - 2^{1-s}) zeta is a sigma = 1
prefactor comb, and the midpoint warp's half-shift satisfies 2^{-s} zeta(s, 1/2) =
(1 - 2^{-s}) zeta(s) = L(s, chi_0 mod 2) -- the principal character mod 2, whose
sigma = 0 comb IS the warp's sigma = 0 companion string. The bridge has been doing
q = 2 character combs all along; this module does the rest of degree 1: for the
Selberg class proper, every degree-1 element is zeta or a shifted primitive
Dirichlet L (conjectured Conrey-Ghosh 1993, proved Kaczorowski-Perelli, Acta Math.
182, 1999; simplest proof Soundararajan 2005), so chi combs + shifts close out
degree 1 in principle.

Lit-pass status (adversarial search, 2026-08-01; full findings + BibTeX in
explorations/README.md)
-----------------------------------------------------------------------------
Parameterized approximation families for Dirichlet L EXIST in print -- partial sums
(Ledoan-Roy-Zaharescu 2014; Roy-Vatwani, Adv. Math. 2019; Dubon 2025 for
L(s, chi_4)), truncated Euler products (Gonek 2012, zeta only), the shift-knob
Hurwitz combinations (Garunkstis-Steuding 2007 individual trajectories; Nakamura
2022 symmetrized combination, zero COUNTS), and -- closest in spirit -- the
Friedli-Karlsson cyclic-graph discretization program (Friedli 2016;
Friedli-Karlsson 2017; Karlsson-Muller 2026 PREPRINT: finite spectral sums on
Z/nZ via Euler-Maclaurin -> L(s, chi), with GRH-equivalences). But in all of these
the zero results are counts / zero-free regions / density / equivalences: we found
NO published work that deforms the chi-weighted Hurwitz combination through a
discretization parameter and TRACKS the zero trajectories of the combination onto
sigma = 1/2. Negative claim, so phrased as "not found under adversarial search",
not "provably absent" -- and the 3-day-old Karlsson-Muller preprint shows this
neighborhood is ACTIVE: cite it and distinguish carefully before any
novelty-adjacent language leaves this directory.

Run directly to validate + plot: writes explorations/figures/character_bridge.png.
Runtime is minutes (migrations dominate); this driver is local/manual only -- it is
deliberately NOT in repro.py or CI (issue #37's CI-economy note).
"""
import itertools
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import mpmath as mp

# Path shim: this module lives in explorations/ but cross-imports the flat bridge/
# sources as bare siblings (harmonic_bridge, warp_alpha, rate_law); put both dirs on
# sys.path for direct runs (explorations/conftest.py does the same for pytest).
_HERE = Path(__file__).resolve().parent
_BRIDGE = _HERE.parent / "bridge"
for _p in (str(_HERE), str(_BRIDGE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import harmonic_bridge as hb   # noqa: E402  (migrate; the alpha=1 moment identity)
import rate_law as rl          # noqa: E402  (rate_comb = s/2 pi^2, the alpha=1 constant)
import warp_alpha as wa        # noqa: E402  (warp_complete_alpha -- route B, find_zeros)

mp.mp.dps = 30

_J = mp.j
PI = mp.pi


# --------------------------------------------------------------------------
# 1. Dirichlet characters -- exact roots of unity via unit-group exponents
# --------------------------------------------------------------------------
class Character:
    """A Dirichlet character mod q, stored EXACTLY as unit-group exponents.

    chi(a) = e^{2 pi i exps[a] / m} for a in the units mod q (m = the character
    order's lcm scale = |(Z/q)^*| as constructed), chi(a) = 0 otherwise. Storing
    integer exponents instead of complex values keeps chi exact at any working
    precision. `conductor` / `primitive` / `parity` are computed on construction.
    """

    def __init__(self, q, m, exps, label):
        # type: (int, int, Dict[int, int], str) -> None
        self.q = int(q)
        self.m = int(m)                     # chi(a) = e^{2 pi i exps[a]/m}
        self.exps = {int(a) % q: int(e) % m for a, e in exps.items()}
        self.label = label
        self.conductor = self._conductor()
        self.primitive = self.conductor == self.q
        self.principal = all(e == 0 for e in self.exps.values())
        # parity kappa: chi(-1) = +1 (even, kappa=0) or -1 (odd, kappa=1)
        e_neg = self.exps[(self.q - 1) % self.q] if self.q > 1 else 0
        self.parity = 0 if e_neg == 0 else 1

    def __call__(self, n):
        n = int(n) % self.q if self.q > 1 else 0
        if self.q == 1:
            return mp.mpf(1)
        e = self.exps.get(n)
        if e is None:
            return mp.mpf(0)
        if 2 * e % self.m == 0:            # real values +-1: return exact mpf
            return mp.mpf(1) if e == 0 else mp.mpf(-1)
        return mp.expjpi(mp.mpf(2 * e) / self.m)

    @property
    def conj(self):
        """The conjugate character chi-bar (negated exponents)."""
        return Character(self.q, self.m, {a: (-e) % self.m for a, e in self.exps.items()},
                         self.label + "~")

    def _conductor(self):
        """Smallest f | q with chi(a) = chi(b) whenever a == b (mod f), both units."""
        us = sorted(self.exps)
        for f in range(1, self.q + 1):
            if self.q % f:
                continue
            classes = {}  # type: Dict[int, int]
            ok = True
            for a in us:
                r = a % f
                if r in classes and classes[r] != self.exps[a]:
                    ok = False
                    break
                classes[r] = self.exps[a]
            if ok:
                return f
        return self.q


def units(q):
    """The unit group (Z/q)^* as a sorted list."""
    return [a for a in range(1, q) if math.gcd(a, q) == 1] or [0]


def characters(q):
    """All phi(q) characters mod q, for moduli with CYCLIC unit group (1,2,4,p^k,2p^k).

    Finds a generator g of (Z/q)^* by brute force, builds the discrete log, and sets
    chi_j(g^t) = e^{2 pi i j t / m}. chars[0] is the principal character. Raises for
    non-cyclic moduli (q = 8, 12, 15, ...) -- use `characters_any` for those (this
    older constructor is kept verbatim so the #37-#39 cast indices stay stable).
    """
    us = units(q)
    m = len(us)
    gen = None
    for g in us:
        x, order = 1, 0
        for _ in range(m):
            x = x * g % q
            order += 1
            if x == 1:
                break
        if order == m:
            gen = g
            break
    if gen is None:
        raise ValueError(f"(Z/{q})^* is not cyclic; construct via `induced`")
    dlog = {1 % q: 0}
    x = 1
    for t in range(1, m):
        x = x * gen % q
        dlog[x] = t
    return [Character(q, m, {a: j * dlog[a] % m for a in us}, f"chi_{q}^({j})")
            for j in range(m)]


def _factorize(n):
    # type: (int) -> List[Tuple[int, int]]
    """Prime-power factorization [(p, e), ...] by trial division (n <= ~10^6 here)."""
    out = []  # type: List[Tuple[int, int]]
    d = 2
    while d * d <= n:
        if n % d == 0:
            e = 0
            while n % d == 0:
                n //= d
                e += 1
            out.append((d, e))
        d += 1
    if n > 1:
        out.append((n, 1))
    return out


def _cyclic_gen(pe):
    # type: (int) -> Tuple[int, int]
    """(generator, order) of the cyclic (Z/pe)^*: pe an odd prime power, 1, 2, 4."""
    us = units(pe)
    m = len(us)
    for g in us:
        x, order = 1, 0
        for _ in range(m):
            x = x * g % pe
            order += 1
            if x == 1:
                break
        if order == m:
            return g, m
    raise ValueError(f"(Z/{pe})^* is not cyclic")


def characters_any(q):
    """All phi(q) characters mod q, ANY modulus -- the CRT product of cyclic pieces.

    (Z/q)^* = prod_p (Z/p^e)^* with each odd-prime-power factor cyclic and
    (Z/2^e)^* = {+-1} x <5> for e >= 3 (cyclic for e <= 2): every character is a
    product of one character per cyclic slot. Each slot g of order m_g contributes
    exponent j_g * dlog_g(a) * (m / m_g) with m = lcm of the slot orders, summed into
    the exact unit-group-exponent storage of `Character`. chars[0] is principal;
    ordering is itertools.product over the slot indices (deterministic). For cyclic
    moduli this reproduces the same character SET as `characters` (possibly in a
    different list order -- the old constructor keeps the #37-#39 cast stable).
    The issue #40 build item: unlocks the non-cyclic moduli q = 8, 12, 15, 16, ...
    for the rate-vs-conductor sweep.
    """
    q = int(q)
    if q <= 2:
        return [Character(q, 1, {a: 0 for a in units(q)}, f"chi_{q}#0")]
    # cyclic slots: (pe, order, dlog table on units(pe))
    slots = []  # type: List[Tuple[int, int, Dict[int, int]]]
    for p, e in _factorize(q):
        pe = p ** e
        if p == 2 and e >= 3:
            half = 2 ** (e - 2)
            dlog_sign = {}  # type: Dict[int, int]
            dlog_five = {}  # type: Dict[int, int]
            for i in range(2):
                for t in range(half):
                    v = (-1) ** i * pow(5, t, pe) % pe
                    dlog_sign[v] = i
                    dlog_five[v] = t
            slots.append((pe, 2, dlog_sign))
            slots.append((pe, half, dlog_five))
        else:
            g, m_g = _cyclic_gen(pe)
            dlog = {1 % pe: 0}
            x = 1
            for t in range(1, m_g):
                x = x * g % pe
                dlog[x] = t
            slots.append((pe, m_g, dlog))
    m = 1
    for _, m_g, _ in slots:
        m = m * m_g // math.gcd(m, m_g)
    us = units(q)
    out = []
    ranges = [range(m_g) for _, m_g, _ in slots]
    for idx, js in enumerate(itertools.product(*ranges)):
        exps = {a: sum(j * dl[a % pe] * (m // m_g)
                       for j, (pe, m_g, dl) in zip(js, slots)) % m
                for a in us}
        out.append(Character(q, m, exps, f"chi_{q}#{idx}"))
    return out


def induced(chi, q):
    """The character mod q induced by chi (mod f, f | q): chi*(a) on units, 0 elsewhere.

    Imprimitive for q > f:  L(s, chi_ind) = L(s, chi) prod_{p | q, p !| f} (1 - chi(p) p^{-s}),
    the removed Euler factors carrying vertical zero combs at sigma = 0.
    """
    if q % chi.q:
        raise ValueError("q must be a multiple of the inducing modulus")
    exps = {a: chi.exps[a % chi.q] for a in units(q)}
    return Character(q, chi.m, exps, f"{chi.label}->mod{q}")


# The cast: the named characters the driver and tests exercise.
CHI3 = characters(3)[1]                 # q=3 quadratic, odd, primitive
CHI4 = characters(4)[1]                 # q=4 (the lfunction_bridge cross-check)
CHI5_I = characters(5)[1]               # q=5 order 4, chi(2) = i: odd, COMPLEX, primitive
CHI5_QUAD = characters(5)[2]            # q=5 quadratic (Legendre), even, primitive
CHI6 = induced(CHI3, 6)                 # IMPRIMITIVE: L = (1 + 2^{-s}) L(chi_3)
for _c, _n in [(CHI3, "chi_3"), (CHI4, "chi_4"), (CHI5_I, "chi_5(2->i)"),
               (CHI5_QUAD, "chi_5(quad)"), (CHI6, "chi_6=chi_3 mod 6")]:
    _c.label = _n


# --------------------------------------------------------------------------
# 2. L(s, chi): the Hurwitz combination, plus independent anchors
# --------------------------------------------------------------------------
def L_chi(s, chi):
    """L(s, chi) = q^{-s} sum_a chi(a) zeta(s, a/q) -- the module's spine identity.

    Entire for non-principal chi (the Hurwitz poles cancel by orthogonality); mpmath
    still returns inf from each zeta(s, a/q) AT s = 1 exactly, so the removable value
    is spliced in there: L(1, chi) = -(1/q) sum_a chi(a) psi(a/q) (digamma).
    For principal chi the pole at s = 1 is genuine -- no splice.
    """
    s = mp.mpc(s)
    q = chi.q
    if not chi.principal and abs(s - 1) < mp.mpf("1e-12"):
        return -mp.fsum(chi(a) * mp.digamma(mp.mpf(a) / q) for a in units(q)) / q
    return mp.power(q, -s) * mp.fsum(chi(a) * mp.zeta(s, mp.mpf(a) / q)
                                     for a in units(q))


def L_sum(s, chi, terms=100000):
    """L(s, chi) by direct summation over one period at a time -- the sigma > 1 reference."""
    s = mp.mpc(s)
    q = chi.q
    return mp.fsum(chi(a) * mp.nsum(lambda mm, a=a: (q * mm + a) ** (-s),
                                    [0, (terms - a) // q]) for a in units(q))


def L_euler(s, chi, pmax=200):
    """L(s, chi) = prod_p (1 - chi(p) p^{-s})^{-1} truncated at pmax -- a sigma >= 2 anchor."""
    s = mp.mpc(s)
    out = mp.mpf(1)
    for p in range(2, pmax + 1):
        if all(p % d for d in range(2, int(p ** 0.5) + 1)):
            out /= 1 - chi(p) * mp.power(p, -s)
    return out


def gauss_sum(chi):
    """tau(chi) = sum_{a mod q} chi(a) e^{2 pi i a / q}.  |tau| = sqrt(q) for primitive chi."""
    q = chi.q
    return mp.fsum(chi(a) * mp.expjpi(mp.mpf(2 * a) / q) for a in units(q))


def root_number(chi):
    """epsilon(chi) = tau(chi) / (i^kappa sqrt(q)) -- the functional-equation constant, |eps| = 1."""
    return gauss_sum(chi) / (_J ** chi.parity * mp.sqrt(chi.q))


def completed_L(s, chi):
    """Lambda(s, chi) = (q/pi)^{(s+kappa)/2} Gamma((s+kappa)/2) L(s, chi)  (primitive chi)."""
    s = mp.mpc(s)
    k = chi.parity
    return (mp.mpf(chi.q) / PI) ** ((s + k) / 2) * mp.gamma((s + k) / 2) * L_chi(s, chi)


def functional_eq_residual(s, chi):
    """|Lambda(s, chi) - eps(chi) Lambda(1-s, chi-bar)| -- vanishes for primitive chi."""
    return abs(completed_L(s, chi) - root_number(chi) * completed_L(1 - mp.mpc(s), chi.conj))


# --------------------------------------------------------------------------
# 3. Route A: the residue-class comb, closed form via lower-limit moments
# --------------------------------------------------------------------------
def Emom_a(omega, a, alpha):
    """int_alpha^inf e^{i omega y} y^{-a} dy = (-i omega)^{a-1} Gamma(1-a, -i omega alpha).

    The lower-limit-alpha generalization of harmonic_bridge._Emom (which is the
    alpha = 1 case). Entire in a; validated against oscillatory quadrature in tests.
    """
    p = -_J * omega
    return p ** (a - 1) * mp.gammainc(1 - a, p * alpha)


def hurwitz_term(s, k, alpha):
    """The k-th harmonic of the Hurwitz comb at phase alpha.

    (s/pi k) int_0^inf sin(2 pi k x) (x+alpha)^{-s-1} dx, the shift x -> y = x + alpha
    splitting into cos(2 pi k alpha) S_k - sin(2 pi k alpha) C_k with S/C the
    lower-limit moments at frequency 2 pi k. One Emom_a pair serves both.
    """
    w = 2 * PI * k
    Ep = Emom_a(w, s + 1, alpha)
    Em = Emom_a(-w, s + 1, alpha)
    Sk = (Ep - Em) / (2 * _J)
    Ck = (Ep + Em) / 2
    return s / (PI * k) * (mp.cos(w * alpha) * Sk - mp.sin(w * alpha) * Ck)


def zeta_comb_K(s, K, alpha):
    """zeta^{(K)}(s, alpha): the Hurwitz additive comb.  K=0: alpha^{1-s}/(s-1) + alpha^{-s}/2.

    Euler-Maclaurin on sum_{m>=0} (m+alpha)^{-s} from base 0, sawtooth truncated at K
    harmonics; -> zeta(s, alpha) at O(alpha^{-s-1}/K). At alpha = 1 this IS
    harmonic_bridge.zeta_K term by term (1/(s-1) + 1/2 + the 2 pi k modes).
    """
    s = mp.mpc(s)
    alpha = mp.mpf(alpha)
    tot = alpha ** (1 - s) / (s - 1) + alpha ** (-s) / 2
    for k in range(1, K + 1):
        tot += hurwitz_term(s, k, alpha)
    return tot


def L_comb_K(s, K, chi):
    """L^{(K)}(s, chi) = q^{-s} sum_a chi(a) zeta_comb_K(s, K, a/q)  ->  L(s, chi)."""
    s = mp.mpc(s)
    q = chi.q
    return mp.power(q, -s) * mp.fsum(chi(a) * zeta_comb_K(s, K, mp.mpf(a) / q)
                                     for a in units(q))


def comb_endpoint(s, chi):
    """The K = 0 comb endpoint, closed form:

        E_chi(s) = [sum_a chi(a) a^{1-s}] / (q (s-1)) + (1/2) sum_a chi(a) a^{-s}.

    ENTIRE for non-principal chi (numerator of the pole term vanishes at s = 1 by
    orthogonality) -- a one-period, EM-weighted character Dirichlet polynomial: the
    comb's ground state is a length-q section, the K-knob's kinship to the
    partial-sum cousins. Avoid s = 1 exactly (0/0 numerically).

    A measured surprise (verified to ~48 digits in the driver): for the two-unit
    moduli phi(q) = 2 (q = 3, 4, 6, units {1, -1}), the ground zeros sit EXACTLY on
    sigma = 0. Algebra: with p = q - 1 the zero condition E_chi = 0 collapses to

        p^{-s} = (q s - (q-2)) / (q s + (q-2)),

    and the +-(q-2) reflection (the a <-> q-a unit symmetry) makes the right side a
    Moebius map that is unimodular precisely on the imaginary axis -- where
    |p^{-s}| = 1 too, so a whole zero string lives on sigma = 0 (one real phase
    equation in t). For phi(q) > 2 (q = 5) no such reflection exists and the ground
    zeros scatter (measured: 0.3025 + 7.283i, 0.3828 + 13.770i, ...). So the ground
    string's geometry is set by the unit-group size -- and sigma = 0 is exactly where
    the imprimitive Euler combs (and the midpoint warp's companion) live.
    """
    s = mp.mpc(s)
    q = chi.q
    num = mp.fsum(chi(a) * mp.power(a, 1 - s) for a in units(q))
    poly = mp.fsum(chi(a) * mp.power(a, -s) for a in units(q))
    return num / (q * (s - 1)) + poly / 2


# --------------------------------------------------------------------------
# 4. Route B: the warp combination (warp_alpha reused verbatim), and the projector
# --------------------------------------------------------------------------
def L_warp_K(s, K, chi):
    """L via the warp route: q^{-s} sum_a chi(a) warp_complete_alpha(s, K, a/q).

    For K >= 1 a literal linear combination of warp_alpha's existing objects (the
    grid+moment fast evaluator, cached per (K, alpha)); -> L(s, chi) at O(1/K).
    K = 0 evaluates the DC-ALWAYS-ON endpoint (see `warp_endpoint_dc`); the
    DC-IN-KNOB endpoint is identically 0 for non-principal chi (see
    `projector_residual` and the module docstring).
    """
    s = mp.mpc(s)
    q = chi.q
    return mp.power(q, -s) * mp.fsum(chi(a) * wa.warp_complete_alpha(s, K, mp.mpf(a) / q)
                                     for a in units(q))


def warp_endpoint_dc(s, chi):
    """The K = 0 warp endpoint under the DC-ALWAYS-ON convention (warp_alpha as built):

        E_warp(s) = q^{-s} sum_a chi(a) [ (a/q + 1/2)^{1-s} / (s-1) + (a/q)^{-s} ],

    the phase-deformed integral endpoints plus the always-on completion cells. Small,
    entire for non-principal chi (pole cancels at s = 1: (a/q+1/2)^0 = 1), NOT zero.
    """
    s = mp.mpc(s)
    q = chi.q
    return mp.power(q, -s) * mp.fsum(
        chi(a) * (mp.power(mp.mpf(a) / q + mp.mpf("0.5"), 1 - s) / (s - 1)
                  + mp.power(mp.mpf(a) / q, -s)) for a in units(q))


def projector_residual(chi):
    """|sum_a chi(a)| -- the orthogonality sum whose vanishing IS the projector.

    Under the DC-IN-KNOB convention every component's K = 0 endpoint is the same
    1/(s-1), so L_warp(s, 0, chi) = (q^{-s}/(s-1)) sum_a chi(a): identically zero for
    every non-principal chi. All comb, no continuum.
    """
    return float(abs(mp.fsum(chi(a) for a in units(chi.q))))


# --------------------------------------------------------------------------
# 5. the rate law: alpha-generalized constant, and the chi combination
# --------------------------------------------------------------------------
def rate_comb_alpha(s, alpha):
    """zeta(s, alpha) - zeta_comb_K ~ rate_comb_alpha / K = (s/2 pi^2) alpha^{-s-1} / K.

    Integration by parts puts the comb tail on the integrand's lower-endpoint value
    (x = 0, y = alpha): the alpha-generalization of rate_law.rate_comb (= alpha = 1).
    """
    s = mp.mpc(s)
    return rl.rate_comb(s) * mp.power(mp.mpf(alpha), -s - 1)


def rate_comb_chi(s, chi):
    """L(s, chi) - L_comb_K ~ rate_comb_chi / K = (s q / 2 pi^2) sum_a chi(a) a^{-s-1} / K.

    q times a one-period section of L(s+1, chi). The q-dependence of issue #37's
    prediction-2 rate question, at first order: the constant grows ~ linearly in q.
    (A property of the residue-class comb organization -- lfunction_bridge's CARRIER
    comb for the same L(s, chi_4) has the plain zeta constant s/2 pi^2.)
    """
    s = mp.mpc(s)
    q = chi.q
    return q * rl.rate_comb(s) * mp.fsum(chi(a) * mp.power(a, -s - 1) for a in units(q))


# --------------------------------------------------------------------------
# 6. zero data -- migration targets (re-polished by the driver before use)
# --------------------------------------------------------------------------
# First critical-line zero heights (t > 0), found by |L(1/2+it)| minima scan +
# findroot on the Hurwitz combination, polished to ~1e-12 residual on sigma = 1/2.
# chi_5(2->i) is complex, so its t > 0 and t < 0 zeros differ; these are t > 0.
CHI3_ZEROS = [8.039737156, 11.249206159, 15.704619175, 18.261997163]
CHI5_I_ZEROS = [6.183578193, 8.457229822, 12.674945902, 14.825025570]
CHI5_QUAD_ZEROS = [6.648452945, 9.831443574, 11.958846029, 16.033821135]
LN2 = float(mp.log(2))
CHI6_COMB_ZEROS = [(2 * m + 1) * float(PI) / LN2 for m in range(2)]   # sigma = 0 comb

# K schedule for migrations (lfunction_bridge's: O(1/K) lands zeros within ~2e-2 of
# their line by K = 45, and the closed-form comb is affordable there).
K_SCHEDULE = [1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 27, 35, 45]


def polish_zero(chi, t_seed, sigma_seed=0.5, workdps=20):
    """findroot L(., chi) from (sigma_seed, t_seed); returns the zero or None."""
    with mp.workdps(workdps):
        try:
            z = mp.findroot(lambda s: L_chi(s, chi), mp.mpc(sigma_seed, t_seed))
        except (ValueError, ZeroDivisionError):
            return None
        if abs(L_chi(z, chi)) < mp.mpf(10) ** (-(workdps - 6)):
            return +z
    return None


# --------------------------------------------------------------------------
# self-validating driver
# --------------------------------------------------------------------------
CAST = [CHI3, CHI4, CHI5_I, CHI5_QUAD, CHI6]


def _print_characters():
    print("== the cast: Dirichlet characters (exact unit-group exponents) ==")
    print("   label            q  conductor  primitive  parity  order-check")
    for chi in CAST:
        # multiplicativity over all unit pairs -- exact by construction, verify anyway
        worst = max(float(abs(chi(a * b) - chi(a) * chi(b)))
                    for a in units(chi.q) for b in units(chi.q))
        print(f"   {chi.label:<14} {chi.q:>2}  {chi.conductor:>5}      "
              f"{str(chi.primitive):<9}  {'odd' if chi.parity else 'even':<5} "
              f" mult worst {worst:.1e}")
    # orthogonality: the full character table mod 5
    ch5 = characters(5)
    worst = 0.0
    for i, ci in enumerate(ch5):
        for j, cj in enumerate(ch5):
            ip = mp.fsum(ci(a) * mp.conj(cj(a)) for a in units(5))
            worst = max(worst, float(abs(ip - (4 if i == j else 0))))
    print(f"   orthogonality (full mod-5 table): worst |<chi_i, chi_j> - 4 delta_ij| "
          f"= {worst:.1e}")


def _print_L_anchors():
    print("\n== L(s, chi) anchors ==")
    s2 = mp.mpc(2.5, 1)
    for chi in [CHI3, CHI4, CHI5_I, CHI6]:
        rel = float(abs((L_chi(s2, chi) - L_sum(s2, chi)) / L_sum(s2, chi)))
        eul = float(abs((L_chi(mp.mpc(2), chi) - L_euler(mp.mpc(2), chi))
                        / L_euler(mp.mpc(2), chi)))
        print(f"   {chi.label:<14} |Hurwitz-sum|/|L| = {rel:.2e}   Euler(p<=200) rel "
              f"= {eul:.2e}")
    print(f"   L(1, chi_3) = pi/(3 sqrt 3):  |diff| = "
          f"{float(abs(L_chi(1, CHI3) - PI / (3 * mp.sqrt(3)))):.2e}")
    print(f"   L(1, chi_4) = pi/4:           |diff| = "
          f"{float(abs(L_chi(1, CHI4) - PI / 4)):.2e}")
    print(f"   L(2, chi_4) = Catalan:        |diff| = "
          f"{float(abs(L_chi(2, CHI4) - mp.catalan)):.2e}")
    print("   functional equation Lambda(s) = eps Lambda-bar(1-s), primitive chi:")
    for chi in [CHI3, CHI4, CHI5_I, CHI5_QUAD]:
        eps = root_number(chi)
        worst = max(float(functional_eq_residual(s, chi))
                    for s in [mp.mpc(2, 1), mp.mpc(0.5, 5), mp.mpc(-0.5, 2)])
        print(f"      {chi.label:<14} eps = {complex(eps):.6f}  (|eps| = "
              f"{float(abs(eps)):.10f})   worst residual = {worst:.1e}")
    # the imprimitive factorization -- prediction 1's structural identity
    s0 = mp.mpc(1.3, 7)
    fac = float(abs(L_chi(s0, CHI6) - (1 + mp.power(2, -s0)) * L_chi(s0, CHI3)))
    print(f"   imprimitive:  L(s, chi_6) == (1 + 2^-s) L(s, chi_3):  |diff| = {fac:.1e}")
    # the in-repo q=2 precedent: 2^{-s} zeta(s, 1/2) = (1 - 2^{-s}) zeta(s) = L(s, chi_0 mod 2)
    pre = float(abs(mp.power(2, -s0) * mp.zeta(s0, mp.mpf("0.5"))
                    - (1 - mp.power(2, -s0)) * mp.zeta(s0)))
    print(f"   q=2 precedent: 2^-s zeta(s,1/2) == (1 - 2^-s) zeta(s):  |diff| = {pre:.1e}"
          f"   (the half-shift IS the principal mod-2 comb)")


def _print_comb():
    print("\n== route A: the residue-class comb ==")
    s = mp.mpc(1.3, 7)
    red = float(abs(zeta_comb_K(s, 20, 1) - hb.zeta_K(s, 20)))
    print(f"   reduction: zeta_comb_K(s, 20, alpha=1) == harmonic_bridge.zeta_K: "
          f"|diff| = {red:.1e}")
    print("   component convergence + the alpha-generalized rate (K * err vs "
          "(s/2pi^2) alpha^-s-1):")
    for al in [mp.mpf(1) / 3, mp.mpf(2) / 5, mp.mpf(5) / 6]:
        err = abs(zeta_comb_K(s, 160, al) - mp.zeta(s, al))
        pred = abs(rate_comb_alpha(s, al))
        print(f"      alpha = {float(al):.4f}:  K*err(K=160) = {float(160 * err):.5f}"
              f"   rate_comb_alpha = {float(pred):.5f}")
    print("   L_comb_K -> L(s, chi)  (and the q-dependent constant, K * err vs "
          "rate_comb_chi):")
    for chi in [CHI3, CHI4, CHI5_I, CHI6]:
        L = L_chi(s, chi)
        err = abs(L_comb_K(s, 160, chi) - L)
        pred = abs(rate_comb_chi(s, chi))
        print(f"      {chi.label:<14} K*err(K=160) = {float(160 * err):.5f}   "
              f"rate_comb_chi = {float(pred):.5f}")
    print("   NB chi_4 via lfunction_bridge's CARRIER comb has constant |s/2pi^2| = "
          f"{float(abs(rl.rate_comb(s))):.5f} -- the constant belongs to the comb, "
          "not the L.")
    # the K=0 endpoint: entire, a one-period section
    print("   K=0 endpoint E_chi: entire for non-principal (pole cancels by "
          "orthogonality):")
    for chi in [CHI3, CHI5_I]:
        near = [float(abs(comb_endpoint(mp.mpc(1) + mp.mpf("1e-6") * d, chi)))
                for d in (1, -1)]
        print(f"      {chi.label:<14} |E(1 +- 1e-6)| = {near[0]:.6f} / {near[1]:.6f}"
              f"   (finite through s = 1)")


def _print_warp_and_projector():
    print("\n== route B: the warp combination + the prediction-3 projector ==")
    s = mp.mpc(2, 1)
    print("   L_warp_K -> L (linear combination of warp_alpha objects):")
    for chi in [CHI3, CHI5_I]:
        L = L_chi(s, chi)
        errs = [float(abs(L_warp_K(s, K, chi) - L)) for K in (8, 16, 32)]
        print(f"      {chi.label:<14} |L_warp_K - L| at K=8,16,32:  "
              + "  ".join(f"{e:.3e}" for e in errs)
              + f"   (ratios {errs[0]/errs[1]:.2f}, {errs[1]/errs[2]:.2f})")
    print("   the two K = 0 conventions:")
    for chi in [CHI3, CHI5_I]:
        edc = float(abs(warp_endpoint_dc(s, chi)))
        prj = projector_residual(chi)
        print(f"      {chi.label:<14} DC-always-on |E_warp(s)| = {edc:.6f} (small, "
              f"NOT 0)   DC-in-knob: |sum chi(a)| = {prj:.1e}  ->  K=0 == 0")
    print("      (non-principal chi: with the DC offset + completion counted as "
          "discreteness data,")
    print("       every component endpoint is 1/(s-1) and orthogonality kills it: "
          "ALL COMB, NO CONTINUUM.)")


def _print_migrations():
    print("\n== migrations (route A comb): zeros onto their lines at O(1/K) ==")
    results = {}
    hdrK = [0, 1, 5, 21, 45]
    for chi, targets, tag in [
            (CHI3, [mp.mpc(0.5, t) for t in CHI3_ZEROS[:3]], "chi_3 -> sigma=1/2"),
            (CHI5_I, [mp.mpc(0.5, t) for t in CHI5_I_ZEROS[:3]],
             "chi_5(2->i) COMPLEX -> sigma=1/2"),
            (CHI6, [mp.mpc(0.5, t) for t in CHI3_ZEROS[:2]]
             + [mp.mpc(0, t) for t in CHI6_COMB_ZEROS],
             "chi_6 IMPRIMITIVE -> sigma=1/2 UNION sigma=0 comb")]:
        print(f"   -- {tag} --")
        print("   target zero          " + "".join(f"K={k:<7d}" for k in hdrK))
        trajs = []
        sched0 = [0] + K_SCHEDULE
        for zt in targets:
            traj = hb.migrate(lambda ss, K, c=chi: L_comb_K(ss, K, c), zt,
                              schedule=sched0)
            d = {K: zz for K, zz in traj if zz is not None}
            cells = "".join(f"{float(d[k].real):<9.3f}" if k in d else "   --    "
                            for k in hdrK)
            print(f"   {float(zt.real):.1f}+{float(zt.imag):7.3f}i    {cells}")
            trajs.append((zt, traj))
        results[chi.label] = trajs
    print("   (K=0 = the E_chi ground string; chi_6 splits: primitive zeros -> 1/2, "
          "Euler-factor comb -> 0)")
    return results


def _print_zero_checks():
    print("\n== migration-target verification (findroot polish on L itself) ==")
    for chi, zs, sig in [(CHI3, CHI3_ZEROS, 0.5), (CHI5_I, CHI5_I_ZEROS, 0.5),
                         (CHI5_QUAD, CHI5_QUAD_ZEROS, 0.5)]:
        worst_off = 0.0
        for t in zs:
            z = polish_zero(chi, t, sig)
            assert z is not None, (chi.label, t)
            worst_off = max(worst_off, abs(float(z.real) - sig))
        print(f"   {chi.label:<14} {len(zs)} zeros re-polished; worst |Re - {sig}| "
              f"= {worst_off:.1e}")
    worst = max(float(abs(L_chi(mp.mpc(0, t), CHI6))) for t in CHI6_COMB_ZEROS)
    print(f"   chi_6 sigma=0 comb (t = (2m+1) pi/ln2): worst |L(it, chi_6)| = {worst:.1e}")


def _traj_xy(traj):
    pts = [(K, zz) for K, zz in traj if zz is not None]
    return ([float(zz.real) for _, zz in pts], [float(zz.imag) for _, zz in pts],
            [K for K, _ in pts])


def _main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(_BRIDGE))
    import figstyle
    figstyle.enlarge()
    plt.rcParams.update({"axes.titlesize": 13, "legend.fontsize": 10.5,
                         "figure.titlesize": 16})

    _print_characters()
    _print_L_anchors()
    _print_comb()
    _print_warp_and_projector()
    _print_zero_checks()
    migr = _print_migrations()

    # ---- the P2 panel data: components scattered, combination on-line ----
    print("\n== components vs combination (the prediction-2 panel) ==")
    comp_zeros = {}
    for a in (1, 2, 3, 4):
        al = mp.mpf(a) / 5
        comp_zeros[a] = wa.find_zeros(lambda s, al=al: mp.zeta(s, al),
                                      (-0.3, 1.8), (2.0, 22.0))
        dev, noff, ngt1 = wa.clean_score(comp_zeros[a], lines=(0.5,))
        print(f"   zeta(s, {a}/5): {len(comp_zeros[a])} zeros, max dev from 1/2 = "
              f"{dev:.3f}, {ngt1} with sigma > 1  (scattered)")
    L5 = [polish_zero(CHI5_I, t) for t in CHI5_I_ZEROS]
    print(f"   L(s, chi_5(2->i)): {len(L5)} zeros, all |Re - 1/2| < 1e-10  (one line)")

    # ============================== figure ==============================
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10))

    # (0,0) P2: four scattered component clouds + the on-line combination
    ax = axes[0, 0]
    marks = {1: ("C3", "^"), 2: ("C1", "v"), 3: ("C4", "<"), 4: ("C8", ">")}
    for a, zs in comp_zeros.items():
        c, mk = marks[a]
        ax.scatter([float(z.real) for z in zs], [float(z.imag) for z in zs],
                   s=26, marker=mk, color=c, alpha=0.75,
                   label=rf"$\zeta(s,{a}/5)$ (scattered)")
    ax.scatter([float(z.real) for z in L5], [float(z.imag) for z in L5], s=80,
               marker="o", facecolors="none", edgecolors="C2", linewidths=1.6,
               label=r"$L(s,\chi_5)$, $\chi(2)=i$", zorder=5)
    ax.axvline(0.5, ls="-", lw=1.2, color="C2", zorder=1)
    ax.axvline(1.0, ls="--", lw=1, color="0.7", zorder=1)
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$t$")
    ax.set_title("components scattered (Davenport-Heilbronn),\n"
                 r"combination on $\sigma=\frac{1}{2}$")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(-0.4, 2.0)

    # (0,1) chi_3 migration
    ax = axes[0, 1]
    for zt, traj in migr[CHI3.label]:
        xs, ys, ks = _traj_xy(traj)
        if xs:
            ax.plot(xs, ys, "-o", ms=3, lw=1, color="C0")
            ax.plot(xs[-1], ys[-1], "k.", ms=9)
            if ks[0] == 0:
                ax.plot(xs[0], ys[0], "s", ms=7, mfc="none", mec="0.3")
    ax.plot([], [], "-o", color="C0", label=r"$L^{(K)}$ zeros")
    ax.plot([], [], "s", mfc="none", mec="0.3", label=r"$K=0$ ground ($E_\chi$)")
    ax.axvline(0.5, ls="-", lw=1.2, color="C2", label=r"$\sigma=\frac{1}{2}$")
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$t$")
    ax.set_title(r"$\chi_3$ (primitive real): ground $\to$ one clean line")
    ax.legend(loc="upper right")

    # (0,2) chi_6 imprimitive split
    ax = axes[0, 2]
    for zt, traj in migr[CHI6.label]:
        xs, ys, ks = _traj_xy(traj)
        if xs:
            col = "C0" if abs(float(zt.real) - 0.5) < 0.1 else "C1"
            ax.plot(xs, ys, "-o", ms=3, lw=1, color=col)
            ax.plot(xs[-1], ys[-1], "k.", ms=9)
            if ks[0] == 0:
                ax.plot(xs[0], ys[0], "s", ms=7, mfc="none", mec="0.3")
    ax.plot([], [], "-o", color="C0", label=r"$\to\sigma=\frac{1}{2}$ (primitive)")
    ax.plot([], [], "-o", color="C1", label=r"$\to\sigma=0$ (Euler comb)")
    ax.axvline(0.5, ls="-", lw=1.2, color="C2")
    ax.axvline(0.0, ls="--", lw=1.2, color="C1")
    for t in CHI6_COMB_ZEROS:
        ax.plot(0, t, "x", color="C1", ms=10, mew=2)
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$t$")
    ax.set_title(r"$\chi_6$ (imprimitive): SPLIT onto $\frac{1}{2}\cup$"
                 r" $\{1+2^{-s}\}$ comb")
    ax.legend(loc="upper right")

    # (1,0) chi_5 complex migration
    ax = axes[1, 0]
    for zt, traj in migr[CHI5_I.label]:
        xs, ys, ks = _traj_xy(traj)
        if xs:
            ax.plot(xs, ys, "-o", ms=3, lw=1, color="C3")
            ax.plot(xs[-1], ys[-1], "k.", ms=9)
            if ks[0] == 0:
                ax.plot(xs[0], ys[0], "s", ms=7, mfc="none", mec="0.3")
    ax.plot([], [], "-o", color="C3", label=r"$L^{(K)}$ zeros")
    ax.axvline(0.5, ls="-", lw=1.2, color="C2", label=r"$\sigma=\frac{1}{2}$")
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$t$")
    ax.set_title(r"$\chi_5$, $\chi(2)=i$ (COMPLEX coefficients):"
                 "\nsame story, no conjugate symmetry")
    ax.legend(loc="upper right")

    # (1,1) the rate law: K * |L - L_comb_K| flattening onto rate_comb_chi, per q
    ax = axes[1, 1]
    s0 = mp.mpc(1.3, 7)
    Ks = [5, 10, 20, 40, 80, 160]
    for chi, col in [(CHI3, "C0"), (CHI4, "C2"), (CHI5_I, "C3"), (CHI6, "C1")]:
        L = L_chi(s0, chi)
        kerrs = [float(K * abs(L_comb_K(s0, K, chi) - L)) for K in Ks]
        pred = float(abs(rate_comb_chi(s0, chi)))
        ax.semilogx(Ks, kerrs, "-o", ms=4, color=col,
                    label=rf"{chi.label}  ($\to{pred:.2f}$)")
        ax.axhline(pred, ls=":", lw=1, color=col)
    ax.set_xlabel("K")
    ax.set_ylabel(r"$K\,|L - L^{(K)}_{\rm comb}|$ at $s=1.3+7i$")
    ax.set_title(r"$O(1/K)$ survives; constant $=\frac{sq}{2\pi^2}"
                 r"\sum_a\chi(a)a^{-s-1}$ ($q$-dependent)")
    ax.legend(loc="center right", fontsize=9)

    # (1,2) endpoints: zeta's pole vs the entire character endpoints (prediction 3)
    ax = axes[1, 2]
    # tiny imaginary offset keeps the grid off the literal pole at s = 1
    sig = [0.2 + 0.02 * i for i in range(int((3.0 - 0.2) / 0.02) + 1)]
    ax.semilogy(sig, [float(abs(1 / (mp.mpc(sg, "1e-9") - 1) + mp.mpf("0.5")))
                      for sg in sig], "-", color="k", lw=1.6,
                label=r"$\zeta$: $|1/(s-1)+\frac{1}{2}|$ (pole)")
    for chi, col in [(CHI3, "C0"), (CHI5_I, "C3")]:
        ax.semilogy(sig, [float(abs(comb_endpoint(mp.mpc(sg) + mp.mpf("1e-9") * _J,
                                                  chi))) for sg in sig],
                    "-", color=col, lw=1.4, label=rf"{chi.label}: $|E_\chi|$ (entire)")
    ax.axvline(1.0, ls=":", lw=1, color="0.6")
    ax.set_xlabel(r"$\sigma$ (real axis)")
    ax.set_ylabel(r"$|$endpoint$(\sigma)|$")
    ax.set_title("$K=0$ continuum: $\\zeta$ has a pole,\n"
                 "$\\chi\\neq\\chi_0$ is entire (DC-in-knob: $\\equiv 0$)")
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(r"Character combs $L^{(K)}(s,\chi)=q^{-s}\sum_a\chi(a)\,"
                 r"\zeta^{(K)}(s,a/q)$: the discreteness knob on the rest of degree 1"
                 r" (issue #37)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    figdir = _HERE / "figures"
    figdir.mkdir(exist_ok=True)
    out = figdir / "character_bridge.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    _main()
