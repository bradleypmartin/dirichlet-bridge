r"""
Continuous Dirichlet-eta integral  F(s) = int_1^inf cos(pi x) x^{-s} dx.

The continuous-eta endpoint of the discrete <-> continuous bridge. Treat the eta *series*
eta(s) = sum (-1)^{n-1} n^{-s} as a Riemann sum (sign (-1)^n = cos(pi n) at the
integers) and ask for the underlying *integral*. Unlike the bland zeta-side
endpoint  int_1^inf x^{-s} dx = 1/(s-1)  (no zeros at all), this continuous eta is
genuinely structured.

Closed form
-----------
Using  int_1^inf x^{-s} e^{-p x} dx = p^{s-1} Gamma(1-s, p)  (upper incomplete
gamma) with cos(pi x) = (e^{i pi x} + e^{-i pi x})/2, so p = -/+ i*pi:

    F(s) = (1/2) [ (-i pi)^{s-1} Gamma(1-s, -i pi) + (i pi)^{s-1} Gamma(1-s, i pi) ]

(`F_closed`; validated to full precision against `F_quad`, an oscillatory
quadrature, in __main__).

The off-critical zero string and its laws (the Phase-0 deliverable)
------------------------------------------------------------------
For large t = Im s the two conjugate prefactors split exponentially,
|(-/+ i pi)^{s-1}| = pi^{sigma-1} e^{+/- pi t/2}. Pushing the large-parameter
(a = 1-s -> infinity, z = +/-i pi fixed) incomplete-gamma asymptotics through the
cancellation leaves a clean two-term reduction (`F_asym`):

    F(s) ~ 1/(1-s)  +  (1/2) (-i pi)^{s-1} Gamma(1-s).

The first term is the bland continuous pole; the second is the oscillatory term
that carries the zeros. Setting F = 0 and balancing magnitude vs phase gives three
laws, confirmed numerically over 8 < t < 160 (2-3 digits for the sigma-law, 3 for
spacing, 4-5 for the counting constant):

  * real-part law  (`sigma_law`):  pi^{sigma-1} sqrt(2 pi) t^{3/2-sigma} = 2, i.e.
        sigma(t) = (3/2 ln t - 1/2 ln 2pi) / ln(t/pi)   ->  3/2   (logarithmically).
    The zeros are NOT on a critical line; their real parts drift down toward 3/2.

  * spacing law  (`spacing_law`):  Delta t ~ 2 pi / ln(t/pi).
    A slowly *compressing* gap (~ 1/ln t) -- NOT the naive t_n ~ log n we had
    guessed from an earlier quick look.

  * counting law  (`counting_law`):  N(t) = (t/2pi) ln( t / (pi e) ) + 5/8 + o(1).
    This is a Riemann-von Mangoldt law: the SAME zero-density structure as the
    genuine zeta/eta zeros (pi where zeta has 2pi, constant 5/8 where zeta has
    7/8) -- but realized off the critical line. That is the point of this continuous
    endpoint: the continuous eta already has "non-trivial" zeros with the right
    density; what it lacks (and what restoring discreteness must supply) is pinning
    them to Re s = 1/2.

Run directly to validate + plot: writes bridge/figures/cont_eta.png.
"""
import math
from pathlib import Path
from typing import List

import mpmath as mp

mp.mp.dps = 30

_IPI = mp.j * mp.pi
_E = math.e


# --------------------------------------------------------------------------
# the object
# --------------------------------------------------------------------------
def F_closed(s):
    """F(s) via the upper-incomplete-gamma closed form (the primary evaluator)."""
    s = mp.mpc(s)
    return (((-_IPI) ** (s - 1)) * mp.gammainc(1 - s, -_IPI)
            + ((_IPI) ** (s - 1)) * mp.gammainc(1 - s, _IPI)) / 2


def F_quad(s):
    """F(s) by direct oscillatory quadrature (period-2 cosine) -- the reference."""
    s = mp.mpc(s)
    f = lambda x: mp.cos(mp.pi * x) * x ** (-s)
    return mp.quadosc(f, [1, mp.inf], period=2)


def F_asym(s):
    """Two-term asymptotic reduction  1/(1-s) + (1/2)(-i pi)^{s-1} Gamma(1-s)."""
    s = mp.mpc(s)
    return 1 / (1 - s) + mp.mpf("0.5") * ((-_IPI) ** (s - 1)) * mp.gamma(1 - s)


# --------------------------------------------------------------------------
# the derived laws  (floats; t = Im s)
# --------------------------------------------------------------------------
def sigma_law(t):
    """Predicted real part of the zero at height t (magnitude balance) -> 3/2."""
    lt = math.log(t)
    return (1.5 * lt - 0.5 * math.log(2 * math.pi)) / (lt - math.log(math.pi))


def spacing_law(t):
    """Predicted gap to the next zero at height t:  2 pi / ln(t/pi)."""
    return 2 * math.pi / math.log(t / math.pi)


def counting_law(t):
    """Predicted number of zeros with 0 < Im <= t (Riemann-von Mangoldt form)."""
    return t / (2 * math.pi) * math.log(t / (math.pi * _E)) + 0.625


# --------------------------------------------------------------------------
# zero finding (continuation seeded by the laws -- one root solve per zero)
# --------------------------------------------------------------------------
def _polish(t_seed):
    """findroot near (sigma_law(t_seed), t_seed); return the root or None."""
    try:
        z = mp.findroot(F_closed, mp.mpc(sigma_law(t_seed), t_seed))
    except (ValueError, ZeroDivisionError):
        return None
    if abs(F_closed(z)) < mp.mpf(10) ** (-mp.mp.dps + 8) and z.imag > 1:
        return z
    return None


def find_zeros(t_max, t_first=10.6714):
    """Zeros of F with 0 < Im up to ~t_max, walked upward via the spacing law.

    The first zero is seeded near t_first; thereafter the next seed is the law
    prediction t + spacing_law(t), which lands within ~0.01 of the true zero, so a
    single findroot per zero suffices. The walk stops once the next *seed* would
    pass t_max, so the cut is fuzzy by a fraction of one spacing: the last root
    kept can sit slightly above t_max.
    """
    zeros = []  # type: List[mp.mpc]
    z = _polish(t_first)
    if z is None:                       # robustness fallback: small scan
        for ts in [t_first - 0.5, t_first + 0.5, t_first + 1.5]:
            z = _polish(ts)
            if z is not None:
                break
    if z is None:
        raise RuntimeError("could not locate the first zero")
    zeros.append(z)
    t = float(z.imag)
    while True:
        t_next = t + spacing_law(t)
        if t_next > t_max:
            break
        z = _polish(t_next)
        if z is None or float(z.imag) <= t + 0.1:   # stalled -> nudge upward
            z = _polish(t_next + 0.3)
        if z is None or float(z.imag) <= t + 0.1:
            break
        zeros.append(z)
        t = float(z.imag)
    return zeros


# --------------------------------------------------------------------------
# self-validating driver
# --------------------------------------------------------------------------
def _main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figstyle
    figstyle.enlarge()

    print("== F_closed vs F_quad (oscillatory quadrature reference) ==")
    worst = 0.0
    for s in [mp.mpc(2, 1), mp.mpc(1.5, 2), mp.mpc(0.7, 10), mp.mpc(1.8, 25)]:
        rel = float(abs((F_closed(s) - F_quad(s)) / F_quad(s)))
        worst = max(worst, rel)
        print(f"   s={complex(s):>16}  rel err = {rel:.2e}")
    print(f"   worst closed-vs-quad rel err = {worst:.2e}")

    print("\n== F_asym (two-term reduction) vs F_closed ==")
    for t in [20, 40, 80, 150]:
        s = mp.mpc(sigma_law(t), t)
        rel = float(abs((F_asym(s) - F_closed(s)) / F_closed(s)))
        print(f"   t={t:>4}  rel err = {rel:.2e}   (-> 0 as t grows)")

    t_max = 120.0
    zeros = find_zeros(t_max)
    ts = [float(z.imag) for z in zeros]
    ss = [float(z.real) for z in zeros]
    print(f"\n== {len(zeros)} zeros with 0 < Im <= {t_max:.0f} ==")
    print("    n        t      sigma   |  gap     law      |  N    law")
    max_gap_rel = 0.0
    for i, (t, sg) in enumerate(zip(ts, ss)):
        if i == 0:
            print(f"   {i+1:>3}  {t:8.4f}  {sg:7.4f}  |   --        --     "
                  f"|  {i+1:>2}  {counting_law(t):6.3f}")
            continue
        gap = ts[i] - ts[i - 1]
        law = spacing_law(0.5 * (ts[i] + ts[i - 1]))
        max_gap_rel = max(max_gap_rel, abs(gap - law) / gap)
        print(f"   {i+1:>3}  {t:8.4f}  {sg:7.4f}  |  {gap:6.4f}  {law:6.4f}  "
              f"|  {i+1:>2}  {counting_law(t):6.3f}")
    print(f"\n   worst spacing-law rel err = {max_gap_rel:.2e}")
    print(f"   sigma range = [{min(ss):.4f}, {max(ss):.4f}]  (-> 3/2 from above)")
    offs = [(i + 1) - counting_law(t) for i, t in enumerate(ts) if t > 20]
    print(f"   counting offset N - (t/2pi)ln(t/pi e) = "
          f"{sum(offs)/len(offs):.4f} +/- {max(offs)-min(offs):.0e}  (-> 5/8)")

    # ---- figure ----
    tt = [8 + 0.5 * k for k in range(int((t_max - 8) / 0.5) + 1)]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    ax[0].plot(ss, ts, "o", ms=4, color="C0", label="zeros of F(s)")
    ax[0].plot([sigma_law(t) for t in tt], tt, "-", color="C3", lw=1.2,
               label=r"$\sigma$-law")
    ax[0].axvline(1.5, ls="--", color="0.5", lw=1, label=r"$\sigma=3/2$")
    ax[0].axvline(0.5, ls=":", color="C2", lw=1, label=r"critical line $1/2$")
    ax[0].set_xlabel(r"$\sigma=\mathrm{Re}\,s$"); ax[0].set_ylabel(r"$t=\mathrm{Im}\,s$")
    ax[0].set_title("zero string (off the critical line)")
    ax[0].set_xlim(0.3, 2.3); ax[0].legend(loc="upper right")

    gaps = [ts[i] - ts[i - 1] for i in range(1, len(ts))]
    mids = [0.5 * (ts[i] + ts[i - 1]) for i in range(1, len(ts))]
    ax[1].plot(mids, gaps, "o", ms=4, color="C0", label="actual gap")
    ax[1].plot(tt, [spacing_law(t) for t in tt], "-", color="C3", lw=1.2,
               label=r"$2\pi/\ln(t/\pi)$")
    ax[1].set_xlabel(r"$t$"); ax[1].set_ylabel(r"$\Delta t$")
    ax[1].set_title("spacing law (compresses like $1/\\ln t$)")
    ax[1].legend()

    ax[2].step(ts, range(1, len(ts) + 1), where="post", color="C0",
               label=r"actual $N(t)$")
    ax[2].plot(tt, [counting_law(t) for t in tt], "-", color="C3", lw=1.2,
               label=r"$\frac{t}{2\pi}\ln\frac{t}{\pi e}+\frac{5}{8}$")
    ax[2].set_xlabel(r"$t$"); ax[2].set_ylabel(r"$N(t)$")
    ax[2].set_title("Riemann-von Mangoldt counting law")
    ax[2].legend()

    fig.suptitle(r"Continuous Dirichlet-eta  $F(s)=\int_1^\infty\cos(\pi x)\,"
                 r"x^{-s}\,dx$")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    figdir = Path(__file__).resolve().parent / "figures"
    figdir.mkdir(exist_ok=True)
    out = figdir / "cont_eta.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    _main()
