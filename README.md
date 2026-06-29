# dirichlet-bridge

**Interpolating from the bland continuous integral `∫₁^∞ x^{-s} dx = 1/(s−1)` to
the full Dirichlet sum `ζ(s)`, and watching the non-trivial zeros be _born_ and
_migrate_ onto the critical line.**

The interpolation knob is *how many Fourier harmonics `K` of the discreteness
kernel* (the Euler–Maclaurin sawtooth ≡ Abel–Plana comb) you keep. At `K = 0`
the object is the pure integral `1/(s−1)` — ζ's pole, but **no non-trivial
zeros at all**. As `K → ∞` you recover `ζ(s)`, and the zeros appear and condense
onto `Re s = ½`.

> **What this is — and isn't.** This is a *computational / experimental-mathematics*
> study: a unified, reproducible re-derivation that threads several **known**
> endpoints (`1/(s−1)`, `ζ(s)`, the half-shifted `(2^s−1)ζ(s) = ζ(s,½)`, and the
> general `ζ(s,α)`) onto one explicit discreteness-restoration knob. It is **not** a
> new theorem about where zeros live. The individual ingredients are classical;
> the contribution is the synthesis, the unifying `O(1/K)` rate law, and a couple
> of spectral readings. Prior art is cited explicitly (see *Relation to known
> work* below).

## The arc (six modules)

| Module | What it shows |
|---|---|
| `cont_eta.py` | The rich continuous endpoint `F(s)=∫₁^∞ cos(πx)x^{-s}dx` — an incomplete-gamma closed form whose **off-critical** zero string already obeys the Riemann–von Mangoldt counting law `N(t)=(t/2π)ln(t/πe)+5/8`. The right zero *density*, not yet pinned to ½. |
| `harmonic_bridge.py` | Restore discreteness *additively*: Euler–Maclaurin sawtooth ≡ Abel–Plana Bose comb, term-by-term. The **zero-migration map** — ζ zeros *born* onto σ=½; the η ground string *splits* onto σ=½ (ζ zeros) and σ=1 (the `1−2^{1-s}` prefactor zeros). |
| `warp_bridge.py` | The *nonlinear* `n*`-warp deliberately targeting the half-shift `ζ(s,½)=(2^s−1)ζ(s)`: zeros climb onto σ=½ **from below**, plus a **σ=0 companion** line from the `2^s−1` factor (the functional-equation mirror of η's σ=1 comb). The σ=½ + σ=0 bijection is *measured*, not asserted. |
| `warp_alpha.py` | The general-phase `n+α` warp → `ζ(s,α)`, exhibiting the **Saias–Weingartner `P·L` dichotomy**: clean vertical lines only at `α ∈ {0, ½, 1}`; generic α is Davenport–Heilbronn-scattered off-line. |
| `rate_law.py` | The **unified `O(1/K)` rate / birth-`K` law** across all routes (one sawtooth-tail mechanism), the approach-side `= sign(Re a₁)` criterion, the overshoot ⟺ `Re a₁ < 0` rule, and the Gram's-law vertical-migration duality. |
| `eta_two_component.py` | The η zero set as a two-component **crystal × GUE** spectrum: a rigid σ=1 comb (`t = 2πk/ln2`) superimposed on the σ=½ GUE ζ-zeros, with the comb frequencies `m·ln2` tied to the `p = 2` prime-power lattice in the explicit formula. |

The first five are pure `mpmath`; `eta_two_component.py` additionally reads the
cached Riemann zeros in `data/riemann_zeros.csv` and reuses a few spectral-statistics
helpers (`gue_spacing`, `spectral_rigidity`, `zero_form_factor`, `cone_log_prime`)
that are vendored into `bridge/` alongside it.

Two small companion drivers illustrate the warp itself (see [`RESULTS.md`](RESULTS.md)):
`warp_coordinate.py` draws the coordinate `n* = x + φ_K(x)` bending from a straight line
into the midpoint staircase as `K` grows (Gibbs wobble and all), and
`warp_phase_compare.py` contrasts the midpoint staircase (`α = ½`, half-integers) with the
integer one (`α = 1`, counting numbers) — and why a pure-`x` start forces the former.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                  # numpy, scipy, mpmath, matplotlib, pytest (pinned)
# (or, with uv:  uv venv --python 3.9 .venv  &&  uv pip install -r requirements.txt)

pytest                       # fast suite (slow high-precision checks deselected by default)
pytest -m slow               # the expensive end-to-end regressions

python repro.py              # regenerate ALL figures from scratch (a few minutes)
python bridge/cont_eta.py    # or run one driver: it self-validates and writes bridge/figures/<name>.png
```

Targets **Python 3.9** (kept 3.9-compatible: `typing.Union`, not PEP-604 `X | Y`
runtime unions). Every driver anchors its data reads and figure writes to the
repo root, so it runs the same regardless of the working directory.

**Read next:** [`RESULTS.md`](RESULTS.md) walks through every figure ("what it
shows / why it matters"); [`CONTRIBUTING.md`](CONTRIBUTING.md) covers setup and the
layout conventions.

## Layout

```
bridge/        the six drivers + the vendored spectral-statistics helpers (flat; bare cross-imports)
  figures/     generated PNGs (gitignored)
data/          riemann_zeros.csv (cached non-trivial zeros)
tests/         one test module per driver
conftest.py    puts bridge/ on sys.path for pytest
```

## Relation to known work (prior art it is distinct *from*)

The endpoints and most ingredients are classical — this package cites them rather
than claiming them:

- The half-shift `(2^s−1)ζ(s)=ζ(s,½)`, its σ=0 companion at `t=2πk/ln2`, and the
  migration of zeros onto σ=½ as a parameter varies: **Garunkštis–Steuding** (Math.
  Comp. 76, 2007), **Gonek** (Springer LNM 899, 1981).
- The `α ∈ {0,½,1}` "clean lines only" dichotomy: **Saias–Weingartner** (Acta Arith.
  140, 2009); generic off-line `ζ(s,α)` zeros: **Davenport–Heilbronn** (1936) /
  **Cassels** (1961).
- The σ=1 eta comb at `t=2πk/ln2`: **Sondow** (Amer. Math. Monthly 110, 2003);
  **Beliakov–Matiyasevich** (Exp. Math., 2015); DLMF §25.2.3.
- The closest "knob" cousin (partial-sum *length* `N`, not the harmonic count `K`):
  **Gonek–Ledoan** (IMRN 2010), **Gonek–Montgomery** (IMRN 2013).
- Prime-power log-frequencies (smallest `log 2`) in the zeta-zero number variance,
  in a regime where GUE fails: **Berry** (1988), proved by **Lugar–Milinovich–
  Quesada-Herrera** (2022).
- The de Bruijn–Newman backward-heat flow (**Rodgers–Tao**, Forum Math. Pi 8, 2020)
  *rearranges* an always-infinite zero set toward all-real — a categorically
  different deformation from this *creation-from-a-zeroless-endpoint* knob.

A full annotated bibliography (`references.bib`) and a figure-by-figure walkthrough
accompany the preprint.

## Provenance

This repository is the self-contained **bridge arc**, extracted from the author's
larger research project; it carries the six arc drivers plus the vendored
spectral-statistics helpers they depend on, and nothing else.

## License

MIT — see [`LICENSE`](LICENSE).
