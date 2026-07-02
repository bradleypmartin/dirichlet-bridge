# dirichlet-bridge

**Interpolating from the bland continuous integral `∫₁^∞ x^{-s} dx = 1/(s−1)` to
the full Dirichlet sum `ζ(s)`, and watching the non-trivial zeros be _born_ and
_migrate_ onto the critical line.**

The interpolation knob is *how many Fourier harmonics `K` of the discreteness
kernel* (the Euler–Maclaurin sawtooth ≡ Abel–Plana comb) you keep. At `K = 0`
the object is the pure integral `1/(s−1)` — ζ's pole, but **no non-trivial
zeros at all**. As `K → ∞` you recover `ζ(s)`, and the zeros appear and condense
onto `Re s = ½`.

## Summary and scope

This is a *computational / experimental-mathematics* study: a unified,
reproducible re-derivation that threads several **known** endpoints (`1/(s−1)`,
`ζ(s)`, the half-shifted `(2^s−1)ζ(s) = ζ(s,½)`, and the general `ζ(s,α)`) onto
one explicit discreteness-restoration knob. It is **not** a new theorem about
where zeros live. The individual ingredients are classical; the contribution is
the synthesis, the unifying `O(1/K)` rate law, and a couple of spectral
readings. Prior art is cited explicitly (see *Relation to known work* below).

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

Six small companion drivers extend the arc (see [`RESULTS.md`](RESULTS.md)):
`comb_vs_warp.py` draws the two restoration routes side by side at the level of the
integrand — the additive comb `x^{-s} + s·φ_K(x)·x^{-s-1}` (kernel *added*; `x` linear)
vs the warp `(x + φ_K(x))^{-s}` (kernel *composed into the argument*; `x` warped) — the
single clearest picture of how the two methods differ, both built from the same `φ_K`;
`warp_coordinate.py` draws the coordinate `n* = x + φ_K(x)` bending from a straight line
into the midpoint staircase as `K` grows (Gibbs wobble and all); `warp_phase_compare.py`
contrasts the midpoint staircase (`α = ½`, half-integers) with the integer one (`α = 1`,
counting numbers) — and why a pure-`x` start forces the former; `warp_eta.py` warps the **η**
integrand `cos(πx)x^{-s}` and finds the midpoint `α = ½` *annihilated* (`cos(π(m+½)) = 0`), so
only `α = 1` survives — reproducing the σ=½ ∪ σ=1 split by a second route; `stability.py`
shows each `K`-truncated interpolant is meromorphic on **all of ℂ** (no abscissa — it converges
at σ<0 too), the cost being *height* `t`, not `Re s`; and `trivial_zeros.py` follows that
stability out to σ<0 to show where the **trivial** zeros come from — a rising tide `+|s|/2π²K`
that carves `−2, −4, −6, …` onto the negative real axis, the first *inherited* from the bland
endpoint's lone zero and the rest *born* as conjugate pairs that pinch onto the axis.

Five further drivers carry the preprint's two appendices — the four **beyond-ζ/η**
cases (Appendix A) and the **geometric-series miniature** (Appendix B):

- `jonquiere_zeros.py` reproduces **Fornberg–Kölbig 1975**, the program's direct
  ancestor: the polylogarithm's argument as a deformation knob, whose zero
  trajectories only *graze* `σ = ½` before the pole emerging at `x = 1` absorbs
  them — the sharpest contrast with the bridge's *converge-and-stay*.
- `lfunction_bridge.py` runs the bridge's own `K`-knob + warp on the primitive
  Dirichlet L-function `L(s,χ₄)`: zeros born onto the *single* line `σ = ½` with
  **no companion** (primitivity — cleaner than η), the warp landing on the Lerch
  value `2^s·L(s,χ₄)`.
- `hurwitz_lerch_zeros.py` reproduces the closest *published* parameter-sweep
  analogues — the Garunkštis–Steuding Hurwitz `α`-trajectories (stable zeros
  wandering on/off ½; unstable ones landing on the σ=0 comb) and the
  Garunkštis–Tamošiūnas Lerch `λ=α` diagonal — organized by the **Lerch family
  square** whose corners are the bridge's four objects.
- `epstein_zeros.py` reproduces the reverse-direction foil (**Travěnec–Šamaj
  2022**): the Epstein dimension-knob pitchfork at `d* ≈ 9.24555`, zeros born
  *off* the critical line `σ = d/4` and migrating *away* — the opposite arrow.
- `geometric_bridge.py` runs the whole construction on the geometric series
  `Σ sⁿ = 1/(1−s)`, where the comb *telescopes* and the warp *factorizes* in
  elementary closed form — no special functions, no zeros; an illustrative
  consistency check, not a result.

## Quickstart

```bash
# any Python 3.9-3.12 (the pinned wheels don't cover 3.13+)
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                  # numpy, scipy, mpmath, matplotlib, pytest (pinned)
# (or, with uv:  uv venv --python 3.9 .venv  &&  uv pip install -r requirements.txt
#  -- then activate the same way; a uv-created venv has no pip inside, so use
#  `uv pip ...` for any later installs)

pytest                       # fast suite (slow high-precision checks deselected by default)
pytest -m slow               # the expensive end-to-end regressions

python repro.py              # regenerate ALL 18 figures from scratch (~22 min)
python bridge/cont_eta.py    # or run one driver: it self-validates and writes bridge/figures/<name>.png
```

Targets **Python 3.9** and runs unchanged through **3.12** (the pinned wheels
cover 3.9–3.12 only; on 3.13+ the numpy/scipy install fails). The code is kept
3.9-compatible (`typing.Union`, not PEP-604 `X | Y` runtime unions). Every
driver anchors its data reads and figure writes to the repo root, so it runs
the same regardless of the working directory. Windows note: a fresh PowerShell
may refuse `Activate.ps1` under its default execution policy — either run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or skip activation
and call `.venv\Scripts\python.exe` / `.venv\Scripts\pytest.exe` directly
(everything here works unactivated; `repro.py` launches its drivers via
`sys.executable`).

**Read next:** [`RESULTS.md`](RESULTS.md) walks through every figure ("what it
shows / why it matters"); [`CONTRIBUTING.md`](CONTRIBUTING.md) covers setup and the
layout conventions.

## Layout

```
bridge/        the six core drivers + six companions + five appendix drivers + vendored spectral-statistics helpers (flat; bare cross-imports)
  figures/     the 18 canonical PNGs (committed; other generated output gitignored)
data/          riemann_zeros.csv (cached non-trivial zeros)
tests/         one test module per driver
paper/         arXiv preprint: LaTeX source, references.bib, built PDF
conftest.py    puts bridge/ on sys.path for pytest
```

## Relation to known work (prior art it is distinct *from*)

The endpoints and most ingredients are classical — this package cites them rather
than claiming them:

- The **direct ancestor** of the whole deform-and-track program: **Fornberg–Kölbig**
  (Math. Comp. 29, 1975) sweep the polylogarithm's argument `x` from `−1` to `1`
  (endpoints `−η` and `ζ`) and follow the zero trajectories — in 1975. On their knob
  the zeros only *graze* `σ = ½` before being absorbed into the emerging pole; the
  bridge's harmonic-count knob is one on which that chased convergence actually
  happens (`jonquiere_zeros.py` reproduces FK's computation and draws the contrast).
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
- The closest published parameter-sweep analogues — **Garunkštis–Steuding** (above;
  re-framed as the nearest deform-and-track sibling, its "stable" zeros wandering
  on/off ½) and **Garunkštis–Tamošiūnas** (Lith. Math. J. 57, 2017; the Lerch `λ=α`
  diagonal) — and the reverse-direction foil **Travěnec–Šamaj** (Appl. Math. Comput.
  413, 2022; Epstein zeros born *off* the line, migrating away) are all *reproduced*
  in-repo and distinguished in the preprint's Appendix A.

The preprint (LaTeX source, the annotated `references.bib`, and the built PDF)
lives in [`paper/`](paper/) — a computational / experimental-mathematics write-up
of the whole arc. See [`paper/README.md`](paper/README.md) to build it (one
command with [Tectonic](https://tectonic-typesetting.github.io/)) or to package an
arXiv upload.

## Provenance

This repository began as the self-contained **bridge arc**, extracted from the
author's larger research project: the six core arc drivers, their companions, and
the vendored spectral-statistics helpers they depend on. The five appendix drivers
(the beyond-ζ/η reproductions and the geometric-series miniature) and the preprint
were developed here.

## License

MIT — see [`LICENSE`](LICENSE).
