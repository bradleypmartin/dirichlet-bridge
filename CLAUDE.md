# dirichlet-bridge — Project Notes

## What this is
A standalone, reproducible study of the **discrete ↔ continuous bridge**:
interpolating from the bland continuous integral `∫₁^∞ x^{-s} dx = 1/(s−1)`
(which keeps ζ's pole but has **no** non-trivial zeros, no functional equation,
no arithmetic) to the full Dirichlet sum `ζ(s)`. The interpolation knob is *how
many Fourier harmonics `K` of the discreteness kernel* (the Euler–Maclaurin
sawtooth ≡ Abel–Plana comb) you keep. As `K` grows, the non-trivial zeros are
**born** and **migrate** onto `Re s = ½`. Extracted from a larger research
project; this repo is the self-contained bridge arc.

## Honest framing — read this before claiming anything
This is a **methods / experimental-mathematics** study (a decision made up
front): a computational / expository **synthesis**, **not** a new theorem. Two
adversarial literature passes — full findings + BibTeX live in **issue #2's two
comments** — concluded *"novel as a package."*

**Classical (cite, do not claim):** the half-shift `(2^s−1)ζ(s)=ζ(s,½)` and its
`σ=0` companion (Garunkštis–Steuding 2007; Gonek 1981); the `α∈{0,½,1}` `P·L`
dichotomy (Saias–Weingartner 2009); the `σ=1` eta comb at `t=2πk/ln2` (Sondow
2003; Beliakov–Matiyasevich 2015; DLMF §25.2.3); the partial-sum-length-`N`
cousins (Gonek–Ledoan 2010 / Gonek–Montgomery 2013).

**The one deflator to foreground-and-distinguish:** Berry (1988), proved by
Lugar–Milinovich–Quesada-Herrera (2022), already places prime-power
log-frequencies (smallest `= log 2`) in the zeta-zero **number variance** in a
regime where **GUE fails**. So the "p=2-ghost" reading's surviving novelty is
*only* the comb⊎GUE **superposition** model + **single-prime isolation** + the
**eta tie-in** — not the appearance of `log 2` itself.

**Genuinely un-anticipated (the contribution):** the `K`-knob with a zeroless
`1/(s−1)` endpoint, the EM-sawtooth ≡ Abel–Plana zero-birth identity, the
deliberately-engineered warp landing on `ζ(s,½)`, and the unified `O(1/K)` rate
law (+ the Gram's-law vertical-migration coupling).

## Environment & running
Targets **Python 3.9**, runs on 3.9–3.12 (the pins ship no 3.13+ wheels) — keep
code 3.9-compatible (`typing.Union`, *not* PEP 604 `X | Y` runtime unions).
Per-machine venv (gitignored):

- Setup: `python -m venv .venv` then `pip install -r requirements.txt`
  (numpy, scipy, mpmath, matplotlib, pytest — pinned).
- Fast suite: `pytest` (142 tests, ~95 s; the slow high-precision checks are
  deselected by default via `addopts = -m "not slow"` in `pytest.ini`).
- Slow suite: `pytest -m slow`.
- Each driver self-validates and writes a figure, e.g.
  `python bridge/cont_eta.py` → `bridge/figures/cont_eta.png`.

## File map (one line each)
`bridge/` is a **flat** source dir — every module lives here so the bare
cross-imports resolve as siblings; the root `conftest.py` puts it on `sys.path`
for pytest.

- `cont_eta.py` — the continuous-η endpoint `F(s)=∫₁^∞ cos(πx)x^{-s}dx`; the
  off-critical Riemann–von Mangoldt zero string.
- `comb_vs_warp.py` — the two restoration routes drawn **side by side at the level
  of the integrand** (issue #11): the additive comb `x^{-s}+s·φ_K(x)·x^{-s-1}`
  (kernel added, `x` linear) vs. the warp `(x+φ_K(x))^{-s}` (kernel composed into the
  argument, `x` warped), both from the same `φ_K`. Validates that each integrand's
  area is `harmonic_bridge.zeta_K` / `warp_bridge.warp_K` (depends on both). A preview
  figure for §sec:comb/§sec:warp; the *coordinate* view is `warp_coordinate.py`.
- `harmonic_bridge.py` — additive comb (EM sawtooth ≡ Abel–Plana, term-by-term);
  the zero-migration map (ζ born onto σ=½; η ground string splits σ=½ ∪ σ=1).
- `warp_bridge.py` — the nonlinear midpoint-warp → `ζ(s,½)=(2^s−1)ζ`; the σ=0
  companion; the measured σ=½ + σ=0 bijection.
- `warp_alpha.py` — the general-phase `n+α` warp → `ζ(s,α)`; the
  Saias–Weingartner `P·L` dichotomy.
- `warp_coordinate.py` — the warp map itself: `n* = x+φ_K(x)` bending from the
  straight line into the midpoint staircase as `K` grows (Gibbs wobble and all);
  the *coordinate*-space companion of `comb_vs_warp.py`.
- `warp_phase_compare.py` — the midpoint staircase (`α=½`) vs. the integer one
  (`α=1`), and why a pure-`x` start forces the former.
- `warp_eta.py` — warps the **η** integrand `cos(πx)x^{-s}` (supplement to the
  warp): the midpoint `α=½` is annihilated (`cos(π(m+½))=0`), only `α=1` survives →
  `η=(1−2^{1-s})ζ`, reproducing the σ=½ ∪ σ=1 split by a second route. Reuses the
  `warp_bridge` moment machinery (cos-weighted moments + alternating `−η` tail).
- `rate_law.py` — the unified `O(1/K)` rate / birth-`K` law; the Gram's-law
  vertical-axis duality.
- `stability.py` — the stability corollary of the rate law: each finite-`K`
  interpolant is meromorphic on all of ℂ (no abscissa; converges at σ<0), and the
  cost is **height** `t`, not `Re s` (error ∼ `|s|/(2π²K)`).
- `trivial_zeros.py` — where the **trivial** zeros come from (issue #10): on the
  negative real axis the same `O(1/K)` tail acts as a rising tide `+|s|/(2π²K)` that
  lifts ζ and carves zeros onto `−2n`. The first zero `−2` is *inherited* (the bland
  endpoint's lone zero at `−1` slides onto it); the rest are *born* as conjugate
  pairs that pinch onto the axis near the odd-integer midpoints and split. Deep dips
  are factorially deep → born first; the shallow `{−4,−6}` pair is born last
  (`K≈62`). Reuses `rate_law.disp_coeff` verbatim at `ρ=−2n` (depends on
  `harmonic_bridge`, `rate_law`).
- `jonquiere_zeros.py` — reproduces **Fornberg–Kölbig 1975** (the direct ancestor,
  issue #15): the polylogarithm deformation `F(x,s)=Σ x^k k^{-s}` (`=mpmath.polylog`,
  endpoints `F(1)=ζ`, `F(−1)=−η`) and its complex-zero trajectories as `x` sweeps
  `(−1,1)`. Reproduces FK's two classes (P=0 → the pole `s=1`; the rest → ζ zeros, e.g.
  `s0+(x,1,1)→s₁`, `(2,1)→s₂`, `(1,2)→s₃`), the **transient σ=½ brush + spiral about
  `s₁`** (Figs. 6–7 — the foil: FK *approach but never land*, vs our `O(1/K)`), the
  σ=1 log-2 comb `1+2πim/log2` (Eq. 23, an earlier deform-and-track appearance than
  Sondow 2003), the trivial-zero real-axis trajectories → `−2N` (§6, cousin of
  `trivial_zeros.py`), the argument-principle counts `Z(0.1)=27`,`Z(−0.1)=26` (Eq. 25),
  and the `α±` counting constants (Eqs. 39–40). Self-contained (no bridge cross-imports).
  Tracks zeros with a cancellation-free x→−1 series + a two-phase `|x|` schedule; near
  `x=±1` it swaps `polylog` for the Eq. 26 expansion (speed). Feeds preprint work in #18.
- `lfunction_bridge.py` — the **fork-2 headline** (issue #20): the `K`-knob + warp on a
  genuine primitive **Dirichlet L-function** `L(s,χ₄)=Σχ₄(n)n^{-s}`. Key identity
  `χ₄(n)=sin(πn/2)`, so χ₄ is mean-zero periodic like η: the bland endpoint is *not* zeroless
  but the **structured** `G(s)=∫₁^∞ sin(πx/2)x^{-s}dx=Smom(π/2,s)` (an off-critical RvM string,
  σ→3/2 — cont_eta with π→π/2). So χ₄ is **η-type** — zeros migrate from the `G+½` ground
  string — but `L(s,χ₄)` is *primitive* (no prefactor·ζ), so there is **no companion**: the
  whole ground string maps 1:1 onto the *single* line σ=½ (density matches on the nose, q=4
  supplying η's doubling). Comb `L_comb_K=G+½−Σ_k D_k/(πk)→L` (`D_k` at the odd-quarter
  frequencies `(2k±½)π`), zeros born-onto-½ at `O(1/K)`. Warp of the **phase-½ carrier**
  `sin(πy)` (the α=½-survivor dual of `warp_eta`'s cos/α=1) lands on `2^s L(s,χ₄)=L(½,½,s)`
  (Lerch tie-in), load-bearing `+2^s`, single line (2^s never vanishes). Documented aside: warping
  the *genuine* χ₄ carrier `sin(πx/2)` at midpoints lands on a **mod-8** character
  `2^s(√2/2)L(s,χ₋₈)` (a period-q carrier's half-integer samples reveal period-2q). Reuses
  `harmonic_bridge` (Smom/Cmom, migrate) + `warp_bridge` (grid+moment machinery; sin-weighted).
- `hurwitz_lerch_zeros.py` — reproduces the closest **published** deform-and-track analogues
  (issue #21): **Garunkštis–Steuding 2007** (track `ζ(s,α)` zeros as the shift `α:1→½`; the
  "stable zero" = trajectory starts+ends on `σ=½`, vs the "unstable" ones landing on the `σ=0`
  companion comb `2πim/log2`) and **Garunkštis–Tamošiūnas 2017** (the Lerch `λ=α` diagonal
  `ζ→2^s L(s,χ₄)`, stays on `σ=½`). The organizing picture is the **Lerch family square**
  `L(λ,α,s)=lerchphi(e^{2πiλ},s,α)`: the four bridge objects (`ζ`, half-shift, `η`, `2^s L(χ₄)`)
  at the corners, and its three edges = the three published ancestors — the `λ=1` edge is G–S
  (`σ=0` companion), the `α=1` edge is Fornberg–Kölbig's polylog knob (`jonquiere_zeros.py`;
  `σ=1` companion), the diagonal is G–T (no companion). Contrasts the *parameter*-sweep (wanders
  on/off ½ between two zero-rich ends) with our K-knob (born onto ½ from a zeroless endpoint,
  reusing `harmonic_bridge.zeta_K`). Evaluator gotcha: `lerchphi` diverges at `z→1`, spliced with
  the Hurwitz zeta. Self-contained (only `harmonic_bridge` for the K-knob contrast panel).
- `epstein_zeros.py` — the **reverse-direction foil** (issue #22): reproduces **Travěnec–Šamaj
  2022** (dimension knob; Bétermin–Šamaj–Travěnec 2021 shape knob is the cited companion) on the
  hypercubic Epstein zeta `Z_d(s)=Σ'_{n∈ℤᵈ}|n|^{-2s}`. Zeros
  are born **off** the critical line `σ=d/4` and migrate **away** — the *opposite* arrow to our
  zeroless→born→**onto**-½, and the sharpest single foil for the directional taxonomy
  (`LITERATURE.md` §4.0/§4.5). Evaluator is the Terras/theta completed formula `Λ_d(s)=π^{-s}Γ(s)Z_d
  =∫₁^∞[t^{s-1}+t^{d/2-s-1}](ψ(t)ᵈ−1)dt+1/(s−d/2)−1/s` (`ψ`=Jacobi θ₃; `ψᵈ` for real `d` *is* the
  analytic dimension continuation), fast + entire-except-poles, validated vs the direct lattice sum
  and the closed forms `Z_1=2ζ(2s)`, `Z_2=4ζ(s)β(s)`. The mechanism is a **pitchfork at the critical
  dimension `d*=9.24555…`** (root of `g(d)=Λ_d(d/4)`, reproduced to 10 digits): for `d<d*` the lowest
  pair sits on `σ=d/4` at `d/4±it₁(d)` with `t₁→0`; at `d*` they collide at the real center; for `d>d*`
  they split into a real pair `d/4±δ(d)` migrating to the strip edges `{0,d/2}` (σ₋→0, σ₊→d/2 at d=25).
  An argument-principle count (0 real zeros for `d<d*`, 2 for `d>d*`) certifies the split independently.
  Contrast panel reuses `harmonic_bridge.zeta_K`+`migrate` (born-onto-½). Self-contained (only
  `harmonic_bridge` for the contrast).
- `geometric_bridge.py` — the **geometric-series miniature** (epic #29 / issue #30): the whole
  bridge on `Σ_{n≥0} s^n = 1/(1-s)` (continuous shadow `∫₀^∞ s^x dx = −1/ln s`), where every step is
  an **elementary closed form** and there are **no special functions and no zeros**. The comb
  `g_K = −1/ln s + ½ − Σ_{k≤K} 2 ln s/((ln s)²+4π²k²)` telescopes *exactly* to `1/(1-s)` via the
  `coth` partial fraction (= Bernoulli generating function); the rate is the arc's `O(1/K)` law with
  **`s ↦ ln s`** (so the height/cost coordinate is `|ln s|`); **no abscissa** (converges for `|s|>1`,
  Abel-summing `1+2+4+…=−1`, Grandi `½`); the warp *factorizes* into one scalar `J_K = ∫₀¹ s^{u−½+φ_K}du
  → 1` (`warp_K = (1/(1-s))·J_K`, no Hurwitz zeta); and the same harmonics run backwards (ascending
  comb → shadow). Classical throughout (Hardy 1949 *Divergent Series*; `coth`/Bernoulli) — an
  **illustrative consistency check, not a result**; lives in the preprint as **Appendix B**
  (`app:geometric`). Self-contained (no bridge cross-imports).
- `eta_two_component.py` — the η zeros as a crystal × GUE spectrum; the p=2-ghost.
  Reads `data/riemann_zeros.csv`.
- vendored helpers used only by `eta_two_component`: `gue_spacing` (→
  `maass_loader`), `spectral_rigidity`, `zero_form_factor`, `cone_log_prime`.
- `figstyle.py` — shared Matplotlib font bump (`figstyle.enlarge()`); each driver
  calls it before plotting so the embedded figures stay legible when scaled down in
  the preprint. The dense/long-title figures (`rate_law`, `eta_two_component`,
  `warp_coordinate`, `warp_eta`, `comb_vs_warp`, `jonquiere_zeros`, `geometric_bridge`)
  override `axes.titlesize`/`figure.titlesize` locally so titles don't overrun. Changing
  sizes here means re-running `python repro.py`.

`data/riemann_zeros.csv` — cached non-trivial zeros (the **only** runtime data
dependency). `tests/` — one test module per driver. `paper/` — the arXiv preprint
(#5): `main.tex`, `references.bib`, `make_arxiv.py`, built `main.pdf`; figures are
referenced from `bridge/figures/` via `\graphicspath` (not copied).

## Gotchas
- The flat `bridge/` layout is deliberate: bare imports (`import harmonic_bridge`,
  `from gue_spacing import …`) resolve because the modules are siblings, and
  `Path(__file__).resolve().parents[1]` → repo root keeps data/figure paths
  correct. Don't move modules into subfolders without revisiting both.
- Each cross-importing module keeps a short header that re-asserts its own
  directory on `sys.path` (robustness for direct runs); the parent project's
  stale bootstrap dirs (`maass/`, `prime_zero/`, …) were cleaned out under #4.
- Only `riemann_zeros.csv` is needed at runtime; the Maass eigenvalue CSV is not.
- `_private/` is a gitignored local stash for full-text papers (`_private/papers/`)
  and scratch material we keep organized but never commit (copyright + bulk).
  Reproductions cite the paper; they don't ship it. The FK1975 PDF lives there.

## Status & open work (issues in this repo)
The original arc (#1 epic; #2 literature; #3 extraction; #4 presentation &
reproducibility; #5 arXiv preprint; #6 outreach) and the appendix epics
(#16/#20/#21/#22 → Appendix A; #29/#30 → Appendix B) are all **closed**.
Still-load-bearing pointers into the closed issues:
- **#2 — Literature & novelty hardening.** Its two comments carry the BibTeX,
  the honest "what's known / what we add" statement, the Berry/LMQ deflator, and
  a ready-to-paste Rodgers–Tao contrast paragraph. **Read these before touching
  the paper's framing.**
- **#5 — arXiv preprint**: the draft lives in `paper/` — `main.tex` (~29-page
  methods/experimental-math write-up; Appendix A = the four beyond-ζ/η rhyming
  cases from #18, Appendix B = the geometric-series miniature from #30), a
  verified `references.bib` (36 entries), `make_arxiv.py` packager, and a built
  `main.pdf`. Build with `tectonic main.tex` from `paper/` (Tectonic + Poppler
  live in `~/.local/bin` / installed via winget). The 18 figures are pulled
  straight from `bridge/figures/` via `\graphicspath`, so no figure is duplicated.

Open — the two pre-external-review polish passes. PRs *reference* but never
close these ("Part of #N"); Brad does his own final pass after merge, then closes:
- **#26 — figure refinement pass.**
- **#32 — code pass**: tests passing, multi-platform + clear onboarding docs,
  efficiency/comment balance, Brad sign-off.

> Bare issue numbers like `#117`/`#132` that appear *inside the issue text* refer
> to findings in the originating research project, not to issues in this repo.
