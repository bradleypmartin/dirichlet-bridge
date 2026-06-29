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
Targets **Python 3.9** — keep code 3.9-compatible (`typing.Union`, *not* PEP 604
`X | Y` runtime unions). Per-machine venv (gitignored):

- Setup: `python -m venv .venv` then `pip install -r requirements.txt`
  (numpy, scipy, mpmath, matplotlib, pytest — pinned).
- Fast suite: `pytest` (66 tests, ~60 s; the slow high-precision checks are
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
- `harmonic_bridge.py` — additive comb (EM sawtooth ≡ Abel–Plana, term-by-term);
  the zero-migration map (ζ born onto σ=½; η ground string splits σ=½ ∪ σ=1).
- `warp_bridge.py` — the nonlinear midpoint-warp → `ζ(s,½)=(2^s−1)ζ`; the σ=0
  companion; the measured σ=½ + σ=0 bijection.
- `warp_alpha.py` — the general-phase `n+α` warp → `ζ(s,α)`; the
  Saias–Weingartner `P·L` dichotomy.
- `rate_law.py` — the unified `O(1/K)` rate / birth-`K` law; the Gram's-law
  vertical-axis duality.
- `eta_two_component.py` — the η zeros as a crystal × GUE spectrum; the p=2-ghost.
  Reads `data/riemann_zeros.csv`.
- vendored helpers used only by `eta_two_component`: `gue_spacing` (→
  `maass_loader`), `spectral_rigidity`, `zero_form_factor`, `cone_log_prime`.
- `figstyle.py` — shared Matplotlib font bump (`figstyle.enlarge()`); each driver
  calls it before plotting so the embedded figures stay legible when scaled down in
  the preprint. The dense/long-title figures (`rate_law`, `eta_two_component`,
  `warp_coordinate`) override `axes.titlesize`/`figure.titlesize` locally so titles
  don't overrun. Changing sizes here means re-running `python repro.py`.

`data/riemann_zeros.csv` — cached non-trivial zeros (the **only** runtime data
dependency). `tests/` — one test module per driver. `paper/` — the arXiv preprint
(#5): `main.tex`, `references.bib`, `make_arxiv.py`, built `main.pdf`; figures are
referenced from `bridge/figures/` via `\graphicspath` (not copied).

## Gotchas
- The flat `bridge/` layout is deliberate: bare imports (`import harmonic_bridge`,
  `from gue_spacing import …`) resolve because the modules are siblings, and
  `Path(__file__).resolve().parents[1]` → repo root keeps data/figure paths
  correct. Don't move modules into subfolders without revisiting both.
- Modules were copied **byte-identical** from the originating project, so each
  cross-importing one still carries a self-bootstrapping `sys.path` header that
  adds sibling dir names that don't exist here (`maass/`, `prime_zero/`, …) —
  harmless; cleanup is tracked in **#4**.
- Only `riemann_zeros.csv` is needed at runtime; the Maass eigenvalue CSV is not.

## Status & open work (issues in this repo)
- **#1 — Epic** (the hub; full scope + dependency graph).
- **#2 — Literature & novelty hardening — DONE.** Its two comments carry the
  BibTeX, the honest "what's known / what we add" statement, the Berry/LMQ
  deflator, and a ready-to-paste Rodgers–Tao contrast paragraph. **Read these
  before drafting the paper.**
- **#3 — Repo extraction — DONE** (this repo).
- **#4 — Presentation & reproducibility** (open): `RESULTS.md` walkthrough,
  one-command figure regen, and the self-bootstrap-header cleanup above.
- **#5 — arXiv preprint** (open): **draft complete in `paper/`** — `main.tex`
  (14-page methods/experimental-math write-up), a verified `references.bib` (30
  entries), `make_arxiv.py` packager, and a built `main.pdf`. Build with
  `tectonic main.tex` from `paper/` (Tectonic + Poppler live in `~/.local/bin` /
  installed via winget). Framing/citations come from #2; the 9 figures are pulled
  straight from `bridge/figures/` via `\graphicspath`, so no figure is duplicated.
- **#6 — Senior-collaborator outreach** (open): one-pager + draft email; depends
  on #2/#4/#5.

> Bare issue numbers like `#117`/`#132` that appear *inside the issue text* refer
> to findings in the originating research project, not to issues in this repo.
