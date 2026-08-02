# explorations/ — post-manuscript experimental arcs

This directory holds **experimental work that is deliberately disjoint from the
frozen manuscript arc** (`bridge/` + `paper/` + `RESULTS.md`). Nothing here is
referenced by the paper, `repro.py`, or CI. Tests live in `explorations/tests/`
and are **excluded from the default suite** (`pytest.ini`'s `testpaths = tests`);
run them explicitly:

```
pytest explorations/tests            # fast (~4 s)
pytest explorations/tests -m slow    # + the migration regression (~1 min)
```

Drivers are local/manual only (issue #37's CI-economy note): run them directly,
e.g. `python explorations/character_bridge.py` (minutes; writes
`explorations/figures/character_bridge.png`).

If an arc here matures, the intended landing zone is a **second paper or a new
appendix**, never edits to the frozen manuscript.

---

## `character_bridge.py` — character combs: the K-knob on L(s, χ) (issue #37)

![Character combs](figures/character_bridge.png)

The discreteness-restoration knob extended to Dirichlet L-functions through the
(textbook) Hurwitz decomposition, `L(s,χ) = q^{-s} Σ_a χ(a) ζ(s, a/q)`, with the
Hurwitz components carrying the bridge's K-knob: the additive residue-class comb
(closed form via lower-limit incomplete-gamma moments; reduces **exactly** to
`harmonic_bridge.zeta_K` at α = 1) and the warp route (a literal linear
combination of `warp_alpha.warp_complete_alpha` calls). Everything below is
measured by the driver, which self-validates its identities (functional-equation
residuals ~1e-30, including the complex root number for χ₅) before plotting.

**Findings (2026-08-01, first pass):**

1. **The rate law survives with a q-dependent constant.** Component-wise,
   `ζ(s,α) − ζ_comb_K ~ (s/2π²)·α^{−s−1}/K` (the α-generalization of
   `rate_law.rate_comb`; α = 1 recovers it exactly), and the χ-weighted sum gives
   `L − L_comb_K ~ (sq/2π²)·[Σ_a χ(a) a^{−s−1}]/K` — q times a one-period section
   of `L(s+1, χ)`, so the constant grows ~linearly in the modulus. Note the
   constant belongs to the *comb organization*, not the L-function:
   `lfunction_bridge.py`'s carrier comb for the same `L(s,χ₄)` has the plain zeta
   constant `s/2π²`, the residue-class comb has `(4s/2π²)(1−3^{−s−1})`. Both
   measured.
2. **Order from disorder (issue #37, prediction 2).** Each component `ζ(s,a/5)`
   is Davenport–Heilbronn-scattered (generic phase in the Saias–Weingartner
   dichotomy, per `warp_alpha.py`); the χ₅-weighted combination is a primitive
   L-function whose zeros the migration lands on σ = 1/2 at O(1/K). Critical-line
   order materializes only in the arithmetically-weighted superposition. The
   complex character χ₅(2) = i is the repo's first genuinely complex-coefficient
   migration (no conjugate symmetry in t) — same story.
3. **Imprimitive split (prediction 1, structural part).** For
   χ₆ = χ₃ induced mod 6, `L(s,χ₆) = (1+2^{−s})·L(s,χ₃)`: the Euler-factor comb at
   σ = 0, t = (2m+1)π/ln 2 is verified, and the migration splits the ground
   string onto σ = 1/2 ∪ the σ = 0 comb — the η precedent (rigid comb ×
   critical string) at a genuine imprimitive L. The spectral-statistics
   measurement (crystal × GUE superposition instruments) is deferred to a
   sub-issue.
4. **The projector (prediction 3).** With the DC offset (the k = 0 Fourier mode
   of the shifted sawtooth) and the m = 0 completion cell counted as
   *discreteness data* — i.e. inside the knob — every warp component's K = 0
   endpoint is the same `∫₁^∞ x^{−s} dx = 1/(s−1)`, and orthogonality
   (`Σ_a χ(a) = 0`) kills the combination identically: **a non-principal
   L-function is all comb and no continuum**, born from the zero function. Under
   the DC-always-on convention (warp_alpha as built) the K = 0 endpoint is a
   small entire function instead — both endpoints are computed side by side; the
   right normalization for the small-K zero-birth story is deliberately left
   open (sub-issue).
5. **A measured surprise: the ground string's geometry is set by φ(q).** The
   comb-route K = 0 endpoint `E_χ` (an EM-weighted one-period section — the
   K-knob's kinship to the partial-sum literature) has its zeros **exactly on
   σ = 0** for the two-unit moduli q ∈ {3, 4, 6} (verified to ~48 digits: the
   zero condition collapses to `p^{−s} = (qs−(q−2))/(qs+(q−2))`, a Möbius map
   unimodular precisely on the imaginary axis, by the a ↔ q−a unit reflection).
   For φ(q) > 2 (q = 5) the ground zeros scatter. σ = 0 is also exactly where
   the imprimitive Euler combs and the midpoint warp's companion live.

**Honest framing.** Experimental mathematics; no RH/GRH claims. The Hurwitz
decomposition is classical. What the adversarial lit pass (below) did not find a
published analog for is precisely: deforming the *character-weighted
combination* through a *discretization* parameter and *tracking the zero
trajectories* of the combination onto the critical line.

### Lit-pass record (2026-08-01, adversarial search for issue #37)

> **What's known, what has no published analog (L-function fork).** The degree-1
> classification we lean on — every Selberg-class element of degree 1 is ζ(s) or
> a shifted primitive Dirichlet L-function L(s+iθ,χ) — is Kaczorowski–Perelli
> (Acta Math. 182, 1999), conjectured by Conrey–Ghosh (Duke Math. J. 72, 1993),
> who had proved d = 0 forces F = 1 and that no degree lies in (0,1); the
> (0,1)-nonexistence has still-earlier antecedents in Richert (1957) and Bochner
> (1958) for Dirichlet series with functional equations, and Soundararajan
> (2005) gives the simplest proof of the degree-1 case. (For the *extended*
> class S# the degree-1 answer is linear combinations of shifted L's — quote the
> clean statement only for the Selberg class proper.) On the construction
> itself: parameterized approximation families for Dirichlet L-functions do
> exist in print — partial sums of length N (Ledoan–Roy–Zaharescu 2014 for
> cyclotomic Dedekind zetas, i.e. products of all L(s,χ) mod q; Roy–Vatwani,
> Adv. Math. 2019; Dubon 2025 for β = L(s,χ₄)), truncated Euler products (Gonek
> 2012 for ζ, with L-function analogs only sketched), and — closest in spirit
> to our K-knob — the cyclic-graph discretization program of Friedli and
> Karlsson (Friedli 2016; Friedli–Karlsson 2017; Karlsson–Müller 2026
> *preprint*, which approximates L(s,χ) by finite spectral sums on ℤ/nℤ via
> Euler–Maclaurin expansions). But in every one of these, the zero results are
> of a different kind: zero-free regions, zero counts, density of real parts, or
> GRH-equivalences via asymptotic functional equations of the approximants.
> Nakamura's quadrilateral zeta (J. Number Theory 233, 2022) deforms a
> symmetrized Hurwitz-plus-periodic combination through the shift parameter a
> with zero counts on the line, and Garunkštis–Steuding (2007) track individual
> Hurwitz zeros through the shift — yet we found no published work that deforms
> a *character-weighted combination* of Hurwitz zetas through a *discretization*
> parameter and *tracks the zero trajectories* of the combination onto the
> critical line. As always with a negative claim, this is "not found under
> adversarial search" (arXiv, journals, citation-chasing; MathSciNet/zbMATH not
> directly searchable), not "provably absent" — and the 2026 Karlsson–Müller
> preprint shows this exact neighborhood is active, so any write-up must cite it
> and distinguish carefully.

Load-bearing BibTeX for this arc is kept in `explorations/references-37.bib`.

### Deferred (sub-issues of #37)

- Spectral statistics for imprimitive L: crystal × GUE superposition via the
  `eta_two_component.py` instruments (needs bulk L-zero computation).
- Normalized zero-birth from the identically-zero endpoint: the right K-family
  normalization (per-K L² norm? leading harmonic?), and whether birth-K is
  qualitatively different from ζ's.
- Rate-vs-conductor sweep: the measured `(sq/2π²)Σχ(a)a^{−s−1}` constant across
  many q; connects to the weil-positivity-lab conductor thread.
