# Literature & prior art — the discrete ↔ continuous bridge

A standing map of what is **known** (cite, do not claim), what is the **closest
prior art** (foreground and distinguish), and what is **genuinely un-anticipated**
(the contribution), for the dirichlet-bridge program.

This document consolidates three efforts:

1. two adversarial **novelty deep-research passes** (issue #2; full findings + BibTeX
   in its two comments) — focused on ζ and η;
2. the **closest-ancestor analysis** of Fornberg–Kölbig (1975), surfaced via the
   originating advisor (issue #12);
3. the **broadening pass** (issue #12 fork 1) — does this approach generalize to
   Dirichlet series *beyond* ζ/η? *(§4: ~50-year deform-and-track lineage exists, but
   the zeroless→born→onto-½ combination is un-anticipated beyond ζ/η.)*

> **Honest framing (decided up front).** This is a **methods / experimental-
> mathematics** study — a computational / expository **synthesis**, *not* a new
> theorem. Both adversarial passes concluded the work is *"novel as a package."*
> The sections below are written to keep us on exactly that footing.

### How this document is maintained

This is a **living, persistently-tracked** artifact, not a one-off pass. Any work on
the project that meets one of these **gating criteria** should **update this file**
(prior art, closest ancestors, and the novelty ledger in §5):

1. it touches **Dirichlet-series approximations or transitions** — any knob,
   smoothing, truncation, interpolation, or deformation;
2. it touches **ζ / η / L-function** zeros or values;
3. it makes — or even gestures toward — a **novelty claim**.

Newly surfaced prior art (e.g. an ancestor a collaborator flags) is itself a trigger
to revise. Keep this in sync with the two adversarial passes (issue #2) and the
preprint's framing (issue #5).

---

## 0. The program in one paragraph

We interpolate from the bland continuous integral
`∫₁^∞ x^{−s} dx = 1/(s−1)` — which keeps ζ's pole but has **no** non-trivial zeros,
no functional equation, no arithmetic — to the full Dirichlet sum `ζ(s)`. The knob
is *how many Fourier harmonics `K` of the discreteness kernel* (the Euler–Maclaurin
sawtooth ≡ Abel–Plana comb) we keep. As `K` grows the non-trivial zeros are **born**
and **migrate** onto `Re s = ½`. A companion **nonlinear warp** of the integrand
lands deliberately on `ζ(s,½) = (2^s−1)ζ(s)` and, on the η integrand, reproduces the
`σ=½ ∪ σ=1` split by a second route.

---

## 1. The honest claim, itemised

**Classical — cite, do not claim:**

- the half-shift `(2^s−1)ζ(s)=ζ(s,½)`, its `σ=0` companion at `t=2πk/ln2`, and the
  migration of Hurwitz-type zeros onto `σ=½` as a parameter varies — `Gonek1981`,
  `GarunkstisSteuding2007`; counting law `GarunkstisTamosiunas2013`;
- the `α∈{0,½,1}` "clean lines only" `P·L` dichotomy — `SaiasWeingartner2009`;
  generic off-line `ζ(s,α)` zeros — `DavenportHeilbronn1936`, `Cassels1961`;
- the `σ=1` eta comb at `t=2πk/ln2` — `Sondow2003`, tabulated by
  `BeliakovMatiyasevich2015`, elementary via `DLMF §25.2.3`;
- the closest "knob" cousins use partial-sum **length `N`**, none from a zeroless
  endpoint — `GonekLedoan2010`, `GonekMontgomery2013`, `PlattTrudgian2016`; zeros of
  combinations of Euler products for `σ>1` — `BookerThorne2014`, `Righetti2016`.

**The deflator to foreground (the `p=2`-ghost reading):** `Berry1988`, proved by
`LugarMilinovichQuesadaHerrera2022` — prime-power log-frequencies (smallest `=log 2`)
already sit in the zeta-zero **number variance** in a regime where **GUE fails**. So
the appearance of `log 2` is *not* ours; only the comb⊎GUE **superposition** model,
**single-prime isolation**, and the **eta tie-in** survive.

**Genuinely un-anticipated (the contribution):**

- the **`K`-knob** with a *zeroless* `1/(s−1)` endpoint;
- the **EM-sawtooth ≡ Abel–Plana** zero-birth identity;
- the deliberately-engineered **warp** landing on `ζ(s,½)`;
- the unified **`O(1/K)` rate law** + the **Gram's-law vertical-migration coupling**.

---

## 2. The direct ancestor — the polylogarithm deformation (Fornberg–Kölbig 1975)

> **`FornbergKolbig1975`** — B. Fornberg & K. S. Kölbig, *Complex Zeros of the
> Jonquière or Polylogarithm Function*, **Math. Comp. 29 (130), 1975, 582–599**,
> [doi:10.2307/2005579](https://doi.org/10.2307/2005579). (CERN; CDC 7600.)

This is the **closest intellectual ancestor of the whole program** and must be
foregrounded. It studies the zeros, in the `s`-plane, of the **polylogarithm /
Jonquière function**

```
F(x, s) = Σ_{k≥1} x^k / k^s        (|x| < 1),
```

treating the argument `x ∈ (−1, 1)` as a **deformation knob** and numerically
**tracking the zero trajectories** as `x` varies — exactly the "deform ζ and watch
the zeros" move, done in 1975. The endpoints are ours: `F(1,s)=ζ(s)`,
`F(−1,s)=−η(s)`.

> **Reproduced in code (issue #15).** `bridge/jonquiere_zeros.py` now reproduces FK1975
> on a laptop in ~40 s: the two zero-trajectory classes, the transient σ=½ brush +
> spiral about `s₁` (Figs. 6–7), the σ=1 log-2 comb (Eq. 23), the trivial-zero
> trajectories → `−2N` (§6), the argument-principle counts `Z(0.1)=27`/`Z(−0.1)=26`
> (Eq. 25), and the `α±` counting constants (Eqs. 39–40) — the code-side foil the
> preprint's contrast figure draws on (#18). The full-text PDF lives in
> `_private/papers/` (gitignored).

### 2.1 What FK1975 already contains (four correspondences)

| FK1975 | bridge analogue |
|---|---|
| Zero trajectories approach `σ=½` as `x→1` (Figs. 2, 6, 7) | the zero-migration map |
| `x→−1` endpoint zeros at `1 + 2πi m/log 2` (Eq. 23, the η side) | the **σ=1 log-2 comb** (`warp_eta.py`, `harmonic_bridge.py`) |
| Real-axis trajectories → `−2m` (§6) | the **trivial zeros** (`trivial_zeros.py`, #10) |
| Asymptote counting law vs Riemann–von Mangoldt, constants in `log 2`, `γ` (§7) | the rate/counting laws (`rate_law.py`), loosely |

It even carries a ζ↔η self-similar functional equation `F(x,s)=2^{1−s}F(x²,s)−F(−x,s)`
(Eq. 28) echoing our `2^{1−s}` structure.

The σ=1 comb (Eq. 23) is notable: it is an appearance of the `1+2πik/log2` points in
a **deform-and-track** setting **28 years before** the `Sondow2003` we currently cite
for that comb. (Stated precisely: the *points* are elementary zeros of `1−2^{1−s}`;
Sondow's cited result is about η specifically. FK is the earlier appearance *in this
deformation context*, not a displacement of Sondow's statement.)

### 2.2 The headline distinction — transient brush vs. genuine convergence

In FK's deformation the moving zeros **do not converge onto `σ=½`**. Per Fornberg
(email, 2026) and FK §5 (Figs. 6–7): the zeros are **born far out in the left
half-plane**, migrate rightward, and are **all eventually absorbed into the emerging
pole at `s=1`** — some *bypassing* ζ's zeros, others *approaching extremely closely
before being "ejected" and spiralling off*, also ending at `s=1`. (Mechanism: the
`x→1` expansion, FK Eq. 26, carries a branch term `Γ(1−s)(−log x)^{s−1}`, Eq. 27,
whose phase winds as `log(−log x)→−∞`; and `F(x,s)` is *entire* for `|x|<1`, so the
`s=1` pole only emerges at `x=1`, swallowing the migrating zeros.)

*(Nuance, lest a reviewer catch it: at the **other** endpoint `x→−1` some trajectories
**do** coincide with ζ's nontrivial zeros — but only because `F(−1,s)=−η(s)` already
carries them. That is the endpoint **having** the zeros, not a dynamical convergence
onto the line; the genuine dynamical approach, `x→+1`, is the transient/spiral one.)*

The bridge's `K`-harmonic knob does the thing FK's `x`-knob **set out toward but
could not reach**: the zeros **land on `σ=½` and stay**, converging cleanly at
`O(1/K)`. So the honest one-liner is:

> *Fornberg–Kölbig (1975) is the direct ancestor — same deform-ζ-and-track-the-zeros
> program — but on their Abel-damping knob the zeros only graze `σ=½` before being
> absorbed into the pole; the bridge replaces that knob with a harmonic truncation of
> the Euler–Maclaurin ≡ Abel–Plana kernel, off a genuinely zeroless `1/(s−1)`
> endpoint, under which the zeros actually **converge** onto the critical line.*

This is a **stronger** novelty statement than "we use a different knob" — it is "we
use a knob on which the convergence the classic deformation chased actually happens."

### 2.3 The five things that stay distinct

1. **Different knob** — FK's `x` is convergence-factor / Abel-mean damping (the
   polylog *is* the Abel mean of ζ); ours is truncation of the EM ≡ Abel–Plana
   discreteness kernel, with the EM≡AP identity FK has nothing of.
2. **Cleaner endpoint** — FK's `x→0` limit degenerates (zeros pushed to `σ=−∞`); ours
   is the named, finite, genuinely **zeroless** integral `1/(s−1)`.
3. **Clean convergence vs. transient near-miss** — §2.2.
4. **No warp** — the engineered warp onto `ζ(s,½)`, the general-α dichotomy, the σ=0
   companion are absent in FK.
5. **No spectral reading** — the two-component crystal×GUE / p=2-ghost is absent.

A planned code reproduction of FK1975's picture (issue **#15**) will exhibit the
§2.2 contrast directly.

---

## 3. Prior art for ζ and η specifically (the two adversarial passes, #2)

Both passes: 0 claims killed across ~25 verified; net **"novel as a package."**

### 3.1 The `K`-knob / zero-birth
Partially anticipated **direction**, novel **knob**. The cousins all use partial-sum
length `N` and none start from a zeroless endpoint: `GonekLedoan2010`,
`GonekMontgomery2013`, `PlattTrudgian2016`, `BeliakovMatiyasevich2015`. The
opposite-direction neighbours *resum the entire* EM remainder (no truncation, no
zeros): `CostinGaroufalidis2008`; Tao's 2010 EM exposition treats term-count as
accuracy only. **The `K`-harmonic-of-the-kernel knob + zeroless `1/(s−1)` endpoint +
EM-sawtooth ≡ Abel–Plana identity: no prior art found.**

### 3.2 The warp → `ζ(s,½)`
End structure already in the literature (cite): half-shift target, σ=0 companion at
`t=2πk/ln2`, `a=½`'s distinguished status, migration onto `σ=½` —
`GarunkstisSteuding2007`, `Gonek1981`. Only the **engineered nonlinear warp
mechanism** that lands on it is residual.

### 3.3 The general-α dichotomy
**Already in the literature — cite, don't claim:** it *is* `SaiasWeingartner2009`'s
`P·L` dichotomy plus classical Hurwitz off-line zeros (`DavenportHeilbronn1936`,
`Cassels1961`). Value distribution of `ζ(s;α)` for algebraic-irrational α:
`SourmelidisSteuding2022`.

### 3.4 The unified `O(1/K)` rate law + Gram coupling
**No direct prior art** (the softest negative — adjacent literature is large). Only a
*contrast*: the Taylor sections of ξ converge **super-exponentially**, not
algebraically (`JenkinsMcLaughlin2016`) — a different approximant family. The
**Gram's-law / approach-side coupling found zero prior art** and is the sharpest
un-anticipated bit.

### 3.5 The η two-component crystal × GUE / `p=2` ghost
The **comb component is classical** (`Sondow2003`, `BeliakovMatiyasevich2015`,
`DLMF §25.2.3`). The **spectral reading is not found**, but is the **most exposed**
claim — foreground-and-distinguish the deflator (`Berry1988`,
`LugarMilinovichQuesadaHerrera2022`; §1). Surviving contribution: the comb⊎GUE
**superposition** statistics, **single-prime isolation** (`p=2`), and the **eta
tie-in** of the `σ=1` comb to the `p=2` lattice via the explicit formula. Related
"quasicrystal" readings supply only the *crystal* half as a single spectrum:
`Lagarias2006`, `MadisonMadisonKozyrev2023`, `Shaughnessy2024` (construction only),
`MoranLedezma2023` (value-distribution, not zero-statistics).

### 3.6 Two contrasts the paper draws explicitly
- **de Bruijn–Newman heat flow** (`RodgersTao2020`, `Polymath2019DBN`): transports a
  *fixed-cardinality* infinite zero set onto the real axis — *real-ification*, not
  *creation*. Our ζ side is creation-from-zeroless; even the nearer η side is a
  density-conserving **split onto two vertical lines**, not a heat flow.
- **Taylor-section / resummation routes**: `JenkinsMcLaughlin2016` (super-exponential
  Taylor sections of ξ), `CostinGaroufalidis2008` (Borel/resurgent resummation of the
  *entire* EM remainder) — both run opposite to our truncate-the-kernel `O(1/K)`.

A **deform-and-track homotopy** numerical study of the Davenport–Heilbronn
counterexample's off-line zeros: `BalanzarioSanchezOrtiz2007` (the previous closest
deform-and-track citation, now superseded by `FornbergKolbig1975` and
`GarunkstisSteuding2007`; see the full beyond-ζ/η taxonomy in **§4**, where it is the
*reverse-direction* mirror of the bridge).

---

## 4. Broadening beyond ζ/η — issue #12, fork 1

*Source: a fact-checked deep-research pass (6 angles, 22 sources, 25 claims verified,
**24 confirmed / 1 refuted**), 2026-06-30. Confidence notes in §4.9.*

### 4.0 Verdict

The "deform a Dirichlet series by a parameter and watch its zeros move" idea is a
**well-established ~50-year lineage that reaches well beyond ζ/η** — but **no** prior
work combines the bridge's three signature ingredients:

> **(i)** a genuinely **zeroless endpoint** (`∫₁^∞ x^{−s}dx = 1/(s−1)`, pole only);
> **(ii)** zeros **born** from that zeroless start; **(iii)** migration **onto**
> `Re s = ½`.

Every ancestor found either deforms between two endpoints that **both already carry
infinitely many zeros** (no birth), or **births/pushes zeros off and away** from the
line. So the **zeroless-endpoint → born-onto-½** framing appears **un-anticipated for
L-function families other than ζ/η**. The two closest explicit ancestors to
foreground are `FornbergKolbig1975` (argument knob, §2) and `GarunkstisSteuding2007`
(Hurwitz α-trajectory + "stable zero" critical-line framing, §4.2).

**The organizing insight — a directional taxonomy of the lineage:**

| Direction of the zeros | Family / knob | Closest sources |
|---|---|---|
| zeroless → **born → onto ½** *(the bridge)* | ζ, η — harmonic count `K` | *this work* |
| zero-rich → zero-rich; trajectories **wander on/off ½** | polylog (arg `x`), Hurwitz (`α`), Lerch (`λ,α`) | `FornbergKolbig1975`, `GarunkstisSteuding2007`, `GarunkstisTamosiunas2017` |
| **born off ½ → migrate away** | Davenport–Heilbronn (homotopy `τ`), Epstein (dim `d`, shape `Δ`) | `BalanzarioSanchezOrtiz2007`, `TravenecSamaj2022` |

Our motion is the only one whose arrow points **onto** the line from **nothing**.

### 4.1 Polylogarithm / Jonquière (argument knob) — the ancestor + a documented negative
- **`FornbergKolbig1975`** — the ancestor; full treatment in §2.
- **Campbell (2012)**, *Polylogarithm Approaches to Riemann Zeta Zeroes*, arXiv:1212.2246
  — **false positive / documented negative.** In the polylog title-space but uses
  *static* visible-point-vector product identities: for the nontrivial zeros it imposes
  the critical line **by hand** (`s=½+iT`, `t=½−iT` so `s+t=1`); for the trivial zeros
  it tries the argument knob `Li_s(y)` as `y→1` and explicitly judges it *"not very
  illuminating, as the limit of the polylogarithm approaches infinity."* No zero birth
  or migration; cites Lewin/Borwein, **not** Fornberg–Kölbig. Useful as the explicit
  *thing-that-looks-like-the-bridge-but-isn't*, and as a recorded negative result on the
  argument→1 route for trivial zeros. **UNRELATED.**
- **Gap:** no continuation of FK's *numerical argument-knob trajectory* program was found
  past 2012 (see §4.8).

### 4.2 Hurwitz `ζ(s,α)` — the closest published analogue
- **`GarunkstisSteuding2007`** — R. Garunkštis & J. Steuding, *On the distribution of
  zeros of the Hurwitz zeta-function*, **Math. Comp. 76 (257), 2007, 323–337**, doi
  `10.1090/S0025-5718-06-01882-5`. Presents **computer plots of `ζ(s,α)` zero
  trajectories** as the shift `α` is deformed continuously from `α=1` (`=ζ`) to `α=½`
  (the half-shift), and defines a zero as **"stable" iff its trajectory starts and ends
  on `Re s=½`.** This is the **single closest published analogue** of "zeros migrate
  to/from ½ as a parameter varies." *Already in `references.bib`* (cited for the
  half-shift target) — **re-frame it as the closest deform-and-track analogue**, not
  only the half-shift source. Caveat: both endpoints already carry infinitely many zeros
  (no birth), and the `α=½` endpoint is itself flagged as established ζ/η prior art.
  **CLOSEST COUSIN (after FK).**

### 4.3 Lerch `L(λ,α,s)` — the deformation family that literally contains the bridge's objects
- The two-parameter Lerch family `L(λ,α,s)=Σ_{m≥0} e^{2πiλm}(m+α)^{−s}` **contains our
  objects as special parameter values**: `L(1,1,s)=ζ`, `L(1,½,s)=(2^s−1)ζ` (half-shift),
  `L(½,1,s)=(1−2^{1−s})ζ=η`, and `L(½,½,s)=2^s L(s,χ₄)` (**a Dirichlet L-function**).
  (All four verified to machine precision.)
- **`GarunkstisTamosiunas2017`** — R. Garunkštis & R. Tamošiūnas, *Symmetry of zeros of
  the Lerch zeta-function for equal parameters*, **Lith. Math. J. 57 (4), 2017,
  433–440**, doi `10.1007/s10986-017-9373-0` (arXiv:1901.10790; companion arXiv:1902.03064).
  For **equal parameters `λ=α`** the nontrivial zeros lie *extremely close to* / *almost
  symmetric about* `σ=½`; gives **parametric trajectory sweeps `½≤λ≤1`** with the `(1,1)`
  endpoint = ζ. **COUSIN** — a static structural-symmetry special point, largely
  numerical ("almost restored," not a proven RH analogue), not a birth-from-zeroless.
- **`GarunkstisPanavas2022`** — *The zeros of the Lerch zeta-function are uniformly
  distributed modulo one*, Ukr. Math. J. 73 (9), doi `10.1007/s11253-022-01999-2` — a
  zero-*distribution* result, not deform-and-track. Earlier: Garunkštis–Steuding, *On
  the zero distributions of Lerch zeta-functions*, Analysis (Munich) 22 (2002), 1–12.

### 4.4 Davenport–Heilbronn — a homotopy in the **reverse** direction
- **`BalanzarioSanchezOrtiz2007`** — *Zeros of the Davenport–Heilbronn Counterexample*,
  **Math. Comp. 76 (260), 2007, 2045–2049**, doi `10.1090/S0025-5718-07-01999-0`.
  Convex homotopy `f_τ=(1−τ)f₀+τf₁` from `f₀=(1+√5·5^{−s})ζ(s)` (zeros **all on** `½`)
  into the Davenport–Heilbronn series `f₁` (zeros **off** `½`), with a continuity-of-zeros
  theorem tracking each starting zero across. *Already cited.* **The mirror image of the
  bridge:** zeros start *on* the line and migrate *off* as the periodic structure is
  switched on. **COUSIN (reverse direction).**

### 4.5 Epstein / lattice zeta — a geometric knob; zeros born **off** and pushed **away**
- **`TravenecSamaj2022`** — I. Travěnec & L. Šamaj, *Generation of off-critical zeros
  for hypercubic Epstein zeta-functions*, **Appl. Math. Comput. 413, 2022** (arXiv:1909.07112).
  Uses **spatial dimension `d` as a continuous knob** (analytically continued); off-critical
  zeros are **born** at edge points, and past **`d>9.24555`** a conjugate pair of real
  off-critical zeros emerges and migrates to the strip boundaries `0,d` as `d→∞`.
- **`BeterminSamajTravenec2021`** — *Interplay between critical and off-critical zeros of
  2D Epstein zeta functions*, arXiv:2110.09368 — rectangular-lattice `ζ⁽²⁾(s,Δ)` with the
  **shape `Δ`** as the deform knob (+ follow-up arXiv:2307.06002).
- **NEW family** and the closest *literal zero-birth-via-a-continuous-knob* — but the
  arrow points **off and away** from `½`, and Epstein zeta lacks an Euler product.
  **COUSIN (reverse direction).**

### 4.6 Partial-sum (guise c) generalizations beyond η
- **Dirichlet λ and β** — *A note on the density of zeros of partial sums of the
  Dirichlet lambda, beta and eta functions*, **Eur. J. Math., 2025**, doi
  `10.1007/s40879-025-00821-0`. By Bohr's equivalence theorem, partial-sum zeros of the
  Dirichlet λ and β functions are dense in the **same** critical interval as ζ's —
  pushing the Gonek–Ledoan / η partial-sum program to two more series. **COUSIN (guise c).**
- **Dedekind ζ_K (cyclotomic)** — *Zeros of partial sums of the Dedekind zeta function of
  a cyclotomic field* (J. Number Theory; arXiv:1307.xxxx): truncated ideal Dirichlet series
  `ζ_{K,X}(s)=Σ_{‖a‖≤X}‖a‖^{−s}` — guise (c) for a **higher-degree** L-function.
  **COUSIN (guise c).**
- **Miyagawa (2017)**, arXiv:1704.01850 — a new **approximate functional equation** for
  the Hurwitz and Lerch zeta-functions; guise-(c) *machinery* for L-functions beyond ζ/η,
  but contains **no** zeros / deformation / migration. Tooling, not deform-and-track.

### 4.7 The Tao continuous model (the conceptual framework, no zeros)
- **`Tao2017Continuous`** — *Continuous approximations to arithmetic functions* (2017):
  replaces a discrete arithmetic function on ℕ with a locally-integrable `f` on `[1,∞)`,
  with `D[f](s)=∫₁^∞ f(t)t^{−s}dt` the analogue of `Σ f(n)n^{−s}`; maps pole structure to
  summatory asymptotics. **The general-arithmetic-function sibling of our bland
  endpoint** — but it does **not** track zeros, and **no published follow-up connecting
  it to zeros was found.** Conceptual ancestor of the *zeroless-∫-endpoint* idea, not of
  the zero dynamics.
- Tao's 2010 Euler–Maclaurin / Bernoulli / ζ / analytic-continuation post treats term
  count as **accuracy**, not a zero knob (consistent with the #2 pass).

### 4.8 Gaps / open trails (what the search did *not* find)
1. **No deform-and-track for genuine higher-rank objects** — Dirichlet `L(s,χ)` *directly*
   (only via Lerch specialization), Dedekind ζ (only partial-sum), **automorphic / GL(n)**,
   and the **Selberg class** were not reached; the trail stops at Hurwitz/Lerch/DH/Epstein.
2. **No prior work matches all three bridge ingredients** (zeroless endpoint + born + onto
   ½) for any family beyond ζ/η.
3. **No continuation of FK's argument-knob numerical program past Campbell 2012** — is
   there a modern study that genuinely tracks `Li_s(x)` argument-knob trajectories landing
   on `Re s=½`? (Open.)

### 4.9 Source confidence
24 of 25 verified claims confirmed (the one refutation was a stronger negative
characterization of arXiv:1902.03064, dropped on a 1–2 vote). Two AMS *Math. Comp.* PDFs
(`GarunkstisSteuding2007`, `BalanzarioSanchezOrtiz2007`) sat behind a Cloudflare
challenge, so their load-bearing quotes rest on publisher abstracts (CrossRef/ADS) plus
the authors' own preprints rather than a second full-text read — verify the exact
`f_τ=(1−τ)f₀+τf₁` line against the article body before quoting it in the preprint. The
Lerch parameter-control claims carried some 2–1 votes but with high verifier evidence and
unanimous sibling claims. (Full per-claim verdicts: task `w5x6bcw1x` output.)

---

## 5. Consolidated novelty ledger

| Claim | Status | Nearest prior art | What survives |
|---|---|---|---|
| `K`-knob + zeroless endpoint + EM≡Abel–Plana | **survives** | length-`N` cousins; `CostinGaroufalidis2008` (opposite) | the whole knob |
| Engineered warp → `ζ(s,½)` | survives (mechanism) | `GarunkstisSteuding2007`, `Gonek1981` (target) | the warp construction |
| General-α dichotomy | **cite, don't claim** | `SaiasWeingartner2009` | — |
| `O(1/K)` rate + Gram coupling | survives | `JenkinsMcLaughlin2016` (contrast) | rate law; Gram coupling sharpest |
| η crystal×GUE / `p=2` ghost | survives (exposed) | `Berry1988`, `LugarMilinovichQuesadaHerrera2022` | superposition + single-prime + eta tie-in |
| Deform-ζ-and-track-zeros (overall) | **ancestor exists** | **`FornbergKolbig1975`** | clean `O(1/K)` *convergence* (vs FK's transient brush) |
| Generalise beyond ζ/η | **survives** (§4) | `GarunkstisSteuding2007` (Hurwitz α-track); `TravenecSamaj2022` (Epstein, reverse) | zeroless→born→**onto** ½ unmatched for any family ≠ ζ/η |

---

## 6. New BibTeX to add (beyond `paper/references.bib`)

```bibtex
@article{FornbergKolbig1975,
  author    = {Fornberg, Bengt and K{\"o}lbig, K. S.},
  title     = {Complex Zeros of the {Jonqui\`ere} or Polylogarithm Function},
  journal   = {Mathematics of Computation},
  volume    = {29},
  number    = {130},
  pages     = {582--599},
  year      = {1975},
  doi       = {10.2307/2005579},
  publisher = {American Mathematical Society}
}

@misc{Tao2017Continuous,
  author       = {Tao, Terence},
  title        = {Continuous approximations to arithmetic functions},
  year         = {2017},
  howpublished = {Blog post, \url{https://terrytao.wordpress.com/2017/11/09/continuous-approximations-to-arithmetic-functions/}},
  note         = {Accessed 2026}
}

@article{GarunkstisTamosiunas2017,
  author  = {Garunk{\v{s}}tis, Ram{\=u}nas and Tamo{\v{s}}i{\=u}nas, Raivydas},
  title   = {Symmetry of Zeros of the {Lerch} Zeta-Function for Equal Parameters},
  journal = {Lithuanian Mathematical Journal},
  volume  = {57},
  number  = {4},
  pages   = {433--440},
  year    = {2017},
  doi     = {10.1007/s10986-017-9373-0},
  note    = {arXiv:1901.10790; companion arXiv:1902.03064}
}

@article{TravenecSamaj2022,
  author  = {Trav{\v{e}}nec, Igor and {\v{S}}amaj, Ladislav},
  title   = {Generation of Off-Critical Zeros for Hypercubic {Epstein} Zeta-Functions},
  journal = {Applied Mathematics and Computation},
  volume  = {413},
  pages   = {126611},
  year    = {2022},
  note    = {arXiv:1909.07112}
}

@article{BeterminSamajTravenec2021,
  author  = {B{\'e}termin, Laurent and {\v{S}}amaj, Ladislav and Trav{\v{e}}nec, Igor},
  title   = {Interplay Between Critical and Off-Critical Zeros of Two-Dimensional {Epstein} Zeta Functions},
  year    = {2021},
  note    = {arXiv:2110.09368; follow-up arXiv:2307.06002}
}

@article{GarunkstisPanavas2022,
  author  = {Garunk{\v{s}}tis, Ram{\=u}nas and Panavas, Tadas},
  title   = {The Zeros of the {Lerch} Zeta-Function Are Uniformly Distributed Modulo One},
  journal = {Ukrainian Mathematical Journal},
  volume  = {73},
  number  = {9},
  year    = {2022},
  doi     = {10.1007/s11253-022-01999-2}
}
```

> Already in `paper/references.bib` but **due for re-framing** as deform-and-track
> ancestors (not only their currently-cited roles): `GarunkstisSteuding2007` (the
> closest Hurwitz α-trajectory analogue) and `BalanzarioSanchezOrtiz2007` (the
> reverse-direction homotopy). Guise-(c) cousins beyond η worth a one-line mention if
> the partial-sum thread is expanded: the Dirichlet λ/β partial-sum density paper
> (Eur. J. Math. 2025, doi 10.1007/s40879-025-00821-0) and the cyclotomic Dedekind
> partial-sum study. Campbell (2012, arXiv:1212.2246) is a recorded **negative** — cite
> only if explicitly distinguishing the bridge from static polylog-identity work.

---

*Existing verified BibTeX (30 entries) lives in `paper/references.bib`; the full
per-claim verdicts and the ready-to-paste Rodgers–Tao contrast paragraph are in the
two comments on issue #2.*
