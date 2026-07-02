# Contributing / how to run

A short guide to setting the repo up, reproducing the results, and the couple of
conventions worth knowing before you edit.

## Environment

Targets **Python 3.9** and keeps the code 3.9-compatible (`typing.List`/`Union`,
**not** PEP-604 `X | Y` runtime unions). Any of **3.9–3.12** works; the pins in
`requirements.txt` (numpy, scipy, mpmath, matplotlib, pytest) ship wheels for
those versions only, so on 3.13+ the install fails at numpy/scipy.

Either toolchain works:

```bash
# with uv (fast):
uv venv --python 3.9 .venv
uv pip install -r requirements.txt
. .venv/bin/activate          # Windows: .venv\Scripts\activate

# or with stock venv + pip:
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Don't mix the two command sets: a uv-created venv contains no `pip`, so later
installs into it must also go through `uv pip ...`. The `.venv/` is per-machine
and gitignored. Windows note: a fresh PowerShell may refuse `Activate.ps1`
under its default execution policy — either run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or skip activation
and call `.venv\Scripts\python.exe` / `.venv\Scripts\pytest.exe` directly
(everything below works unactivated; `repro.py` launches its drivers via
`sys.executable`).

## Run the tests

```bash
pytest                 # 142 fast self-checks (~2 min); slow ones deselected by default
pytest -m slow         # the expensive high-precision regressions
```

`pytest.ini` sets `addopts = -m "not slow"`, so a bare `pytest` skips the slow
suite. That default also silently deselects a slow test you name explicitly —
to run one, override the marker filter: `pytest -m slow tests/test_x.py::test_y`
(or `-m ""` for everything). The root `conftest.py` puts `bridge/` on
`sys.path` so the flat cross-imports resolve during collection.

## Reproduce the figures

```bash
python repro.py            # regenerate all eighteen figures (~28 min; the migration drivers dominate)
python repro.py -v         # also stream each driver's validation output
python repro.py rate_law   # just one driver
```

Every driver also runs standalone and writes its own figure(s) to
`bridge/figures/`, e.g. `python bridge/cont_eta.py`. Each one self-validates
(asserts its identities to full precision) before plotting.

The eighteen canonical figures are committed (they are embedded in `RESULTS.md`
and pulled into the preprint via `\graphicspath`).
Regenerating them is reproducible; note that a re-run may show a PNG diff that is
**only metadata** (matplotlib stamps a creation time) even when the plot is
pixel-identical.

## Layout conventions (read before moving files)

`bridge/` is a deliberately **flat** source directory. Every module is a sibling,
so the bare cross-imports (`import cont_eta`, `from gue_spacing import …`) resolve
two ways: a direct `python bridge/<module>.py` run gets `bridge/` on `sys.path`
automatically (Python adds the script's own directory), and pytest gets it from
the root `conftest.py`. Each module re-asserts its own directory on `sys.path` in a
short header for robustness. **Don't move modules into subpackages** without
revisiting both mechanisms and the `Path(__file__).resolve().parents[1]` → repo
root anchoring used for data/figure paths.

The six **arc drivers** (`cont_eta`, `harmonic_bridge`, `warp_bridge`,
`warp_alpha`, `rate_law`, `eta_two_component`) are the substance. Six
**companions** extend them: `comb_vs_warp` (the two routes at the level of the
integrand), `warp_coordinate` (the warp map itself, linear → midpoint staircase),
`warp_phase_compare` (the `α = ½` vs `α = 1` trade-off), `warp_eta` (the warp on
the η integrand), `stability` (no abscissa; height is the cost), and
`trivial_zeros` (the trivial zeros born on the negative axis). Five **appendix
drivers** carry the preprint's appendices: `jonquiere_zeros`, `lfunction_bridge`,
`hurwitz_lerch_zeros`, `epstein_zeros` (Appendix A, beyond ζ/η), and
`geometric_bridge` (Appendix B, the elementary miniature). `figstyle` is the
shared Matplotlib font bump every driver calls before plotting. The remaining
modules (`gue_spacing`, `spectral_rigidity`, `zero_form_factor`, `cone_log_prime`,
`maass_loader`) are vendored spectral-statistics helpers used only by
`eta_two_component`. The only runtime data dependency is
`data/riemann_zeros.csv`.

## Tests for a new or changed driver

There is one test module per driver under `tests/`. If a driver writes a new
figure, add it to the `DRIVERS` table in `repro.py` (stem, figure filename(s),
runtime note) and, if it should be committed and embedded, add a `!`-exception for
it in `.gitignore` and a section in `RESULTS.md`.
