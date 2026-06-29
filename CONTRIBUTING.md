# Contributing / how to run

A short guide to setting the repo up, reproducing the results, and the couple of
conventions worth knowing before you edit.

## Environment

Targets **Python 3.9** and keeps the code 3.9-compatible (`typing.List`/`Union`,
**not** PEP-604 `X | Y` runtime unions). Dependencies are pinned in
`requirements.txt` (numpy, scipy, mpmath, matplotlib, pytest).

Either toolchain works:

```bash
# with uv (fast):
uv venv --python 3.9 .venv
uv pip install -r requirements.txt

# or with stock venv + pip:
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The `.venv/` is per-machine and gitignored.

## Run the tests

```bash
pytest                 # 66 fast self-checks (~1.5 min); slow ones deselected by default
pytest -m slow         # the expensive high-precision regressions
```

`pytest.ini` sets `addopts = -m "not slow"`, so a bare `pytest` skips the slow
suite. The root `conftest.py` puts `bridge/` on `sys.path` so the flat
cross-imports resolve during collection.

## Reproduce the figures

```bash
python repro.py            # regenerate all nine figures (~12 min; harmonic/warp dominate)
python repro.py -v         # also stream each driver's validation output
python repro.py rate_law   # just one driver
```

Every driver also runs standalone and writes its own figure(s) to
`bridge/figures/`, e.g. `python bridge/cont_eta.py`. Each one self-validates
(asserts its identities to full precision) before plotting.

The nine canonical figures are committed (they are embedded in `RESULTS.md`).
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
`warp_alpha`, `rate_law`, `eta_two_component`) are the substance. `warp_coordinate`
and `warp_phase_compare` are small **illustrative companions** to the warp drivers —
they draw the warp map itself (linear → midpoint staircase) and the half-integer
vs. integer (`α = ½` vs `α = 1`) trade-off, and assert no new result. The remaining
modules (`gue_spacing`, `spectral_rigidity`, `zero_form_factor`, `cone_log_prime`,
`maass_loader`) are vendored spectral-statistics helpers used only by
`eta_two_component`. The only runtime data dependency is
`data/riemann_zeros.csv`.

## Tests for a new or changed driver

There is one test module per driver under `tests/`. If a driver writes a new
figure, add it to the `DRIVERS` table in `repro.py` (stem, figure filename(s),
runtime note) and, if it should be committed and embedded, add a `!`-exception for
it in `.gitignore` and a section in `RESULTS.md`.
