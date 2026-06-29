# Preprint — build instructions

The arXiv-ready LaTeX source for the *discrete ↔ continuous bridge* preprint
(repo issue #5). A computational / experimental-mathematics write-up of the arc
walked figure-by-figure in [`../RESULTS.md`](../RESULTS.md).

## Files

| File | What it is |
|---|---|
| `main.tex` | The paper (single source file). |
| `references.bib` | Annotated bibliography (verified; see issue #2 for provenance). |
| `main.pdf` | The built PDF (committed deliverable). |
| `make_arxiv.py` | Packages a self-contained arXiv upload into `arxiv/` (gitignored). |

The canonical figures live in [`../bridge/figures/`](../bridge/figures/) and are
committed there. `main.tex` finds them via `\graphicspath{{figures/}{../bridge/figures/}}`,
so a **local build needs nothing copied** — it reads them straight from
`../bridge/figures/`.

## Local build (Tectonic)

[Tectonic](https://tectonic-typesetting.github.io/) is a single self-contained
binary that fetches whatever TeX packages it needs on first run — no system TeX
install required. It runs the full LaTeX + BibTeX passes automatically.

```bash
# from this paper/ directory:
tectonic main.tex          # -> main.pdf
```

Install Tectonic (Windows, no admin): download the `x86_64-pc-windows-msvc` zip
from the [releases page](https://github.com/tectonic-typesetting/tectonic/releases),
unzip, and put `tectonic.exe` on your PATH (e.g. `~/.local/bin`). On macOS/Linux:
`brew install tectonic` / `cargo install tectonic` / your package manager.

A traditional TeX Live / MiKTeX toolchain works too:

```bash
latexmk -pdf main.tex      # runs pdflatex + bibtex as needed
```

## arXiv upload

arXiv builds from a flat, self-contained source tree and cannot follow the `../`
path to `../bridge/figures/`. `make_arxiv.py` assembles a clean upload directory
(`arxiv/`) with `main.tex`, `references.bib`, and a local `figures/` copy, then
prints the `tar` command:

```bash
python make_arxiv.py       # -> arxiv/  (and a ready-to-upload arxiv.tar.gz)
```

Suggested arXiv categories: **math.NT** (primary), cross-listed **math.HO**
(history and overview / experimental mathematics), reflecting the
methods/expository framing.
