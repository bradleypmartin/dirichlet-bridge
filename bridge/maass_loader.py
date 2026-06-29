"""Loaders for the Maass-eigenvalue CSVs.

`load_maass_eigenvalues` reads the LMFDB level-1 PSL_2(Z) dump
(data/maass_eigenvalues_psl2z.csv). `load_gq_maass_eigenvalues` reads the
generalized Hecke-triangle-group G_q spectra cached by compute_gq_maass.py
(data/maass_eigenvalues_g{q}.csv, issue #44) -- a different, lighter schema.

Both return a list of dict rows with at least a float `tau`, the raw decimal
string `tau_str`, and an int `symmetry` (0 = even, 1 = odd), so that the
gue_spacing.py helpers (`split_taus_by_parity`, the unfolding / spacing
pipeline) work uniformly across PSL_2(Z) and G_q.

Example:
    >>> rows = load_maass_eigenvalues()
    >>> rows[0]["tau"]
    9.533695261353557
    >>> g5 = load_gq_maass_eigenvalues(5)
    >>> g5[0]["q"], g5[0]["symmetry"]
    (5, 1)
"""

from __future__ import annotations

import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_PATH = DATA_DIR / "maass_eigenvalues_psl2z.csv"


def load_maass_eigenvalues(path: Path | None = None) -> list[dict]:
    path = Path(path) if path is not None else DEFAULT_PATH
    rows: list[dict] = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                {
                    "n": int(row["n"]),
                    "tau": float(row["tau"]),
                    "tau_str": row["tau"],
                    "tau_error_str": row["tau_error"],
                    "tau_prec_bits": int(row["tau_prec_bits"]),
                    "symmetry": int(row["symmetry"]),
                    "fricke": int(row["fricke"]),
                    "lmfdb_label": row["lmfdb_label"],
                    "source": row["source"],
                }
            )
    return rows


def load_gq_maass_eigenvalues(q: int | None = None,
                              path: Path | None = None) -> list[dict]:
    """Load a cached G_q Maass spectrum (compute_gq_maass.py, issue #44).

    Pass either `q` (resolving data/maass_eigenvalues_g{q}.csv) or an explicit
    `path`. The G_q CSV schema is lighter than the LMFDB one -- columns
    n, tau, symmetry, parity, q, automorphy_residual, N_weyl, source -- so the
    full LMFDB loader would choke; this thin reader parses those fields. Keeps
    `tau_str` (raw decimal) for any later mpmath use, and the per-level
    `automorphy_residual` so callers can trim borderline levels."""
    if path is None:
        if q is None:
            raise ValueError("pass q or an explicit path")
        path = DATA_DIR / f"maass_eigenvalues_g{q}.csv"
    path = Path(path)
    rows: list[dict] = []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            rows.append(
                {
                    "n": int(row["n"]),
                    "tau": float(row["tau"]),
                    "tau_str": row["tau"],
                    "symmetry": int(row["symmetry"]),
                    "parity": row["parity"],
                    "q": int(row["q"]),
                    "automorphy_residual": float(row["automorphy_residual"]),
                    "N_weyl": float(row["N_weyl"]),
                    "source": row["source"],
                }
            )
    return rows
