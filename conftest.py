"""Put the flat ``bridge/`` source directory on ``sys.path`` so the package's
bare cross-imports (``import cont_eta``, ``from gue_spacing import ...``)
resolve under pytest, regardless of the working directory.

Direct runs (``python bridge/cont_eta.py``) don't need this — Python already
puts the script's own directory on ``sys.path`` — but pytest collects from the
repo root, so we add ``bridge/`` here. Mirrors the path-shim convention of the
parent project this work was extracted from.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "bridge"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
