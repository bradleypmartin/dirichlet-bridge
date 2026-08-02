"""Put explorations/ and the flat bridge/ source dir on sys.path for pytest.

Mirrors the root conftest.py convention: the explorations modules cross-import the
bridge sources as bare siblings (harmonic_bridge, warp_alpha, rate_law). These tests
are OUTSIDE the default suite (pytest.ini's `testpaths = tests` excludes them; CI
never runs them -- issue #37's CI-economy note): run with `pytest explorations/tests`.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent / "bridge")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
