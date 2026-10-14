"""Smoke test: every pipeline module imports without errors."""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

MODULES = [
    "config", "download_data", "inspect_data", "clean_data", "explore",
    "stat_tests", "make_charts", "model", "model_charts",
]


def test_modules_import():
    for name in MODULES:
        importlib.import_module(name)
