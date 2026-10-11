"""Rebuild the whole project with one command.

Usage (from the project folder):
    python run_project.py

Steps:
    1. Download the raw file if needed and check its hash   (src/download_data.py)
    2. Inspect the raw data                                 (src/inspect_data.py)
    3. Clean the data and run the cleaning checks           (src/clean_data.py)

Every step rebuilds its output files, so the script is safe to rerun.
It stops with an error if a check fails.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from clean_data import CheckFailed, clean_data  # noqa: E402
from config import TABLES_DIR  # noqa: E402
from download_data import download_data  # noqa: E402
from inspect_data import inspect_data  # noqa: E402

STEPS = 3


def step(number, title):
    print(f"\n[{number}/{STEPS}] {title}")


def main():
    step(1, "Getting the raw data")
    download_data()

    step(2, "Inspecting the raw data")
    inspect_data(verbose=False)
    print(f"Saved results to {TABLES_DIR / 'inspection'}")

    step(3, "Cleaning")
    clean_data(verbose=False)

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except (CheckFailed, ValueError, FileNotFoundError) as error:
        print(f"\nSTOPPED: {error}")
        sys.exit(1)
