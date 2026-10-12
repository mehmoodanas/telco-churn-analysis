"""Rebuild the whole project with one command.

Usage (from the project folder):
    python run_project.py

Steps:
    1. Download the raw file if needed and check its hash   (src/download_data.py)
    2. Inspect the raw data                                 (src/inspect_data.py)
    3. Clean the data and run the cleaning checks           (src/clean_data.py)
    4. Build the exploratory analysis tables and check them (src/explore.py)
    5. Run the statistical tests                            (src/stat_tests.py)
    6. Make the charts                                      (src/make_charts.py)

Every step rebuilds its output files, so the script is safe to rerun.
It stops with an error if a check fails.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from clean_data import CheckFailed, clean_data  # noqa: E402
from config import FIGURES_DIR, TABLES_DIR  # noqa: E402
from download_data import download_data  # noqa: E402
from explore import run_exploration  # noqa: E402
from inspect_data import inspect_data  # noqa: E402
from make_charts import make_charts  # noqa: E402
from stat_tests import run_tests  # noqa: E402

STEPS = 6


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

    step(4, "Exploratory analysis")
    run_exploration(verbose=False)
    print(f"Saved tables to {TABLES_DIR / 'analysis'}")

    step(5, "Statistical tests")
    tests = run_tests()
    print(f"{len(tests)} tests, {int(tests['significant_after_holm'].sum())} significant after Holm correction")

    step(6, "Charts")
    charts = make_charts()
    print(f"Saved {len(charts)} charts to {FIGURES_DIR}")

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except (CheckFailed, ValueError, FileNotFoundError) as error:
        print(f"\nSTOPPED: {error}")
        sys.exit(1)
