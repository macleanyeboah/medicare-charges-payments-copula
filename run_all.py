"""Reproduce every table and figure: python run_all.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import analysis  # noqa: E402
import figures  # noqa: E402

if __name__ == "__main__":
    analysis.main()
    figures.main()
