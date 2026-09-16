"""Run all EDA scripts in order. Usage: python3 main.py"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCRIPTS = [
    "overview.py",
    "distributions.py",
    "spatial_temporal.py",
    "correlations.py",
    "functional_readiness.py",
]

for name in SCRIPTS:
    path = HERE / name
    print(f"\n{'=' * 20} {name} {'=' * 20}")
    result = subprocess.run([sys.executable, str(path)])
    if result.returncode != 0:
        print(f"\n{name} failed, stopping.")
        sys.exit(result.returncode)

print("\nDone. Outputs written next to the scripts in", HERE)
