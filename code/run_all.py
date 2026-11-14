"""Rerun the whole study and rebuild the tables and figures.

Usage: python code/run_all.py <DATA_DIR>
DATA_DIR is the folder with train_FD00x.txt, test_FD00x.txt and RUL_FD00x.txt.
Outputs go to runs/ (raw chunks), merged/ (one CSV per subset) and out/ (tables, figures).
The run can be stopped and started again; finished chunks are skipped.
Takes about one hour on two CPU cores. Add --sensitivity to also rerun Section 4.7.
"""
import sys, glob, subprocess
from pathlib import Path
import pandas as pd

if len(sys.argv) < 2:
    sys.exit(__doc__)
DATA = sys.argv[1]
ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
draws = {"FD001": 20, "FD003": 20, "FD002": 10, "FD004": 10}

def run(*args):
    print(">", " ".join(map(str, args)), flush=True)
    subprocess.run([PY, *map(str, args)], check=True, cwd=ROOT)

for fd, n in draws.items():
    run("code/run_experiment.py", DATA, fd, f"runs/{fd}", n)
(ROOT / "merged").mkdir(exist_ok=True)
for fd in draws:
    files = sorted(glob.glob(str(ROOT / f"runs/{fd}/k*_n*.csv")))
    pd.concat(pd.read_csv(f) for f in files).to_csv(ROOT / f"merged/{fd}.csv", index=False)
run("code/analyse.py", "merged", "out")
run("code/stats_tests.py", "merged")

if "--sensitivity" in sys.argv:
    for fd, n in {"FD001": 10, "FD003": 10, "FD004": 5}.items():
        run("code/sensitivity.py", DATA, fd, f"runs_sensitivity/{fd}", n)
    run("code/analyse_sensitivity.py", "runs_sensitivity", "merged")
print("Done. Tables in out/tables.txt, figures in out/, tests in results_tests.csv")
