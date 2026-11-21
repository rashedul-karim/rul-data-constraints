"""Paired Wilcoxon tests with Holm adjustment within each family of 22 cells,
sign tests across cells, and the one-change-at-a-time logging vs retrofit comparison.
Usage: python stats_tests.py <MERGED_DIR>"""
import sys, glob, numpy as np, pandas as pd
from scipy.stats import wilcoxon, binomtest
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{sys.argv[1]}/FD00*.csv"))])
full = d.groupby("fd").n.max().to_dict()
def family(label, A, B):
    rows = []
    for fd in full:
        for n in sorted(d[d.fd == fd].n.unique()):
            if n == full[fd]: continue
            s = lambda m, se, k: d[(d.fd == fd) & (d.n == n) & (d.model == m) & (d.sensors == se) & (d.k == k)].set_index("rep").rmse.sort_index()
            a, b = s(*A), s(*B)
            rows.append(dict(contrast=label, fd=fd, n=n, median_diff=float(np.median(b - a)), p=wilcoxon(a, b).pvalue))
    R = pd.DataFrame(rows).sort_values("p").reset_index(drop=True); m = len(R)
    R["p_holm"] = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * R.p))
    pos = int((R.median_diff > 0).sum())
    print(f"{label}: {m} cells, Holm-significant {int((R.p_holm < 0.05).sum())}, positive {pos}, sign test p {binomtest(pos, m).pvalue:.2g}")
    return R
out = pd.concat([family("S4 minus S14 (GBM, k=1)", ("GBM", "S14", 1), ("GBM", "S4", 1)),
                 family("k10 minus k1 (GBM, S14)", ("GBM", "S14", 1), ("GBM", "S14", 10)),
                 family("GBM minus RF (S14, k=1)", ("RF", "S14", 1), ("GBM", "S14", 1))])
out.sort_values(["contrast", "fd", "n"]).to_csv("results_tests.csv", index=False)
g = d[d.model == "GBM"].groupby(["fd", "n", "sensors", "k"]).rmse.median()
print("\nOne change at a time, from S4 with k=20 (GBM median RMSE):")
for fd, N in full.items():
    for n in sorted(d[d.fd == fd].n.unique()):
        base = g[(fd, n, "S4", 20)]
        print(fd, n, "start", round(base, 1), "retrofit gain", round(base - g[(fd, n, "S14", 20)], 1), "logging gain", round(base - g[(fd, n, "S4", 1)], 1))
