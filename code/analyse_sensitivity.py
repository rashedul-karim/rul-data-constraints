"""Hyperparameter sensitivity: compare four alternative settings with the paper's settings
on a reduced design (fleet 2, 10, 50, full; sensors S14, S4; interval 1, 10), same draws.
Usage: python code/analyse_sensitivity.py <SENSITIVITY_DIR> <MERGED_DIR>
SENSITIVITY_DIR holds either FD00x.csv files (as in results/sensitivity) or one
folder per subset with the chunk files written by sensitivity.py."""
import sys, glob, numpy as np, pandas as pd
SENS, MERGED = sys.argv[1], sys.argv[2]
out = []
for fd in ["FD001", "FD002", "FD003", "FD004"]:
    import os
    if os.path.exists(f"{SENS}/{fd}.csv"):
        s = pd.read_csv(f"{SENS}/{fd}.csv")
    else:
        fs = glob.glob(f"{SENS}/{fd}/k*_n*.csv")
        if len(fs) < 8: continue          # subset not run, or run not finished
        s = pd.concat(pd.read_csv(f) for f in fs)
    s = s[s.model != "MeanLife"]
    p = pd.read_csv(f"{MERGED}/{fd}.csv"); p = p[p.model.isin(["RF", "GBM"])]
    key = s[["n", "k", "rep", "sensors"]].drop_duplicates()
    p = p.merge(key, on=["n", "k", "rep", "sensors"]); p["model"] = p.model + "_paper"
    d = pd.concat([s, p]); N = d.n.max()
    cells = d.groupby(["n", "k", "sensors"]).rep.nunique()
    print(fd, "draws per cell:", cells.groupby("n").max().to_dict(), "rows", len(d))
    for m, g in d.groupby("model"):
        med = g.groupby(["n", "sensors", "k"]).rmse.median()
        ref = med[(N, "S14", 1)]
        x = g[g.n < N].copy(); x["y"] = np.log(x.rmse); gm = x.y.mean(); sst = ((x.y - gm) ** 2).sum()
        sh = {}
        for f in ["n", "sensors", "k"]:
            mu = x.groupby(f).y.agg(["mean", "size"]); sh[f] = 100 * (mu["size"] * (mu["mean"] - gm) ** 2).sum() / sst
        out.append(dict(fd=fd, model=m, full=ref, n10=100 * (med[(10, "S14", 1)] / ref - 1), n2=100 * (med[(2, "S14", 1)] / ref - 1),
                        S4=100 * (med[(N, "S4", 1)] / ref - 1), k10=100 * (med[(N, "S14", 10)] / ref - 1),
                        sh_fleet=sh["n"], sh_k=sh["k"], sh_sens=sh["sensors"],
                        add=ref + (med[(10, "S14", 1)] - ref) + (med[(N, "S4", 1)] - ref) + (med[(N, "S14", 10)] - ref), obs=med[(10, "S4", 10)]))
o = pd.DataFrame(out).set_index(["fd", "model"]).round(1)
o["order_ok"] = (o.sh_fleet > o.sh_k) & (o.sh_k > o.sh_sens)
print(o.to_string())
o.to_csv(f"{SENS}/sensitivity_summary.csv")
