"""Summaries, tables and figures from the experiment CSVs.
Usage: python analyse.py <RESULTS_DIR> <OUT_DIR>"""
import sys, json, glob, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{RES}/FD00*.csv"))], ignore_index=True)
FDS = sorted(d.fd.unique())
SETS = ["S14", "S7", "S4", "S2"]
MODELS = ["Ridge", "RF", "GBM"]
full_n = d.groupby("fd").n.max().to_dict()
out = {}

def q(x):
    return pd.Series({"med": x.median(), "p10": x.quantile(.1), "p90": x.quantile(.9), "runs": len(x)})

# T3: favourable case (full fleet, S14, k=1)
t3 = d[(d.k == 1) & (d.sensors.isin(["S14", "none"]))]
t3 = t3[t3.apply(lambda r: r.n == full_n[r.fd], axis=1)]
out["T3_full"] = t3.pivot_table(index="fd", columns="model", values=["rmse", "score", "late"]).round(2)

# T4: learning curve, S14, k=1 (median [p10-p90])
lc = d[(d.k == 1) & (d.sensors.isin(["S14", "none"]))].groupby(["fd", "model", "n"]).rmse.apply(q).unstack()
out["T4_learning_curve"] = lc.round(1)

# T5: GBM median RMSE by fleet size x sensors (k=1) and fleet size x interval (S14)
g = d[d.model == "GBM"]
out["T5a_n_by_sensors_k1"] = g[g.k == 1].groupby(["fd", "n", "sensors"]).rmse.median().unstack()[SETS].round(1)
out["T5b_n_by_k_S14"] = g[g.sensors == "S14"].groupby(["fd", "n", "k"]).rmse.median().unstack().round(1)
out["T5c_sensors_by_k_full"] = g[g.apply(lambda r: r.n == full_n[r.fd], axis=1)].groupby(["fd", "sensors", "k"]).rmse.median().unstack().loc[(slice(None), SETS), :].round(1)

# best model per cell (median RMSE)
cell = d[d.model.isin(MODELS)].groupby(["fd", "n", "sensors", "k", "model"]).rmse.median().unstack()
best = cell.idxmin(axis=1)
out["T6_best_model_share_by_n"] = (best.groupby(["fd", "n"]).value_counts(normalize=True).unstack().fillna(0) * 100).round(0)
out["T6b_best_model_share_overall"] = (best.groupby("fd").value_counts(normalize=True).unstack().fillna(0) * 100).round(0)

# sensors vs no sensors: share of draws where learned model beats MeanLife, by n (S14,k=1) and worst case (S2,k=20)
ml = d[d.model == "MeanLife"].set_index(["fd", "k", "n", "rep"]).rmse
rows = []
for (fd, s, k, n, m), grp in d[d.model.isin(MODELS)].groupby(["fd", "sensors", "k", "n", "model"]):
    ref = ml.loc[[(fd, k, n, r) for r in grp.rep]].values
    rows.append(dict(fd=fd, sensors=s, k=k, n=n, model=m, win=np.mean(grp.rmse.values < ref) * 100,
                     gain=np.median(ref - grp.rmse.values)))
win = pd.DataFrame(rows)
out["T7_win_vs_meanlife_worstcase_S2_k20"] = win[(win.sensors == "S2") & (win.k == 20)].pivot_table(index=["fd", "n"], columns="model", values="win").round(0)
out["T7b_win_vs_meanlife_S14_k1"] = win[(win.sensors == "S14") & (win.k == 1)].pivot_table(index=["fd", "n"], columns="model", values="win").round(0)

# variance shares (log RMSE), balanced part: n below full fleet
rows = []
for fd in FDS:
    for m in MODELS:
        x = d[(d.fd == fd) & (d.model == m) & (d.n < full_n[fd])].copy()
        x["y"] = np.log(x.rmse)
        gm = x.y.mean(); sst = ((x.y - gm) ** 2).sum()
        ss = {}
        for f in ["n", "sensors", "k"]:
            mu = x.groupby(f).y.agg(["mean", "size"])
            ss[f] = (mu["size"] * (mu["mean"] - gm) ** 2).sum()
        cm = x.groupby(["n", "sensors", "k"]).y.agg(["mean", "size"])
        ss_cells = (cm["size"] * (cm["mean"] - gm) ** 2).sum()
        ss_int = ss_cells - sum(ss.values())
        ss_res = sst - ss_cells
        rows.append(dict(fd=fd, model=m, fleet=100 * ss["n"] / sst, sensors=100 * ss["sensors"] / sst,
                         interval=100 * ss["k"] / sst, interactions=100 * ss_int / sst, draws=100 * ss_res / sst))
out["T8_variance_shares_pct"] = pd.DataFrame(rows).set_index(["fd", "model"]).round(1)

# marginal cost of each constraint for GBM, relative to the favourable corner, in cycles and %
rows = []
for fd in FDS:
    gf = g[g.fd == fd]
    ref = gf[(gf.n == full_n[fd]) & (gf.sensors == "S14") & (gf.k == 1)].rmse.median()
    def med(n, s, k): return gf[(gf.n == n) & (gf.sensors == s) & (gf.k == k)].rmse.median()
    N = full_n[fd]
    for lab, val in [("ref", ref), ("n=50", med(50, "S14", 1)), ("n=20", med(20, "S14", 1)), ("n=10", med(10, "S14", 1)),
                     ("n=5", med(5, "S14", 1)), ("n=2", med(2, "S14", 1)),
                     ("S7", med(N, "S7", 1)), ("S4", med(N, "S4", 1)), ("S2", med(N, "S2", 1)),
                     ("k=5", med(N, "S14", 5)), ("k=10", med(N, "S14", 10)), ("k=20", med(N, "S14", 20)),
                     ("n=10,S4,k=10", med(10, "S4", 10)), ("n=5,S4,k=20", med(5, "S4", 20)), ("n=2,S2,k=20", med(2, "S2", 20))]:
        rows.append(dict(fd=fd, setting=lab, rmse=val, pct=100 * (val / ref - 1)))
cost = pd.DataFrame(rows)
out["T9_cost_rmse"] = cost.pivot(index="setting", columns="fd", values="rmse").round(1)
out["T9_cost_pct"] = cost.pivot(index="setting", columns="fd", values="pct").round(0)

# additivity check: observed combined penalty vs sum of single penalties (GBM), n=10,S4,k=10
rows = []
for fd in FDS:
    c = cost[cost.fd == fd].set_index("setting").rmse
    add = c["ref"] + (c["n=10"] - c["ref"]) + (c["S4"] - c["ref"]) + (c["k=10"] - c["ref"])
    rows.append(dict(fd=fd, observed=c["n=10,S4,k=10"], additive=add))
out["T10_additivity"] = pd.DataFrame(rows).set_index("fd").round(1)

# smallest fleet whose median RMSE is within 10% / 20% of the full-fleet RMSE, per sensors, k (GBM)
rows = []
for (fd, s, k), grp in g.groupby(["fd", "sensors", "k"]):
    m = grp.groupby("n").rmse.median()
    ref = m.loc[full_n[fd]]
    r = dict(fd=fd, sensors=s, k=k)
    for tol in (10, 20):
        okn = [n for n in m.index if m.loc[n] <= ref * (1 + tol / 100)]
        r[f"within{tol}"] = min(okn)
    rows.append(r)
out["T11_min_fleet"] = pd.DataFrame(rows).pivot_table(index=["fd", "sensors"], columns="k", values=["within10", "within20"]).loc[(slice(None), SETS), :]

# paired tests: Ridge vs GBM at each n (S14,k=1); S14 vs S4 (GBM,k=1); k=1 vs k=10 (GBM,S14)
def paired(a, b):
    a, b = a.sort_index(), b.sort_index()
    if len(a) < 6 or np.allclose(a.values, b.values): return np.nan, np.nan
    return float(np.median(b.values - a.values)), float(wilcoxon(a.values, b.values).pvalue)
rows = []
for fd in FDS:
    for n in sorted(d[d.fd == fd].n.unique()):
        if n == full_n[fd]: continue
        def s(model, sens, k):
            return d[(d.fd == fd) & (d.n == n) & (d.model == model) & (d.sensors == sens) & (d.k == k)].set_index("rep").rmse
        for lab, a, b in [("Ridge minus GBM", s("GBM", "S14", 1), s("Ridge", "S14", 1)),
                          ("RF minus GBM", s("GBM", "S14", 1), s("RF", "S14", 1)),
                          ("S4 minus S14 (GBM)", s("GBM", "S14", 1), s("GBM", "S4", 1)),
                          ("k=10 minus k=1 (GBM)", s("GBM", "S14", 1), s("GBM", "S14", 10))]:
            md, p = paired(a, b)
            rows.append(dict(fd=fd, n=n, contrast=lab, median_diff=md, p=p))
pt = pd.DataFrame(rows)
out["T12_paired_diff"] = pt.pivot_table(index=["contrast", "n"], columns="fd", values="median_diff").round(1)
out["T12_paired_p"] = pt.pivot_table(index=["contrast", "n"], columns="fd", values="p")

# late predictions and score for GBM at selected settings; fit time
out["T13_late_pct_GBM_S14_k1"] = (g[(g.sensors == "S14") & (g.k == 1)].groupby(["fd", "n"]).late.median().unstack() * 100).round(0)
out["T13b_late_pct_GBM_by_k_full"] = (g[(g.sensors == "S14")].groupby(["fd", "n", "k"]).late.median().unstack() * 100).round(0)
out["T14_fit_seconds_fullfleet_S14_k1"] = d[(d.sensors == "S14") & (d.k == 1)][d.apply(lambda r: r.n == full_n[r.fd], axis=1)].pivot_table(index="fd", columns="model", values="fit_s").round(2)
out["T15_spread_p90_minus_p10_GBM_S14_k1"] = g[(g.sensors == "S14") & (g.k == 1)].groupby(["fd", "n"]).rmse.apply(lambda x: x.quantile(.9) - x.quantile(.1)).unstack().round(1)
out["T16_runs"] = d.groupby("fd").size().to_frame("rows")

with open(f"{OUT}/tables.txt", "w") as f:
    for k_, v in out.items():
        f.write(f"\n===== {k_} =====\n{v.to_string()}\n")
pd.to_pickle(out, f"{OUT}/tables.pkl")

# ---------- figures ----------
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
COL = {"MeanLife": "#888888", "Ridge": "#1b9e77", "RF": "#d95f02", "GBM": "#7570b3"}
fig, ax = plt.subplots(1, len(FDS), figsize=(3.1 * len(FDS), 3.0), sharey=False)
ax = np.atleast_1d(ax)
for a, fd in zip(ax, FDS):
    for m in ["MeanLife", "Ridge", "RF", "GBM"]:
        s = lc.loc[(fd, m)]
        a.plot(s.index, s.med, marker="o", ms=3, color=COL[m], label="Mean life (no sensors)" if m == "MeanLife" else m)
        a.fill_between(s.index, s.p10, s.p90, color=COL[m], alpha=.15, lw=0)
    a.set_xscale("log"); a.set_xticks(sorted(lc.loc[(fd, "GBM")].index)); a.set_xticklabels(sorted(lc.loc[(fd, "GBM")].index))
    a.minorticks_off(); a.set_title(fd); a.set_xlabel("Training engines (N)")
ax[0].set_ylabel("Test RMSE (cycles)"); ax[0].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_learning_curves.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(2, len(FDS), figsize=(3.1 * len(FDS), 5.2))
ax = ax.reshape(2, -1)
for j, fd in enumerate(FDS):
    A = out["T5a_n_by_sensors_k1"].loc[fd]; B = out["T5b_n_by_k_S14"].loc[fd]
    for i, (M, xl) in enumerate([(A, "Sensor set (k = 1)"), (B, "Recording interval k (S14)")]):
        a = ax[i, j]; im = a.imshow(M.values, cmap="viridis_r", aspect="auto")
        a.set_xticks(range(M.shape[1])); a.set_xticklabels(M.columns); a.set_yticks(range(M.shape[0])); a.set_yticklabels(M.index)
        vmid = (np.nanmin(M.values) + np.nanmax(M.values)) / 2
        for r in range(M.shape[0]):
            for c in range(M.shape[1]):
                a.text(c, r, f"{M.values[r, c]:.1f}", ha="center", va="center", fontsize=7,
                       color="white" if M.values[r, c] > vmid else "black")
        a.set_xlabel(xl); a.spines[:].set_visible(False)
        if j == 0: a.set_ylabel("Training engines (N)")
        if i == 0: a.set_title(fd)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_heatmaps.png", dpi=200); plt.close(fig)

vs = out["T8_variance_shares_pct"].xs("GBM", level="model")
fig, a = plt.subplots(figsize=(5.2, 2.6))
left = np.zeros(len(vs)); cols = ["#33608c", "#d1802f", "#4f9d69", "#9a9a9a", "#d9d9d9"]
labs = ["Fleet size", "Sensor coverage", "Recording interval", "Interactions", "Which engines were drawn"]
for c, col, lab in zip(["fleet", "sensors", "interval", "interactions", "draws"], cols, labs):
    a.barh(vs.index, vs[c], left=left, color=col, label=lab, edgecolor="white")
    for y, (v, l) in enumerate(zip(vs[c], left)):
        if v >= 6: a.text(l + v / 2, y, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if col != "#d9d9d9" else "black")
    left += vs[c].values
a.invert_yaxis(); a.set_xlabel("Share of variance in log RMSE (%)"); a.set_xlim(0, 100)
a.legend(frameon=False, fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_variance_shares.png", dpi=200); plt.close(fig)
print(open(f"{OUT}/tables.txt").read())
