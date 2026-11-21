"""Journal-style figures for the paper (Figures 1 to 8).

Usage: python code/figures.py <DATA_DIR> <RESULTS_DIR> <OUT_DIR>
DATA_DIR    folder with the NASA C-MAPSS text files
RESULTS_DIR folder with FD001.csv ... FD004.csv and results_tests.csv (the results/ folder)
OUT_DIR     where the figures are written, as PDF (vector) and PNG (600 dpi)
"""
import sys, zlib
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import lightgbm as lgb

DATA, RES, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "lines.linewidth": 0.9, "lines.markersize": 3.5, "axes.grid": False,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
W1, W2 = 3.35, 6.9                      # single and double column width, inches (85 and 175 mm)
FDS = ["FD001", "FD002", "FD003", "FD004"]
FULL = {"FD001": 100, "FD002": 260, "FD003": 100, "FD004": 249}
MODEL_STYLE = {   # monochrome, distinguishable in print
    "GBM": dict(color="0.0", marker="o", ls="-", mfc="0.0", label="Boosted trees"),
    "RF": dict(color="0.35", marker="^", ls="-", mfc="white", label="Random forest"),
    "Ridge": dict(color="0.55", marker="s", ls="--", mfc="white", label="Ridge regression"),
    "MeanLife": dict(color="0.0", marker="", ls=":", mfc="none", label="Mean life (no sensors)"),
}
GREYS4 = ["0.0", "0.3", "0.55", "0.75"]
MARK4 = ["o", "s", "^", "D"]


def panel(ax, letter, text=None):
    s = f"({letter})" + (f" {text}" if text else "")
    ax.set_title(s, loc="left", fontsize=8, pad=3)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=600)
    plt.close(fig)
    print("wrote", name)


def log_fleet_axis(ax, ns):
    ax.set_xscale("log")
    ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
    ax.minorticks_off()


d = pd.concat([pd.read_csv(RES / f"{fd}.csv") for fd in FDS], ignore_index=True)
tests = pd.read_csv(RES / "results_tests.csv")


# ---------------------------------------------------------------- experiment code (for Figures 1 and 2)
def load_experiment(fd):
    src = (HERE / "run_experiment.py").read_text()
    head = src.split("units_all = np.sort(train.unit.unique())")[0]
    ns = {"__name__": "exp"}
    argv = sys.argv
    sys.argv = ["run_experiment.py", DATA, fd, str(OUT / "_unused")]
    try:
        exec(compile(head, "run_experiment.py", "exec"), ns)
    finally:
        sys.argv = argv
    return ns


# ================================================================ Figure 1: features under two intervals
E = load_experiment("FD001")
tr = E["train"]
life = tr.groupby("unit").cycle.max()
u = int((life - 200).abs().idxmin())
g = tr[tr.unit == u]
x, y = g.cycle.values, g.s4.values
t = 150
fig, axes = plt.subplots(1, 2, figsize=(W2, 2.2), sharey=True)
for ax, k, letter in zip(axes, [1, 20], "ab"):
    keep = (x - 1) % k == 0
    ax.plot(x, y, color="0.8", lw=0.6, zorder=1)
    ax.plot(x[keep & (x <= t)], y[keep & (x <= t)], "o", ms=2.2 if k == 1 else 3.5, mfc="0.0" if k == 20 else "0.3",
            mec="none", zorder=3)
    base_m = keep & (x <= 20); lev_m = keep & (x > t - 30) & (x <= t); tr_m = keep & (x > t - 60) & (x <= t)
    if base_m.any():
        ax.hlines(y[base_m].mean(), 1, 20, color="0.0", lw=1.4, zorder=4)
    if lev_m.any():
        ax.hlines(y[lev_m].mean(), t - 29, t, color="0.0", lw=1.4, ls=(0, (1, 1)), zorder=4)
    if tr_m.sum() >= 2:
        b, c = np.polyfit(x[tr_m], y[tr_m], 1); xx = np.array([t - 59, t])
        ax.plot(xx, b * xx + c, color="0.0", lw=1.0, ls="--", zorder=4)
    ax.axvline(t, color="0.0", lw=0.5)
    ax.set_xlim(0, life[u] + 5)
    ax.set_xlabel("Flight cycle")
    n_rec = int(tr_m.sum())
    panel(ax, letter, f"every {'flight' if k == 1 else '20th flight'}; {n_rec} records in the trend window")
axes[0].set_ylabel("T50 (°R)")
handles = [Line2D([], [], color="0.0", lw=1.4, label="Baseline (first 20 cycles)"),
           Line2D([], [], color="0.0", lw=1.4, ls=(0, (1, 1)), label="Level (last 30 cycles)"),
           Line2D([], [], color="0.0", lw=1.0, ls="--", label="Trend (last 60 cycles)"),
           Line2D([], [], color="0.0", lw=0.5, label=f"Prediction time (cycle {t})")]
fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(w_pad=1.0)
save(fig, "fig1_features")


# ================================================================ Figure 2: predicted against true RUL, FD001
def fit_predict(E, n, sens, k, rep):
    units_all = np.sort(E["train"].unit.unique())
    FXtr, F1tr, age_tr, unit_tr, life_tr, st_tr = E["build"](E["train"], k, False)
    FXte, F1te, age_te, _, _, _ = E["build"](E["test"], k, True)
    lb = dict(zip(units_all, life_tr))
    rul_tr = np.minimum(np.array([lb[v] for v in unit_tr]) - age_tr, E["CAP"])
    seed = zlib.crc32(f"FD001-{n}-{rep}".encode())
    sel = units_all if n == len(units_all) else np.random.default_rng(seed).choice(units_all, n, replace=False)
    m = np.isin(unit_tr, sel)
    mu, sd = E["norm_stats"](st_tr, sel)
    sidx = [E["ALL_S"].index(s) for s in E["SENSOR_SETS"][sens]]
    Xtr = E["features"](FXtr[m], F1tr[m], mu, sd, sidx, age_tr[m].astype(float))
    Xte = E["features"](FXte, F1te, mu, sd, sidx, age_te.astype(float))
    mdl = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=1, verbose=-1,
                            random_state=seed % 2**31).fit(Xtr, rul_tr[m])
    return np.clip(mdl.predict(Xte), 0, E["CAP"])


truth = np.minimum(E["rul_test"], E["CAP"])
gb = d[(d.fd == "FD001") & (d.model == "GBM")]
r10 = gb[(gb.n == 10) & (gb.sensors == "S14") & (gb.k == 1)].set_index("rep").rmse
rep10 = int((r10 - r10.median()).abs().idxmin())          # the draw closest to the median
cases = [("a", "All data", 100, "S14", 1, 0), ("b", f"10 engines (median draw)", 10, "S14", 1, rep10),
         ("c", "Sensor set S4", 100, "S4", 1, 0), ("d", "Every 20th flight", 100, "S14", 20, 0)]
fig, axes = plt.subplots(1, 4, figsize=(W2, 1.95), sharex=True, sharey=True)
for ax, (letter, lab, n, s, k, rep) in zip(axes, cases):
    p = fit_predict(E, n, s, k, rep)
    e = p - truth
    ax.fill_between([0, 130], [10, 140], [130, 130], color="0.9", lw=0, zorder=0)
    ax.plot([0, 130], [0, 130], color="0.0", lw=0.6, zorder=1)
    ax.plot(truth, p, "o", ms=2.3, mfc="white", mec="0.0", mew=0.5, zorder=2)
    ax.set_xlim(0, 130); ax.set_ylim(0, 130); ax.set_aspect("equal")
    ax.set_xticks([0, 40, 80, 120]); ax.set_yticks([0, 40, 80, 120])
    ax.set_xlabel("True RUL (cycles)")
    panel(ax, letter, lab)
    ax.text(0.97, 0.05, f"RMSE {np.sqrt(np.mean(e ** 2)):.1f}\nlate >10: {np.mean(e > 10) * 100:.0f}%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    print("fig2", lab, "RMSE", round(float(np.sqrt(np.mean(e ** 2))), 2))
axes[0].set_ylabel("Predicted RUL (cycles)")
fig.tight_layout(w_pad=0.6)
save(fig, "fig2_predicted_vs_true")


# ================================================================ Figure 3: learning curves
def q(x):
    return pd.Series({"med": x.median(), "lo": x.quantile(0.1), "hi": x.quantile(0.9)})


lc = d[(d.k == 1) & d.sensors.isin(["S14", "none"])].groupby(["fd", "model", "n"]).rmse.apply(q).unstack()
fig, axes = plt.subplots(1, 4, figsize=(W2, 2.0), sharey=False)
for ax, fd, letter in zip(axes, FDS, "abcd"):
    ns = sorted(lc.loc[(fd, "GBM")].index)
    for m in ["MeanLife", "Ridge", "RF", "GBM"]:
        s = lc.loc[(fd, m)]; st = MODEL_STYLE[m]
        if m == "GBM":
            ax.fill_between(s.index, s.lo, s.hi, color="0.87", lw=0, zorder=0)
        ax.plot(s.index, s.med, color=st["color"], ls=st["ls"], marker=st["marker"], mfc=st["mfc"],
                mec=st["color"], mew=0.6, label=st["label"], zorder=2)
    log_fleet_axis(ax, ns)
    ax.set_ylim(0, None)
    ax.set_xlabel("Training engines")
    panel(ax, letter, fd)
axes[0].set_ylabel("Test RMSE (cycles)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(w_pad=0.8)
save(fig, "fig3_learning_curves")


# ================================================================ Figure 4: sensor sets and intervals across fleet sizes
g = d[d.model == "GBM"].groupby(["fd", "n", "sensors", "k"]).rmse.median()
fig, axes = plt.subplots(2, 4, figsize=(W2, 3.7), sharex="col")
for j, fd in enumerate(FDS):
    ns = sorted(d[d.fd == fd].n.unique())
    for i, (levels, key, labels) in enumerate([(["S14", "S7", "S4", "S2"], "sensors", ["S14", "S7", "S4", "S2"]),
                                               ([1, 5, 10, 20], "k", ["$k$ = 1", "$k$ = 5", "$k$ = 10", "$k$ = 20"])]):
        ax = axes[i, j]
        for lv, c, mk, lab in zip(levels, GREYS4, MARK4, labels):
            ys = [g[(fd, n, lv, 1)] if key == "sensors" else g[(fd, n, "S14", lv)] for n in ns]
            ax.plot(ns, ys, color=c, marker=mk, mfc="white" if c != "0.0" else c, mec=c, mew=0.6, label=lab)
        log_fleet_axis(ax, ns)
        ax.set_ylim(5, 45)
        if i == 1: ax.set_xlabel("Training engines")
        if j == 0: ax.set_ylabel("Median RMSE (cycles)")
        panel(ax, "abcdefgh"[i * 4 + j], fd)
    if j == 3:
        axes[0, j].legend(loc="upper right", title="Sensors, $k$ = 1", title_fontsize=7, handlelength=1.6)
        axes[1, j].legend(loc="upper right", title="Interval, S14", title_fontsize=7, handlelength=1.6)
fig.tight_layout(h_pad=0.6, w_pad=0.8)
save(fig, "fig4_sensors_and_interval")


# ================================================================ Figure 5: paired differences with Holm adjustment
spec = [("S4 minus S14 (GBM, k=1)", ("GBM", "S14", 1), ("GBM", "S4", 1), "(a) 4 sensors minus 14 sensors"),
        ("k10 minus k1 (GBM, S14)", ("GBM", "S14", 1), ("GBM", "S14", 10), "(b) Every 10th flight minus every flight"),
        ("GBM minus RF (S14, k=1)", ("RF", "S14", 1), ("GBM", "S14", 1), "(c) Boosted trees minus random forest")]
fig, axes = plt.subplots(1, 3, figsize=(W2, 2.2), sharex=True)
offs = dict(zip(FDS, [0.94, 0.98, 1.02, 1.06]))
for ax, (lab, A, B, title) in zip(axes, spec):
    ax.axhline(0, color="0.0", lw=0.5)
    for fd, c, mk in zip(FDS, GREYS4, MARK4):
        for n in sorted(d[d.fd == fd].n.unique()):
            if n == FULL[fd]: continue
            sel = lambda m, s, k: d[(d.fd == fd) & (d.n == n) & (d.model == m) & (d.sensors == s) & (d.k == k)].set_index("rep").rmse.sort_index()
            diff = sel(*B) - sel(*A)
            row = tests[(tests.contrast == lab) & (tests.fd == fd) & (tests.n == n)].iloc[0]
            xv = n * offs[fd]
            ax.plot([xv, xv], [diff.quantile(0.25), diff.quantile(0.75)], color=c, lw=0.7)
            ax.plot(xv, diff.median(), marker=mk, color=c, mec=c, mew=0.6,
                    mfc=c if row.p_holm < 0.05 else "white", ms=3.5)
    log_fleet_axis(ax, [2, 5, 10, 20, 50, 100])
    ax.set_xlabel("Training engines")
    ax.set_title(title, loc="left", fontsize=8, pad=3)
axes[0].set_ylabel("Paired difference in RMSE (cycles)")
h = [Line2D([], [], color=c, marker=mk, mfc=c, mec=c, ls="", label=fd) for fd, c, mk in zip(FDS, GREYS4, MARK4)]
h += [Line2D([], [], color="0.0", marker="o", mfc="0.0", ls="", label="Holm-adjusted $p$ < 0.05"),
      Line2D([], [], color="0.0", marker="o", mfc="white", ls="", label="not significant"),
      Line2D([], [], color="0.0", lw=0.7, label="interquartile range over draws")]
fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, columnspacing=1.2)
fig.tight_layout(w_pad=1.0)
save(fig, "fig5_paired_differences")


# ================================================================ Figure 6: variance shares
def shares(x, f):
    x = x.copy(); x["y"] = f(x.rmse); gm = x.y.mean(); sst = ((x.y - gm) ** 2).sum(); out = {}
    for fac in ["n", "k", "sensors"]:
        mu = x.groupby(fac).y.agg(["mean", "size"]); out[fac] = (mu["size"] * (mu["mean"] - gm) ** 2).sum()
    cm = x.groupby(["n", "sensors", "k"]).y.agg(["mean", "size"]); cells = (cm["size"] * (cm["mean"] - gm) ** 2).sum()
    out["inter"] = cells - sum(out.values()); out["draw"] = sst - cells
    return {k_: 100 * v / sst for k_, v in out.items()}


comp = [("n", "Fleet size", "0.15", None), ("k", "Recording interval", "0.45", None),
        ("sensors", "Sensor coverage", "0.75", None), ("inter", "Interactions", "white", "////"),
        ("draw", "Engine draw (within cells)", "white", None)]
fig, axes = plt.subplots(1, 2, figsize=(W2, 1.6), sharey=True)
for ax, (f, lab), letter in zip(axes, [(np.log, "log RMSE"), (lambda v: v, "RMSE")], "ab"):
    for i, fd in enumerate(FDS):
        sh = shares(d[(d.fd == fd) & (d.model == "GBM") & (d.n < FULL[fd])], f)
        left = 0
        for key, name, col, hatch in comp:
            ax.barh(i, sh[key], left=left, color=col, edgecolor="0.0", lw=0.5, hatch=hatch, height=0.62,
                    label=name if i == 0 else None)
            if sh[key] >= 9:
                ax.text(left + sh[key] / 2, i, f"{sh[key]:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if col in ("0.15", "0.45") else "0.0")
            left += sh[key]
    ax.set_yticks(range(4)); ax.set_yticklabels(FDS)
    ax.set_xlim(0, 100); ax.set_xlabel(f"Share of variance in {lab} (%)")
    ax.tick_params(axis="y", length=0)
    panel(ax, letter)
axes[0].invert_yaxis()                  # shared y axis: invert once, FD001 on top
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 1.02), handlelength=1.2, columnspacing=1.0)
fig.tight_layout(w_pad=1.0)
save(fig, "fig6_variance_shares")


# ================================================================ Figure 7: additivity check
rows = []
for fd in FDS:
    N = FULL[fd]; ref = g[(fd, N, "S14", 1)]
    for n in sorted(d[d.fd == fd].n.unique()):
        for s in ["S14", "S7", "S4", "S2"]:
            for k in [1, 5, 10, 20]:
                act = int(n != N) + int(s != "S14") + int(k != 1)
                if act < 2: continue
                pa = g[(fd, n, "S14", 1)] + g[(fd, N, s, 1)] + g[(fd, N, "S14", k)] - 2 * ref
                rows.append(dict(fd=fd, act=act, obs=g[(fd, n, s, k)], add=pa))
a = pd.DataFrame(rows); err = (a["add"] - a.obs).abs()
fig, ax = plt.subplots(figsize=(W1, W1 * 0.9))
lim = [8, 48]
ax.fill_between(lim, [lim[0] - 2.8, lim[1] - 2.8], [lim[0] + 2.8, lim[1] + 2.8], color="0.9", lw=0, zorder=0)
ax.plot(lim, lim, color="0.0", lw=0.6, zorder=1)
for act, mk, mfc in [(2, "o", "white"), (3, "^", "0.0")]:
    s = a[a.act == act]
    ax.plot(s["add"], s.obs, mk, ms=2.6, mfc=mfc, mec="0.0", mew=0.5, ls="",
            label=f"{act} constraints combined ($n$ = {len(s)})")
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
ax.set_xlabel("Additive prediction (cycles)"); ax.set_ylabel("Observed median RMSE (cycles)")
ax.legend(loc="upper left")
ax.text(0.97, 0.04, f"{len(a)} cells\nmedian |error| {err.median():.1f}\n90th percentile {err.quantile(0.9):.1f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
print("fig7 cells", len(a), "median", round(err.median(), 2), "p90", round(err.quantile(0.9), 2))
save(fig, "fig7_additivity")


# ================================================================ Figure 8: logging against retrofit, one change at a time
fig, axes = plt.subplots(1, 4, figsize=(W2, 1.9), sharey=True)
for ax, fd, letter in zip(axes, FDS, "abcd"):
    ns = sorted(d[d.fd == fd].n.unique())
    retro = [g[(fd, n, "S4", 20)] - g[(fd, n, "S14", 20)] for n in ns]
    logg = [g[(fd, n, "S4", 20)] - g[(fd, n, "S4", 1)] for n in ns]
    ax.plot(ns, logg, color="0.0", marker="o", mfc="0.0", label="Record every flight (4 sensors)")
    ax.plot(ns, retro, color="0.45", marker="s", mfc="white", mec="0.45", mew=0.6, ls="--",
            label="Add 10 sensors (every 20th flight)")
    ax.axhline(0, color="0.0", lw=0.4)
    log_fleet_axis(ax, ns)
    ax.set_ylim(-1, 16); ax.set_yticks([0, 4, 8, 12, 16])
    ax.set_xlabel("Training engines")
    panel(ax, letter, fd)
axes[0].set_ylabel("Reduction in RMSE (cycles)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(w_pad=0.6)
save(fig, "fig8_logging_vs_retrofit")


import shutil
shutil.rmtree(OUT / "_unused", ignore_errors=True)   # empty folder left by load_experiment
