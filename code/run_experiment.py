"""
Factorial experiment: fleet size x sensor coverage x recording interval
on NASA C-MAPSS (FD001-FD004), with low-compute models.

Writes one CSV per recording interval and fleet size into OUT_DIR and skips files that
already exist, so an interrupted run can be restarted.

Usage: python code/run_experiment.py <DATA_DIR> <FD00x> <OUT_DIR> [draws] [time_budget_s]
"""
import sys, time, zlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

DATA, FD, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 20
BUDGET = float(sys.argv[5]) if len(sys.argv) > 5 else 1e9
import os
os.makedirs(OUT, exist_ok=True)
CAP = 125
LEVEL_W, SLOPE_W, BASE_W = 30, 60, 20
FLEETS = [2, 5, 10, 20, 50, 100]
INTERVALS = [1, 5, 10, 20]
# sensor index (1-based, Saxena et al. 2008): 2 T24, 3 T30, 4 T50, 7 P30, 8 Nf, 9 Nc,
# 11 Ps30, 12 phi, 13 NRf, 14 NRc, 15 BPR, 17 htBleed, 20 W31, 21 W32
SENSOR_SETS = {
    "S14": [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21],
    "S7": [2, 3, 4, 7, 8, 9, 12],
    "S4": [4, 8, 9, 12],
    "S2": [4, 9],
}
ALL_S = SENSOR_SETS["S14"]
cols = ["unit", "cycle", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]


def load(name):
    d = pd.read_csv(f"{DATA}/{name}_{FD}.txt", sep=r"\s+", header=None, names=cols)
    return d


train, test = load("train"), load("test")
rul_test = np.loadtxt(f"{DATA}/RUL_{FD}.txt")

# operating regime = rounded operational settings
def regime_key(d):
    return (d.os1.round(0).astype(int).astype(str) + "_" + d.os2.round(2).astype(str)
            + "_" + d.os3.round(0).astype(int).astype(str))

keys = sorted(regime_key(train).unique())
kmap = {k: i for i, k in enumerate(keys)}
train["reg"] = regime_key(train).map(kmap)
test["reg"] = regime_key(test).map(kmap)
assert test.reg.notna().all()
test["reg"] = test.reg.astype(int)
R, S = len(keys), len(ALL_S)
scols = [f"s{i}" for i in ALL_S]


def functionals(cyc, Y, k, only_last=False):
    """Linear functionals of the recorded series.
    cyc: cycles 1..T, Y: [T, m] series. Records exist where (cycle-1) % k == 0.
    Returns level, base, slope arrays [T or 1, m] evaluated at each cycle t
    using only records with cycle <= t."""
    T = len(cyc)
    avail = ((cyc - 1) % k == 0).astype(float)
    Ya = Y * avail[:, None]
    c = cyc.astype(float)
    cs_n = np.concatenate([[0], np.cumsum(avail)])
    cs_y = np.vstack([np.zeros((1, Y.shape[1])), np.cumsum(Ya, 0)])
    cs_c = np.concatenate([[0], np.cumsum(avail * c)])
    cs_cc = np.concatenate([[0], np.cumsum(avail * c * c)])
    cs_cy = np.vstack([np.zeros((1, Y.shape[1])), np.cumsum(Ya * c[:, None], 0)])
    t_idx = np.arange(1, T + 1) if not only_last else np.array([T])
    lo_l = np.maximum(t_idx - LEVEL_W, 0)
    n_l = cs_n[t_idx] - cs_n[lo_l]
    level = (cs_y[t_idx] - cs_y[lo_l]) / n_l[:, None]
    hi_b = np.minimum(t_idx, BASE_W)
    base = cs_y[hi_b] / cs_n[hi_b][:, None]
    lo_s = np.maximum(t_idx - SLOPE_W, 0)
    n = cs_n[t_idx] - cs_n[lo_s]
    sc = cs_c[t_idx] - cs_c[lo_s]
    scc = cs_cc[t_idx] - cs_cc[lo_s]
    sy = cs_y[t_idx] - cs_y[lo_s]
    scy = cs_cy[t_idx] - cs_cy[lo_s]
    sxx = scc - sc * sc / n
    num = scy - (sc / n)[:, None] * sy
    ok = (n >= 2) & (sxx > 1e-9)
    slope = np.where(ok[:, None], num / np.where(ok, sxx, 1.0)[:, None], 0.0)
    return level, base, slope


def build(d, k, only_last):
    """Per engine: functionals of x*1_r (per regime, per sensor) and of 1_r."""
    FX, F1, age, unit, life, stats = [], [], [], [], [], {}
    for u, g in d.groupby("unit", sort=True):
        cyc = g.cycle.values
        X = g[scols].values.astype(float)
        reg = g.reg.values
        ind = np.zeros((len(g), R))
        ind[np.arange(len(g)), reg] = 1.0
        Y = np.concatenate([(X[:, None, :] * ind[:, :, None]).reshape(len(g), R * S), ind], 1)
        lv, bs, sl = functionals(cyc, Y, k, only_last)
        f = np.stack([lv, bs, sl], -1)  # [t, R*S+R, 3]
        FX.append(f[:, :R * S, :].reshape(-1, R, S, 3).astype(np.float32))
        F1.append(f[:, R * S:, :].astype(np.float32))
        tt = cyc if not only_last else cyc[-1:]
        age.append(tt); unit.append(np.full(len(tt), u)); life.append(cyc[-1])
        # sufficient statistics of recorded values per regime (for normalisation)
        av = (cyc - 1) % k == 0
        cnt = np.array([(av & (reg == r)).sum() for r in range(R)], float)
        sm = np.stack([X[av & (reg == r)].sum(0) for r in range(R)])
        sq = np.stack([(X[av & (reg == r)] ** 2).sum(0) for r in range(R)])
        stats[u] = (cnt, sm, sq)
    return (np.concatenate(FX), np.concatenate(F1), np.concatenate(age),
            np.concatenate(unit), np.array(life), stats)


def norm_stats(stats, units):
    cnt = sum(stats[u][0] for u in units)
    sm = sum(stats[u][1] for u in units)
    sq = sum(stats[u][2] for u in units)
    gm = sm.sum(0) / cnt.sum()
    gs = np.sqrt(np.maximum(sq.sum(0) / cnt.sum() - gm ** 2, 0))
    mu = np.where(cnt[:, None] > 0, sm / np.maximum(cnt, 1)[:, None], gm[None, :])
    var = sq / np.maximum(cnt, 1)[:, None] - mu ** 2
    sd = np.sqrt(np.maximum(var, 0))
    sd = np.where(cnt[:, None] >= 3, sd, gs[None, :])
    sd = np.where(sd > 1e-6 * (np.abs(mu) + 1), sd, 1.0)
    return mu, sd


def features(FX, F1, mu, sd, sidx, age):
    # normalised functional = sum_r (F[x*1_r] - mu_r * F[1_r]) / sd_r
    Z = ((FX[:, :, sidx, :] - mu[None, :, sidx, None] * F1[:, :, None, :])
         / sd[None, :, sidx, None]).sum(1)          # [n, s, 3]
    level, base, slope = Z[..., 0], Z[..., 1], Z[..., 2]
    return np.concatenate([level, level - base, slope, age[:, None]], 1)


def score_fn(err):
    return float(np.sum(np.where(err < 0, np.exp(-err / 13) - 1, np.exp(err / 10) - 1)))


units_all = np.sort(train.unit.unique())
n_all = len(units_all)
fleets = [n for n in FLEETS if n < n_all] + [n_all]
y_test = np.minimum(rul_test, CAP)
t0 = time.time()
for k in INTERVALS:
    todo = [n for n in fleets if not os.path.exists(f"{OUT}/k{k}_n{n}.csv")]
    if not todo:
        continue
    FXtr, F1tr, age_tr, unit_tr, life_tr, st_tr = build(train, k, False)
    FXte, F1te, age_te, unit_te, _, _ = build(test, k, True)
    life_by_unit = dict(zip(units_all, life_tr))
    rul_tr = np.minimum(np.array([life_by_unit[u] for u in unit_tr]) - age_tr, CAP)
    for n in todo:
        if time.time() - t0 > BUDGET:
            print("INCOMPLETE", FD, flush=True); sys.exit(0)
        rows = []
        reps = 1 if n == n_all else REPS
        for rep in range(reps):
            seed = zlib.crc32(f"{FD}-{n}-{rep}".encode())  # same draw for every k, sensor set, model
            rng = np.random.default_rng(seed)
            sel = units_all if n == n_all else rng.choice(units_all, n, replace=False)
            m = np.isin(unit_tr, sel)
            mu, sd = norm_stats(st_tr, sel)
            ytr = rul_tr[m]
            mean_life = np.mean([life_by_unit[u] for u in sel])
            pred0 = np.clip(mean_life - age_te, 0, CAP)
            base_row = dict(fd=FD, k=k, n=n, rep=rep)
            e = pred0 - y_test
            rows.append(dict(base_row, sensors="none", model="MeanLife", rmse=np.sqrt(np.mean(e ** 2)),
                             mae=np.mean(np.abs(e)), score=score_fn(e), bias=e.mean(),
                             late=np.mean(e > 10), fit_s=0.0, n_rows=int(m.sum())))
            for sname, sset in SENSOR_SETS.items():
                sidx = [ALL_S.index(s) for s in sset]
                Xtr = features(FXtr[m], F1tr[m], mu, sd, sidx, age_tr[m].astype(float))
                Xte = features(FXte, F1te, mu, sd, sidx, age_te.astype(float))
                sc = StandardScaler().fit(Xtr)
                models = {
                    "Ridge": (Ridge(alpha=1.0), True),
                    "RF": (RandomForestRegressor(n_estimators=50, min_samples_leaf=5, max_features=0.33,
                                                 n_jobs=1, random_state=seed % 2**31), False),
                    "GBM": (lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                              min_child_samples=20, subsample=0.8, subsample_freq=1,
                                              colsample_bytree=0.8, n_jobs=1, verbose=-1,
                                              random_state=seed % 2**31), False),
                }
                for mname, (mdl, scale) in models.items():
                    a = sc.transform(Xtr) if scale else Xtr
                    b = sc.transform(Xte) if scale else Xte
                    t1 = time.time()
                    mdl.fit(a, ytr)
                    ft = time.time() - t1
                    p = np.clip(mdl.predict(b), 0, CAP)
                    e = p - y_test
                    rows.append(dict(base_row, sensors=sname, model=mname, rmse=np.sqrt(np.mean(e ** 2)),
                                     mae=np.mean(np.abs(e)), score=score_fn(e), bias=e.mean(),
                                     late=np.mean(e > 10), fit_s=ft, n_rows=int(m.sum())))
        pd.DataFrame(rows).to_csv(f"{OUT}/k{k}_n{n}.tmp", index=False)
        os.replace(f"{OUT}/k{k}_n{n}.tmp", f"{OUT}/k{k}_n{n}.csv")
        print(FD, "k", k, "n", n, "done", round(time.time() - t0), "s", flush=True)
print("FINISHED", FD, round(time.time() - t0), "s")
