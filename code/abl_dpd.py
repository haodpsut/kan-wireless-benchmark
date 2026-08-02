"""
DPD ablations that stress-test (steelman) the KAN before we conclude against it:
  A. BUDGET SWEEP    -- does the MLP>KAN gap close if KANs get more parameters?
  B. KAN HYPERPARAMS -- give the KAN its best shot: sweep Fourier K and spline grid.
  C. SAMPLE EFFICIENCY -- do KANs win in the low-data regime (a distinct claimed edge)?
All matched-capacity where a head-to-head is drawn, all multi-seed.
"""
import numpy as np, torch, csv, os
from harness import (KANreg, KANspline, MLPreg, nparams, match_width, train_reg, paired_diff_ci)
from exp_dpd import make_dataset, L

SEEDS = range(5)


def run(models_fn, ntr=None):
    accs = {k: [] for k in models_fn}
    for seed in SEEDS:
        Xtr, Ytr = make_dataset(seed); Xte, Yte = make_dataset(100 + seed)
        if ntr:
            Xtr, Ytr = Xtr[:ntr], Ytr[:ntr]
        for k, mk in models_fn.items():
            accs[k].append(train_reg(mk(), Xtr, Ytr, Xte, Yte, seed))
    return accs


if __name__ == "__main__":
    din = 2 * L
    os.makedirs("results", exist_ok=True)

    # ---- A. budget sweep ----
    with open("results/abl_dpd_budget.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["budget_H", "params", "model", "nmse_mean", "nmse_std"])
        for H in [8, 24, 48, 96]:
            p = nparams(KANreg(din, H, 2, 4))
            hs = match_width(p, lambda h: KANspline(din, h, 2), 4, 96)
            hm = match_width(p, lambda h: MLPreg(din, h, 2))
            mods = {"KANfourier": lambda: KANreg(din, H, 2, 4),
                    "KANspline": lambda: KANspline(din, hs, 2),
                    "MLP": lambda: MLPreg(din, hm, 2)}
            r = run(mods)
            for k in mods:
                a = np.array(r[k]); w.writerow([H, p, k, round(a.mean(), 3), round(a.std(), 3)])
            print(f"budget H={H} (~{p}p): MLP={np.mean(r['MLP']):.2f} "
                  f"fKAN={np.mean(r['KANfourier']):.2f} sKAN={np.mean(r['KANspline']):.2f} dB")

    # ---- B. KAN hyperparameter sweep (best shot for the KAN) ----
    with open("results/abl_dpd_kanhp.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["variant", "hp", "params", "nmse_mean", "nmse_std"])
        for K in [2, 4, 8]:
            r = run({"f": lambda K=K: KANreg(din, 24, 2, K)})
            a = np.array(r["f"]); p = nparams(KANreg(din, 24, 2, K))
            w.writerow(["fourier_K", K, p, round(a.mean(), 3), round(a.std(), 3)])
            print(f"Fourier K={K}: {a.mean():.2f} dB ({p}p)")
        for grid in [4, 6, 10]:
            r = run({"s": lambda g=grid: KANspline(din, 12, 2, g)})
            a = np.array(r["s"]); p = nparams(KANspline(din, 12, 2, grid))
            w.writerow(["spline_grid", grid, p, round(a.mean(), 3), round(a.std(), 3)])
            print(f"spline grid={grid}: {a.mean():.2f} dB ({p}p)")

    # ---- C. sample efficiency ----
    with open("results/abl_dpd_samples.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["ntrain", "model", "nmse_mean", "nmse_std"])
        H = 24; p = nparams(KANreg(din, H, 2, 4)); hm = match_width(p, lambda h: MLPreg(din, h, 2))
        for ntr in [1000, 4000, 16000, 63995]:
            mods = {"KANfourier": lambda: KANreg(din, H, 2, 4),
                    "MLP": lambda: MLPreg(din, hm, 2)}
            r = run(mods, ntr=ntr)
            for k in mods:
                a = np.array(r[k]); w.writerow([ntr, k, round(a.mean(), 3), round(a.std(), 3)])
            d, lo, hi = paired_diff_ci(r["KANfourier"], r["MLP"])
            print(f"ntrain={ntr}: MLP={np.mean(r['MLP']):.2f} fKAN={np.mean(r['KANfourier']):.2f} "
                  f"diff={d:+.2f}[{lo:+.2f},{hi:+.2f}] dB")
    print("wrote abl_dpd_{budget,kanhp,samples}.csv")
