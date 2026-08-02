"""
Dump a few representative learned KAN edge functions and their cubic fits, so the
interpretability claim can be shown concretely (Fig. fig_edges): the edges really are simple
curves. Reuses the trained DPD Fourier-KAN from the interpretability setup.
"""
import numpy as np, torch, csv, os
from harness import KANreg, train_reg
from exp_dpd import make_dataset, L
from exp_interp import edge_functions, poly_fit_all, DEG

if __name__ == "__main__":
    din, H, K = 2 * L, 24, 4
    Xtr, Ytr = make_dataset(0); Xte, Yte = make_dataset(100)
    kan = KANreg(din, H, 2, K); train_reg(kan, Xtr, Ytr, Xte, Yte, 0)
    lo, hi = np.percentile(Xtr, 1), np.percentile(Xtr, 99)
    grid = torch.linspace(float(lo), float(hi), 120)
    F = edge_functions(kan, grid); P, R2 = poly_fit_all(F, grid)
    g = grid.numpy()
    # pick 3 edges spanning the R^2 range (highest, median, lowest) for an honest sample
    flat = [(j, i, R2[j, i]) for j in range(F.shape[0]) for i in range(F.shape[1])]
    flat.sort(key=lambda t: t[2])
    picks = [flat[-1], flat[len(flat) // 2], flat[0]]
    os.makedirs("results", exist_ok=True)
    with open("results/exp_edges.csv", "w", newline="") as fp:
        w = csv.writer(fp); w.writerow(["edge", "r2", "x", "f_kan", "f_cubic"])
        for eid, (j, i, r2) in enumerate(picks):
            fc = np.polyval(P[j, i], g)
            for x, fk, fcv in zip(g, F[j, i], fc):
                w.writerow([eid, round(float(r2), 4), round(float(x), 4),
                            round(float(fk), 5), round(float(fcv), 5)])
    print("picked edges R^2:", [round(float(p[2]), 4) for p in picks])
    print("wrote results/exp_edges.csv")
