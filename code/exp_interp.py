"""
Interpretability probe (the KAN selling point, tested honestly).

KAN's headline claim: the learned univariate edge functions are simple enough to read off
and replace by closed-form symbols, giving an interpretable model "for free". We test two
things a claim of interpretability must actually satisfy:

  (Q1) SIMPLICITY: are the learned first-layer edge functions well-approximated by a
       human-readable low-order (<=3) polynomial?  -> per-edge R^2 distribution.
  (Q2) FIDELITY (is it free?): if we DISTILL layer 1 into those polynomials (symbolic
       surrogate) and re-evaluate, does task accuracy (NMSE) survive, or does the claimed
       interpretability throw away performance?  -> NMSE(full KAN) vs NMSE(symbolic KAN).

We run this on the DPD regime (the KAN-favourable smooth task). Fourier-KAN is used because
its edges are analytic; the probe logic is variant-agnostic.
"""
import numpy as np, torch, csv, os
from harness import KANreg, train_reg
from exp_dpd import make_dataset, nmse_db, L

DEG = 3


def edge_functions(kan, grid):
    """Return f_ij(x) sampled on grid for every layer-1 edge. coef: [2,h,din,K]."""
    c = kan.l1.coef.detach()
    K = c.shape[-1]
    kk = torch.arange(1, K + 1).float()
    cos = torch.cos(grid[:, None] * kk); sin = torch.sin(grid[:, None] * kk)   # [G,K]
    h, din = c.shape[1], c.shape[2]
    F = np.zeros((h, din, len(grid)), np.float64)
    for j in range(h):
        for i in range(din):
            F[j, i] = (cos @ c[0, j, i] + sin @ c[1, j, i]).numpy()
    return F                                       # [h, din, G]


def poly_fit_all(F, grid):
    """Fit degree-DEG polynomial to every edge; return coeffs [h,din,DEG+1] and R^2 [h,din]."""
    g = grid.numpy(); h, din, G = F.shape
    P = np.zeros((h, din, DEG + 1)); R2 = np.zeros((h, din))
    for j in range(h):
        for i in range(din):
            y = F[j, i]; p = np.polyfit(g, y, DEG); yh = np.polyval(p, g)
            ss_res = ((y - yh) ** 2).sum(); ss_tot = ((y - y.mean()) ** 2).sum() + 1e-12
            P[j, i] = p; R2[j, i] = 1 - ss_res / ss_tot
    return P, R2


class SymbolicKAN(torch.nn.Module):
    """KAN with layer-1 edges replaced by their fitted polynomials (symbolic surrogate).
    Layer 2 is kept exact, so any degradation is attributable to the layer-1 symbolic step."""
    def __init__(self, kan, P):
        super().__init__()
        self.P = torch.tensor(P, dtype=torch.float32)          # [h,din,DEG+1]
        self.bias1 = kan.l1.bias.detach().clone()
        self.l2 = kan.l2

    def forward(self, x):                                       # x: [B,din]
        B = x.shape[0]; h, din, _ = self.P.shape
        xp = torch.stack([x ** (DEG - d) for d in range(DEG + 1)], -1)  # [B,din,DEG+1]
        # edge_ij(x_i) = sum_d P[j,i,d] x_i^(DEG-d); h1_j = sum_i edge_ij + bias
        h1 = torch.einsum("bid,jid->bj", xp, self.P) + self.bias1
        return self.l2(torch.tanh(h1))


if __name__ == "__main__":
    din, H, K = 2 * L, 24, 4
    Xte, Yte = make_dataset(100)
    med_r2s, wmed_r2s, fracs, degs = [], [], [], []
    last_R2 = None
    for seed in range(5):                                  # multi-seed, like the rest of the paper
        Xtr, Ytr = make_dataset(seed)
        kan = KANreg(din, H, 2, K)
        nmse_full = train_reg(kan, Xtr, Ytr, Xte, Yte, seed)
        lo, hi = np.percentile(Xtr, 1), np.percentile(Xtr, 99)
        grid = torch.linspace(float(lo), float(hi), 200)
        F = edge_functions(kan, grid)
        P, R2 = poly_fit_all(F, grid)
        # variance-weighted R^2: weight each edge by its function variance (amplitude), so
        # near-flat edges cannot inflate the simplicity score (addresses the R^2-blindness point)
        wvar = F.var(axis=2)                               # [h,din]
        wmed = float((R2 * wvar).sum() / (wvar.sum() + 1e-12))
        sym = SymbolicKAN(kan, P)
        with torch.no_grad():
            pred = sym(torch.tensor(Xte)).numpy()
        nmse_sym = nmse_db(pred, Yte)
        med_r2s.append(float(np.median(R2))); wmed_r2s.append(wmed)
        fracs.append(float((R2 > 0.95).mean())); degs.append(nmse_sym - nmse_full)
        last_R2 = R2
        print(f"  seed{seed}: medR2={np.median(R2):.3f} wR2={wmed:.3f} "
              f"frac>0.95={100*(R2>0.95).mean():.1f}% degr={nmse_sym-nmse_full:+.2f}dB")

    med_r2 = float(np.mean(med_r2s)); wmed_r2 = float(np.mean(wmed_r2s))
    frac_simple = float(np.mean(fracs)); deg = float(np.mean(degs)); deg_std = float(np.std(degs))
    print(f"[Q1] median edge R^2={med_r2:.3f} | variance-weighted R^2={wmed_r2:.3f} | "
          f"frac>0.95={100*frac_simple:.1f}% (5 seeds)")
    print(f"[Q2] symbolic degradation = {deg:.2f} +- {deg_std:.2f} dB (5 seeds)")

    os.makedirs("results", exist_ok=True)
    with open("results/exp_interp.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "value"])
        w.writerow(["median_edge_R2_deg3", round(med_r2, 4)])
        w.writerow(["variance_weighted_R2", round(wmed_r2, 4)])
        w.writerow(["frac_edges_R2_gt_0.95", round(frac_simple, 4)])
        w.writerow(["symbolic_degradation_db", round(deg, 3)])
        w.writerow(["symbolic_degradation_std", round(deg_std, 3)])
    with open("results/edge_r2.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["r2"]); [w.writerow([round(float(v), 4)]) for v in last_R2.ravel()]
    print("wrote results/exp_interp.csv, results/edge_r2.csv")
