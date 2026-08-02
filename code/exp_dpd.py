"""
Experiment DPD: power-amplifier behavioral modeling (the smooth, low-dimensional
nonlinear-regression regime where KAN is *claimed* to help most). Honest test.

Ground truth PA = Wiener-Hammerstein: FIR h1 -> static Saleh AM/AM+AM/PM -> FIR h2.
This is NOT a memory polynomial, so the classical memory-polynomial baseline does not
get an unfair "true model class" advantage; all three learners approximate.

Task: given a memory window of the complex input {x[n-L+1..n]} predict the PA output y[n].
Metric: NMSE in dB (lower is better), 5 seeds, bootstrap CI, paired KAN-vs-baseline diff.

Baselines: (a) memory polynomial fit by least squares (classical DPD workhorse),
(b) parameter-matched MLP, (c) KAN. Interpretability probe: can the KAN's learned edge
function be fit by a simple closed form (low-order poly)? -> demonstrable-interpretability R^2.
"""
import numpy as np, torch, csv, os
from harness import (KANreg, KANspline, MLPreg, nparams, match_width, train_reg,
                     bootstrap_ci, paired_diff_ci)

L = 5                      # memory window length (samples) -> din = 2L
SPS = 4                    # samples per symbol
NSYM = 8000               # symbols per realization


def rrc(beta, sps, span=6):
    n = np.arange(-span * sps, span * sps + 1) / sps
    with np.errstate(divide="ignore", invalid="ignore"):
        h = (np.sin(np.pi * n * (1 - beta)) + 4 * beta * n * np.cos(np.pi * n * (1 + beta))) \
            / (np.pi * n * (1 - (4 * beta * n) ** 2))
    h[n == 0] = 1 - beta + 4 * beta / np.pi
    t = np.abs(n) == 1 / (4 * beta)
    h[t] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
    return h / np.sqrt((h ** 2).sum())


def gen_signal(seed, backoff_db=6.0):
    rng = np.random.default_rng(seed)
    # QAM16 symbols
    m = np.array([-3, -1, 1, 3])
    C = (m[:, None] + 1j * m[None, :]).ravel(); C /= np.sqrt((np.abs(C) ** 2).mean())
    syms = C[rng.integers(0, 16, NSYM)]
    up = np.zeros(NSYM * SPS, complex); up[::SPS] = syms
    x = np.convolve(up, rrc(0.25, SPS), "same")
    x /= np.sqrt((np.abs(x) ** 2).mean())
    x *= 10 ** (-backoff_db / 20)                     # input backoff
    return x


def pa_wiener_hammerstein(x):
    h1 = np.array([1.0, 0.3, -0.1], complex)          # input memory FIR
    h2 = np.array([1.0, 0.2, 0.05, -0.03], complex)   # output memory FIR
    u = np.convolve(x, h1, "same")
    r = np.abs(u)
    # Saleh AM/AM and AM/PM (smooth, saturating, standard)
    aa, ba = 2.0, 1.2
    ap, bp = 4.0, 9.0
    A = aa * r / (1 + ba * r ** 2)
    P = ap * r ** 2 / (1 + bp * r ** 2)
    v = A * np.exp(1j * (np.angle(u) + P))
    y = np.convolve(v, h2, "same")
    return y


def make_dataset(seed):
    x = gen_signal(seed); y = pa_wiener_hammerstein(x)
    N = len(x)
    X = np.zeros((N - L, 2 * L), np.float32)
    Y = np.zeros((N - L, 2), np.float32)
    for i in range(L, N):
        win = x[i - L:i]
        X[i - L] = np.concatenate([win.real, win.imag]).astype(np.float32)
        Y[i - L] = [y[i - 1].real, y[i - 1].imag]
    # scale target to unit power for stable NMSE
    s = np.sqrt((Y ** 2).sum(1).mean()); Y = Y / s
    return X, Y


def memory_poly_ls(Xtr, Ytr, Xte, K=7):
    """Generalized Memory Polynomial (GMP) baseline, the genuine classical DPD/PA workhorse
    (Morgan et al. 2006). Basis = aligned terms x[n-m]|x[n-m]|^(k-1) PLUS lagging/leading
    cross-envelope terms x[n-m]|x[n-m']|^(k-1) for m' near m, odd k up to K. Fit by
    well-conditioned least squares (light ridge, column-normalized). This is a STRONG
    baseline, not a diagonal-only strawman."""
    def basis(X):
        xc = np.stack([X[:, tap] + 1j * X[:, L + tap] for tap in range(L)], 1)  # [N,L]
        mag = np.abs(xc)
        cols = []
        for m in range(L):
            for mp in range(L):                        # cross-lag envelope (GMP), incl. m'=m
                if abs(m - mp) <= 2:
                    for k in range(1, K + 1, 2):
                        cols.append(xc[:, m] * mag[:, mp] ** (k - 1))
        return np.stack(cols, 1)
    Btr, Bte = basis(Xtr).astype(np.complex128), basis(Xte).astype(np.complex128)
    scale = np.sqrt((np.abs(Btr) ** 2).mean(0)) + 1e-9
    Btr_n, Bte_n = Btr / scale, Bte / scale
    yc = (Ytr[:, 0] + 1j * Ytr[:, 1]).astype(np.complex128)
    F = Btr_n.shape[1]
    G = Btr_n.conj().T @ Btr_n
    lam = 1e-6 * np.real(np.trace(G)) / F              # light ridge: strong, well-conditioned
    w = np.linalg.solve(G + lam * np.eye(F), Btr_n.conj().T @ yc)
    pred = np.nan_to_num(Bte_n @ w)
    return np.stack([pred.real, pred.imag], 1).astype(np.float32)


def nmse_db(pred, Y):
    return 10 * np.log10(((pred - Y) ** 2).sum() / (Y ** 2).sum())


if __name__ == "__main__":
    din, H, K = 2 * L, 24, 4
    p_fkan = nparams(KANreg(din, H, dout=2, K=K))
    h_skan = match_width(p_fkan, lambda h: KANspline(din, h, 2), lo=4, hi=64)  # matched budget
    p_skan = nparams(KANspline(din, h_skan, 2))
    h_mlp = match_width(p_fkan, lambda h: MLPreg(din, h, 2))
    p_mlp = nparams(MLPreg(din, h_mlp, 2))
    print(f"DPD PA-modeling | din={din} L={L} | Fourier-KAN={p_fkan} | "
          f"spline-KAN(h={h_skan})={p_skan} | MLP(h={h_mlp})={p_mlp} | all matched to ~{p_fkan}")

    res = {"KANspline": [], "KANfourier": [], "MLP": [], "MemPoly": []}
    for seed in range(5):
        Xtr, Ytr = make_dataset(seed); Xte, Yte = make_dataset(100 + seed)
        res["KANspline"].append(train_reg(KANspline(din, h_skan, 2), Xtr, Ytr, Xte, Yte, seed))
        res["KANfourier"].append(train_reg(KANreg(din, H, 2, K), Xtr, Ytr, Xte, Yte, seed))
        res["MLP"].append(train_reg(MLPreg(din, h_mlp, 2), Xtr, Ytr, Xte, Yte, seed))
        res["MemPoly"].append(nmse_db(memory_poly_ls(Xtr, Ytr, Xte), Yte))
        print(f"  seed{seed}: sKAN={res['KANspline'][-1]:.2f}  fKAN={res['KANfourier'][-1]:.2f}  "
              f"MLP={res['MLP'][-1]:.2f}  MemPoly={res['MemPoly'][-1]:.2f} dB")

    os.makedirs("results", exist_ok=True)
    with open("results/exp_dpd.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "params", "nmse_mean", "nmse_std", "ci_lo", "ci_hi"])
        for name, p in [("KANspline", p_skan), ("KANfourier", p_fkan), ("MLP", p_mlp), ("MemPoly", "-")]:
            mean, std, lo, hi = bootstrap_ci(res[name])
            w.writerow([name, p, round(mean, 3), round(std, 3), round(lo, 3), round(hi, 3)])
            print(f"{name}: NMSE {mean:.2f}±{std:.2f} dB  CI[{lo:.2f},{hi:.2f}]")
    for a in ["KANspline", "KANfourier"]:
        dk, lo, hi = paired_diff_ci(res[a], res["MLP"])
        print(f"{a}-MLP paired NMSE diff: {dk:+.2f} dB  CI[{lo:+.2f},{hi:+.2f}]  (neg = KAN better)")
    print("wrote results/exp_dpd.csv")
