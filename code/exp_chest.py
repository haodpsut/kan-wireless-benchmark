"""
Regime 3: OFDM channel estimation (a third, distinct task type: estimation, not PA modeling
or classification). Comb pilots observe the channel frequency response at a subset of
subcarriers under noise; the task is to estimate the channel at all subcarriers. This is a
smooth interpolation-style regression with a genuine STRONG classical baseline (LMMSE), the
Wiener estimator that is optimal in the linear-Gaussian setting -- so KANs face the real
workhorse, not a strawman.

Metric: channel NMSE in dB. Matched-capacity KAN (spline, Fourier) vs MLP; classical LMMSE and
linear interpolation baselines. Multi-seed. Operating SNR chosen non-saturated.
"""
import numpy as np, torch, csv, os
from harness import KANreg, KANspline, MLPreg, nparams, match_width, train_reg, bootstrap_ci, paired_diff_ci

N = 64                 # subcarriers
PILOT_STEP = 4         # comb pilots -> N/PILOT_STEP pilots
NP = N // PILOT_STEP
L = 8                  # channel taps (exponential power-delay profile)
SNR_DB = 10.0          # non-saturated operating point
NTRAIN, NTEST = 20000, 8000


def gen_channels(n, seed):
    rng = np.random.default_rng(seed)
    pdp = np.exp(-np.arange(L) / 3.0); pdp /= pdp.sum()      # exponential PDP
    h = (rng.normal(size=(n, L)) + 1j * rng.normal(size=(n, L))) * np.sqrt(pdp / 2)
    H = np.fft.fft(h, N, axis=1)                             # freq response [n, N]
    return H


def add_noise(H, seed):
    rng = np.random.default_rng(seed + 777)
    p = (np.abs(H) ** 2).mean()
    sigma2 = p / (10 ** (SNR_DB / 10))
    noise = (rng.normal(size=H.shape) + 1j * rng.normal(size=H.shape)) * np.sqrt(sigma2 / 2)
    return H + noise, sigma2


def to_feat(Hp):                                             # complex pilots -> real features
    return np.concatenate([Hp.real, Hp.imag], 1).astype(np.float32)


def make_dataset(seed, ntrain=NTRAIN):
    H = gen_channels(ntrain, seed)
    Yp, _ = add_noise(H[:, ::PILOT_STEP], seed)              # noisy pilot observations
    X = to_feat(Yp)
    Y = np.concatenate([H.real, H.imag], 1).astype(np.float32)  # full channel target
    return X, Y


def nmse_db(pred, Y):
    return 10 * np.log10(((pred - Y) ** 2).sum() / (Y ** 2).sum())


def lmmse(Xtr, Ytr, Xte, seed):
    """Wiener/LMMSE estimator: W = R_{H,Hp}(R_{Hp,Hp}+sigma^2 I)^-1, learned from training
    covariances. The optimal linear estimator; the genuine classical baseline."""
    ntr = len(Xtr)
    Hp = (Xtr[:, :NP] + 1j * Xtr[:, NP:]).astype(np.complex128)          # [ntr, NP] noisy pilots
    H = (Ytr[:, :N] + 1j * Ytr[:, N:]).astype(np.complex128)            # [ntr, N] clean
    # complex covariances R = E[x y^H] -> sample form X^T conj(Y) / n
    R_HHp = (H.T @ Hp.conj()) / ntr                                      # E[H Hp^H]  [N, NP]
    R_HpHp = (Hp.T @ Hp.conj()) / ntr                                    # E[Hp Hp^H] [NP, NP]
    W = R_HHp @ np.linalg.inv(R_HpHp + 1e-9 * np.eye(NP))                # LMMSE gain [N, NP]
    Hp_te = (Xte[:, :NP] + 1j * Xte[:, NP:]).astype(np.complex128)
    est = Hp_te @ W.T                                                    # [nte, N]
    return np.concatenate([est.real, est.imag], 1).astype(np.float32)


def lininterp(Xte):
    """Linear interpolation of pilot observations across subcarriers (weak classical baseline)."""
    Hp = Xte[:, :NP] + 1j * Xte[:, NP:]
    xp = np.arange(0, N, PILOT_STEP); xf = np.arange(N)
    out = np.stack([np.interp(xf, xp, Hp[i]) for i in range(len(Hp))], 0)
    return np.concatenate([out.real, out.imag], 1).astype(np.float32)


if __name__ == "__main__":
    din, dout, H, K = 2 * NP, 2 * N, 24, 4
    p = nparams(KANreg(din, H, dout, K))
    hs = match_width(p, lambda h: KANspline(din, h, dout), 4, 96)
    hm = match_width(p, lambda h: MLPreg(din, h, dout))
    print(f"ChEst | din={din} dout={dout} | fKAN={p} sKAN={nparams(KANspline(din,hs,dout))} "
          f"MLP(h={hm})={nparams(MLPreg(din,hm,dout))}")
    res = {"KANspline": [], "KANfourier": [], "MLP": [], "LMMSE": [], "LinInterp": []}
    for seed in range(5):
        Xtr, Ytr = make_dataset(seed); Xte, Yte = make_dataset(200 + seed, NTEST)
        res["KANspline"].append(train_reg(KANspline(din, hs, dout), Xtr, Ytr, Xte, Yte, seed))
        res["KANfourier"].append(train_reg(KANreg(din, H, dout, K), Xtr, Ytr, Xte, Yte, seed))
        res["MLP"].append(train_reg(MLPreg(din, hm, dout), Xtr, Ytr, Xte, Yte, seed))
        res["LMMSE"].append(nmse_db(lmmse(Xtr, Ytr, Xte, seed), Yte))
        res["LinInterp"].append(nmse_db(lininterp(Xte), Yte))
        print(f"  seed{seed}: sKAN={res['KANspline'][-1]:.2f} fKAN={res['KANfourier'][-1]:.2f} "
              f"MLP={res['MLP'][-1]:.2f} LMMSE={res['LMMSE'][-1]:.2f} interp={res['LinInterp'][-1]:.2f}")
    os.makedirs("results", exist_ok=True)
    with open("results/exp_chest.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "nmse_mean", "nmse_std", "ci_lo", "ci_hi"])
        for m in ["LMMSE", "MLP", "KANspline", "KANfourier", "LinInterp"]:
            mean, std, lo, hi = bootstrap_ci(res[m])
            w.writerow([m, round(mean, 3), round(std, 3), round(lo, 3), round(hi, 3)])
            print(f"{m}: {mean:.2f}+-{std:.2f} dB CI[{lo:.2f},{hi:.2f}]")
    dk, lo, hi = paired_diff_ci(res["KANspline"], res["MLP"])
    print(f"KANspline-MLP: {dk:+.2f} dB CI[{lo:+.2f},{hi:+.2f}]")
    dl, lo2, hi2 = paired_diff_ci(res["MLP"], res["LMMSE"])
    print(f"MLP-LMMSE: {dl:+.2f} dB CI[{lo2:+.2f},{hi2:+.2f}]")
    print("wrote results/exp_chest.csv")
