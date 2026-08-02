"""
Shared honest-measurement harness for the KAN-in-wireless research paper.

Design goals (the whole point of the paper):
  1. PARAMETER-MATCHED comparison: KAN vs MLP are sized to the SAME parameter budget,
     so "KAN is more efficient" can never be an artifact of a small-KAN-vs-large-MLP.
  2. MULTI-SEED with bootstrap confidence intervals: no single-seed wins.
  3. Task-appropriate strong baselines (CNN for I/Q, memory-polynomial for DPD): KAN
     is not allowed to look good only against a weak baseline.
  4. Whatever it shows, we report. The finding is regime-dependent by design.

KAN variant = Fourier-KAN (learnable truncated Fourier series on each edge). This is a
standard KAN family; params per layer = in*out*2*K + out.
"""
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
DEVICE = "cpu"  # small models; CPU is deterministic and avoids MPS complex/einsum quirks


# ----------------------------------------------------------------------------- KAN / MLP
class FourierKANLayer(nn.Module):
    """One KAN layer: y_j = sum_i sum_{k=1..K} [a_ijk cos(k x_i) + b_ijk sin(k x_i)] + bias_j."""
    def __init__(self, din, dout, K=4):
        super().__init__()
        self.din, self.dout, self.K = din, dout, K
        self.coef = nn.Parameter(0.1 * torch.randn(2, dout, din, K) / (din * K) ** 0.5)
        self.bias = nn.Parameter(torch.zeros(dout))

    def forward(self, x):                       # x: [B, din]
        kk = torch.arange(1, self.K + 1, device=x.device).float()
        xk = x[:, None, :, None] * kk           # [B,1,din,K]
        basis = torch.cat([torch.cos(xk), torch.sin(xk)], dim=1)  # [B,2,din,K]
        return torch.einsum("bcik,coik->bo", basis, self.coef) + self.bias


class KANreg(nn.Module):
    """Fourier-KAN for regression (linear head)."""
    def __init__(self, din, h, dout, K=4):
        super().__init__()
        self.l1 = FourierKANLayer(din, h, K)
        self.l2 = FourierKANLayer(h, dout, K)

    def forward(self, x):
        return self.l2(torch.tanh(self.l1(x)))


class SplineKANLayer(nn.Module):
    """Canonical B-spline KAN layer (efficient-kan style): each edge = SiLU base + B-spline.
    This is the KAN variant most reviewers mean; params = din*dout*(1 + grid+k)."""
    def __init__(self, din, dout, grid=6, k=3, rng=(-2.5, 2.5)):
        super().__init__()
        self.din, self.dout, self.grid_size, self.k = din, dout, grid, k
        h = (rng[1] - rng[0]) / grid
        knots = torch.arange(-k, grid + k + 1) * h + rng[0]           # [grid+2k+1]
        self.register_buffer("grid", knots.expand(din, -1).contiguous())
        self.base_weight = nn.Parameter(0.1 * torch.randn(dout, din))
        self.spline_weight = nn.Parameter(0.1 * torch.randn(dout, din, grid + k))

    def b_splines(self, x):                       # x: [B,din] -> [B,din,grid+k]
        g = self.grid                             # [din, G]
        x = x.unsqueeze(-1)
        b = ((x >= g[:, :-1]) & (x < g[:, 1:])).float()
        for kk in range(1, self.k + 1):
            b = ((x - g[:, :-(kk + 1)]) / (g[:, kk:-1] - g[:, :-(kk + 1)]) * b[:, :, :-1]
                 + (g[:, kk + 1:] - x) / (g[:, kk + 1:] - g[:, 1:-kk]) * b[:, :, 1:])
        return b

    def forward(self, x):
        base = torch.nn.functional.silu(x) @ self.base_weight.t()
        spline = torch.einsum("bic,oic->bo", self.b_splines(x), self.spline_weight)
        return base + spline


class KANspline(nn.Module):
    """2-layer canonical spline-KAN for regression or classification."""
    def __init__(self, din, h, dout, grid=6, k=3):
        super().__init__()
        self.l1 = SplineKANLayer(din, h, grid, k)
        self.l2 = SplineKANLayer(h, dout, grid, k)

    def forward(self, x):
        return self.l2(self.l1(x))


class MLPreg(nn.Module):
    def __init__(self, din, h, dout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, dout))

    def forward(self, x):
        return self.net(x)


class KANcls(nn.Module):
    def __init__(self, din, h, nc, K=4):
        super().__init__()
        self.l1 = FourierKANLayer(din, h, K)
        self.l2 = FourierKANLayer(h, nc, K)

    def forward(self, x):
        return self.l2(torch.tanh(self.l1(x)))


class MLPcls(nn.Module):
    def __init__(self, din, h, nc):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, nc))

    def forward(self, x):
        return self.net(x)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def match_width(target_params, build_fn, lo=4, hi=512):
    """Pick the hidden width whose param count is closest to target_params."""
    best, bh = 1e18, lo
    for h in range(lo, hi):
        d = abs(nparams(build_fn(h)) - target_params)
        if d < best:
            best, bh = d, h
    return bh


# ----------------------------------------------------------------------------- training
def train_reg(model, Xtr, Ytr, Xte, Yte, seed, epochs=60, lr=2e-3, bs=256):
    """Train a regressor; return test NMSE in dB (lower is better)."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.MSELoss()
    ds = torch.utils.data.TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr))
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True)
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xte)).numpy()
    err = ((pred - Yte) ** 2).sum()
    sig = (Yte ** 2).sum()
    return 10 * np.log10(err / sig)


def train_cls(model, Xtr, ytr, Xte, yte, seed, epochs=40, lr=2e-3, bs=256):
    from sklearn.metrics import accuracy_score
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True)
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xte)).argmax(1).numpy()
    return accuracy_score(yte, pred)


# ----------------------------------------------------------------------------- statistics
def bootstrap_ci(vals, B=2000, alpha=0.05, seed=0):
    """Bootstrap CI of the mean over seeds."""
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, float)
    means = [rng.choice(v, len(v), replace=True).mean() for _ in range(B)]
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(v.std()), float(lo), float(hi)


def paired_diff_ci(a, b, B=2000, alpha=0.05, seed=0):
    """Bootstrap CI of the paired difference a-b across seeds (KAN vs baseline)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    means = [rng.choice(d, len(d), replace=True).mean() for _ in range(B)]
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(lo), float(hi)
