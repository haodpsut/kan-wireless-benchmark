"""
Symmetric MLP hyperparameter sweep (answers the "you tuned the KAN but not the MLP" fairness
objection). At the same DPD budget where Table IV sweeps 6 KAN configs, we sweep the MLP over
depth and width/activation and report the best. If the best MLP is no better than the default
MLP already reported, the comparison was fair; the KAN was not beaten by an untuned baseline.
"""
import numpy as np, torch, torch.nn as nn, csv, os
from harness import KANreg, MLPreg, nparams, match_width, train_reg
from exp_dpd import make_dataset, L


class MLPflex(nn.Module):
    """MLP with configurable depth and activation, sized to a target width."""
    def __init__(self, din, h, dout, depth=2, act="relu"):
        super().__init__()
        A = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[act]
        layers = [nn.Linear(din, h), A()]
        for _ in range(depth - 1):
            layers += [nn.Linear(h, h), A()]
        layers += [nn.Linear(h, dout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def match_flex(target, din, dout, depth, act):
    best, bh = 1e18, 8
    for h in range(4, 200):
        d = abs(nparams(MLPflex(din, h, dout, depth, act)) - target)
        if d < best:
            best, bh = d, h
    return bh


if __name__ == "__main__":
    din = 2 * L; target = nparams(KANreg(din, 24, 2, 4))
    os.makedirs("results", exist_ok=True)
    configs = [(1, "relu"), (2, "relu"), (3, "relu"), (2, "gelu"), (2, "tanh"), (4, "relu")]
    rows = []
    for depth, act in configs:
        h = match_flex(target, din, 2, depth, act)
        p = nparams(MLPflex(din, h, 2, depth, act))
        accs = []
        for seed in range(5):
            Xtr, Ytr = make_dataset(seed); Xte, Yte = make_dataset(100 + seed)
            accs.append(train_reg(MLPflex(din, h, 2, depth, act), Xtr, Ytr, Xte, Yte, seed))
        a = np.array(accs)
        rows.append([f"depth{depth}-{act}", h, p, round(a.mean(), 3), round(a.std(), 3)])
        print(f"MLP depth={depth} act={act} h={h} ({p}p): {a.mean():.2f}+-{a.std():.2f} dB")
    with open("results/exp_mlp_sweep.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["config", "h", "params", "nmse_mean", "nmse_std"]); w.writerows(rows)
    best = min(rows, key=lambda r: r[3])
    print(f"best MLP config: {best[0]} = {best[3]} dB (default depth2-relu is the one reported)")
    print("wrote results/exp_mlp_sweep.csv")
