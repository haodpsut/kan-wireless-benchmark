"""
Compute-cost audit (addresses the "matched parameters hide KAN's higher compute" objection).
At MATCHED parameter count we measure inference latency and training-step time for each model,
so the paper reports compute, not just parameter count. KANs are expected to be slower per
parameter (trig/spline basis per edge) than an MLP; we quantify it honestly.
"""
import numpy as np, torch, csv, os, time
from harness import KANreg, KANspline, MLPreg, KANcls, KANspline as KS, MLPcls, nparams, match_width
from exp_amc import CNNc, WIN
import torch.nn as nn

torch.set_num_threads(4)


def latency_ms(model, x, iters=50):
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(x)                       # warmup
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        return 1000.0 * (time.perf_counter() - t0) / iters


def step_ms(model, x, y, lossf, iters=30):
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for _ in range(3):
        opt.zero_grad(); lossf(model(x), y).backward(); opt.step()
    t0 = time.perf_counter()
    for _ in range(iters):
        opt.zero_grad(); lossf(model(x), y).backward(); opt.step()
    return 1000.0 * (time.perf_counter() - t0) / iters


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    rows = []
    B = 256

    # ---- DPD-sized (regression) at matched budget ----
    din = 10; H, K = 24, 4
    p = nparams(KANreg(din, H, 2, K))
    hs = match_width(p, lambda h: KANspline(din, h, 2), 4, 64)
    hm = match_width(p, lambda h: MLPreg(din, h, 2))
    x = torch.randn(B, din); y = torch.randn(B, 2); mse = nn.MSELoss()
    for name, m in [("MLP", MLPreg(din, hm, 2)), ("KANfourier", KANreg(din, H, 2, K)),
                    ("KANspline", KANspline(din, hs, 2))]:
        rows.append(["DPD", name, nparams(m), round(latency_ms(m, x), 3), round(step_ms(m, x, y, mse), 3)])

    # ---- AMC-sized (classification) at matched budget ----
    dinc = 2 * WIN; Hc, Kc, nc = 32, 4, 11
    pc = nparams(KANcls(dinc, Hc, nc, Kc))
    hsc = match_width(pc, lambda h: KS(dinc, h, nc), 4, 48)
    hmc = match_width(pc, lambda h: MLPcls(dinc, h, nc))
    xc = torch.randn(B, dinc); yc = torch.randint(0, nc, (B,)); ce = nn.CrossEntropyLoss()
    for name, m in [("CNN", CNNc(nc)), ("MLP", MLPcls(dinc, hmc, nc)),
                    ("KANfourier", KANcls(dinc, Hc, nc, Kc)), ("KANspline", KS(dinc, hsc, nc))]:
        rows.append(["AMC", name, nparams(m), round(latency_ms(m, xc), 3), round(step_ms(m, xc, yc, ce), 3)])

    with open("results/exp_compute.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["task", "model", "params", "infer_ms_per_batch", "train_ms_per_step"])
        w.writerows(rows)
    # headline: KAN inference slowdown vs matched MLP
    d = {(r[0], r[1]): r[3] for r in rows}
    print(f"DPD infer slowdown: fKAN {d[('DPD','KANfourier')]/d[('DPD','MLP')]:.1f}x, "
          f"sKAN {d[('DPD','KANspline')]/d[('DPD','MLP')]:.1f}x vs MLP")
    print(f"AMC infer slowdown: fKAN {d[('AMC','KANfourier')]/d[('AMC','MLP')]:.1f}x, "
          f"sKAN {d[('AMC','KANspline')]/d[('AMC','MLP')]:.1f}x vs MLP")
    for r in rows: print(r)
    print("wrote results/exp_compute.csv")
