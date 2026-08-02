# kan-wireless-benchmark

Reproducibility artifact for **"When Does a Kolmogorov--Arnold Network Help in Wireless?
A Parameter-Matched, Non-Saturated Benchmark."**

An honest-measurement benchmark of Kolmogorov--Arnold networks (KANs) against
parameter-matched baselines across two wireless regimes (power-amplifier behavioral modeling
and automatic modulation classification on RML2016.10a), plus an interpretability audit and
four stress-test ablations. Every number in the paper is produced by a script here and traced
to `paper/claims.json`.

## Setup
```bash
pip install torch numpy scipy scikit-learn matplotlib
# AMC experiments need the public RML2016.10a dataset:
#   place RML2016.10a_dict.pkl under data/RML2016.10a/
```

## Result -> script map
| Paper item | Script | Output CSV |
|---|---|---|
| Table II (DPD, matched capacity) | `code/exp_dpd.py` | `results/exp_dpd.csv` |
| Table III (AMC, RML2016.10a low-SNR) | `code/exp_amc.py` | `results/exp_amc.csv` |
| Interpretability audit (Q1/Q2) | `code/exp_interp.py` | `results/exp_interp.csv`, `edge_r2.csv` |
| Learned-edge figure | `code/exp_edges.py` | `results/exp_edges.csv` |
| Table IV (KAN hyperparameter sweep) | `code/abl_dpd.py` | `results/abl_dpd_kanhp.csv` |
| Budget & sample-efficiency sweeps | `code/abl_dpd.py` | `results/abl_dpd_{budget,samples}.csv` |
| SNR sweep & AMC budget sweep | `code/abl_amc.py` | `results/abl_amc_{snr,budget}.csv` |
| Per-modulation analysis | `code/exp_amc_permod.py` | `results/exp_amc_permod.csv` |

`code/harness.py` implements the KAN variants (spline + Fourier), the MLP, the
matched-capacity sizing, the training loop, and the bootstrap statistics shared by all
experiments. `code/plots.py` renders the figures in the paper house style (requires the
`plot-figure-dph` styling module); the numbers themselves are fully reproduced by the
experiment scripts above and stored as CSVs regardless of the plotting layer.

## Reproduce
```bash
cd code
python3 exp_dpd.py          # DPD / PA behavioral modeling
python3 exp_amc.py          # AMC on RML2016.10a
python3 exp_interp.py       # interpretability audit
python3 abl_dpd.py          # DPD ablations (budget / K,grid / samples)
python3 abl_amc.py          # AMC ablations (SNR sweep / budget)
python3 exp_amc_permod.py   # per-modulation analysis
```
All random seeds are fixed. Minor (<0.1 dB) variation on the spline-KAN path is possible on
Apple MPS; the reported numbers are the CSVs in `results/`.

## License
Released for review and reproduction.
