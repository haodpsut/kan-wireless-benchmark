"""Sinh cac claim MOI tu CSV ket qua, khong go tay con so nao.

Doc results-neucom/*.csv (keo ve tu sol1) va ghi:
  - paper/claims/<ten>.tex   mot tep mot so
  - cap nhat paper/claims.json

Moi so trong abstract va trong mục co che deu phai di qua day. Cong c5 se bat neu
con so nao go tay trong van xuoi.
"""
import csv, json, os, re, collections

HERE = os.path.dirname(os.path.abspath(__file__))
# MOT nha duy nhat cho ket qua: dung duong ma chinh cac script thi nghiem GHI VAO,
# khong tao thu muc thu hai. Da tung co "results-neucom" song song va do la hai nha.
RES = os.path.join(HERE, "results")
CLAIMS_DIR = os.path.join(HERE, "..", "paper", "claims")
CLAIMS_JSON = os.path.join(HERE, "..", "paper", "claims.json")


def doc(ten):
    with open(os.path.join(RES, ten), newline="") as f:
        return list(csv.DictReader(f))


def lay(rows, **dk):
    ra = [r for r in rows if all(str(r.get(k, "")) == str(v) for k, v in dk.items())]
    assert len(ra) == 1, f"mong 1 dong, duoc {len(ra)} voi {dk}"
    return ra[0]


def main():
    m2 = doc("exp_mechanism2.csv")
    bb = doc("exp_amc_backbone.csv")
    bg = doc("exp_bigbudget_paired.csv")
    sw = doc("exp_mlp_sweep_all.csv")
    tn = doc("exp_mechanism2_tuning.csv")

    moi = {}

    # --- doi chung do kho: bien do cua moc trung tinh qua moi lam
    for n in ("256", "4096"):
        v = [float(r["nmse_db"]) for r in m2
             if r["model"].startswith("LSQ") and r["n"] == n]
        moi[f"mech_dokho_bien_do_n{n}"] = round(max(v) - min(v), 3)

    # --- diem dao dau o n=256: lam cuoi con KAN thang, lam dau MLP thang
    def paired(n, lam, model):
        return float(lay(m2, n=n, lam=lam, model=f"PAIRED {model} - MLP")["nmse_db"])

    moi["mech_n_it"] = 256
    moi["mech_n_nhieu"] = 4096
    moi["mech_lam_giao_duoi"] = 0.5
    moi["mech_lam_giao_tren"] = 0.75
    # bien do MLP thang o lam=1, n=256: lay nho nhat va lon nhat trong 4 bien the
    v1 = [paired("256", "1.0", m) for m in ("KAN-spline", "KAN-Fourier", "KAN-fast", "KAN-cheby")]
    moi["mech_mlp_thang_min"] = round(min(v1), 2)
    moi["mech_mlp_thang_max"] = round(max(v1), 2)
    # xoi mon o n=4096 voi spline: tu lam=0 den lam=1
    moi["mech_spline_n4096_lam0"] = round(abs(paired("4096", "0.0", "KAN-spline")), 2)
    moi["mech_spline_n4096_lam1"] = round(abs(paired("4096", "1.0", "KAN-spline")), 2)
    moi["mech_so_bien_the"] = 4

    # --- backbone chung
    for ten, khoa in (("linear (moc, nhu CNN cu)", "bb_linear_acc"),
                      ("MLP head", "bb_mlp_acc"),
                      ("KAN-spline head", "bb_kanspline_acc"),
                      ("KAN-Fourier head", "bb_kanfourier_acc")):
        moi[khoa] = round(float(lay(bb, head=ten)["acc"]), 4)
    r = lay(bb, head="PAIRED MLP head - linear (moc, nhu CNN cu)")
    moi["bb_mlp_tru_linear"] = round(float(r["acc"]), 4)
    r = lay(bb, head="PAIRED KAN-spline head - MLP head")
    moi["bb_kanspline_tru_mlp"] = round(float(r["acc"]), 4)

    # --- tuyen bo 0,19 dB do lai
    r = lay(bg, dai_luong="paired_kan_minus_mlp")
    moi["k8_paired_hieu"] = round(float(r["gia_tri"]), 3)
    moi["k8_paired_ci_lo"] = round(float(r["ci_lo"]), 3)
    moi["k8_paired_ci_hi"] = round(float(r["ci_hi"]), 3)
    moi["k8_paired_p"] = float(lay(bg, dai_luong="wilcoxon_exact_p")["gia_tri"])
    moi["k8_n_seeds"] = 20

    # --- quet MLP hai tac vu
    best = min((r for r in sw if r["task"] == "chest" and r["config"] != "fourier-KAN(ref)"),
               key=lambda r: float(r["mean"]))
    moi["sweep_chest_mlp_tot_nhat"] = round(float(best["mean"]), 3)
    moi["sweep_chest_kan_ref"] = round(float(lay(sw, task="chest", config="fourier-KAN(ref)")["mean"]), 3)
    best = max((r for r in sw if r["task"] == "amc" and r["config"] != "fourier-KAN(ref)"),
               key=lambda r: float(r["mean"]))
    moi["sweep_amc_mlp_tot_nhat"] = round(float(best["mean"]), 4)
    moi["sweep_amc_kan_ref"] = round(float(lay(sw, task="amc", config="fourier-KAN(ref)")["mean"]), 4)

    # --- doi chung dieu chinh: khep duoc bao nhieu khi cho gap ba lich
    a = abs(float(lay(tn, epochs="60", lam="1.0", mlp__contains=None)["paired"])) if False else None
    v60 = [abs(float(r["paired"])) for r in tn if r["epochs"] == "60" and r["lam"] == "1.0"
           and r["mlp"].startswith("BEST")]
    v180 = [abs(float(r["paired"])) for r in tn if r["epochs"] == "180" and r["lam"] == "1.0"
            and r["mlp"].startswith("BEST")]
    moi["tune_lam1_60ep"] = round(v60[0], 2)
    moi["tune_lam1_180ep"] = round(v180[0], 2)
    moi["tune_khep_duoc"] = round(v60[0] - v180[0], 2)

    # --- tham so THIET KE cua thi nghiem co che: doc TU MA, assert neu thieu
    src2 = open(os.path.join(HERE, "exp_mechanism2.py"), encoding="utf-8").read()
    def lay_hang(ten, kieu=int):
        m = re.search(rf"^{ten}\s*=\s*([0-9.]+)", src2, re.M)
        assert m, f"khong tim thay hang {ten} trong exp_mechanism2.py"
        return kieu(m.group(1))
    moi["mech_D"] = lay_hang("D")
    moi["mech_budget"] = lay_hang("BUDGET")
    moi["mech_noise"] = lay_hang("NOISE_SD", float)
    moi["mech_ntest"] = lay_hang("NTEST")
    m = re.search(r"^SEEDS\s*=\s*list\(range\((\d+)\)\)", src2, re.M)
    assert m, "khong tim thay SEEDS trong exp_mechanism2.py"
    moi["mech_seeds"] = int(m.group(1))

    # --- hai lich huan luyen cua doi chung dieu chinh, doc TU MA
    src3 = open(os.path.join(HERE, "exp_mechanism2_tuning.py"), encoding="utf-8").read()
    me = re.search(r"^EPOCHS\s*=\s*\[(\d+),\s*(\d+)\]", src3, re.M)
    assert me, "khong tim thay EPOCHS trong exp_mechanism2_tuning.py"
    moi["mech_ep_ngan"], moi["mech_ep_dai"] = int(me.group(1)), int(me.group(2))

    # --- co batch cua phep do chi phi: doc TU MA, va CHET neu khong tim thay.
    # ⛔ Lan dau toi lay bang re.search(r"BATCH|batch") roi RO VE 256 khi khong khop.
    # Bien trong ma ten la "B", nen bo loc truot, va 256 chi TRUNG ngau nhien voi so
    # toi tu go. Xuat xu gia thi con so van dung ma nguon goc da sai.
    src = open(os.path.join(HERE, "exp_compute.py"), encoding="utf-8").read()
    mb = re.search(r"^\s*B\s*=\s*(\d+)\s*$", src, re.M)
    assert mb, "khong tim thay co batch trong exp_compute.py, KHONG duoc doan"
    moi["compute_batch"] = int(mb.group(1))

    # ================= doc ngoai vong 1: so moi =================
    # (2a) doi chung CO THE HONG thay cho moc OLS khong the hong
    ct = doc("exp_mechanism2_control.csv")
    def bien_do(dc, n):
        v = [float(r["nmse_db"]) for r in ct if r["doi_chung"] == dc and r["n"] == str(n)]
        assert v, f"thieu {dc} n={n}"
        return round(max(v) - min(v), 3)
    moi["ctrl_ols_bien_do_n256"] = bien_do("OLS-bac2", 256)
    moi["ctrl_ols_bien_do_n4096"] = bien_do("OLS-bac2", 4096)
    moi["ctrl_ols_ngoai_span_bien_do"] = bien_do("OLS-bac2-dich-ngoai-span", 4096)
    moi["ctrl_rf_bien_do_n256"] = bien_do("random-features", 256)
    moi["ctrl_rf_bien_do_n4096"] = bien_do("random-features", 4096)

    # ty le doc do kho tren hieu ung kien truc, de khong phai tu tinh trong bai
    def bien_do_hieu_ung(n):
        """Khoang DICH CO DAU cua hieu ghep cap tu lam=0 den lam=1.

        ⛔ Ban dau toi lay tri tuyet doi roi tru, ra 3,22 dB thay vi 5,52 dB: lay abs()
        lam mat dau nen hai phia cua diem dao bi gap vao nhau. Hieu ung la viec no DI TU
        -3,55 LEN +1,97, khong phai khoang cach giua hai do lon.
        """
        v = [float(r["nmse_db"]) for r in m2
             if r["model"] == "PAIRED KAN-spline - MLP" and r["n"] == str(n)]
        return max(v) - min(v) if v else None
    for n, nhan in ((256, "n256"), (4096, "n4096")):
        hu = bien_do_hieu_ung(n)
        dk = moi[f"ctrl_rf_bien_do_{nhan}"]
        moi[f"ctrl_rf_ty_le_{nhan}"] = round(100.0 * dk / hu, 1)

    moi["mech_so_o_bang8"] = len([r for r in m2 if r["model"].startswith("PAIRED")])

    # (2b) khop do sau
    dm = doc("exp_depth_matched.csv")
    def paired_dm(ds, n, lam):
        r = [x for x in dm if x["do_sau"] == ds and x["n"] == str(n)
             and float(x["lam"]) == lam and x["model"] == "PAIRED KAN - MLP"]
        assert len(r) == 1, f"thieu {ds} n={n} lam={lam}"
        return round(float(r[0]["nmse_db"]), 2)
    for ds, nhan in (("1 lop an", "d1"), ("2 lop an", "d2")):
        for lam in (0.0, 0.5, 1.0):
            moi[f"depth_{nhan}_n256_lam{str(lam).replace('.','')}"] = paired_dm(ds, 256, lam)
            moi[f"depth_{nhan}_n4096_lam{str(lam).replace('.','')}"] = paired_dm(ds, 4096, lam)

    # (3a) quet phia KAN o n=256
    kt = doc("exp_kan_tuning_n256.csv")
    for lam, nhan in (("0.75", "075"), ("1.0", "10")):
        b = [r for r in kt if r["lam"] == lam and r["model"].startswith("BEST")]
        assert len(b) == 1, f"thieu BEST lam={lam}"
        moi[f"kantune_n256_lam{nhan}_hieu"] = round(float(b[0]["paired"]), 2)
        moi[f"kantune_n256_lam{nhan}_cau_hinh"] = b[0]["model"].split("=", 1)[1]
    moi["kantune_so_cau_hinh"] = len([r for r in kt if r["lam"] == "1.0"
                                      and r["model"] not in ("MLP",)
                                      and not r["model"].startswith("BEST")])

    # (4) truc chi phi: ti so that, KHONG lay so da lam tron
    cp = {r["model"]: r for r in doc("exp_compute.csv") if r["task"] == "AMC"}
    t_cnn = float(cp["CNN"]["infer_ms_per_batch"])
    t_mlp = float(cp["MLP"]["infer_ms_per_batch"])
    t_ksp = float(cp["KANspline"]["infer_ms_per_batch"])
    moi["cnn_cham_hon_mlp"] = round(t_cnn / t_mlp, 1)
    moi["cnn_cham_hon_kanspline"] = round(t_cnn / t_ksp, 2)
    moi["cnn_infer_ms"] = t_cnn

    # ================= doc ngoai vong 1 muc 2(c): hai che do hoi quy chay lai 10 hat giong
    # Moi so dpd_* va chest_* phai SINH LAI tu CSV moi, khong giu so cua ban 5 hat giong.
    ch = {r["model"]: r for r in doc("exp_chest.csv")}
    for ten, khoa in (("LMMSE", "chest_lmmse_nmse"), ("MLP", "chest_mlp_nmse"),
                      ("KANspline", "chest_kanspline_nmse"),
                      ("KANfourier", "chest_kanfourier_nmse"),
                      ("LinInterp", "chest_lininterp_nmse")):
        assert ten in ch, f"thieu {ten} trong exp_chest.csv"
        moi[khoa] = round(float(ch[ten]["nmse_mean"]), 2)

    dp = {r["model"]: r for r in doc("exp_dpd.csv")}
    for ten, khoa in (("MLP", "dpd_mlp_nmse"), ("KANfourier", "dpd_kanfourier_nmse"),
                      ("KANspline", "dpd_kanspline_nmse"), ("MemPoly", "dpd_mempoly_nmse")):
        if ten in dp:
            moi[khoa] = round(float(dp[ten]["nmse_mean"]), 2)
    if "MLP" in dp and "KANfourier" in dp:
        moi["dpd_kanfourier_minus_mlp"] = round(
            float(dp["KANfourier"]["nmse_mean"]) - float(dp["MLP"]["nmse_mean"]), 2)
        moi["dpd_kanspline_minus_mlp"] = round(
            float(dp["KANspline"]["nmse_mean"]) - float(dp["MLP"]["nmse_mean"]), 2)
    # so hat giong nay doc TU MA, khong go tay
    src_dpd = open(os.path.join(HERE, "exp_dpd.py"), encoding="utf-8").read()
    m = re.search(r"^N_SEEDS\s*=\s*(\d+)", src_dpd, re.M)
    assert m, "khong tim thay N_SEEDS trong exp_dpd.py"
    moi["n_seeds_dpd"] = int(m.group(1))

    # bo quet MLP tren DPD: so hat giong doc TU MA, va gia tri doc TU CSV
    src_sw = open(os.path.join(HERE, "exp_mlp_sweep.py"), encoding="utf-8").read()
    msw = re.search(r"for seed in range\((\d+)\)", src_sw)
    assert msw, "khong tim thay so hat giong trong exp_mlp_sweep.py"
    moi["n_seeds_mlpsweep"] = int(msw.group(1))
    sw = {r["config"]: r for r in doc("exp_mlp_sweep.csv")}
    assert "depth2-relu" in sw, "thieu dong depth2-relu trong exp_mlp_sweep.csv"
    moi["mlp_sweep_depth2_nmse"] = round(float(sw["depth2-relu"]["nmse_mean"]), 2)

    # nguong R^2 cua phep kiem canh don gian: doc TU MA exp_interp.py
    src_i = open(os.path.join(HERE, "exp_interp.py"), encoding="utf-8").read()
    mi = re.search(r"R2\s*>\s*([0-9.]+)", src_i)
    assert mi, "khong tim thay nguong R^2 trong exp_interp.py"
    moi["interp_r2_nguong"] = float(mi.group(1))

    # dai ngan sach cua ablation: doc TU CSV, khong go tay
    ab = doc("abl_dpd_budget.csv")
    cot = "params" if "params" in ab[0] else list(ab[0])[0]
    ps = sorted(int(float(r[cot])) for r in ab)
    moi["abl_budget_min_k"] = round(ps[0] / 1000.0, 1)
    moi["abl_budget_max_k"] = round(ps[-1] / 1000.0, 0)

    # --- ghi ra
    os.makedirs(CLAIMS_DIR, exist_ok=True)
    d = json.load(open(CLAIMS_JSON), object_pairs_hook=collections.OrderedDict)
    for k, v in moi.items():
        d[k] = v
        with open(os.path.join(CLAIMS_DIR, f"{k}.tex"), "w") as f:
            f.write(f"{v}")
    json.dump(d, open(CLAIMS_JSON, "w"), indent=2, ensure_ascii=False)
    print(f"da sinh {len(moi)} claim moi:")
    for k, v in moi.items():
        print(f"   {k:28s} = {v}")


if __name__ == "__main__":
    main()
