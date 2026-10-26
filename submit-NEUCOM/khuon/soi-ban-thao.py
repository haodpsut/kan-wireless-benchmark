"""Chay 5 cong cua khuon Elsevier tren BAN THAO HIEN TAI (con la khuon IEEE).

Muc dich khong phai de no xanh, ma de co BAO CAO HONG: biet truoc phai sua gi khi
chuyen khuon. Cong nao CHAN vi thieu hien vat thi dung la ket qua can biet.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "cong"))
BAN_THAO = os.path.join(HERE, "..", "..", "paper", "main.tex")

import c1_dem_tu, c2_highlights, c3_hien_vat, c4_em_dash, c5_so_go_tay

PDF = os.path.join(HERE, "..", "..", "paper", "main.pdf")
# ⛔ CHAN cham tren ban dung CU: da xay ra mot lan, latexmk tra ma 12 ma cong van bao
# 6/6 xanh, vi chung doc main.pdf cua lan dung TRUOC.
if os.path.exists(PDF) and os.path.getmtime(PDF) < os.path.getmtime(BAN_THAO):
    print("⛔ CHAN: main.pdf CU HON main.tex. Dich lai truoc khi cham cong.")
    raise SystemExit(2)
print(f"ban thao: {os.path.relpath(BAN_THAO, HERE)}\n")
ma = []
ma.append(c1_dem_tu.chay(BAN_THAO))
ma.append(c2_highlights.chay(os.path.join(HERE, "..", "..", "paper", "highlights.tex")))
# Neurocomputing dat tran abstract 250 tu, khong phai 300 nhu mac dinh Elsevier chung
ma.append(c3_hien_vat.chay(BAN_THAO, han_abstract=250))
ma.append(c4_em_dash.chay(BAN_THAO, os.path.join(HERE, "..", "..", "paper", "main.pdf")))
ma.append(c5_so_go_tay.chay(BAN_THAO))

# cong 6: bang trong bai phai sinh lai duoc tu CSV. c5 bo qua tabular nen day la cho
# bao dam cho so trong bang, va no da duoc chung minh bang mot ca tiem (-99.99).
import subprocess
r = subprocess.run([sys.executable, os.path.join(HERE, "..", "..", "code",
                                                 "sinh_bang_ketqua.py"), "--kiem"],
                   capture_output=True, text=True)
print(r.stdout.strip().split("\n")[-1] if r.stdout.strip() else "c6: khong chay duoc")
ma.append(0 if r.returncode == 0 else 1)

print(f"\n=> {sum(1 for m in ma if m == 0)}/6 cong XANH,"
      f" {sum(1 for m in ma if m == 1)} DO, {sum(1 for m in ma if m == 2)} CHAN")
