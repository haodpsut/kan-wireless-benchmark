#!/usr/bin/env python3
"""Cong 1: so TU chu chay, khong phai so TRANG.

Elsevier khong co gioi han trang, va ban nop la mot cot gian dong doi, nen
"12 trang hai cot" khong do duoc. Muc tieu: 8000 den 10000 tu.
Thieu thi CHAN viec nop. Vuot thi chi canh bao, vi Elsevier khong cam dai.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chung

TOI_THIEU = 8000
MUC_TREN = 10000


def chay(goc="main.tex"):
    if not os.path.isfile(goc):
        return chung.in_ket("c1 so tu", chung.CHAN, ["khong thay %s" % goc])
    dong = chung.nap_tex(goc)
    # macro so la MOT tu khi in ra; bo han thi dem thieu
    sotu = chung.dem_tu(chung.thay_macro_bang_mot_tu(chung.van_xuoi(dong)))
    bao = ["chu chay: %d tu (muc tieu %d den %d)" % (sotu, TOI_THIEU, MUC_TREN)]
    if sotu < TOI_THIEU:
        bao.append("THIEU %d tu. Ban 6 trang IEEE thuong chi khoang 3500 tu." % (TOI_THIEU - sotu))
        return chung.in_ket("c1 so tu", chung.DO, bao)
    if sotu > MUC_TREN:
        bao.append("vuot muc tren, chi la canh bao")
    return chung.in_ket("c1 so tu", chung.XANH, bao)


if __name__ == "__main__":
    sys.exit(chay(*sys.argv[1:]))
