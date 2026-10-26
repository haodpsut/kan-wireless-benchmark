#!/usr/bin/env python3
"""Cong 3: hien vat Elsevier bat buoc, va phai DUOC DIEN, khong chi co mat.

Bai hoc da tra gia: cong kiem co mat muc ma khong kiem muc da dien thi xanh tren
mot bai con nguyen van mau.
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chung

MAU_CON_SOT = ["TODO", "FIXME", "XXX", "TBD", "DIA-CHI-KHO", "TIEU DE BAI",
               "Thay doan nay bang tom tat that", "tu khoa mot", "Noi dung muc",
               "Gach dau dong mot"]

DOI_MUC = {
    "CRediT": r"CRediT authorship contribution statement",
    "canh tranh loi ich": r"Declaration of competing interest",
    "san co du lieu": r"Data availability",
}


def chay(goc="main.tex", han_abstract=300):
    if not os.path.isfile(goc):
        return chung.in_ket("c3 hien vat", chung.CHAN, ["khong thay %s" % goc])
    dong = chung.nap_tex(goc)
    van = "\n".join(d for _, _, d in dong)
    bao, loi = [], False

    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", van, flags=re.S)
    if not m:
        bao.append("thieu abstract"); loi = True
    else:
        _tho = chung.van_xuoi([("abstract", 1, m.group(1))], bo_moi_truong=False)
        n = chung.dem_tu(chung.thay_macro_bang_mot_tu(_tho))
        bao.append("abstract %d tu (doi 80 den %d)" % (n, han_abstract))
        if not (80 <= n <= han_abstract):
            loi = True

    m = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", van, flags=re.S)
    if not m:
        bao.append("thieu keyword"); loi = True
    else:
        k = [x for x in re.split(r"\\sep", m.group(1)) if x.strip()]
        bao.append("keyword: %d tu khoa (doi it nhat 3)" % len(k))
        if len(k) < 3:
            loi = True

    for ten, mau in DOI_MUC.items():
        if not re.search(mau, van):
            bao.append("thieu muc: %s" % ten); loi = True

    for p in MAU_CON_SOT:
        for tep, so, d in dong:
            if p in d:
                bao.append("con van mau %r tai %s:%d" % (p, tep, so)); loi = True
                break
    return chung.in_ket("c3 hien vat", chung.DO if loi else chung.XANH, bao)


if __name__ == "__main__":
    sys.exit(chay(*sys.argv[1:]))
