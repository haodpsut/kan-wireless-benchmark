#!/usr/bin/env python3
"""Cong 4: khong em-dash, do tren HAI nguon: nguon .tex va BAN IN pdf.

Do mot nguon la khong du: ban in co the mang em-dash do goi sinh ra
(vi du --- trong .tex bien thanh mot dau trong pdf).
Khong co pdf thi CHAN, khong phai xanh.
"""
import sys, os, re, subprocess, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chung


def chay(goc="main.tex", pdf="main.pdf"):
    if not os.path.isfile(goc):
        return chung.in_ket("c4 em-dash", chung.CHAN, ["khong thay %s" % goc])
    dong = chung.nap_tex(goc)
    bao, loi, chan = [], False, False

    hit_tex = []
    for tep, so, d in dong:
        if "\u2014" in d:
            hit_tex.append("%s:%d em-dash that" % (tep, so))
        if re.search(r"(?<!-)---(?!-)", d):
            hit_tex.append("%s:%d chuoi --- sinh ra em-dash" % (tep, so))
    bao.append("nguon tex: %d cho" % len(hit_tex))
    bao.extend("    " + h for h in hit_tex[:10])
    if hit_tex:
        loi = True

    if not os.path.isfile(pdf):
        bao.append("nguon pdf: CHAN, chua co %s" % pdf)
        chan = True
    elif not shutil.which("pdftotext"):
        bao.append("nguon pdf: CHAN, khong co pdftotext")
        chan = True
    else:
        van = subprocess.run(["pdftotext", pdf, "-"], capture_output=True, text=True).stdout
        n = van.count("\u2014")
        bao.append("nguon pdf: %d em-dash" % n)
        if n:
            loi = True

    if loi:
        return chung.in_ket("c4 em-dash", chung.DO, bao)
    if chan:
        return chung.in_ket("c4 em-dash", chung.CHAN, bao)
    return chung.in_ket("c4 em-dash", chung.XANH, bao)


if __name__ == "__main__":
    sys.exit(chay(*sys.argv[1:]))
