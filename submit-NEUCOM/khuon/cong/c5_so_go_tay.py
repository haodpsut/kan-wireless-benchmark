#!/usr/bin/env python3
"""Cong 5: moi so trong van xuoi phai den tu MACRO, khong go tay.

Da tra gia: 49% so trong mot bai la go tay, va bang go tay thi chay lai thi
nghiem khong cap nhat. Cong nay bat ca so PHAN TRAM, vi do dung la cho bo do
cu cua nha bi mu.

Danh sach cho phep o so-cho-phep.txt duoc in ra moi lan, va neu no gach qua
NGUONG_CHO_PHEP muc thi cong canh bao: danh sach dang lam thay viec cua bai.
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chung

NGUONG_CHO_PHEP = 20
BO_LENH = ("cite", "citep", "citet", "ref", "eqref", "label", "includegraphics",
           "input", "include", "usepackage", "documentclass", "bibliography",
           "bibliographystyle", "graphicspath", "affiliation", "ead", "journal",
           "newcommand", "renewcommand", "author", "address", "url", "href",
           "DOI", "doi", "texttt", "verb")


def nap_cho_phep(tep):
    mau = []
    if os.path.isfile(tep):
        for d in open(tep, encoding="utf-8"):
            d = d.strip()
            if d and not d.startswith("#"):
                mau.append(re.compile(d))
    return mau


def chay(goc="main.tex"):
    if not os.path.isfile(goc):
        return chung.in_ket("c5 so go tay", chung.CHAN, ["khong thay %s" % goc])
    thu_muc = os.path.dirname(os.path.abspath(__file__))
    mau_cho = nap_cho_phep(os.path.join(thu_muc, "so-cho-phep.txt"))

    dong = chung.nap_tex(goc)
    # Tep SINH RA tu du lieu khong phai van xuoi: so trong do la ket qua, khong phai so
    # go tay. Nhan dien bang DAU O DONG DAU chu khong mien tru ca thu muc, de mot bang
    # go tay dat nham cho van bi kiem.
    tep_sinh_ra = {t for t, so, d in dong
                   if so == 1 and re.match(r"\s*%+\s*SINH TU\b", d)}
    # Cat lenh tren TOAN VAN (giu so dong) vi doi so co the trai nhieu dong.
    toan = "\n".join(d for _, _, d in dong)
    for ten in BO_LENH:
        toan = chung.bo_lenh_can_bang_giu_dong(toan, ten)
    sach = toan.split("\n")
    assert len(sach) == len(dong), "lech so dong sau khi cat lenh"

    bat, cho = [], []
    trong_moi_truong = False
    # So trong PREAMBLE la tuy chon goi va plumbing (12pt, \endlinechar=-1, ...), khong
    # bao gio la so do duoc. Da do duoc: \CatchFileDef{...}{\endlinechar=-1} bi bat oan.
    trong_than = False
    for (tep, so, _), s in zip(dong, sach):
        # Nguon so cua bai KHONG phai van xuoi: so.tex, hoac mot tep moi so trong claims/.
        # Bat chung la bao oan, va da do duoc: 12/53 lan bat o ban KAN-wireless.
        if "\\begin{document}" in s:
            trong_than = True
            continue
        if not trong_than:
            continue
        if tep.endswith("so.tex") or re.search(r"(^|[\\/])claims[\\/]", tep):
            continue
        if tep in tep_sinh_ra:
            continue
        # Moi truong toan HIEN THI trai nhieu dong: phai bo theo TRANG THAI, khong
        # the bo bang re.sub tung dong. Da do duoc: "10\\log_{10}" trong equation
        # bi bat oan vi bo loc chi xu ly $...$ mot dong.
        # So trong TABULAR la du lieu trinh bay, khong phai van xuoi. Bao dam dung cho
        # chung KHONG phai "den tu macro" ma la "sinh lai duoc tu CSV", va do la viec
        # cua mot cong rieng. Neu de c5 bat ca bang thi no bao 120 ca va khong ai doc.
        if re.search(r"\\begin\{(equation|align|gather|eqnarray|multline|displaymath|tabular)\*?\}", s):
            trong_moi_truong = True
        if trong_moi_truong:
            if re.search(r"\\end\{(equation|align|gather|eqnarray|multline|displaymath|tabular)\*?\}", s):
                trong_moi_truong = False
            continue
        # ⛔ Ban cu BO CA cum $...$, nen moi so go tay trong che do toan deu VO HINH.
        # Da do duoc tren ban KAN-wireless: "$0.8$--$2.2$~dB" va "$-16.27$" nam trong
        # chu thich hinh, deu la so DO DUOC, deu cu sau khi chay lai thi nghiem, va c5
        # khong thay gi. Nay: van bo LENH va ky hieu trong toan, nhung GIU lai so co
        # phan THAP PHAN, vi so nguyen trong toan thuong la chi so hay so mu, con so
        # thap phan gan nhu luon la mot phep do.
        def _loc_toan(m):
            noi = m.group(1)
            noi = re.sub(r"\\[A-Za-z]+", " ", noi)           # bo lenh
            noi = re.sub(r"[_^]\{?[0-9A-Za-z]*\}?", " ", noi)  # bo chi so va so mu
            # Trong che do toan, DAU PHAY la dau ngan cach danh sach: $[-6,6]$ la mot
            # khoang, khong phai so 6,6. Chi coi DAU CHAM la dau thap phan o day.
            giu = re.findall(r"-?[0-9]+\.[0-9]+", noi)
            return " " + " ".join(giu) + " "
        s = re.sub(r"\$([^$]*)\$", _loc_toan, s)
        s = re.sub(r"\\\[.*?\\\]", " ", s)
        # Ten dinh danh co chua so (RML2016.10a, IEEE802.11, GPT-4) khong phai so DO DUOC.
        # Da do duoc: "RML2016.10a" bi bat 4 lan o ban KAN-wireless.
        s = re.sub(r"[A-Za-z][A-Za-z0-9]*(?:[.\-][0-9A-Za-z]+)+", " ", s)
        # Dinh danh BAT DAU bang so cung khong phai so do duoc: 16-QAM, 4-PAM, 802.11.
        # Da do duoc: "16-QAM" bi bat oan o ban KAN-wireless.
        s = re.sub(r"[0-9]+(?:[.\-][0-9]+)*-[A-Za-z][A-Za-z0-9]*", " ", s)
        # Dinh danh dang SO LIEN CHU: 6G, 5G, 8PSK, 16QAM. Khong co dau noi nen bo loc
        # tren khong bat duoc, va chung dang bi DANH SACH CHO PHEP gach ho, lam danh
        # sach phong len 22 muc va che mat viec cua bai.
        s = re.sub(r"\b[0-9]+[A-Za-z]+[A-Za-z0-9]*", " ", s)
        # Nhan liet ke trong ngoac: "(1) Matched capacity." Day la cau truc van ban,
        # khong phai so do duoc. Cung ho voi cot LaTeX trong \multicolumn va \cmidrule.
        s = re.sub(r"\((?:[0-9]+|[ivxlc]+)\)", " ", s)
        # So trong TIEU DE MUC la so thu tu, khong phai so do duoc: "Regime 1: ...".
        # Chu thich bang thi NGUOC LAI: do dung la cho so do duoc an, nen KHONG bo.
        s = re.sub(r"\\(?:sub)*section\*?\{[^}]*\}", " ", s)
        s = re.sub(r"\\(?:multicolumn|cmidrule|cline|hhline)\s*(\([^)]*\))?\s*\{[^}]*\}", " ", s)
        s = re.sub(r"[0-9.]+\\(linewidth|textwidth|columnwidth|cm|mm|pt|in|em|ex)", " ", s)
        for m in re.finditer(r"(?<![A-Za-z0-9\\])[0-9]+(?:[.,][0-9]+)*%?", s):
            t = m.group(0)
            if any(p.match(t) for p in mau_cho):
                cho.append("%s:%d  %s" % (tep, so, t))
            else:
                bat.append("%s:%d  %s" % (tep, so, t))

    bao = ["so go tay bi bat: %d" % len(bat)]
    bao.extend("    " + b for b in bat[:15])
    bao.append("so duoc danh sach cho phep gach: %d (nguong canh bao %d)" % (len(cho), NGUONG_CHO_PHEP))
    if len(cho) > NGUONG_CHO_PHEP:
        bao.append("CANH BAO: danh sach cho phep dang gach qua nhieu, soi lai so-cho-phep.txt")
    if bat:
        return chung.in_ket("c5 so go tay", chung.DO, bao)
    return chung.in_ket("c5 so go tay", chung.XANH, bao)


if __name__ == "__main__":
    sys.exit(chay(*sys.argv[1:]))
