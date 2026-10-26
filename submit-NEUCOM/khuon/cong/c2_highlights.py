#!/usr/bin/env python3
"""Cong 2: Highlights theo dung luat Elsevier: 3 den 5 gach, moi gach toi da 85 ky tu."""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chung

TOI_DA_KY_TU = 85
IT_NHAT, NHIEU_NHAT = 3, 5


def chay(tep="highlights.tex"):
    if not os.path.isfile(tep):
        return chung.in_ket("c2 highlights", chung.CHAN, ["khong thay %s" % tep])
    dong = chung.nap_tex(tep)
    van = "\n".join(d for _, _, d in dong)
    # Luat cua Elsevier la do dai BAN IN, nen macro phai duoc thay bang GIA TRI truoc
    # khi dem. Ban cu dem ca duong dan trong \input{claims/...} thanh ky tu, nen mot
    # gach 75 ky tu bi bao la 103. Da do duoc tren bai KAN-wireless.
    thu_muc = os.path.dirname(os.path.abspath(tep))
    def _gia_tri(m):
        d = m.group(1)
        if not d.endswith(".tex"):
            d += ".tex"
        duong = os.path.join(thu_muc, d)
        if os.path.isfile(duong):
            return open(duong, encoding="utf-8").read().strip()
        return m.group(0)
    van = re.sub(r"\\(?:input|include)\{([^}]+)\}", _gia_tri, van)
    muc = re.findall(r"\\item\s+(.+?)(?=\n\s*\\item|\n\s*\\end\{|\Z)", van, flags=re.S)
    muc = [re.sub(r"\s+", " ", m).strip() for m in muc if m.strip()]
    bao, loi = [], False
    if not (IT_NHAT <= len(muc) <= NHIEU_NHAT):
        bao.append("co %d gach, luat doi %d den %d" % (len(muc), IT_NHAT, NHIEU_NHAT))
        loi = True
    for i, m in enumerate(muc, 1):
        tho = chung.van_xuoi([("highlights.tex", i, m)], bo_moi_truong=False).strip()
        tho = re.sub(r"\s+", " ", tho)
        n = len(tho)
        dau = "DAI" if n > TOI_DA_KY_TU else "ok "
        bao.append("%s gach %d: %d ky tu" % (dau, i, n))
        if n > TOI_DA_KY_TU:
            loi = True
    return chung.in_ket("c2 highlights", chung.DO if loi else chung.XANH, bao)


if __name__ == "__main__":
    sys.exit(chay(*sys.argv[1:]))
