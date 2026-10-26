"""Ham dung chung cho cac cong. Khong co logic phan quyet o day."""
import os, re, sys

XANH, DO, CHAN = 0, 1, 2

MOI_TRUONG_BO = ("equation", "equation*", "align", "align*", "gather", "gather*",
                 "eqnarray", "eqnarray*", "tabular", "table", "table*", "figure",
                 "figure*", "verbatim", "lstlisting", "minted", "thebibliography",
                 "itemize_RETAIN_PLACEHOLDER")

GIU_DOI_SO = {"section", "subsection", "subsubsection", "paragraph", "emph",
              "textbf", "textit", "texttt", "caption", "title", "item"}


def bo_chu_thich(dong):
    """Bo phan chu thich LaTeX, giu lai \\% da escape."""
    ra, i = [], 0
    while i < len(dong):
        c = dong[i]
        if c == "\\" and i + 1 < len(dong):
            ra.append(dong[i:i + 2]); i += 2; continue
        if c == "%":
            break
        ra.append(c); i += 1
    return "".join(ra)


def nap_tex(goc):
    """Doc main.tex va moi \\input/\\include, tra ve [(tep, so_dong, noi_dung)].

    Bo qua chu thich. Khong theo \\input nam trong chu thich.
    """
    goc = os.path.abspath(goc)
    thu_muc = os.path.dirname(goc)
    ket, da_doc = [], set()

    def di(duong):
        duong = os.path.abspath(duong)
        if duong in da_doc or not os.path.isfile(duong):
            return
        da_doc.add(duong)
        ten = os.path.relpath(duong, thu_muc)
        with open(duong, encoding="utf-8") as f:
            for k, dong_tho in enumerate(f, 1):
                dong = bo_chu_thich(dong_tho.rstrip("\n"))
                # ⛔ Loi da tra gia: ban cu gap BAT KY dong co \input thi nap tep con
                # roi "continue", tuc BO CA DONG. Ma bai cua nha dat macro so NGAY GIUA
                # van xuoi (\input{claims/x}), nen moi dong co so bi xoa khoi tam nhin
                # cua MOI cong: c1/c3 dem thieu tu, va c5 KHONG THAY so go tay nam cung
                # dong voi mot macro. Nay chi nap tep con khi dong CHI CO mot lenh input.
                chi_co_input = re.fullmatch(r"\s*\\(?:input|include)\{([^}]+)\}\s*", dong)
                if chi_co_input:
                    con = chi_co_input.group(1)
                    if not con.endswith(".tex"):
                        con += ".tex"
                    di(os.path.join(thu_muc, con))
                    continue
                ket.append((ten, k, dong))

    di(goc)
    return ket


def _bo_moi_truong(van):
    for ten in MOI_TRUONG_BO:
        if ten.endswith("PLACEHOLDER"):
            continue
        van = re.sub(r"\\begin\{%s\}.*?\\end\{%s\}" % (re.escape(ten), re.escape(ten)),
                     " ", van, flags=re.S)
    return van


def van_xuoi(dong_list, bo_moi_truong=True):
    """Gop thanh van xuoi: bo toan, bo moi truong khong phai chu, bo lenh."""
    van = "\n".join(d for _, _, d in dong_list)
    if bo_moi_truong:
        van = _bo_moi_truong(van)
    van = re.sub(r"\$\$.*?\$\$", " ", van, flags=re.S)
    van = re.sub(r"\$[^$]*\$", " ", van)
    van = re.sub(r"\\\(.*?\\\)", " ", van, flags=re.S)
    van = re.sub(r"\\\[.*?\\\]", " ", van, flags=re.S)
    # lenh khong giu doi so: bo ca lenh va moi {..} [..] ngay sau
    def thay(m):
        ten = m.group(1)
        return " " + m.group(0)[len("\\" + ten):] if ten in GIU_DOI_SO else " "
    for _ in range(6):
        van_moi = re.sub(r"\\([A-Za-z@]+)\*?(?:\[[^\]]*\])*", thay, van)
        if van_moi == van:
            break
        van = van_moi
    van = van.replace("{", " ").replace("}", " ")
    van = re.sub(r"\\[^A-Za-z]", " ", van)
    return van


def dem_tu(van):
    return len(re.findall(r"[A-Za-z\u00C0-\u1EF9][A-Za-z\u00C0-\u1EF9'\-]*", van))


def in_ket(ten, trang_thai, cac_dong):
    nhan = {XANH: "XANH", DO: "DO", CHAN: "CHAN"}[trang_thai]
    print("[%s] %s" % (nhan, ten))
    for d in cac_dong:
        print("    " + d)
    return trang_thai


def bo_lenh_can_bang(s, ten):
    """Bo \\ten, moi doi so [..] va mot khoi {..} CAN BANG ngoac.

    Bo loc \\{[^}]*\\} cat do khi doi so long ngoac, vi du
    \\affiliation[dau]{organization={X}, addressline={566 ...}} se con lai so nha.
    """
    ra, i, mau = [], 0, "\\" + ten
    while i < len(s):
        if s.startswith(mau, i) and not (i + len(mau) < len(s) and s[i + len(mau)].isalpha()):
            i += len(mau)
            if i < len(s) and s[i] == "*":
                i += 1
            while i < len(s) and s[i] == "[":
                sau = s.find("]", i)
                if sau < 0:
                    return "".join(ra)
                i = sau + 1
            if i < len(s) and s[i] == "{":
                sau, do_sau = i + 1, 1
                while sau < len(s) and do_sau:
                    if s[sau] == "\\":
                        sau += 2; continue
                    if s[sau] == "{":
                        do_sau += 1
                    elif s[sau] == "}":
                        do_sau -= 1
                    sau += 1
                i = sau
            ra.append(" ")
            continue
        ra.append(s[i]); i += 1
    return "".join(ra)


def bo_lenh_can_bang_giu_dong(s, ten):
    """Nhu bo_lenh_can_bang nhung THAY bang dung so dong da bo, de giu so dong khop.

    Can cho cong quet theo dong: \\affiliation{...} co the trai nhieu dong.
    """
    ra, i, mau = [], 0, "\\" + ten
    while i < len(s):
        if s.startswith(mau, i) and not (i + len(mau) < len(s) and s[i + len(mau)].isalpha()):
            bat_dau = i
            i += len(mau)
            if i < len(s) and s[i] == "*":
                i += 1
            while i < len(s) and s[i] == "[":
                sau = s.find("]", i)
                if sau < 0:
                    i = len(s); break
                i = sau + 1
            if i < len(s) and s[i] == "{":
                sau, do_sau = i + 1, 1
                while sau < len(s) and do_sau:
                    if s[sau] == "\\":
                        sau += 2; continue
                    if s[sau] == "{":
                        do_sau += 1
                    elif s[sau] == "}":
                        do_sau -= 1
                    sau += 1
                i = sau
            ra.append("\n" * s.count("\n", bat_dau, i))
            continue
        ra.append(s[i]); i += 1
    return "".join(ra)


def thay_macro_bang_mot_tu(van, dau="0"):
    """Doi moi \\input{...} NAM TRONG VAN XUOI thanh mot tu, de dem tu cho DUNG.

    Vi sao can: macro so la mot tu khi IN RA. Neu bo han thi dem THIEU, va voi
    Neurocomputing (abstract <= 250 tu) thi dem thieu la nguy: bai 249 tu theo cong
    co the la 259 tu tren ban in.
    """
    return re.sub(r"\\(?:input|include)\{[^}]*\}", " %s " % dau, van)
