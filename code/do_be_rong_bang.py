"""Do BE RONG THAT cua moi bang va so voi be rong cot, bang \\typeout.

Vi sao can: bang tran khoi cot KHONG chac sinh canh bao Overfull. Ca da xay ra:
Table 8 rong 266,31pt trong cot 252,0pt, tran 14,31pt, ma log bao "7 Overfull, tat ca
in paragraph" nen tuong la dep, va nhin anh 110 dpi cung khong thay. Hao phat hien.

⇒ Khong doan, khong nhin bang mat: dung chinh TeX do roi so.
"""
import os, re, subprocess, sys, tempfile, json

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, "..", "paper")
NGUON = [os.path.join(PAPER, "main.tex")] + sorted(
    os.path.join(PAPER, "bang", f) for f in os.listdir(os.path.join(PAPER, "bang"))
    if f.endswith(".tex"))
LOP = r"\documentclass[final,5p,times]{elsarticle}"


def thay_macro(s, claims):
    def g(m):
        k = m.group(1).split("/")[-1]
        return str(claims.get(k, "0"))
    s = re.sub(r"\\protect\\input\{([^}]+)\}", g, s)
    return re.sub(r"\\input\{([^}]+)\}", g, s)


def lay_bang(duong, claims):
    """Boc CA THAN FLOAT, khong chi khoi tabular.

    ⛔ Ban dau ham nay chi boc \\begin{tabular}...\\end{tabular}, nen no BO QUA
    \\small va \\setlength{\\tabcolsep} dat NGAY TRUOC tabular trong float. Hau qua:
    sua bang cho nho lai ma bo do van bao y nguyen 271,31pt. Bo do noi doi thi te hon
    khong co bo do.
    """
    s = open(duong, encoding="utf-8").read()
    s = "\n".join(re.sub(r"(?<!\\)%.*", "", d) for d in s.split("\n"))
    ra = []
    for m in re.finditer(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", s, re.S):
        than = m.group(1)
        if "\\begin{tabular}" not in than:
            continue
        nhan = "?"
        lb = re.findall(r"\\label\{([^}]+)\}", than)
        if lb:
            nhan = lb[-1]
        # bo caption/label/centering, giu moi khai bao co anh huong be rong
        than = re.sub(r"\\caption\{(?:[^{}]|\{[^{}]*\})*\}", "", than, flags=re.S)
        than = re.sub(r"\\label\{[^}]*\}", "", than)
        than = re.sub(r"\\centering", "", than)
        ra.append((nhan, thay_macro(than.strip(), claims)))
    # bang nam ngoai moi truong float (hiem) van phai do
    ngoai = re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", "", s, flags=re.S)
    for m in re.finditer(r"\\begin\{tabular\}.*?\\end\{tabular\}", ngoai, re.S):
        ra.append(("(ngoai float)", thay_macro(m.group(0), claims)))
    return ra


def main():
    claims = json.load(open(os.path.join(PAPER, "claims.json")))
    bang = []
    for d in NGUON:
        for nhan, noi in lay_bang(d, claims):
            bang.append((os.path.basename(d), nhan, noi))
    if not bang:
        print("khong thay bang nao"); return 2

    tex = [LOP, r"\usepackage{amsmath,amssymb,booktabs,graphicx,url,multirow}",
           r"\begin{document}",
           r"\typeout{DOCOT columnwidth \the\columnwidth}",
           r"\typeout{DOCOT textwidth \the\textwidth}",
           r"\newsavebox{\hop}"]
    for i, (_, _, noi) in enumerate(bang):
        tex += [r"\savebox{\hop}{" + noi + "}",
                r"\typeout{DOBANG %d \the\wd\hop}" % i]
    tex += [r"\end{document}"]

    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "do.tex")
        open(f, "w", encoding="utf-8").write("\n".join(tex))
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode",
                            "-output-directory", td, f], capture_output=True, text=True)
        out = r.stdout
    cot = re.search(r"DOCOT columnwidth ([0-9.]+)pt", out)
    toan = re.search(r"DOCOT textwidth ([0-9.]+)pt", out)
    if not cot:
        print("⛔ khong doc duoc be rong cot, pdflatex loi:")
        print("\n".join(l for l in out.split("\n") if l.startswith("!"))[:400])
        return 2
    W, WT = float(cot.group(1)), float(toan.group(1))
    print(f"be rong COT = {W:.2f}pt | be rong TOAN TRANG = {WT:.2f}pt\n")
    rong = {int(m.group(1)): float(m.group(2))
            for m in re.finditer(r"DOBANG (\d+) ([0-9.]+)pt", out)}
    xau = 0
    for i, (tep, nhan, _) in enumerate(bang):
        w = rong.get(i)
        if w is None:
            print(f"  ⛔ {tep:22s} {nhan:22s} KHONG DO DUOC"); xau += 1; continue
        if w <= W:
            print(f"  ✅ {tep:22s} {nhan:22s} {w:7.2f}pt  (con thua {W-w:5.2f}pt)")
        elif w <= WT:
            print(f"  ⛔ {tep:22s} {nhan:22s} {w:7.2f}pt  VUOT COT {w-W:5.2f}pt"
                  f"  (vua TOAN TRANG, dung table*)")
            xau += 1
        else:
            print(f"  ⛔ {tep:22s} {nhan:22s} {w:7.2f}pt  VUOT CA TOAN TRANG {w-WT:5.2f}pt")
            xau += 1
    print(f"\n=> {len(bang)-xau}/{len(bang)} bang vua cot")
    return 1 if xau else 0


if __name__ == "__main__":
    sys.exit(main())
