"""Cong kiem so cua bai: doi chieu van xuoi voi claims.json.

⚠ Tep nay LE RA phai co tu truoc: `claims.json` tu khai *"verify_numbers.py checks the
paper text against these"*, nhung trong kho KHONG he co no. Day la lop loi "tieu chi
VIET ra ma khong CAI dat".

Bon phep kiem:
  K1  moi \\input{claims/X} trong bai phai co X trong claims.json va co tep claims/X.tex
  K2  noi dung claims/X.tex phai khop gia tri trong claims.json
  K3  moi khoa trong claims.json nen duoc dung o dau do trong bai (canh bao neu mo)
  K4  ⭐ RANG BUOC SO SANH BAC NHAT: cau nao chua "best/worst/largest/..." ma dung mot
      macro tro vao MOT thanh vien cu the cua mot nhom, thi thanh vien do phai dung la
      cuc tri cua nhom theo chieu da khai trong claims.json["_groups"].

K4 la cong sinh ra tu ca that: mot ban truoc viet "the CNN leads the BEST KAN by
+0.309" trong khi 0,309 = CNN tru KAN-FOURIER, ma Fourier la KAN TE NHAT tren AMC
(0,349 so voi spline 0,468). Khoang cach that la 0,190. Phan bien 2 bat duoc, va bai
bi reject. Ba cong K1-K3 deu XANH tren cau do: con so co that, macro ten dung, gia tri
khop CSV. Chi co CHU "best" la sai.
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, "..", "paper")
TEX = os.path.join(PAPER, "main.tex")
CLAIMS = os.path.join(PAPER, "claims.json")
SUPERLATIVES = r"\b(best|worst|largest|smallest|fastest|slowest|highest|lowest|strongest|weakest)\b"


def doc_claims():
    with open(CLAIMS) as f:
        d = json.load(f)
    groups = d.pop("_groups", {})
    meta = {k: v for k, v in d.items() if k.startswith("_")}
    vals = {k: v for k, v in d.items() if not k.startswith("_")}
    return vals, groups, meta


def cau_cua(text):
    """Cat van xuoi thanh cau, giu nguyen macro."""
    t = re.sub(r"%.*", "", text)
    return re.split(r"(?<=[.;])\s+", t)


def nap_ca_cay():
    """Doc main.tex VA moi tep duoc \\input vao, dung bo nap cua khuon nha.

    ⛔ Ban dau ham main() doc THANG main.tex, nen cong nay:
      - bao oan "khoa khong dung trong bai" voi macro nam trong bang SINH RA, va
      - K4 (rang buoc so sanh bac nhat) KHONG BAO GIO soi cac tep duoc nap vao,
        tuc chinh cho de an loi nhat. Dung lop loi "cong cham mot hien vat khi co nhieu".
    """
    khuon = os.path.join(HERE, "..", "submit-NEUCOM", "khuon", "cong")
    if os.path.isdir(khuon):
        sys.path.insert(0, os.path.abspath(khuon))
        import chung
        return "\n".join(d for _, _, d in chung.nap_tex(TEX))
    print("  canh bao: khong thay bo nap cua khuon, chi doc main.tex (KHONG di theo \\input)")
    return open(TEX, encoding="utf-8").read()


def main():
    vals, groups, _ = doc_claims()
    tex = nap_ca_cay()
    dung = re.findall(r"\\input\{claims/([A-Za-z0-9_]+)\}", tex)
    loi, canh = [], []

    # K1
    for k in sorted(set(dung)):
        if k not in vals:
            loi.append(f"K1: bai dung claims/{k} ma claims.json KHONG co khoa '{k}'")
        if not os.path.exists(os.path.join(PAPER, "claims", f"{k}.tex")):
            loi.append(f"K1: thieu tep claims/{k}.tex")

    # K2
    for k in sorted(set(dung) & set(vals)):
        p = os.path.join(PAPER, "claims", f"{k}.tex")
        if os.path.exists(p):
            got = open(p).read().strip()
            want = str(vals[k])
            if got.rstrip("0").rstrip(".") != want.rstrip("0").rstrip(".") and got != want:
                loi.append(f"K2: claims/{k}.tex = '{got}' nhung claims.json = '{want}'")

    # K3
    for k in sorted(set(vals) - set(dung)):
        canh.append(f"K3: khoa '{k}' co trong claims.json ma khong dung trong bai")

    # K4: rang buoc so sanh bac nhat
    if not groups:
        canh.append("K4: claims.json KHONG khai '_groups' => cong so sanh bac nhat KHONG chay")
    for c in cau_cua(tex):
        if not re.search(SUPERLATIVES, c, re.I):
            continue
        macros = re.findall(r"\\input\{claims/([A-Za-z0-9_]+)\}", c)
        if not macros:
            continue
        for gname, g in groups.items():
            direction = g.get("direction", "max")
            members = g["members"]                      # ten ngan -> khoa claims
            prefix = g.get("prefix", "")                # chong RO CHEO giua cac tac vu
            for sub_name, sub in (g.get("subsets") or {}).items():
                # cong chi no khi tu so sanh bac nhat DINH NGAY vao ten nhom,
                # vi du "best KAN" / "worst KAN", chu khong phai moi cau co chu "best"
                mt = re.search(rf"{SUPERLATIVES}[\s~-]+{sub_name}\b", c, re.I)
                if not mt:
                    continue
                # chi xet macro nam SAU cum so sanh: do la con so ma tu ay CHI PHOI.
                # macro dung truoc no thuoc menh de khac (do la nguon bao dong gia
                # da do duoc: 5/6 lan no o ban dau cua cong nay).
                sau = c[mt.start():]
                macros_sau = re.findall(r"\\input\{claims/([A-Za-z0-9_]+)\}", sau)
                hit = [(m, s) for m in macros_sau if m.startswith(prefix)
                       for s in sub if s in m]
                if not hit:
                    continue
                best = (max if direction == "max" else min)(sub, key=lambda s: vals[members[s]])
                for m, s in hit:
                    if s != best:
                        loi.append(
                            f"K4: cau co tu so sanh bac nhat va nhac '{sub_name}', dung macro "
                            f"'{m}' tro vao '{s}' (= {vals[members[s]]}), nhung cuc tri "
                            f"({direction}) cua '{sub_name}' la '{best}' (= {vals[members[best]]}).\n"
                            f"      cau: {' '.join(c.split())[:140]}")

    for c in canh:
        print("  canh bao:", c)
    if loi:
        print(f"\n⛔ DO: {len(loi)} loi")
        for e in loi:
            print("  -", e)
        sys.exit(1)
    print(f"\n✅ DAT: {len(set(dung))} macro duoc kiem, {len(canh)} canh bao")


if __name__ == "__main__":
    main()
