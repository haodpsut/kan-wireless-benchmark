"""Bai tiem cho verify_numbers.py: cong phai BAT loi that va KHONG bat oan.

⛔ Ban dau tep nay neo ca tiem vao chuoi CUA LOI DANG CON TRONG BAI, nen sau khi
sua bai thi ca tiem "khong tiem duoc" va diem tu 4/4 tut xuong 2/4. Do la lop loi
"ca tiem neo vao chuoi vua bi xoa": bai tiem phai TU TAO ra khuyet diem tren ban
thao HIEN TAI, khong duoc dua vao viec khuyet diem con do.

Nay moi ca deu neo vao ban thao DA SUA va tu tiem loi vao.
"""
import os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, "..", "paper")
TEX = os.path.join(PAPER, "main.tex")

# neo: cau da sua, dung macro cuc tri
NEO = r"best KAN by \$\+\\input\{claims/amc_cnn_minus_bestkan\}\$"

CA = [
    ("A. ban thao HIEN TAI, da dung macro cuc tri", None, None, False),
    ("B. tiem lai dung loi lich su: 'best KAN' tro vao KAN TE NHAT",
     NEO, "best KAN by $+\\input{claims/amc_cnn_minus_kanfourier}$", True),
    ("C. doi chung AM: doi tu so sanh sang MLP thi khong duoc no",
     NEO, "best MLP by $+\\input{claims/amc_cnn_minus_kanfourier}$", False),
    ("D. tiem tren DPD theo chieu min: 'best KAN' tro vao KAN te nhat",
     r"\\section\{Interpretability Audit\}",
     "The best KAN reaches $\\input{claims/dpd_kanspline_nmse}$~dB.\n"
     "\\section{Interpretability Audit}", True),
    ("E. doi chung AM tren DPD: 'best KAN' tro vao KAN TOT NHAT theo chieu min",
     r"\\section\{Interpretability Audit\}",
     "The best KAN reaches $\\input{claims/dpd_kanfourier_nmse}$~dB.\n"
     "\\section{Interpretability Audit}", False),
]


def chay(tex_text):
    with tempfile.TemporaryDirectory() as td:
        shutil.copytree(PAPER, os.path.join(td, "paper"))
        open(os.path.join(td, "paper", "main.tex"), "w", encoding="utf-8").write(tex_text)
        code_dst = os.path.join(td, "code")
        os.makedirs(code_dst)
        shutil.copy(os.path.join(HERE, "verify_numbers.py"), code_dst)
        r = subprocess.run([sys.executable, "verify_numbers.py"], cwd=code_dst,
                           capture_output=True, text=True)
        return r.returncode != 0, r.stdout


def main():
    goc = open(TEX, encoding="utf-8").read()
    dat = 0
    for ten, tim, thay, ky_vong_do in CA:
        if tim is None:
            t = goc
        else:
            t, n = re.subn(tim, lambda m: thay, goc, count=1)
            if n == 0:
                print(f"  ⛔ {ten}\n       KHONG tiem duoc: mau neo khong khop"
                      f" => ca nay VO HIEU, phai sua bai tiem")
                continue
        do, _ = chay(t)
        ok = (do == ky_vong_do)
        dat += ok
        print(f"  {'✅' if ok else '⛔'} {ten}\n       ky vong"
              f" {'DO' if ky_vong_do else 'XANH'}, nhan {'DO' if do else 'XANH'}")
    print(f"\n{dat}/{len(CA)} ca dat")
    sys.exit(0 if dat == len(CA) else 1)


if __name__ == "__main__":
    main()
