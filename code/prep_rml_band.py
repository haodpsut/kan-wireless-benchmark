"""Loc dung dai SNR ma thi nghiem dung, de khoi phai day 611 MB len sol1.

Goi DUNG load_rml() cua exp_amc.py nen khong co duong du lieu thu hai: ket qua la
chinh thu ma ban da cong bo doc ra. In SHA-256 de doi chieu hai dau.
"""
import hashlib, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_amc import load_rml, SNR_LO, SNR_HI

X, y, mods = load_rml()
h = hashlib.sha256()
h.update(np.ascontiguousarray(X).tobytes()); h.update(np.ascontiguousarray(y).tobytes())
print(f"dai SNR [{SNR_LO},{SNR_HI}] dB | X{X.shape} {X.dtype} | y{y.shape} | lop={len(mods)}")
print("SHA-256 (X||y) =", h.hexdigest())
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "rml_band.npz")
np.savez(out, X=X, y=y, mods=np.array(mods))
print("ghi:", os.path.abspath(out), f"{os.path.getsize(out)/1e6:.1f} MB")
