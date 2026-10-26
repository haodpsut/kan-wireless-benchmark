#!/usr/bin/env bash
# Mot lenh: dich lai bai roi chay het cong. Tra ve khac 0 neu co cong DO hoac CHAN.
set -u
cd "$(dirname "$0")"
echo "== dich =="
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null 2>&1
ma_dich=$?
if [ $ma_dich -ne 0 ]; then
  echo "[CHAN] latexmk that bai, doc main.log:"
  grep -E "^! |^l\.[0-9]+" main.log | head -20
  exit 2
fi
echo "[XANH] dich xong: $(ls -l main.pdf | awk '{print $5}') byte"
echo
echo "== cong =="
xau=0
for c in cong/c1_dem_tu.py cong/c2_highlights.py cong/c3_hien_vat.py cong/c4_em_dash.py cong/c5_so_go_tay.py; do
  python3 "$c" || xau=$((xau+1))
done
echo
if [ $xau -eq 0 ]; then echo "TAT CA CONG XANH"; else echo "CO $xau CONG KHONG XANH, chua nop duoc"; fi
exit $xau
