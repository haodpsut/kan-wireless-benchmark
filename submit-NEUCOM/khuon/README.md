# Khuôn Elsevier + cổng kiểm

Dựng 10/10/2026. Dùng cho **mọi bài nộp Elsevier và Springer** trong kế hoạch thu nhập:
hai bài solo Q1 và dây chuyền bài 2 tác giả.

> ⛔ **ĐÂY KHÔNG PHẢI PIPELINE.** Pipeline viết paper có **đúng một nhà** ở
> `working/viet-paper-chuan-dph/`. Thư mục này là **bộ khuôn theo nơi nhận**, cùng loại với
> `paper-lab/phong-cach-ieee-phan-tich/`. Đừng copy script của pipeline vào đây, và đừng
> coi đây là bản thứ hai của pipeline. Lớp lỗi 30 sinh ra từ đúng việc đó.

## Dùng thế nào

```bash
cp -R paper-lab/khuon-elsevier paper-lab/hao-paper-2026/<bai>/elsevier
cd paper-lab/hao-paper-2026/<bai>/elsevier
bash dung-lai.sh          # dịch rồi chạy hết cổng
python3 tiem-loi.py       # thử lại chính các cổng, phải ra 15/15
```

`dung-lai.sh` trả về khác 0 khi còn cổng chưa xanh, nên cắm được vào bất cứ chỗ nào cần chặn.

## Vì sao khuôn này tồn tại

Bốn chỗ Elsevier khác IEEE, mỗi chỗ sai là mất một vòng nộp:

1. **"12 trang hai cột" không tồn tại.** Bản nộp là **một cột, giãn dòng đôi** (tuỳ chọn
   `review` của `elsarticle`), nên trang không đo được gì. Đo bằng **số từ**: 8.000 đến 10.000.
2. Hiện vật IEEE không đòi mà Elsevier đòi: **Highlights** (3 đến 5 gạch, mỗi gạch tối đa
   **85 ký tự**), **CRediT** (bài một tác giả vẫn phải khai), Declaration of competing interest,
   Data availability.
3. Đổi lớp LaTeX từ `IEEEtran` sang `elsarticle` đã từng làm vỡ bốn thứ. Khuôn này dịch được
   sẵn nên không phải đổi lớp giữa lúc đã có hình.
4. Mỗi tạp chí Elsevier có trang **Insights** công bố review time và acceptance rate thật.
   Đọc từ mạng trường, vì ScienceDirect chặn bot (403).

## Năm cổng

| cổng | chặn điều gì | ngưỡng |
|---|---|---|
| `c1_dem_tu.py` | bài quá ngắn cho Elsevier | chữ chạy **≥ 8.000 từ**, trên 10.000 chỉ cảnh báo |
| `c2_highlights.py` | Highlights sai luật | **3 đến 5** gạch, mỗi gạch **≤ 85 ký tự** |
| `c3_hien_vat.py` | thiếu hiện vật, hoặc **có mục mà chưa điền** | abstract 80 đến 300 từ, ≥ 3 từ khoá, đủ ba mục khai báo, **không còn văn mẫu** |
| `c4_em_dash.py` | em-dash, đo trên **hai nguồn** | 0 trong `.tex` (kể cả chuỗi `---`) **và** 0 trong bản in PDF |
| `c5_so_go_tay.py` | số gõ tay trong văn xuôi | mọi số phải đến từ macro trong `so.tex` |

**Ba chỗ cố ý làm khác bộ đo cũ:**

- `c4` đo **cả bản in**, không chỉ nguồn. Một nguồn là không đủ: `---` trong `.tex` sạch về mặt
  ký tự nhưng bản in ra em-dash thật.
- `c4` trả **CHẶN** khi chưa có PDF, không trả xanh. Kiểm 0 đơn vị không phải là sạch.
- `c5` bắt **cả số phần trăm**, vì đó đúng là chỗ bộ đo cũ của nhà bị mù trong suốt thời gian
  nó báo "0 số không truy được". Danh sách cho phép ở `cong/so-cho-phep.txt` được **in ra mỗi
  lần chạy**, và cổng tự cảnh báo nếu danh sách đó gạch quá 20 mục, để nó không lặng lẽ làm
  thay việc của bài.

## Bộ tiêm lỗi: 15 ca, đã chạy

`tiem-loi.py` dựng một bài Elsevier **hợp lệ hoàn chỉnh** trong thư mục tạm rồi tiêm lỗi vào.
Mỗi cổng có **ca phải bắt** và **ca không được bắt oan**, vì siết cổng mà không đo báo động giả
thì chỉ đổi một lỗi lấy một lỗi khác.

```
15/15 ca ĐẠT (10/10/2026)
```

Hai ca đáng nêu riêng:

- `c4 bắt --- và BẢN IN cũng phải thấy`: ca này kiểm **nhánh PDF còn sống**, không chỉ kiểm kết
  luận cuối. Nếu nhánh PDF chết thì cổng vẫn ra ĐỎ nhờ nhánh tex, và lỗi sẽ không ai thấy.
- `c5 KHÔNG bắt năm tháng trong văn xuôi`: ca báo động giả. "công bố năm 2026" phải đi qua được.

**Phép thử chính bộ tiêm (chạy 10/10):** cố ý làm hỏng `c5` cho nó luôn trả XANH, bộ tiêm ra
**14/15 và chỉ đúng ca bị hỏng**. Tức bộ tiêm phân biệt được, không phải luôn in ĐẠT.

## Cổng này KHÔNG kiểm gì

Nói ra để đừng tưởng xanh là xong:

- **Hình**: không kiểm. Dùng `working/viet-paper-chuan-dph/figure-kit` và luật đo trên **ảnh
  render**, vì log LaTeX 0 Overfull không thấy được đường đè chữ.
- **Tham khảo**: không mở nguồn gốc để xác minh. Việc đó vẫn thủ công.
- **Nội dung**: không kiểm tuyên bố có khớp số hay không. Đó là việc của G3 và của đọc ngoài.
- **Phạm vi tạp chí**: không kiểm. Đọc danh sách loại trừ của venue trước, ở vị trí 0.
