# Kế hoạch sửa bài KAN-wireless và nộp Neurocomputing

Lập **25/10/2026**, sau khi `TCCN-TP-26-1667` bị **EiC reject 25/10** (nộp 03/08, 83 ngày, 3 phản
biện, 18 điểm). Bài **MỘT tác giả**. Nguồn bản thảo: `../paper/main.tex` (6.524 từ, 7 bảng,
10 hình, có `claims.json`). Gói đã nộp TCCN: `../NOP-TCCN/`.
Khuôn nơi nhận: `KHUON-NEUCOM.md` (đo từ bài publish + Guide for Authors).

## 0. Venue: Neurocomputing, Hảo chốt 25/10. Bằng chứng và rủi ro

| đo được | giá trị |
|---|---|
| phí đường thuê bao | **0đ** (*no publication fee*), APC 2.660 USD chỉ khi tự bấm OA ⇒ **đừng bấm** |
| hạng | SJR 2025 **1,465**, **Q1** ngành AI liên tục 2016-2025 |
| nhịp | **8 ngày** tới quyết định đầu · 62 ngày sau phản biện · 139 ngày tới nhận |
| tiền lệ **phương pháp KAN** | **32 bài** từ 2025 |
| ⚠ tiền lệ **KAN cho KHÔNG DÂY** | **0 bài** |
| ⚠ tiền lệ **mô hình hoá PA / DPD** | **0 bài** |

⇒ **Rủi ro duy nhất là PHẠM VI**, và nhà đã chết **6 lần** vì đúng lớp này. Cách trung hoà đã ghi
trong `KHUON-NEUCOM.md`: phép thử của tạp chí là *"directly relevant to neural networks or learning
systems"*, nên bài phải viết với **KAN là chủ thể**, ba tác vụ không dây chỉ là **bàn thử để tách
biến**. Viết ngược lại là rơi vào mục loại trừ *pure combination of existing algorithms*.

**Article type: Type 1 Regular article.** Giữ **Type 4 Replication study** làm đường lùi nếu phản
biện lại nói novelty thấp: tạp chí **công khai nhận replication**, nên lời bác *"matched-capacity
KAN-đối-MLP đã có trong [6]"* mất lực ở venue này.

⚠ Occupied-check phải chạy trước khi nộp: `../../kan-wireless-survey/` là bài KAN khác của Hảo, và
`VT-2026-06251` ở TVT đang chờ. TVT là nơi đã đăng nhiều bài KAN-không-dây (KAN-MAE, Adaptive
Transfer DPD) ⇒ nếu Neurocomputing trả về thì TVT **không** phải đường lùi sạch.

## 1. Ba nhóm việc, xếp theo thứ tự PHẢI làm trước

### P0 — Lỗi thật và việc rẻ. Không nộp nếu còn một ô đỏ

| # | điểm phản biện | việc | cổng kiểm |
|---|---|---|---|
| P0.1 | ⛔⛔ **R2.7a** | Bản nộp viết *"the CNN leads the **best** KAN by +0.309"*, mà Bảng III: CNN **0,658**, KAN spline (tốt nhất) **0,468**, KAN Fourier (tệ nhất) **0,349**. Khoảng cách thật **0,190**. **Sửa số và sửa cách sinh số**: macro phải sinh từ `argmax` của nhóm KAN, không phải số dán | cổng mới: mọi từ so sánh bậc nhất (*best/worst/largest*) phải **ràng buộc vào hàng cực trị**, xem lớp lỗi 25/10 |
| P0.2 | ⛔ **R2.7b** | Mọi tuyên bố **giao nhau** phải in **khoảng tin cậy ghép cặp** và sd theo hạt giống. Hiệu 0,19 dB trong khi sd ±0,06 tới ±0,12 là **1,5 tới 3 sd** ⇒ hoặc in CI ghép cặp, hoặc hạ cấp thành *không phân biệt được* | `analysis/rules.py` (sign test, Wilcoxon chính xác, bootstrap CI, Holm) đã có, khớp scipy 1e-9 |
| P0.3 | **R1.4 + R2.5** | Bỏ chữ *"lower bound"* cho suy giảm 1,17 dB do trích xuất ký hiệu, **hoặc** chứng minh nó. Rẻ nhất: hạ cấp thành *"đo được trên lớp đầu, chưa đặc trưng hoá toàn mạng"*, và thêm 2 lớp nữa nếu chạy kịp | cổng tuyên bố: mỗi câu mạnh phải có số và phạm vi |
| P0.4 | **R3.4** + Guide | Link mã **vào abstract** (bài mẫu làm đúng thế) **và** nạp **Zenodo lấy DOI**, vì Guide bắt Option C: GitHub một mình **không đủ** | kiểm link sống từ máy ngoài |
| P0.5 | **R3.5** | Thêm **bảng tham số mô phỏng** đầy đủ: Saleh PA, RML2016.10a, OFDM, và mọi siêu tham số huấn luyện | đối chiếu bảng với `code/harness.py`, không gõ tay |
| P0.6 | **R2.6 + R3.1 + R3.2** | Gộp lại đóng góp (đang quá vụn), **mở rộng Introduction**, **bỏ công thức và ký hiệu khỏi Related Work**, dồn `sec:protocol` vào phần thực nghiệm, gom 3 mục chế độ + ablation + compute lại. Sửa thụt đầu dòng, bớt dấu chấm phẩy và ngoặc kép | đếm mục cấp 1: hiện **16**, mục tiêu **≤ 9** |
| P0.7 | khuôn | Chuyển `elsarticle` hai cột, đánh số mục 1/1.1, **Highlights 3-5 gạch ≤ 85 ký tự tệp riêng**, abstract **≤ 250 từ đo trên bản in**, CRediT, competing interests tệp `.doc` riêng, **khai dùng AI**, Data availability, **Vitae ≤ 100 từ + ảnh**, tách 10 hình thành 10 tệp, bỏ kẻ dọc trong 7 bảng | `paper-lab/khuon-elsevier/` 5 cổng, tiêm 15/15 · cổng em-dash hai nguồn · cổng `Missing character` |

### P1 — Việc quyết định sống chết của bản nộp: MỘT mục cơ chế

**Ba phản biện hỏi cùng một câu** (R1.5, R2.2, R3.3): *vì sao* KAN hơn ở ít dữ liệu và ở SNR cao,
chứ không chỉ *rằng* nó hơn. Không ai bác kết quả. Đây là lỗ hổng chính, và nó cũng là thứ biến
bài từ *bản đồ chế độ* thành bài hợp Neurocomputing (chủ thể là hệ học).

Ba đường, làm được cả ba thì tốt, ít nhất phải có đường 1 và 2:

1. **Giải tích, dung lượng theo chiều**: MLP đặt hàm phi tuyến ở **đỉnh**, KAN đặt ở **cạnh**. Với
   `d` đầu vào và nền `G` mốc, KAN có `O(dG)` bậc tự do **một biến**, nên lớp hàm cộng tính
   `f(x)=Σφᵢ(xᵢ)` được biểu diễn **chính xác**, còn MLP phải xấp xỉ bằng tổ hợp. ⇒ Dự đoán kiểm
   được: **KAN thắng khi hàm đích gần cộng tính, thua khi tương tác giữa các chiều mạnh**.
   Saleh PA **là** hàm gần cộng tính theo biên độ; I/Q của AMC thì **không**.
2. **Thí nghiệm có mục tiêu, tách biến bằng hàm tổng hợp**: sinh dữ liệu với **mức tương tác điều
   khiển được** (từ cộng tính thuần tới tương tác bậc hai đầy đủ), giữ mọi thứ khác cố định, rồi
   đo điểm giao. Nếu điểm giao dịch theo mức tương tác thì cơ chế được chứng minh **trực tiếp**,
   không phải kể chuyện. ⭐ Đây là phép đo mạnh nhất và nó **chạy được trên dữ liệu tổng hợp**,
   không cần phần cứng.
3. **Hiệu quả dữ liệu**: đo đường học theo cỡ tập cho cả hai, và nối nó với số bậc tự do một biến
   ở đường 1.

⚠ Luật nhà: **thực nghiệm TRƯỚC, viết sau**. Kết luận của mục này viết **sau khi** có số của
đường 2, kẻo vấp lại lớp lỗi 29 (kết luận viết trước, macro nói ngược).

### P2 — Độ phủ. Chạy trên sol1, env riêng cho bài này

| # | điểm | việc | ghi chú |
|---|---|---|---|
| P2.1 | **R2.3** ⭐ | **Bất công thật**: ở AMC, CNN khai thác cấu trúc thời gian I/Q còn KAN nhận đầu vào dàn phẳng. Thêm nhánh **cùng một backbone tích chập**, chỉ đổi **đầu phân loại** MLP-đối-KAN ở dung lượng khớp | đây là lỗ hổng thiết kế, không phải đòi hỏi quá đáng |
| P2.2 | **R2.3** | Quét độ sâu và hàm kích hoạt của MLP hiện **chỉ làm cho PA** (Bảng V). Chạy cho **cả ba** tác vụ | `code/exp_mlp_sweep.py` đã có |
| P2.3 | **R1.1** | Thêm biến thể KAN: **FastKAN**, **Chebyshev-KAN**, và **Wav-KAN** nếu kịp | giữ nguyên ngân sách tham số khớp |
| P2.4 | **R2.4** | So với **DPD-KAN** và **envelope time-delay KAN**. Đo được: cả hai sống ở arXiv và hội nghị microwave, ⬜ phải kiểm có mã công khai không | nếu không có mã thì khai rõ là **so theo số đã công bố**, không tái lập |
| P2.5 | **R1.3** | Thêm **một** bộ AMC thứ hai, ưu tiên **RML22** (công khai, cùng họ nên so được) | DeepSense 6G nặng, để limitation |

### P3 — Khai là hạn chế, có lý do, không hứa

- **R1.2 dữ liệu PA đo thật**: không có phần cứng. ⬜ Thử tìm **một bộ PA đo thật công khai**; nếu
  không có thì khai thẳng ở Threats to validity rằng biên của kết luận DPD **chỉ đứng trên Saleh**.
- **R1.6 học liên tục trực tuyến, phân phối trôi**: ngoài phạm vi bản này, đưa vào Future work một
  câu có **điều kiện kiểm được**, đừng hứa suông. ⚠ Neurocomputing đăng nhiều bài continual
  learning nên câu này phải viết cẩn thận, kẻo mời đúng phản biện ấy.

## 2. Thứ tự thi hành

```
1. P0.1 → P0.3  sửa số và hạ cấp tuyên bố          (giấy bút, nửa ngày)
2. P1 đường 2   thí nghiệm tách biến tương tác      (sol1, env rieng)   ← quyết định sống chết
3. P2.1 → P2.3  backbone chung, quét đủ ba tác vụ, thêm biến thể KAN
4. P1 đường 1,3 viết mục cơ chế theo SỐ đã có
5. P0.5 → P0.6  bảng tham số, gộp mục, viết lại Introduction
6. P0.7         chuyển khuôn elsarticle + 5 cổng + Highlights + CRediT + Zenodo DOI
7. P0.2         in CI ghép cặp cho MỌI tuyên bố giao nhau
8. G6X          ĐỌC NGOÀI, lặp tới MINOR          ← cổng bắt buộc, Hảo chạy phiên riêng
9. nộp
```

⚠ **Không đảo bước 2 lên sau bước 4.** Lớp lỗi 29 đã ghi: viết kết luận trước khi có số thì macro
nói ngược văn xuôi.

## 3. Hiện vật phải có trước khi bấm nộp

```
submit-NEUCOM/
├── KE-HOACH-SUA.md        (tệp này)
├── KHUON-NEUCOM.md        khuôn đo được
├── khuon/                 elsarticle + cổng
├── point-to-point.md      18 điểm, mỗi điểm một khối, nêu rõ ĐÃ LÀM / HẠ CẤP / TỪ CHỐI kèm lý do
└── nop-<ngày>/            bản sạch · highlights.docx · declarations.doc · vitae · 10 tệp hình · source.zip
```

⚠ Neurocomputing là **nộp mới**, không phải vòng sửa của TCCN, nên **không gửi thư trả lời TCCN**.
Nhưng `point-to-point.md` vẫn phải viết **cho mình**, để chứng minh từng điểm đã xử, và để nếu
phản biện Neurocomputing hỏi trùng thì có sẵn câu trả lời có số.

⛔ Theo [[feedback-thu-muc-nop-khong-lan-ghi-chu-noi-bo]]: `nop-<ngày>/` chỉ chứa hiện vật gửi đi;
kế hoạch và point-to-point để **ngoài** thư mục ấy.

## 4. Chỗ cần Hảo quyết

1. **Khai quỹ B2025-DN02-25 hay in câu "không nhận tài trợ"?** Guide bắt khai một trong hai.
2. **P2.5 có thêm RML22 không?** Thêm thì mạnh hơn nhưng tốn một vòng chạy đủ ba tác vụ.
3. **P2.4**: nếu DPD-KAN không có mã công khai, chấp nhận **so theo số đã công bố** chứ không tái
   lập, và khai rõ trong bài?
