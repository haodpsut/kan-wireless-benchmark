# Point-to-point 18 điểm phản biện TCCN, và việc đã làm

Lập **25/10/2026**. ⚠ Neurocomputing là **nộp mới**, nên tệp này **không gửi đi**; nó để chứng
minh từng điểm đã xử, và để có câu trả lời có số nếu phản biện mới hỏi trùng.

Mã chạy trên **sol1**, env `kanwl` (torch 2.14.1+cpu), kho `kan-wireless-benchmark` tại `a6b77d7`.
Kết quả thô: `code/results/exp_*.csv` trên sol1 và trong kho local.

## ✅ R2.7a — con số 0,309 nói quá 63%. ĐÃ SỬA GỐC, không chỉ sửa số

Bản nộp: *"the CNN leads the **best** KAN by +0.309"*. Bảng III của chính bài: CNN **0,658**,
KAN-spline **0,468** (KAN tốt nhất), KAN-Fourier **0,349** (tệ nhất). Khoảng cách thật **0,190**.

⭐ Gốc lỗi **không phải** macro sai: macro `amc_cnn_minus_kanfourier = 0.309` **đặt tên đúng**, và
giá trị khớp bảng. Sai là **chữ "best" trong câu** trỏ vào macro của KAN tệ nhất. Mọi cổng kiểu
*số này có trong bảng không* đều xanh trên câu đó.

Đã làm:
1. Thêm macro `amc_cnn_minus_bestkan = 0,190`, **tính từ claims.json** (`amc_cnn_acc −
   amc_kanspline_acc`), không gõ tay. Sửa câu dùng macro này.
2. **Viết `code/verify_numbers.py`** — tệp mà `claims.json` tự khai là *"verify_numbers.py checks
   the paper text against these"* nhưng **không hề có trong kho**. Lớp lỗi: tiêu chí viết ra mà
   không cài đặt. Bốn phép kiểm K1-K4, trong đó **K4 ràng buộc từ so sánh bậc nhất vào hàng cực
   trị** của nhóm khai trong `claims.json["_groups"]`.
3. **Siết K4 hai vòng vì nó bắt oan**: lần đầu 6 lỗi, **5 là báo động giả**. Hai luật thêm: từ so
   sánh phải **dính ngay vào tên nhóm** (*best KAN*, không phải mọi câu có chữ *best*), macro phải
   **cùng tiền tố tác vụ** (chống rò chéo AMC/DPD), và chỉ xét macro nằm **sau** cụm so sánh.
   Sau khi siết: **1 lỗi thật, 0 báo động giả** trên bản thảo chưa sửa.
4. **Bài tiêm `code/tiem_loi_verify.py`, 4 ca, đạt 4/4**: ca lịch sử thật phải ĐỎ · câu đã sửa
   phải XANH · đối chứng âm *"best MLP"* phải XANH · tiêm ngược trên DPD theo chiều `min` phải ĐỎ.
5. Cổng nay **XANH, 51 macro được kiểm** trên bản thảo đã sửa.

## ✅ R2.7b — tuyên bố 0,19 dB không có khoảng tin cậy. ĐÃ ĐO LẠI, và nó SỐNG

`code/exp_bigbudget_paired.py`, **20 hạt giống**, ghép cặp theo hạt giống:

| | bản cũ (5 hạt giống, không ghép cặp) | đo lại |
|---|---|---|
| Fourier-KAN K=8 (4634 tham số) | −17,142 dB | **−17,060 ± 0,116** |
| MLP depth3-relu (4727 tham số) | −16,95 dB | **−16,904 ± 0,157** |
| hiệu | **0,19 dB**, không có CI | **−0,156 dB, CI95 [−0,246, −0,066]** |
| phép thử | không có | **Wilcoxon chính xác p = 0,0073** |

⇒ Phản biện đúng về **thủ tục**, kết luận của bài **không sai**. Ghép cặp khử phương sai giữa các
hạt giống, nên hiệu 0,156 dB **phân biệt được** dù sd của hiệu là 0,207 dB. Câu trong bài sẽ mang
con số, CI và p; chữ *thin* được giữ và nay đã trả giá.

## ✅ R1.5 + R2.2 + R3.3 — ba phản biện hỏi VÌ SAO. Đã có mục cơ chế, ĐO ĐƯỢC

Giả thuyết: MLP đặt phi tuyến ở **đỉnh**, KAN đặt ở **cạnh**, nên lớp hàm **cộng tính** nằm trong
không gian biểu diễn của KAN với `O(dG)` bậc tự do một biến. Dự đoán: lợi thế KAN **giảm theo
cường độ tương tác giữa các chiều**.

⛔ **Bản 1 (`exp_mechanism.py`) CÓ LỖI THIẾT KẾ, không dùng làm kết quả của bài.** Nó trộn phần
cộng tính `sin(w·x+p)` với phần tương tác song tuyến tính. Chuẩn hoá phương sai khử confound biên
độ, nhưng **không khử confound ĐỘ KHÓ**: chính cột MLP tố ra, vì MLP **tốt hơn khi tương tác mạnh
hơn** (−1,54 → −3,01 dB ở n=256; −9,22 → −16,75 dB ở n=4096). Tức `lam` điều khiển **hai thứ**.
Giữ lại trong kho làm bằng chứng của đúng lớp lỗi mà bài này đi cảnh báo.

✅ **Bản 2 (`exp_mechanism2.py`)**: hai phần **cùng lớp hàm bậc 2**, `f_add = Σ aᵢxᵢ²` và
`f_int = Σ c_jk x_j x_k`, nên khác biệt duy nhất là **tính cộng tính**. Độ khó **đo được** bằng
mốc trung tính (bình phương tối thiểu trên đầy đủ đặc trưng bậc 2): **−26,45 dB tại lam=0** so với
**−26,01 dB tại lam=1**, lệch **0,44 dB** thay cho 7,5 dB của bản 1.

**Đối chứng độ khó, bản 2, đo trên mốc trung tính** (bình phương tối thiểu trên đầy đủ đặc trưng
bậc 2): n=256 chạy từ **−25,11 tới −25,05 dB** (lệch **0,06 dB**), n=4096 từ **−26,07 tới −25,99**
(lệch **0,08 dB**) ⇒ `lam` chỉ điều khiển tính cộng tính, không kéo theo độ khó.

Kết quả bản 2 đầy đủ, hiệu ghép cặp 10 hạt giống (âm = KAN tốt hơn, dB):

| lam | KAN-spline | KAN-Fourier | FastKAN | Chebyshev |
|---|---|---|---|---|
| **n=256** | | | | |
| 0,00 | −3,55 | −12,61 | −9,97 | −9,31 |
| 0,25 | −1,80 | −3,60 | −3,51 | −3,31 |
| 0,50 | −0,33 | −0,52 | −0,65 | −0,61 |
| 0,75 | **+0,93** | **+1,53** | **+1,25** | **+1,26** |
| 1,00 | **+1,97** | **+3,24** | **+2,20** | **+2,77** |
| **n=4096** | | | | |
| 0,00 | −10,18 | −8,28 | −8,10 | −6,44 |
| 0,50 | −5,30 | −3,27 | −5,80 | −3,23 |
| 1,00 | −4,75 | −2,28 | −6,62 | −3,30 |

⇒ **Bản đồ HAI TRỤC**: cường độ tương tác quyết định **độ lớn** lợi thế KAN; **ngân sách dữ liệu**
quyết định lợi thế ấy có **đổi dấu** hay không. Ở n=256 đổi dấu giữa lam 0,50 và 0,75, giống nhau
ở **cả bốn** bản cài; ở n=4096 xói mòn rõ rồi **bão hoà** mà không đổi dấu.

(bảng gọn, n=256, giữ lại để đọc nhanh)

| | lam=0 cộng tính thuần | lam=1 tương tác thuần |
|---|---|---|
| KAN-spline | **−2,62** | **+1,97 MLP thắng** |
| KAN-Fourier | **−11,56** | **+3,24 MLP thắng** |
| FastKAN | **−10,29** | **+2,20 MLP thắng** |
| Chebyshev-KAN | **−8,59** | **+2,77 MLP thắng** |

⇒ Dấu **đảo chiều** theo đúng tham số mà giả thuyết nói, trên **cả bốn** bản cài KAN, sau khi độ
khó đã được kiểm soát.

## ✅ R1.1 — bản đồ chế độ có tổng quát sang bản cài KAN khác không? CÓ

`code/variants.py`: **FastKAN** (cơ sở RBF Gauss + nhánh base) và **Chebyshev-KAN** (đa thức
Chebyshev trên `tanh(x)`), số tham số đã đối chiếu **khớp đúng công thức khai**. Chạy trên **cùng
lưới cơ chế**, cùng hàm đích, cùng hạt giống, cùng ngân sách 2300 tham số.
⇒ Cả bốn biến thể **cùng chiều** và cùng điểm đảo dấu. Một thí nghiệm trả hai điểm phản biện.

⭐ Phép đối chiếu chéo miễn phí: cột MLP có trong **cả hai** script, và **trùng khít 12/12 ô** tới
chữ số cuối, nên phép so giữa các biến thể là cùng một hệ quy chiếu.

## ✅ R2.3 — AMC so bất công. ĐÃ CHẠY, và kết quả LẬT MỘT SỐ CỦA BÀI CŨ

`code/exp_amc_backbone.py`, 10 hạt giống, **dùng chung nguyên bộ backbone tích chập của CNN cũ**,
chỉ đổi đầu phân loại, khớp ngân sách giữa các đầu:

| đầu phân loại | tham số toàn mạng | độ chính xác |
|---|---|---|
| linear (đúng như CNN cũ) | 5.291 | 0,6569 ± 0,0059 |
| **MLP** | 16.031 | **0,6893 ± 0,0047** |
| KAN-spline | 15.348 | 0,6755 ± 0,0084 |
| ⛔ KAN-Fourier | 16.179 | **0,2112 ± 0,1907** |

Hiệu ghép cặp:
- **KAN-spline − MLP = −0,0138** [−0,0192, −0,0075] ⇒ **MLP thắng**, nhưng chỉ 1,4 điểm
- **MLP − linear = +0,0324** [+0,0277, +0,0366] ⇒ đổi đầu đã thêm **3,2 điểm**
- KAN-Fourier − linear = −0,4457 ⇒ **sụp**, và sd 0,19 cho thấy huấn luyện **không ổn định**

⇒ Ba kết luận, và một trong đó **bất lợi cho bài cũ**:
1. Khoảng cách ở AMC **không phải** chỉ do inductive bias của tích chập: cho KAN **cùng bộ đặc
   trưng** thì nó vẫn không hơn. Phản biện 2 đúng khi đòi phép thử này, và phép thử **củng cố**
   kết luận của bài.
2. ⛔ **Con số 0,658 của CNN trong bản cũ chưa phải trần của kiến trúc ấy**: chỉ đổi đầu linear
   sang MLP đã lên 0,689. Tức baseline *task-appropriate* của chính bài **bị điều chỉnh ít hơn**
   mức đáng có, đúng lớp lỗi mà bài này đi cảnh báo người khác. Phải khai thẳng.
3. **KAN-Fourier làm đầu phân loại là không dùng được** ở tác vụ này, và đó là một phát hiện
   báo được chứ không phải một thất bại cần giấu.

## (mục cũ) R2.3 — AMC so bất công

Phản biện **đúng**: CNN có bộ trích đặc trưng tích chập còn KAN/MLP nhận đầu vào dàn phẳng 256
chiều, nên chênh lệch có thể đến từ **inductive bias của tích chập**.
`code/exp_amc_backbone.py`: **dùng chung một backbone tích chập** (nguyên bộ của CNN cũ), chỉ đổi
**đầu phân loại** linear / MLP / KAN-Fourier / KAN-spline, khớp ngân sách giữa các đầu.
⬜ Chặn ở việc đưa dữ liệu lên sol1: pickle 611 MB treo hai lần. Đã lọc đúng dải SNR [−6,6] dB
bằng **chính `load_rml()` của bài** (`code/prep_rml_band.py`) ra bản 79,5 MB, SHA-256 nội dung
`a0808c1d…`, và script nạp **dừng hẳn nếu băm lệch**. Đang đẩy từng khúc 8 MB một phiên.

## ✅ R2.3 nửa sau — bộ quét MLP chỉ chạy cho PA. ĐÃ QUÉT CẢ HAI TÁC VỤ CÒN LẠI

`code/exp_mlp_sweep_all.py`, 6 cấu hình (độ sâu 1-4, relu/gelu/tanh), 10 hạt giống, ngân sách khớp:

| tác vụ | KAN tham chiếu | MLP tốt nhất | hiệu ghép cặp |
|---|---|---|---|
| **OFDM channel est.** (30.872 tham số) | Fourier-KAN **−13,114 dB** | depth4-relu **−13,280 dB** | **+0,166 dB** [+0,154, +0,177] ⇒ MLP |
| **AMC** (68.395 tham số) | Fourier-KAN **0,3517** | depth4-relu **0,5010** | **−0,149** [−0,154, −0,144] ⇒ MLP |

⇒ **Cả 6 trên 6 cấu hình MLP đều thắng KAN ở OFDM**, và cả 6 đều thắng ở AMC. Quét đối xứng
**không lật** kết luận của bài, nó **mở rộng** kết luận: ở hai tác vụ này KAN thua cả baseline
chưa điều chỉnh lẫn baseline đã điều chỉnh.

## ✅ Đối chứng thêm mà PHẢN BIỆN CHƯA HỎI, tôi tự thấy phải làm. ĐÃ CHẠY

`code/exp_mechanism2_tuning.py`, trên hàm đích **bản 2**, n=4096, MLP được quét độ sâu lấy cấu
hình tốt nhất, **cả hai bên** cùng lịch 60 rồi 180 epoch:

| | lam = 0,5 | lam = 1,0 |
|---|---|---|
| 60 epoch | KAN hơn **5,30 dB** [5,11, 5,48] | KAN hơn **3,84 dB** [3,45, 4,22] |
| 180 epoch | KAN hơn **5,53 dB** [5,44, 5,61] | KAN hơn **3,21 dB** [3,03, 3,39] |

⇒ Lịch dài gấp ba chỉ khép được **0,62 dB**. Kết quả *KAN thắng khắp mọi lam khi nhiều dữ liệu*
**không phải tạo tác của điều chỉnh không đối xứng**.

## (mục cũ) Đối chứng điều chỉnh
Ở n=4096 bản 1, KAN thắng ở **mọi** lam kể cả tương tác thuần. Nhiều dữ liệu lẽ ra có lợi cho mô
hình ít cấu trúc hơn, nên nghi là **tạo tác của điều chỉnh không đối xứng**.
`code/exp_mechanism_tuning.py`: ở n=4096, quét MLP lấy cấu hình tốt nhất và cho **cả hai bên** cùng
lịch 60 và 180 epoch. Nếu MLP tốt nhất khép được khoảng cách thì phải báo đúng thế.

## ⬜ Chưa làm, và lý do
- **R1.2 dữ liệu PA đo thật**: không có phần cứng. Sẽ khai ở Threats to validity rằng biên kết
  luận DPD **chỉ đứng trên Saleh**, kèm thử tìm một bộ PA đo thật công khai.
- **R1.3 bộ AMC thứ hai (RML22)**: chờ Hảo quyết, vì tốn một vòng chạy đủ ba tác vụ.
- **R2.4 so với DPD-KAN và envelope time-delay KAN**: ⬜ chưa kiểm hai bài đó có mã công khai.
- **R1.4 + R2.5 trích xuất ký hiệu sâu hơn lớp đầu**: sẽ hạ cấp chữ *lower bound* trước, thêm lớp
  nếu còn thời gian.
- **R1.6 học liên tục / phân phối trôi**: ngoài phạm vi, đưa vào Future work có điều kiện kiểm được.
- **R2.6 + R3.1 + R3.2 + R3.5**: gộp đóng góp, mở rộng Introduction, bỏ công thức khỏi Related
  Work, nén **16 mục cấp 1 xuống ≤ 9**, thêm bảng tham số. Việc văn bản, làm cùng lúc đổi khuôn.
- **R3.4 link mã**: đưa vào abstract **và** nạp Zenodo lấy DOI, vì Guide bắt Research data Option C.


---

## Trạng thái hiện vật, 26/10/2026

Tất cả thí nghiệm chạy **một nơi trên sol1** (Hảo chốt 25/10), env `kanwl`, torch 2.14.1+cpu.
Kết quả đã kéo về kho local: `code/results-neucom/` (8 CSV) và `code/logs-neucom/` (8 log).

| hiện vật | dùng cho bài |
|---|---|
| `exp_mechanism2.csv` · `exp_mechanism2_tuning.csv` | ⭐ **CÓ**, mục cơ chế |
| `exp_bigbudget_paired.csv` | ⭐ **CÓ**, sửa tuyên bố 0,19 dB |
| `exp_amc_backbone.csv` | ⭐ **CÓ**, trả R2.3 |
| `exp_mlp_sweep_all.csv` | ⭐ **CÓ**, trả R2.3 nửa sau |
| `exp_mechanism.csv` · `exp_mechanism_variants.csv` · `exp_mechanism_tuning.csv` | ⛔ **KHÔNG**, bản 1 có confound độ khó. Giữ làm bằng chứng của lớp lỗi |

**Cổng và bài tiêm**: `verify_numbers.py` ✅ 51 macro 0 lỗi, chạy khớp ở cả local và sol1 ·
`tiem_loi_verify.py` ✅ **5/5**, sau khi sửa lỗi ca tiêm neo vào chuỗi vừa bị xoá.

⬜ **Chưa làm**: commit và đẩy `kan-wireless-benchmark` lên GitHub (kho **công khai**, chờ Hảo
đồng ý) · viết mục cơ chế vào bản thảo · đổi khuôn elsarticle · Zenodo DOI · ba câu hỏi ở
`KE-HOACH-SUA.md` mục 4.
