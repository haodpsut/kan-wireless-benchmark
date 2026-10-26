# Kết quả đọc ngoài vòng 1

**Ngày:** 26/10/2026 · **Phán quyết: MAJOR REVISION** ⇒ ⛔ theo G6X **chưa được nộp**.

## Dòng ledger

```
| ngày       | bài                 | vòng | phán quyết | số điểm | số điểm mình BÁC | ghi chú |
| 26/10/2026 | KAN-wireless NEUCOM | 1    | Major      | 5 required + 5 câu hỏi | 0 bác nội dung, 1 sửa số | mục 2(a) là chốt chặn |
```

⭐ Người đọc ngoài **tự kiểm số học của bài trước khi phê**, và mọi số họ kiểm đều tái lập:
0,6893 − 0,6569 = 0,0324 · 0,6893 − 0,6755 = 0,0138 · 3,83 − 3,21 = 0,62 · dải Bảng 8 ·
slowdown Bảng 7. Họ còn kiểm p-value: **p = 0,007296 đúng bằng 7650/2²⁰**, tức là p **chính xác**
đạt được ở n = 20 chứ không phải xấp xỉ làm tròn sáu chữ số. Đây là người đọc ngoài tốt.

---

## Bảng xử từng điểm, theo luật G6X bước 2

| # | điểm | tôi kiểm lại bằng gì | đúng/sai | phán quyết |
|---|---|---|---|---|
| **2(a)** | Moc "neutral reference" OLS bậc 2 **không thể hỏng**, vì `f_add ∈ span{xᵢ²}` và `f_int ∈ span{xⱼxₖ}` nên đích **luôn nằm trong span** của OLS ở mọi λ | **Chạy `exp_mechanism2_control.py`**: OLS phẳng **0,138 dB** (n=256) và **0,176 dB** (n=4096); nhưng khi cho đích **ra khỏi span** (thêm sin 3x) thì OLS **vung 147,36 dB** | ✅ **ĐÚNG** | **SỬA**, đây là chốt chặn |
| **2(b)** | Công thức (2)-(4) tả mạng **khác độ sâu**: MLP có `h²` tức 2 lớp ẩn, KAN chỉ `dh + hc` tức 1 lớp ẩn | Dựng mô hình và đếm: `MLPreg` = **2 lớp ẩn**, `KANreg`/`KANspline` = **1 lớp ẩn** | ✅ **ĐÚNG** | **SỬA** |
| **2(c)** | Số hạt giống không nhất quán: 5 cho hồi quy, 10 cho phân loại và cơ chế, 20 cho một phép so | Đọc `claims.json`: `n_seeds_dpd=5`, `n_seeds_amc=10`, `k8_n_seeds=20` | ✅ **ĐÚNG** | **SỬA**, nâng hồi quy lên 10 |
| **3(a)** | Abstract gắn lời bào chữa điều chỉnh **vào sai thí nghiệm**: 0,62 dB là Bảng 9 ở **n=4096**, còn đảo dấu ở **n=256** | Đọc lại abstract và Bảng 9 | ✅ **ĐÚNG** | **SỬA**: chạy đối chứng phía KAN ở n=256, hoặc rút lời bào chữa khỏi abstract |
| **3(b)** | Bảng 8 có **40 ô** khoảng tin cậy, in đậm mà **không hiệu chỉnh bội** | Đếm: 4 biến thể × 5 λ × 2 n = 40 | ✅ **ĐÚNG** | **SỬA**: khai cách xử bội |
| **4** | §9 lập trục "chi phí mới là trục đáng kể" rồi §11.1 lại **tiến cử CNN** bằng trục **số tham số** mà chính bài gọi là sai; và CNN là **chậm nhất cả nghiên cứu** | Đọc `results/exp_compute.csv`: CNN AMC **25,157 ms** · MLP **0,164 ms** · KAN-spline **3,468 ms** | ✅ **ĐÚNG về bản chất**, ⚠ **SỬA SỐ**: tỉ số thật là **153,4×**, không phải 157× (họ lấy 25,2/0,16 đã làm tròn); so với KAN-spline là **7,25×** chứ không phải 7,3× | **SỬA** |
| **6** | §11.4 Threats thiếu: cách dựng đối chứng, chuyện độ sâu, chuyện hạt giống, và **độ hẹp của họ hàm tổng hợp** (một lớp hàm bậc 2, một chiều d=8) | Đọc §11.4 | ✅ **ĐÚNG** | **SỬA** |

⇒ **Bác nội dung: 0 điểm.** Chỉ **một chỗ sửa số** ở mục 4. Tỉ lệ báo động giả của vòng này là
**0/6 về bản chất**, rất thấp so với các vòng trước của nhà.

---

## Kết quả đối chứng MỚI, đã chạy 26/10 trên sol1

`code/exp_mechanism2_control.py` → `results/exp_mechanism2_control.csv`

**[A] Chứng minh moc cũ không thể hỏng**

| đích | n | NMSE theo λ | biên độ |
|---|---|---|---|
| trong span (thiết kế của bài) | 256 | −25,05 → −24,91 → −25,05 | **0,138 dB** |
| trong span | 4096 | −26,21 → −26,03 | **0,176 dB** |
| ⭐ **ngoài span** (thêm sin 3x) | 4096 | −1,64 → **−149,00** | **147,36 dB** |

⇒ OLS **chỉ hỏng khi đích ra khỏi span của nó**. Trong thiết kế của bài thì đích **không bao giờ**
ra khỏi span, nên moc cũ **phẳng theo cấu trúc**. Phản biện đúng hoàn toàn.

**[B] Đối chứng mới, có thể hỏng**: đặc trưng ngẫu nhiên + ridge, **cùng ngân sách 2300 tham số**,
không biết gì về tính cộng tính.

| n | NMSE theo λ (0 → 1) | biên độ |
|---|---|---|
| 256 | −0,44 → −0,92 | **0,474 dB** |
| 4096 | −4,44 → −5,61 | **1,170 dB** |

⇒ **Có một dốc độ khó thật**, không phải 0 như moc cũ gợi ý, nhưng nó **nhỏ so với hiệu ứng đo
được**: biên độ lợi thế KAN là **5,52 dB** ở n=256 (spline −3,55 → +1,97) và **5,43 dB** ở
n=4096 (−10,18 → −4,75). Tức dốc độ khó chiếm **8,6%** và **21,5%** hiệu ứng.
⚠ Và nó chạy **ngược chiều có lợi cho kết luận**: đầu λ cao **dễ hơn** cho mô hình trung tính, mà
KAN vẫn thua ở đó.

---

## Việc phải làm, theo thứ tự

1. ✅ **đã chạy** đối chứng mới (2a). ⬜ viết lại §10.2: bỏ moc OLS khỏi vai trò đối chứng, giữ nó
   như lời khai *đích nằm trong span bậc 2 nên không mô hình nào vướng trần xấp xỉ*, và báo dốc
   độ khó đo được kèm tỉ lệ so với hiệu ứng.
2. ⬜ **2(b)**: khai đúng kiến trúc từng mô hình, và chạy phép so **khớp độ sâu**.
3. ⬜ **2(c)**: nâng hai chế độ hồi quy lên 10 hạt giống.
4. ⬜ **3(a)**: chạy đối chứng điều chỉnh **phía KAN ở n=256**, hoặc rút lời bào chữa khỏi abstract.
5. ⬜ **3(b)**: khai cách xử bội cho Bảng 8.
6. ⬜ **4**: áp trục chi phí vào lời tiến cử CNN ở §11.1, dùng **153,4×** và **7,25×**.
7. ⬜ **6**: mở rộng §11.4.
8. ⬜ sau khi sửa: chạy lại **toàn bộ** cổng, rồi **đọc ngoài vòng 2**.
