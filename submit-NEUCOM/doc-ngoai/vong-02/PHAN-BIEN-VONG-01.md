# Phiếu phản biện vòng 1, nguyên văn (rút gọn phần tóm tắt)

**Phán quyết: Major Revision.** Sáu mục "Required".

## Required

1. **2(a)** Replace the OLS "neutral reference" with a control that can fail, or withdraw the
   claim that the difficulty question has been tested and rest it on the design argument
   explicitly. *Lý do họ nêu:* `f_add ∈ span{xᵢ²}` và `f_int ∈ span{xⱼxₖ}` nên đích nằm trọn
   trong không gian đặc trưng của OLS ở mọi λ; OLS đặc tả đúng, độ chệch 0, phương sai
   `σ²·p/n` với p = 45 cố định ⇒ NMSE phẳng **theo cấu trúc**, và 0,065 dB quan sát được chỉ là
   nhiễu Monte Carlo. *"The control is guaranteed to pass and measures nothing."*
2. **2(b)** Specify the architectures; match depth or show the ordering survives it. *(2) có
   `h²` tức hai lớp ẩn, (3)(4) chỉ có `dh + hc` tức một lớp ẩn, trong khi cả ba được giới thiệu
   là "a two-layer network".* Họ nhắc: bộ quét của chính bài đo độ sâu đáng 0,47 dB trên DPD.
3. **2(c)** Raise the regression regimes to the ten seeds used elsewhere.
4. **3(a)** Run the KAN-side tuning control at n = 256, or restrict the abstract's claim.
   *0,62 dB là Bảng 9 chạy ở n = 4096, còn đảo dấu là ở n = 256; ở đó MLP thắng và đối chứng
   cần có là đối chứng phía KAN, mà bài không có.*
5. **3(b)** State the multiplicity treatment for Table 8 (40 ô, không hiệu chỉnh).
6. **4** Apply §9's compute axis to the CNN recommendation in §11.1. *Bảng 7 cho CNN AMC
   25,2 ms so với MLP 0,16 ms, tức 157×, và chậm hơn 7,3× so với KAN tệ nhất, trong khi §11.1
   tiến cử CNN bằng chính trục số tham số mà bài vừa gọi là sai.*
7. **6** Extend §11.4 to cover the control, the depth specification and the seed counts, và
   nói thêm độ hẹp của họ hàm tổng hợp (một lớp hàm bậc 2, một chiều d = 8).

## Năm câu hỏi họ đặt

1. Can the OLS reference in §10.2 fail? What would a non-flat result from it have looked like?
2. At n = 256, λ = 1 the MLP wins. What tuning control defends that?
3. Do equations (2) and (3) describe networks of the same depth?
4. Table 7 puts the AMC CNN at 25.2 ms. How does §11.1's recommendation survive §9's argument?
5. Why five seeds for the regression regimes?

## Họ ghi nhận là ĐÚNG

Mọi số họ kiểm đều tái lập: 0,6893 − 0,6569 = 0,0324 · 0,6893 − 0,6755 = 0,0138 ·
3,83 − 3,21 = 0,62 · dải Bảng 8 · slowdown Bảng 7. Và **p = 0,007296 đúng bằng 7650/2²⁰**, tức
p chính xác đạt được ở n = 20.

Họ cũng nêu rõ phần đáng giữ: §8 chạy lại phép so cũ thành ghép cặp và tự khai *"the standard
deviation of the paired difference is larger than its mean"*; §5.4 tự khai baseline của chính
bài bị điều chỉnh thiếu 0,0324 và gọi đó là *"uncomfortable for us"*; §8.1 mở rộng quét đối
xứng theo chiều bất lợi cho kiến trúc đang thử; §8 báo cả chỗ KAN thắng ngược headline.
*"That conduct is the paper's main asset and should not be diluted in revision."*
