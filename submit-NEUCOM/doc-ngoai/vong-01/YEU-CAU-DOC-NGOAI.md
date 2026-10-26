# Đọc ngoài vòng 1, bài KAN-wireless đi Neurocomputing

Theo cổng **G6X** của nhà (`viet-paper-chuan-dph/gates/G6X-external-review.md`). Cổng này
**fail-closed**: G7 và việc nộp bị chặn tới khi vòng gần nhất trả **Minor** hoặc **Accept**.

## Cách chạy, ba điều kiện cứng

1. ⛔ **Chạy ở phiên ĐỘC LẬP**, không phải phiên đang viết bài. Lý do: phiên đang viết đã tin
   vào lập luận của chính nó, nên nó kiểm được **tính nhất quán** mà không kiểm được **tiền đề**.
2. ⛔ **Gửi BLIND**: chỉ đưa PDF, **đừng đưa** bộ phản biện TCCN cũ và đừng kể bài từng bị loại.
   Đưa vào là mồi, người đọc sẽ đi xác nhận câu trả lời của mình thay vì tự tìm lỗi.
   (Muốn soi riêng phần trả lời phản biện cũ thì làm **vòng sau**, bằng `YEU-CAU-VONG-SAU.md`.)
3. Đòi **phán quyết theo thang tạp chí** và **danh sách điểm có số**.

⚠ Bản này theo bố cục **đã xuất bản** (hai cột, float nằm xen trong bài) nên **không có số
dòng**. Bảo họ trích vị trí theo **mục, bảng, hình, trang**, đừng theo số dòng.

## Hiện vật trong thư mục này

| tệp | đưa cho phiên ngoài? |
|---|---|
| `KAN-wireless-NEUCOM-doc-ngoai-v1.pdf` | ✅ bắt buộc, **13 trang, hai cột**, đúng bố cục bài đã xuất bản của Neurocomputing |
| `claims.json` | ✅ nên đưa: mỗi con số trong bài có một khoá ở đây, giúp họ bắt câu văn không có số đỡ |
| `highlights.tex` | ✅ nên đưa, Elsevier yêu cầu và nó cũng là tuyên bố |
| `KET-QUA.md` | ⬜ để dán phán quyết về |

---

## Lời nhắc dán vào phiên ngoài (tiếng Anh, copy nguyên khối)

```
You are reviewing a manuscript submitted to Neurocomputing (Elsevier). Act as a demanding
but fair referee. I want you to attack the PREMISES, not only the internal consistency.

Deliver, in this order:

1. VERDICT on the journal scale: Reject / Major revision / Minor revision / Accept.
2. A NUMBERED list of points. For each point give:
   - severity: FATAL / MAJOR / MINOR / COSMETIC
   - your CONFIDENCE: certain / likely / speculative
   - the exact section, table, or line you are objecting to
   - what measurement or change would settle it
3. A short list of what you checked and found SOUND. I need this to calibrate.

Specific things I want stress-tested, because an internal check cannot do it:

a) SCOPE. Neurocomputing excludes "pure combination of existing algorithms, not directly
   relevant to neural networks or learning systems". The paper is framed as a study of an
   architecture family (KAN) with three wireless tasks as a testbed. Does that framing hold,
   or does it read as an application paper that belongs in a communications venue?
b) THE MECHANISM SECTION (Section 10, "Why the Map Looks Like This", page 9). It claims that the share
   of cross-coordinate interaction, not the task identity, governs when a KAN beats an MLP.
   Is the synthetic design a valid test of that claim? Specifically: both components are
   degree-two polynomials and a least-squares reference on the full degree-two feature set is
   used as a difficulty control. Does that control actually remove the confound it claims to
   remove? What would break it?
c) THE TWO-AXIS CLAIM. At 256 samples the KAN advantage reverses sign; at 4096 it erodes but
   does not reverse. Does the conclusion "interaction sets the size, data budget decides
   whether it reverses" follow from the tables, or is it over-read from two sample sizes?
d) STATISTICS. The paper reports paired bootstrap intervals and one exact Wilcoxon test. Are
   the paired comparisons set up correctly? Is any directional claim resting on an interval
   that includes zero, or on an effect smaller than the seed-to-seed spread?
e) CLAIM-TO-CLAIM CONTRADICTIONS. Does any sentence contradict a number in a table, or another
   sentence elsewhere? This class of error is what internal checks keep missing.
f) SELF-CRITICISM THAT GOES TOO FAR OR NOT FAR ENOUGH. The paper admits its own convolutional
   baseline was under-tuned and that an earlier version of one comparison lacked an interval.
   Is that handled honestly, or does it undercut the paper's own results more than the data
   warrants?
g) THE INTERPRETABILITY SECTION. The cost of symbolic extraction is measured on the first
   layer only and the paper no longer calls it a lower bound. Is the weaker phrasing now
   correct, and is the section's conclusion still supported?

Do not fix the writing. Do not summarise the paper back to me. Find what is wrong.
```

---

## Khi phán quyết về, ba việc BẮT BUỘC với từng điểm

Theo G6X bước 2, không được sửa theo cảm nhận:

1. **Kiểm số học trước khi nhận.** Người đọc ngoài cũng sai. Ca đã có: một vòng nêu *"hằng số
   0,25"* như bất biến, kiểm lại thì nó là **45 độ xoay đuôi**, hệ quả của ngưỡng −3 dB đã chọn.
   Nhận một điểm sai vào bài là tự thêm lỗi.
2. **Ghi phán quyết cho TỪNG điểm**: `SỬA` · `BÁC kèm lý do đo được` · `GHI VÀO LIMITATIONS`.
   Không im lặng bỏ qua điểm nào.
3. **Không tự khai thêm lỗi không có trong thư trả lời.**

Và sau khi sửa, coi bản thảo là **CHƯA KIỂM lại**: chạy lại toàn bộ cổng rồi mới đi vòng sau.
Bằng chứng trong sổ: revision-2 của Trans-H sửa xong **đẻ 2 DANGEROUS mới**.

## Điều kiện qua

| phán quyết | được làm gì |
|---|---|
| Reject hoặc Major | ⛔ **KHÔNG nộp.** Sửa, chạy lại cổng, đọc ngoài lại |
| Minor hoặc Accept | ✅ G6X PASS, sang G7 |

⚠ **Minor là điều kiện CẦN, không phải ĐỦ.** Nó chỉ nói người đọc ngoài không còn tìm thấy lỗi
cấu trúc. Nó **không** nói gì về phạm vi venue hay occupied-check. Ba bài từng qua đọc ngoài tử
tế rồi vẫn desk reject vì **scope**.
