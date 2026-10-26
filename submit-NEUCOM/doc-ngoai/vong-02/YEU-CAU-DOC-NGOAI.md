# Đọc ngoài vòng 2, bài KAN-wireless đi Neurocomputing

Vòng 1 trả **Major Revision**, 6 điểm. Đã sửa cả 6. Cổng G6X chỉ qua khi vòng này trả
**Minor** hoặc **Accept**.

## Cách chạy

1. ⛔ **Phiên độc lập**, không phải phiên đang viết bài.
2. ⭐ **Vòng này KHÁC vòng 1: đưa kèm cả phiếu phản biện vòng 1 và bảng xử của tôi.** Lý do:
   câu hỏi chính của vòng này không còn là *bài có lỗi gì* mà là **các câu trả lời có đứng
   không**, và người đọc cần thấy điểm gốc mới phán được. Vòng 1 thì ngược lại, phải gửi blind.
3. Đòi **phán quyết thang tạp chí** và **danh sách điểm có số**, như vòng 1.

## Hiện vật

| tệp | đưa cho phiên ngoài? |
|---|---|
| `KAN-wireless-NEUCOM-doc-ngoai-v2.pdf` | ✅ bản đã sửa |
| `PHAN-BIEN-VONG-01.md` | ✅ nguyên văn phiếu vòng 1 |
| `BANG-XU-VONG-01.md` | ✅ tôi đã xử từng điểm thế nào, kèm số đo |
| `claims.json` | ✅ mọi số trong bài có một khoá ở đây |

---

## Lời nhắc dán vào phiên ngoài

```
You reviewed this manuscript (or a reviewer did) and returned Major Revision with six
required items. The revised manuscript, the original report, and the author's point-by-point
response are attached. Your job now is NOT to re-review the paper from scratch. It is to
decide whether the responses hold.

Deliver, in this order:

1. VERDICT on the journal scale: Reject / Major revision / Minor revision / Accept.
2. For EACH of the six original items, one of:
   - RESOLVED: the response answers it, and here is what convinced me
   - PARTIALLY RESOLVED: what is still missing, concretely
   - NOT RESOLVED: why the response fails
   - OVER-CORRECTED: the fix introduced a new problem or weakens a claim the data supports
3. NEW problems introduced by the revision, numbered, with severity and confidence. Revisions
   create defects; the author expects this list to be non-empty.
4. A short list of what you checked and found sound.

Pay particular attention to these, because they are where a revision of this kind usually
goes wrong:

a) THE NEW CONTROL (Section 10.2). The old "neutral reference" was ordinary least squares on
   the degree-two feature set, which could not fail because the target lies in its span. It is
   replaced by ridge regression on random Fourier features at the same parameter budget. Does
   that control actually have the property the old one lacked, i.e. can it fail? Is the
   reported residual gradient (0.474 dB at n=256, 1.170 dB at n=4096) used honestly when the
   paper converts it into "8.6% and 20.5% of the architecture effect"? Check that ratio's
   denominator: it is the swing of the paired difference from lambda=0 to lambda=1.
b) THE DEPTH-MATCHED EXPERIMENT (Section 10.4). The paper now admits the main tables match
   parameters but not depth, and reports a repeat at two matched depths. Does the new evidence
   support the claim that the ordering survives? The paper also says the crossover moves
   earlier once depth is matched, which CORRECTS a number in the previous version. Is that
   correction stated clearly enough, and is any other sentence in the paper still relying on
   the old crossover location?
c) THE KAN-SIDE TUNING CONTROL (Section 10.5). Eight KAN configurations at the reversal point.
   Is eight enough, and is the sweep symmetric with what the MLP got at the other end?
d) THE MULTIPLICITY STATEMENT in Section 10.3. The paper claims the headline cells survive
   Bonferroni over 40 comparisons and that the near-zero cells do not, and says it does not
   build on the latter. Verify that claim against Table 8 and check the paper keeps that
   promise elsewhere.
e) THE COMPUTE-AXIS PARAGRAPH in Section 11.1. The paper now states the CNN is the most
   expensive model in the study and recommends it anyway, on accuracy. Is the reasoning sound,
   or has the author traded one inconsistency for a hedge?
f) THE EXPANDED THREATS (Section 11.4). Four limits are now stated. Is anything still missing
   that a Neurocomputing referee would raise, and is any of it overstated to look candid?

Do not fix the writing. Find what is wrong.
```

## Khi phán quyết về

Vẫn ba việc bắt buộc của G6X: **kiểm số học trước khi nhận** · ghi `SỬA` hoặc `BÁC kèm lý do đo
được` hoặc `GHI VÀO LIMITATIONS` cho **từng** điểm · **không tự khai thêm lỗi họ không nêu**.
Và sau khi sửa thì coi bản thảo là chưa kiểm lại, chạy lại toàn bộ cổng.

| phán quyết | được làm gì |
|---|---|
| Reject hoặc Major | ⛔ không nộp, sửa rồi đọc ngoài vòng 3 |
| **Minor hoặc Accept** | ✅ G6X PASS, sang G7 rồi nộp |
