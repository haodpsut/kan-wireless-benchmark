# Khuôn Neurocomputing, ĐO từ bài đã publish và Guide for Authors

Lập **25/10/2026**. Hai nguồn, cả hai do Hảo đưa:
1. **Bài mẫu**: `~/Downloads/1-s2.0-S0925231226027177-main.pdf` = *Convergence and stability of
   Q-learning in Hierarchical Reinforcement Learning*, **Neurocomputing 710 (2027) 135319**,
   DOI `10.1016/j.neucom.2026.135319`.
2. **Guide for Authors** của chính tạp chí.

⛔ Luật nhà: khuôn **đo từ bài đã publish**, không đoán từ template. Đã dùng đúng nếp này cho JSS
(bài mẫu `S0164121226003833`) và JNCA (`S1084804526001682`).

## Đo được từ bài mẫu

| đại lượng | bài mẫu | bài mình hiện tại |
|---|---|---|
| số trang | **15** | 10 (khổ IEEE 2 cột) |
| bố cục | **hai cột** | hai cột ⇒ giữ được, xem mục LaTeX dưới |
| tham khảo | **33**, dấu số `[n]` theo thứ tự xuất hiện | cần đếm lại sau khi thêm baseline |
| abstract | **~156 từ** (hạn 250) | cần đo lại |
| **HIGHLIGHTS** in ngay trang 1 | **4 gạch** | ⬜ chưa có |
| ARTICLE INFO | có dòng *Communicated by H. Zhang* + Keywords dọc | nhà xuất bản tự thêm |
| ⭐ link mã **nằm trong abstract** | *"Code and data available at: https://github.com/..."* | ⬜ chưa có, mà phản biện 3 đòi |

### Thứ tự khối CUỐI BÀI, đo trên bài mẫu, copy đúng thứ tự này

```
CRediT authorship contribution statement
Declaration of Generative AI and AI-assisted technologies in the manuscript preparation process
Declaration of competing interest
Acknowledgements
Appendix A, B, C ...
Data availability
References
```

## Ràng buộc từ Guide for Authors

| mục | luật | việc của mình |
|---|---|---|
| **Abstract** | ≤ **250 từ**, tự đứng được, tránh trích dẫn và viết tắt lạ | đo trên BẢN IN, không đo nguồn |
| **Keywords** | **1 tới 7**, tránh từ khoá nhiều chữ có "and"/"of" | chọn 6 |
| **Highlights** | **3 tới 5** gạch, mỗi gạch **≤ 85 ký tự kể cả dấu cách**, **tệp RIÊNG** có chữ `highlights` trong tên | cổng `c2_highlights` của nhà đã kiểm đúng luật này |
| **LaTeX** | ⭐ **hai cột ĐƯỢC PHÉP** cho bản LaTeX (Word thì bắt buộc một cột) | ⇒ **không phải dàn lại một cột**, khác hẳn ca JSS |
| **Tham khảo** | dấu số `[n]`, số theo **thứ tự xuất hiện**; định dạng không bị siết lúc nộp nhưng phải **nhất quán** | bài đang dùng IEEE numeric ⇒ gần như không phải đổi |
| **Mục** | đánh số 1, 1.1, 1.1.1; abstract **không** nằm trong số mục | bài đang theo IEEE, phải đánh số lại |
| **CRediT** | bắt buộc, kể cả bài một tác giả | khai đủ vai |
| **Competing interests** | khai qua **declarations tool**, tải lên tệp `.doc/.docx` riêng | ⬜ hiện vật riêng, đừng nhúng vào bài |
| **Funding** | phải khai; nếu không có thì in đúng câu *"This research did not receive any specific grant..."* | ⬜ Hảo quyết có khai quỹ B2025-DN02-25 hay không |
| ⭐ **Declaration of generative AI** | nếu dùng thì **thêm một mục trước danh sách tham khảo** | ⛔ **phải khai**: kho có commit mang trailer Claude, đúng ca TNSM |
| ⭐⭐ **Research data: Option C** | **BẮT BUỘC nạp dữ liệu vào một repository** rồi **trích dẫn và liên kết trong bài**, hoặc giải thích vì sao không được | ⇒ GitHub **không đủ**, cần **DOI Zenodo**. Việc này trả luôn phản biện 3 điểm 4 |
| **Vitae** | tiểu sử ≤ **100 từ** mỗi tác giả + **ảnh chân dung** dạng tệp hình riêng, định dạng sửa được | ⚠ dễ bỏ sót, không có trong khuôn IEEE |
| **Hình** | tệp RIÊNG từng hình, vector EPS/PDF hoặc ảnh ≥ 300 dpi, mỗi hình một tệp, **cấm gộp nhiều hình vào một tệp** | bài đang có 10 hình, phải tách |
| **Bảng** | text sửa được, **cấm kẻ dọc và cấm tô nền ô** | kiểm lại 7 bảng |
| ⛔ **Số đặc biệt (VSI)** | là một article type riêng | **tránh**, sổ nhà đếm số đặc biệt 4/4 chết |

## ⛔⛔ Danh sách LOẠI TRỪ của tạp chí, đọc kỹ vì nhà đã chết 6 lần vì scope

> *"Neurocomputing does NOT accept submissions on pure dynamical systems, pure control engineering,
> pure fuzzy logics, pure image processing or **pure combination of existing algorithms**, which are
> **not directly relevant to neural networks or learning systems**."*

Phép thử phạm vi của họ **không phải** "bài có thuộc về không dây hay không", mà là
**"bài có nói về mạng nơ-ron / hệ học hay không"**. Bài mình nghiên cứu **một họ kiến trúc
(KAN) đối chứng với MLP ở dung lượng khớp** ⇒ **đạt**, nếu và chỉ nếu viết với **KAN là chủ thể**
và **ba tác vụ không dây là bàn thử**. Viết ngược lại, tức "giải quyết bài toán không dây bằng
KAN", thì rơi đúng vào *pure combination of existing algorithms*.

⇒ **Luật viết cho bản này**: tiêu đề, abstract, mở đầu và kết luận phải nói về **KAN và hệ học**;
PA, AMC, OFDM chỉ xuất hiện như **ba chế độ dữ liệu** để tách biến.

## ⭐⭐ Bảy article type, và BA cái phục vụ đúng sức mạnh của nhà

| type | nội dung | dùng cho |
|---|---|---|
| **1. Regular article** | phải *directly relevant to Neural Networks and/or Learning Systems* | ⭐ **chọn cái này cho bản nộp** |
| **2. Survey/review** | tổng hợp, tích hợp, nêu xu hướng | hướng survey đang **tạm dừng** |
| ⭐ **4. Replication study** | *"replications of previous studies (published within Neurocomputing and other major journals) are welcome... close, clear and valid replications of the original methodology"* | ⭐⭐ **venue CÓ cửa chính thức cho dạng kiểm toán**. Đây là câu trả lời cho phản biện TCCN điểm 1 về novelty, giữ làm **đường lùi** |
| **5. Software publication** (OSP) | phần mềm, toolbox, hệ thật, có giấy phép OSI | kho cổng của nhà, để sau |

⇒ Phát hiện này **sửa kết luận của tôi hôm 23/10** rằng *"Neurocomputing chỉ nhận method-positive"*.
Sai. Tạp chí có **type 4 cho replication** và **type 5 cho phần mềm**. Cái tôi đo hôm đó là
**tiền lệ đã đăng**, không phải **luật nhận bài**, và hai thứ đó khác nhau.
