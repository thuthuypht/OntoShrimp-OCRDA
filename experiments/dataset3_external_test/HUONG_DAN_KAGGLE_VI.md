# HƯỚNG DẪN KAGGLE — FINAL EXTERNAL TEST DATASET 3

## 1. Đây là lần mở Dataset 3 chính thức
Từ thời điểm chạy inference đầu tiên, **không được thay đổi mô hình, hyperparameter, threshold, checkpoint, seed hay preprocessing dựa trên kết quả Dataset 3**.

## 2. Chuẩn bị Input
Cần attach:
- `OCRDA_v7_4_4_FROZEN_DATASET3_EXTERNAL_EVAL_Kaggle`
- `Dataset3_Vietnam_v1`
- một Kaggle Dataset chứa 45 frozen model artifacts của formal suite.

Không cần SDBD/TSBD vì package không huấn luyện lại.

### Nếu cô đã tải các MODEL ZIP sau mỗi formal seed
Gom các file `OCRDA_v7_4_4_FROZEN_MODEL_*_SEED_*.zip` vào một thư mục và upload chúng thành một Kaggle Dataset, ví dụ `OCRDA_v744_FROZEN_45_MODELS`. Có thể upload nhiều file vào cùng dataset. Kaggle có thể tự giải nén ZIP; runner hỗ trợ cả ZIP lẫn `compact_model_fp16.pt` đã giải nén.

### Nếu không còn các frozen model checkpoint
**Không chạy external test.** `REPORTS_FINAL.zip` không chứa weights nên không thể inference. Cần khôi phục model artifacts từ Kaggle Saved Version/Output đã lưu trước đó. Package này cố ý không có fallback retraining.

## 3. Chạy Inventory trước
Copy `KAGGLE_ONE_CELL_INVENTORY_ONLY.txt` vào một cell và Run. Cell này chỉ kiểm tra dataset + 45 checkpoints, chưa tính metric Dataset 3.
Phải thấy:
```
Dataset3 lock: PASS (150 images; 50/class; SHA256 verified)
Frozen checkpoints found: 45 / 45
INVENTORY ONLY: no Dataset3 inference executed.
```
Nếu thiếu checkpoint, runner sẽ in danh sách thiếu.

## 4. Chạy external evaluation
Copy `KAGGLE_ONE_CELL_EXTERNAL_EVAL.txt` vào cell. Mỗi lần Run xử lý đúng **1 seed × 9 methods** theo thứ tự 42, 123, 2026, 3407, 7777.
Sau mỗi seed tải `OCRDA_v7_4_4_DATASET3_EXTERNAL_EVAL_REPORTS_LATEST.zip`.

Nếu đóng Kaggle, upload REPORTS_LATEST thành một Kaggle Dataset input; runner sẽ tự restore các external results đã xong và tiếp tục seed kế.

## 5. Kết quả cuối
Sau seed 7777, tải:
`OCRDA_v7_4_4_DATASET3_EXTERNAL_EVAL_REPORTS_FINAL.zip`
Gửi file này để phân tích bài báo.
