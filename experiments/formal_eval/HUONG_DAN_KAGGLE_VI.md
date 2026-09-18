# Hướng dẫn chạy trên Kaggle

## Input bắt buộc
Chỉ attach 3 input:
- `OCRDA_v7_4_4_FROZEN_BASELINES_ABLATIONS_Kaggle`
- `OntoShrimp-SDBD`
- `OntoShrimp-TSBD`

**Không attach Dataset 3.** Runner sẽ dừng nếu phát hiện Dataset 3.

## Batch A — baselines
Mở `Batch_A_Kaggle.ipynb` hoặc copy `KAGGLE_ONE_CELL_BATCH_A.txt`. Mỗi lần Run xử lý 1 seed và 4 phương pháp baseline. Sau mỗi seed, tải `OCRDA_v7_4_4_FROZEN_BATCH_A_REPORTS_LATEST.zip`, `...RESUME_LATEST.zip`, và các MODEL ZIP của seed đó. Rerun cell cho seed tiếp theo.

## Batch B — frozen Full + ablations
Mở `Batch_B_Kaggle.ipynb` hoặc copy `KAGGLE_ONE_CELL_BATCH_B.txt`. Mỗi lần Run xử lý 1 seed. Full OCRDA-v7.4.4 chạy trước để tạo source-warm cache; 4 ablation bắt buộc dùng cùng cache này. Không được sửa command/hyperparameter trong notebook.

## Resume sau runtime reset
Upload file `...RESUME_LATEST.zip` mới nhất thành một Kaggle Dataset riêng, Add Input cùng 3 input trên, rồi chạy lại đúng cell. Các method/seed đã hoàn tất sẽ được skip. Nếu Batch B cần tiếp tục ablation nhưng cache tạm đã mất, runner sẽ deterministically rerun frozen Full cho seed đó để tái tạo exact cache trước khi tiếp tục.

## Kết thúc
Khi cả hai batch đủ 5 seeds, tải `OCRDA_v7_4_4_FROZEN_EVAL_REPORTS_FINAL.zip` và gửi để tổng hợp bảng/kiểm định. Chỉ sau khi analysis plan được khóa mới mở Dataset 3.
