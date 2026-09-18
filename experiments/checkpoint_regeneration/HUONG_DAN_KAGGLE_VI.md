# HƯỚNG DẪN CHẠY TRÊN KAGGLE

## A. Lần chạy đầu tiên — seed 42

1. Upload package `OCRDA_v7_4_4_REGENERATE_FROZEN_45_MODELS_Kaggle.zip` thành một Kaggle Dataset riêng (Private).
2. Tạo notebook mới và chỉ Add Input: package này, `OntoShrimp-SDBD`, `OntoShrimp-TSBD`.
3. **Không Add Dataset3_Vietnam_v1.**
4. Settings -> Accelerator -> GPU; bật Internet để ImageNet pretrained ConvNeXt-Tiny khả dụng.
5. Copy toàn bộ `KAGGLE_ONE_CELL_INVENTORY_ONLY.txt` vào một cell và chạy. Lần đầu phải thấy `NEXT SEED: 42`.
6. Copy `KAGGLE_ONE_CELL_REGENERATE.txt` vào cell khác và chạy. Một execution chỉ chạy seed 42 (9 methods) rồi dừng.
7. Download ba file: `OCRDA_v7_4_4_REGENERATED_MODELS_SEED_42.zip`, `OCRDA_v7_4_4_REGEN_RESUME_LATEST.zip`, `OCRDA_v7_4_4_REGEN_REPORTS_LATEST.zip`.

## B. Nếu vẫn giữ cùng Kaggle session

Chỉ cần chạy lại cell REGENERATE. Runner thấy seed 42 đủ 9 model và chuyển sang seed 123. Lặp đến 2026, 3407, 7777.

## C. Nếu đã tắt/restart Kaggle

Trước khi chạy lại, upload/Add Input:
- latest `OCRDA_v7_4_4_REGEN_RESUME_LATEST.zip` (Kaggle có thể tự giải nén), và
- **tất cả seed-model bundle đã hoàn thành**, ví dụ `...SEED_42.zip`, sau đó thêm `...SEED_123.zip`, v.v.

Sau đó chạy INVENTORY ONLY. Ví dụ sau seed 42 và 123 phải thấy:
`seed 42: 9/9 COMPLETE`, `seed 123: 9/9 COMPLETE`, `NEXT SEED: 2026`.
Chỉ khi inventory đúng mới chạy cell REGENERATE.

## D. Sau seed 7777

Nếu 45/45 model đang visible, runner tự tạo `OCRDA_v744_FROZEN_45_MODELS_BUNDLE.zip`.
Nếu chưa tạo vì một số seed bundles không attach trong session cuối, Add Input đủ 5 seed bundles và chạy `KAGGLE_ONE_CELL_BUILD_FINAL_BUNDLE.txt`; bước này không train và không cần GPU.

Download `OCRDA_v744_FROZEN_45_MODELS_BUNDLE.zip`. Upload outer ZIP này thành một Kaggle Dataset. Sau đó dùng nó cùng Dataset 3 trong package `OCRDA_v7_4_4_FROZEN_DATASET3_EXTERNAL_EVAL_Kaggle.zip` để chạy inference-only external test.

## Nguyên tắc khoa học

Không rerun một checkpoint chỉ vì target-dev thấp; không chọn checkpoint tốt hơn giữa nhiều regeneration attempts; không thay hyperparameter; không attach Dataset 3 trong notebook regeneration. Nếu một run hoàn thành và model ZIP đã được bảo vệ, runner sẽ skip model đó.
