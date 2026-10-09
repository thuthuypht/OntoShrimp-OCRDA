# -*- coding: utf-8 -*-
"""
eval_3_systems_kaggle.py — Đánh giá 3 hệ thống (YOLO-only / YOLO+Heuristic / YOLO+Ontology)
trên TẬP TEST SẠCH = 30 ca mendeley (TigerShrimpBD), KHÔNG rò rỉ với train.

Giữ NGUYÊN logic ontology của em (import từ ontology_reasoning_pipeline.py).
Chỉ sửa: đường dẫn Kaggle + model mới + tập test sạch.

CÁCH DÙNG TRÊN KAGGLE:
1. Tạo 1 Kaggle Dataset chứa: mendeley_test/, DATN.rdf, ontology_reasoning_pipeline.py
   (hướng dẫn tạo ở tin nhắn kèm theo). Add Input vào notebook.
2. Sửa 3 đường dẫn CONFIG bên dưới cho khớp.
3. Chạy ô này.
"""
import os, sys, json
from pathlib import Path

# ════════════════ CONFIG — SỬA 3 ĐƯỜNG DẪN NÀY ════════════════
# (a) Model 1024 mới vừa train. Nếu eval cùng session train: dùng đường dẫn working.
#     Nếu train ở Commit khác: tải best.pt rồi thêm vào input, trỏ tới đó.
MODEL_PATH = "/kaggle/working/runs/detect/runs_improve/exp_yolo11x_1024/weights/best.pt"

# (b) Thư mục input (Kaggle Dataset em vừa tạo). Đổi <ten-dataset> cho đúng slug.
INPUT_DIR  = Path("/kaggle/input/shrimp-eval-input")

# (c) Tên file ontology + pipeline trong INPUT_DIR
ONTO_PATH    = INPUT_DIR / "DATN.rdf"
PIPELINE_DIR = INPUT_DIR            # nơi chứa ontology_reasoning_pipeline.py
MENDELEY     = INPUT_DIR / "mendeley_test"
CONF_THRESH  = 0.25
# ══════════════════════════════════════════════════════════════

# Tập test SẠCH: chỉ 30 ca mendeley (TigerShrimpBD), không nằm trong train
TEST_SETS = [
    (MENDELEY / "Black Gill", "BlackGillDisease"),
    (MENDELEY / "WSSV",       "WSSV"),
    (MENDELEY / "Healthy",    "Healthy"),
]

print("="*60); print("  KIỂM TRA CẤU HÌNH"); print("="*60)
ok = True
if not os.path.exists(MODEL_PATH):
    print(f"  [THIẾU] model: {MODEL_PATH}"); ok = False
else:
    print(f"  [OK] model: {MODEL_PATH}")
if not ONTO_PATH.exists():
    print(f"  [THIẾU] ontology: {ONTO_PATH}"); ok = False
if not (PIPELINE_DIR / "ontology_reasoning_pipeline.py").exists():
    print(f"  [THIẾU] ontology_reasoning_pipeline.py trong {PIPELINE_DIR}"); ok = False
for folder, _ in TEST_SETS:
    if folder.exists():
        n = len(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
        print(f"  [OK] {folder.name}: {n} ảnh")
    else:
        print(f"  [THIẾU] {folder}"); ok = False
if not ok:
    print("\nLỖI cấu hình — sửa đường dẫn CONFIG rồi chạy lại."); sys.exit(1)

# ── Load model + pipeline ──────────────────────────────────
from ultralytics import YOLO
print("\nLoading YOLO model..."); model = YOLO(str(MODEL_PATH))
print(f"  Classes: {model.names}")

sys.path.insert(0, str(PIPELINE_DIR))
from ontology_reasoning_pipeline import yolo_to_rdf, apply_swrl_rules

# ── 3 hàm phân loại (giữ nguyên logic gốc của em) ──────────
def classify_yolo_only(dets):
    if not dets: return "Healthy"
    ws = any(d["class"]=="white_spot" for d in dets)
    bg = any(d["class"]=="black_gill" for d in dets)
    if ws and bg:
        best = max(dets, key=lambda d: d["confidence"])
        return "WSSV" if best["class"]=="white_spot" else "BlackGillDisease"
    if ws: return "WSSV"
    if bg: return "BlackGillDisease"
    return "Healthy"

def classify_heuristic(dets):
    if not dets: return "Healthy"
    ws = any(d["class"]=="white_spot" for d in dets)
    bg = any(d["class"]=="black_gill" for d in dets)
    if ws and bg: return "CoInfection"
    if ws: return "WSSV"
    if bg: return "BlackGillDisease"
    return "Healthy"

def classify_ontology(dets, image_id):
    try:
        g, case_uri = yolo_to_rdf(str(ONTO_PATH), dets, image_id, "YOLO11x_1024")
        res = apply_swrl_rules(g, case_uri, dets)
        d = res.get("disease") or "Healthy"
        if "CoInfection" in d: return "CoInfection"
        if "WSSV" in d:        return "WSSV"
        if "BlackGill" in d or "BG" in d: return "BlackGillDisease"
        return "Healthy"
    except Exception as e:
        print(f"    [WARN] ontology error: {e}")
        return classify_heuristic(dets)

# ── Inference ──────────────────────────────────────────────
records = []
print("\n" + "="*60); print("  INFERENCE — 30 CA TEST SẠCH (mendeley)"); print("="*60)
for folder, gt in TEST_SETS:
    imgs = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
    print(f"\n[{folder.name}] GT={gt} ({len(imgs)} ảnh)")
    for img in imgs:
        res = model(str(img), conf=CONF_THRESH, verbose=False)
        b = res[0].boxes
        dets = [{"class": model.names[int(c)], "confidence": round(float(cf),4)}
                for c, cf in zip(b.cls.tolist(), b.conf.tolist())]
        rec = {"image": img.name, "folder": folder.name, "gt": gt, "detections": dets,
               "pred_yolo": classify_yolo_only(dets),
               "pred_heuristic": classify_heuristic(dets),
               "pred_ontology": classify_ontology(dets, img.stem)}
        records.append(rec)
        print(f"  {img.name:28s}| {len(dets)}box | Y:{rec['pred_yolo']:16s}| "
              f"H:{rec['pred_heuristic']:16s}| O:{rec['pred_ontology']}")

# ── Metrics ────────────────────────────────────────────────
def compute(records, key):
    classes = ["WSSV","BlackGillDisease","Healthy"]
    acc = sum(1 for r in records if r[key]==r["gt"])/len(records)
    pc = {}
    for c in classes:
        tp = sum(1 for r in records if r["gt"]==c and r[key]==c)
        fp = sum(1 for r in records if r["gt"]!=c and r[key]==c)
        fn = sum(1 for r in records if r["gt"]==c and r[key]!=c)
        p = tp/(tp+fp) if tp+fp else 0.0
        r_ = tp/(tp+fn) if tp+fn else 0.0
        f1 = 2*p*r_/(p+r_) if p+r_ else 0.0
        pc[c] = {"P":round(p,3),"R":round(r_,3),"F1":round(f1,3),"TP":tp,"FP":fp,"FN":fn}
    return {"accuracy":round(acc,3),
            "macro_P":round(sum(v["P"] for v in pc.values())/3,3),
            "macro_R":round(sum(v["R"] for v in pc.values())/3,3),
            "macro_F1":round(sum(v["F1"] for v in pc.values())/3,3),
            "per_class":pc}

m = {"YOLO-only":compute(records,"pred_yolo"),
     "YOLO+Heuristic":compute(records,"pred_heuristic"),
     "YOLO+Ontology":compute(records,"pred_ontology")}

print("\n"+"="*60); print("  KẾT QUẢ — 30 CA TEST SẠCH"); print("="*60)
print(f"{'Hệ thống':<18}{'P':>8}{'R':>8}{'F1':>8}{'Acc':>8}")
for name,v in m.items():
    print(f"{name:<18}{v['macro_P']:>8.3f}{v['macro_R']:>8.3f}{v['macro_F1']:>8.3f}{v['accuracy']:>8.3f}")

out = {"total_cases":len(records),"metrics":m,"records":records}
with open("/kaggle/working/eval_results_clean.json","w",encoding="utf-8") as f:
    json.dump(out,f,ensure_ascii=False,indent=2)
print("\n✓ Đã lưu /kaggle/working/eval_results_clean.json")
print("→ Dùng file này cho bootstrap_test.py. COPY OUTPUT GỬI CLAUDE!")
