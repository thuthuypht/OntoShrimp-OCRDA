# -*- coding: utf-8 -*-
"""
bootstrap_test.py — Kiểm định thống kê (cô yêu cầu) cho kết quả 3 hệ thống.

Dùng bootstrap resampling để tính:
  - Khoảng tin cậy 95% (CI) cho Macro-F1 của từng hệ thống
  - p-value cho chênh lệch "YOLO+Ontology vs YOLO-only" (Ontology có thật sự tốt hơn?)

Vì cỡ mẫu test nhỏ (30 ca), bootstrap giúp định lượng độ tin cậy — trả lời
trực diện góp ý "chưa có kiểm định thống kê" của cô.

CÁCH DÙNG: chạy SAU eval_3_systems_kaggle.py (cần file eval_results_clean.json).
"""
import json, random

RESULTS = "/kaggle/working/eval_results_clean.json"
N_BOOT  = 2000
SEED    = 42
random.seed(SEED)

data = json.load(open(RESULTS, encoding="utf-8"))
records = data["records"]
CLASSES = ["WSSV", "BlackGillDisease", "Healthy"]
SYSTEMS = {"YOLO-only":"pred_yolo", "YOLO+Heuristic":"pred_heuristic", "YOLO+Ontology":"pred_ontology"}

def macro_f1(sample, key):
    f1s = []
    for c in CLASSES:
        tp = sum(1 for r in sample if r["gt"]==c and r[key]==c)
        fp = sum(1 for r in sample if r["gt"]!=c and r[key]==c)
        fn = sum(1 for r in sample if r["gt"]==c and r[key]!=c)
        p = tp/(tp+fp) if tp+fp else 0.0
        r_ = tp/(tp+fn) if tp+fn else 0.0
        f1s.append(2*p*r_/(p+r_) if p+r_ else 0.0)
    return sum(f1s)/len(f1s)

def percentile(xs, q):
    xs = sorted(xs); i = q*(len(xs)-1)
    lo = int(i); frac = i-lo
    return xs[lo] if lo+1>=len(xs) else xs[lo]*(1-frac)+xs[lo+1]*frac

n = len(records)
boot = {name: [] for name in SYSTEMS}
diff_onto_yolo = []  # Ontology - YOLO-only mỗi lần resample

for _ in range(N_BOOT):
    sample = [records[random.randrange(n)] for _ in range(n)]  # resample có hoàn lại
    f1_vals = {name: macro_f1(sample, key) for name, key in SYSTEMS.items()}
    for name in SYSTEMS: boot[name].append(f1_vals[name])
    diff_onto_yolo.append(f1_vals["YOLO+Ontology"] - f1_vals["YOLO-only"])

print("="*64)
print(f"  KIỂM ĐỊNH BOOTSTRAP ({N_BOOT} lần resample, n={n} ca)")
print("="*64)
print(f"{'Hệ thống':<18}{'Macro-F1':>10}{'CI 95% dưới':>14}{'CI 95% trên':>14}")
print("-"*56)
for name in SYSTEMS:
    vals = boot[name]
    mean = sum(vals)/len(vals)
    lo, hi = percentile(vals, 0.025), percentile(vals, 0.975)
    print(f"{name:<18}{mean:>10.3f}{lo:>14.3f}{hi:>14.3f}")

# p-value 1 phía: P(Ontology KHÔNG tốt hơn YOLO-only)
p_val = sum(1 for d in diff_onto_yolo if d <= 0) / N_BOOT
mean_diff = sum(diff_onto_yolo)/len(diff_onto_yolo)
lo_d, hi_d = percentile(diff_onto_yolo,0.025), percentile(diff_onto_yolo,0.975)
print("\n" + "-"*56)
print("So sánh YOLO+Ontology vs YOLO-only (chênh lệch Macro-F1):")
print(f"  Chênh lệch TB = {mean_diff:+.3f}  (CI95%: {lo_d:+.3f} .. {hi_d:+.3f})")
print(f"  p-value (1 phía) = {p_val:.4f}")
if p_val < 0.05:
    print("  => Ontology tốt hơn YOLO-only có Ý NGHĨA THỐNG KÊ (p<0.05).")
else:
    print("  => Chưa đủ ý nghĩa thống kê (p>=0.05) — do cỡ mẫu nhỏ. Nêu trung thực trong báo cáo.")

print("\nCÁCH VIẾT VÀO BÁO CÁO (ví dụ):")
print('  "Với bootstrap 2000 lần trên 30 ca test, Macro-F1 của YOLO+Ontology')
print('   đạt [mean] (CI95%: [lo]–[hi]); chênh lệch so với YOLO-only là [diff]")')
