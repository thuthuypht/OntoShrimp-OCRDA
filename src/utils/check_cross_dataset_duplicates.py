#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-dataset duplicate checker for:
  Dataset 1 = Kaggle / ShrimpDataset
  Dataset 2 = TigerShrimpBD / Mendeley
  Dataset 3 = Vietnam-curated dataset

It is designed primarily to verify that Dataset 3's FINAL EXTERNAL TEST
does not overlap with Dataset 1 or Dataset 2.

Checks:
1) SHA256 exact duplicates
2) Perceptual hash (pHash + dHash) for resized/compressed variants
3) ORB local-feature matching for crop/resize/light editing candidates

Outputs:
- duplicate_pairs_all.csv
- duplicate_pairs_high_confidence.csv
- dataset3_images_to_exclude.csv
- dataset3_clean_keep_list.csv
- summary.txt

Install:
    pip install pillow imagehash opencv-python pandas tqdm

Example:
    python check_cross_dataset_duplicates.py ^
      --d1 "D:\ShrimpDataset" ^
      --d2 "D:\TigerShrimpBD" ^
      --d3 "D:\Dataset3_Vietnam_v1.zip" ^
      --out "D:\DS3_cross_dedup_results"

Notes:
- --d1, --d2, --d3 can be either folders or .zip files.
- For Dataset 3, if the folder
  "02_recommended_final_external_test_balanced_150" exists,
  only that subset is checked by default.
- Use --d3-all to check the entire Dataset 3 instead.
"""

import argparse
import csv
import hashlib
import math
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}

# Conservative default thresholds.
# These produce CANDIDATES; ORB then helps classify severity.
PHASH_CANDIDATE_MAX = 10
DHASH_CANDIDATE_MAX = 10

# Strong perceptual duplicate threshold
PHASH_STRONG_MAX = 4
DHASH_STRONG_MAX = 4

# ORB thresholds
ORB_MIN_GOOD_MATCHES = 18
ORB_STRONG_GOOD_MATCHES = 35
ORB_STRONG_INLIER_RATIO = 0.35

# For exact/near-exact visual duplicates with same overall composition.
COMBINED_PHASH_DHASH_STRONG = 8  # pHash + dHash distance


@dataclass
class ImgRec:
    dataset: str
    path: Path
    relpath: str
    label: str
    width: int
    height: int
    sha256: str
    phash: str
    dhash: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def open_pil_rgb(path: Path):
    with Image.open(path) as im:
        # First frame only for GIF
        try:
            im.seek(0)
        except Exception:
            pass
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB").copy()


def extract_if_zip(src: Path, temp_root: Path, name: str) -> Path:
    if src.is_dir():
        return src
    if src.is_file() and src.suffix.lower() == ".zip":
        dst = temp_root / name
        dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src, "r") as z:
            z.extractall(dst)
        return dst
    raise FileNotFoundError(f"Not a folder or ZIP: {src}")


def infer_label(path: Path, root: Path) -> str:
    """
    Infer class from parent folders where possible.
    Handles BG/Healthy/WSSV and common Vietnamese/English variants.
    """
    parts = [p.lower() for p in path.relative_to(root).parts[:-1]]
    joined = " / ".join(parts)

    bg_terms = ["bg", "black_gill", "blackgill", "den_mang", "đen_mang", "den mang", "đen mang"]
    healthy_terms = ["healthy", "tom_bt", "normal", "binh_thuong", "bình_thường", "khoe", "khỏe"]
    wssv_terms = ["wssv", "white_spot", "whitespot", "dom_trang", "đốm_trắng", "dom trang", "đốm trắng"]

    if any(t in joined for t in bg_terms):
        return "BG"
    if any(t in joined for t in healthy_terms):
        return "Healthy"
    if any(t in joined for t in wssv_terms):
        return "WSSV"

    # Fallback: immediate parent
    return path.parent.name


def collect_images(dataset_name: str, root: Path, restrict_d3_final: bool = False) -> Tuple[Path, List[Path]]:
    scan_root = root

    if restrict_d3_final:
        candidates = list(root.rglob("02_recommended_final_external_test_balanced_150"))
        candidates = [p for p in candidates if p.is_dir()]
        if candidates:
            # Prefer shortest path (closest to extracted root)
            scan_root = sorted(candidates, key=lambda p: len(p.parts))[0]
            print(f"[Dataset3] Restricting to final external test: {scan_root}")
        else:
            print("[Dataset3] Final-test folder not found; scanning whole dataset.")

    files = [
        p for p in scan_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return scan_root, sorted(files)


def build_records(dataset_name: str, scan_root: Path, files: List[Path]) -> List[ImgRec]:
    records = []
    print(f"\nComputing hashes for {dataset_name}: {len(files)} images")

    for p in tqdm(files, desc=dataset_name):
        try:
            img = open_pil_rgb(p)
            w, h = img.size
            ph = str(imagehash.phash(img))
            dh = str(imagehash.dhash(img))
            rec = ImgRec(
                dataset=dataset_name,
                path=p,
                relpath=str(p.relative_to(scan_root)).replace("\\", "/"),
                label=infer_label(p, scan_root),
                width=w,
                height=h,
                sha256=sha256_file(p),
                phash=ph,
                dhash=dh,
            )
            records.append(rec)
        except Exception as e:
            print(f"\n[WARN] Skip unreadable image: {p}\n  {e}")

    return records


def hdist(h1: str, h2: str) -> int:
    return imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2)


def load_cv_gray(path: Path, max_side=1200):
    """
    Robustly read Unicode Windows paths.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    except Exception:
        img = None
    if img is None:
        return None

    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def orb_compare(path1: Path, path2: Path) -> Dict[str, float]:
    """
    ORB + ratio test + RANSAC homography.
    Returns:
      good_matches
      inliers
      inlier_ratio
      kp1
      kp2
    """
    a = load_cv_gray(path1)
    b = load_cv_gray(path2)
    if a is None or b is None:
        return {"good_matches": 0, "inliers": 0, "inlier_ratio": 0.0, "kp1": 0, "kp2": 0}

    orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8, fastThreshold=7)
    kp1, des1 = orb.detectAndCompute(a, None)
    kp2, des2 = orb.detectAndCompute(b, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return {
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "kp1": len(kp1) if kp1 else 0,
            "kp2": len(kp2) if kp2 else 0,
        }

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    inliers = 0
    inlier_ratio = 0.0

    if len(good) >= 8:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        try:
            H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
            if mask is not None:
                inliers = int(mask.ravel().sum())
                inlier_ratio = inliers / max(len(good), 1)
        except cv2.error:
            pass

    return {
        "good_matches": int(len(good)),
        "inliers": int(inliers),
        "inlier_ratio": float(inlier_ratio),
        "kp1": int(len(kp1)),
        "kp2": int(len(kp2)),
    }


def classify_pair(exact: bool, pd: int, dd: int, orb: Dict[str, float]) -> Tuple[str, str]:
    """
    Returns (confidence, reason).
    """
    if exact:
        return "EXACT", "SHA256 identical"

    combined = pd + dd
    gm = orb["good_matches"]
    ir = orb["inlier_ratio"]
    inl = orb["inliers"]

    # Very strong perceptual equivalence.
    if pd <= 2 and dd <= 2:
        return "HIGH", f"very close pHash/dHash ({pd},{dd})"

    # Same composition after resize/JPEG, backed by ORB.
    if combined <= COMBINED_PHASH_DHASH_STRONG and gm >= ORB_MIN_GOOD_MATCHES:
        return "HIGH", f"close hashes + ORB ({gm} good, {inl} inliers, ratio={ir:.2f})"

    # Crop/resize can change pHash substantially; ORB can still catch it.
    if gm >= ORB_STRONG_GOOD_MATCHES and ir >= ORB_STRONG_INLIER_RATIO:
        return "HIGH", f"strong ORB geometric match ({gm} good, {inl} inliers, ratio={ir:.2f})"

    # Conservative medium candidate.
    if pd <= PHASH_STRONG_MAX and dd <= DHASH_STRONG_MAX:
        return "MEDIUM", f"close pHash/dHash ({pd},{dd}), ORB weaker"

    if gm >= ORB_MIN_GOOD_MATCHES and ir >= 0.20:
        return "MEDIUM", f"ORB candidate ({gm} good, {inl} inliers, ratio={ir:.2f})"

    return "LOW", "weak candidate"


def generate_candidates(ds3: List[ImgRec], other: List[ImgRec]) -> List[Tuple[ImgRec, ImgRec, int, int]]:
    """
    Cross-compare Dataset3 images only against another dataset.
    Candidate if exact SHA OR pHash/dHash reasonably close.

    Also uses loose hash thresholds so ORB can verify likely resize/crop variants.
    """
    # exact SHA lookup first
    sha_map = {}
    for b in other:
        sha_map.setdefault(b.sha256, []).append(b)

    candidates = []
    seen = set()

    for a in tqdm(ds3, desc=f"Candidate scan vs {other[0].dataset if other else 'dataset'}"):
        # exact
        for b in sha_map.get(a.sha256, []):
            key = (a.relpath, b.relpath)
            if key not in seen:
                candidates.append((a, b, 0, 0))
                seen.add(key)

        # perceptual
        for b in other:
            key = (a.relpath, b.relpath)
            if key in seen:
                continue
            pd = hdist(a.phash, b.phash)
            dd = hdist(a.dhash, b.dhash)
            if pd <= PHASH_CANDIDATE_MAX or dd <= DHASH_CANDIDATE_MAX:
                candidates.append((a, b, pd, dd))
                seen.add(key)

    return candidates


def compare_ds3_to_dataset(ds3: List[ImgRec], other: List[ImgRec], out_rows: List[Dict]):
    if not other:
        return

    candidates = generate_candidates(ds3, other)
    print(f"Candidates requiring ORB vs {other[0].dataset}: {len(candidates)}")

    for a, b, pd, dd in tqdm(candidates, desc=f"ORB vs {other[0].dataset}"):
        exact = a.sha256 == b.sha256
        orb = {"good_matches": 0, "inliers": 0, "inlier_ratio": 0.0, "kp1": 0, "kp2": 0}

        # No need ORB for exact.
        if not exact:
            orb = orb_compare(a.path, b.path)

        conf, reason = classify_pair(exact, pd, dd, orb)

        # Keep all candidates in full CSV, including LOW.
        out_rows.append({
            "ds3_relpath": a.relpath,
            "ds3_label": a.label,
            "ds3_width": a.width,
            "ds3_height": a.height,
            "other_dataset": b.dataset,
            "other_relpath": b.relpath,
            "other_label": b.label,
            "other_width": b.width,
            "other_height": b.height,
            "exact_sha256": exact,
            "phash_distance": pd,
            "dhash_distance": dd,
            "orb_good_matches": orb["good_matches"],
            "orb_inliers": orb["inliers"],
            "orb_inlier_ratio": round(orb["inlier_ratio"], 4),
            "duplicate_confidence": conf,
            "reason": reason,
            "label_agreement": a.label == b.label,
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d1", required=True, help="Dataset 1 folder or ZIP")
    ap.add_argument("--d2", required=True, help="Dataset 2 folder or ZIP")
    ap.add_argument("--d3", required=True, help="Dataset 3 folder or ZIP")
    ap.add_argument("--out", required=True, help="Output result folder")
    ap.add_argument(
        "--d3-all",
        action="store_true",
        help="Check entire Dataset3 instead of only recommended final external test"
    )
    ap.add_argument(
        "--exclude-medium",
        action="store_true",
        help="Also exclude MEDIUM-confidence matches from DS3 keep list"
    )
    args = ap.parse_args()

    d1_src = Path(args.d1).expanduser().resolve()
    d2_src = Path(args.d2).expanduser().resolve()
    d3_src = Path(args.d3).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    temp_root = Path(tempfile.mkdtemp(prefix="cross_dedup_"))
    try:
        d1_root = extract_if_zip(d1_src, temp_root, "dataset1")
        d2_root = extract_if_zip(d2_src, temp_root, "dataset2")
        d3_root = extract_if_zip(d3_src, temp_root, "dataset3")

        d1_scan, d1_files = collect_images("Dataset1", d1_root, False)
        d2_scan, d2_files = collect_images("Dataset2", d2_root, False)
        d3_scan, d3_files = collect_images("Dataset3", d3_root, not args.d3_all)

        print("\nImage counts:")
        print(f"  Dataset1: {len(d1_files)}")
        print(f"  Dataset2: {len(d2_files)}")
        print(f"  Dataset3 checked: {len(d3_files)}")

        r1 = build_records("Dataset1", d1_scan, d1_files)
        r2 = build_records("Dataset2", d2_scan, d2_files)
        r3 = build_records("Dataset3", d3_scan, d3_files)

        rows = []
        compare_ds3_to_dataset(r3, r1, rows)
        compare_ds3_to_dataset(r3, r2, rows)

        df = pd.DataFrame(rows)
        if len(df) == 0:
            df = pd.DataFrame(columns=[
                "ds3_relpath","ds3_label","other_dataset","other_relpath",
                "exact_sha256","phash_distance","dhash_distance",
                "orb_good_matches","orb_inliers","orb_inlier_ratio",
                "duplicate_confidence","reason","label_agreement"
            ])

        # Sort strongest first
        rank = {"EXACT": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        if len(df):
            df["_rank"] = df["duplicate_confidence"].map(rank).fillna(9)
            df = df.sort_values(
                ["_rank", "phash_distance", "dhash_distance", "orb_inlier_ratio"],
                ascending=[True, True, True, False]
            ).drop(columns="_rank")

        df.to_csv(out / "duplicate_pairs_all.csv", index=False, encoding="utf-8-sig")

        high = df[df["duplicate_confidence"].isin(["EXACT", "HIGH"])].copy()
        high.to_csv(out / "duplicate_pairs_high_confidence.csv", index=False, encoding="utf-8-sig")

        med = df[df["duplicate_confidence"] == "MEDIUM"].copy()
        med.to_csv(out / "duplicate_pairs_manual_review.csv", index=False, encoding="utf-8-sig")

        # Decide exclusions
        exclude_levels = {"EXACT", "HIGH"}
        if args.exclude_medium:
            exclude_levels.add("MEDIUM")

        exclude_paths = set(
            df.loc[df["duplicate_confidence"].isin(exclude_levels), "ds3_relpath"].tolist()
        )

        exclude_df = (
            df[df["duplicate_confidence"].isin(exclude_levels)]
            .sort_values(["ds3_relpath", "duplicate_confidence"])
            .groupby("ds3_relpath", as_index=False)
            .agg({
                "ds3_label": "first",
                "other_dataset": lambda x: "; ".join(sorted(set(x))),
                "other_relpath": lambda x: "; ".join(list(x)[:10]),
                "duplicate_confidence": lambda x: "; ".join(sorted(set(x))),
                "reason": lambda x: " | ".join(list(x)[:5]),
            })
        )
        exclude_df.to_csv(out / "dataset3_images_to_exclude.csv", index=False, encoding="utf-8-sig")

        keep_rows = []
        for r in r3:
            keep_rows.append({
                "ds3_relpath": r.relpath,
                "label": r.label,
                "keep": r.relpath not in exclude_paths,
                "status": "KEEP" if r.relpath not in exclude_paths else "EXCLUDE_OVERLAP",
            })
        keep_df = pd.DataFrame(keep_rows)
        keep_df.to_csv(out / "dataset3_clean_keep_list.csv", index=False, encoding="utf-8-sig")

        # Per-label summary
        total_by_label = keep_df.groupby("label").size().to_dict()
        keep_by_label = keep_df[keep_df["keep"]].groupby("label").size().to_dict()
        excl_by_label = keep_df[~keep_df["keep"]].groupby("label").size().to_dict()

        lines = []
        lines.append("CROSS-DATASET DEDUP SUMMARY\n")
        lines.append(f"Dataset1 images: {len(r1)}")
        lines.append(f"Dataset2 images: {len(r2)}")
        lines.append(f"Dataset3 checked: {len(r3)}")
        lines.append("")
        lines.append(f"Exact matches: {(df['duplicate_confidence']=='EXACT').sum() if len(df) else 0}")
        lines.append(f"High-confidence near duplicates: {(df['duplicate_confidence']=='HIGH').sum() if len(df) else 0}")
        lines.append(f"Medium manual-review pairs: {(df['duplicate_confidence']=='MEDIUM').sum() if len(df) else 0}")
        lines.append(f"Dataset3 images excluded: {len(exclude_paths)}")
        lines.append(f"Dataset3 images retained: {len(r3)-len(exclude_paths)}")
        lines.append("")
        lines.append("Per-class Dataset3:")
        for cls in sorted(total_by_label):
            lines.append(
                f"  {cls}: total={total_by_label.get(cls,0)}, "
                f"exclude={excl_by_label.get(cls,0)}, keep={keep_by_label.get(cls,0)}"
            )
        lines.append("")
        lines.append("Recommended interpretation:")
        lines.append("- EXACT/HIGH: remove from Dataset3 final external test.")
        lines.append("- MEDIUM: visually inspect before deciding.")
        lines.append("- LOW: normally keep unless manual inspection shows overlap.")
        lines.append("- If a Dataset3 image overlaps either Dataset1 or Dataset2, it is NOT independent external test data.")
        lines.append("- After exclusions, re-balance the final test if class counts become unequal.")

        (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

        print("\n" + "\n".join(lines))
        print(f"\nResults saved to: {out}")

    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
