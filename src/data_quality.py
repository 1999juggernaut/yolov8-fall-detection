"""Data-quality pipeline for a YOLO-format dataset.

Steps (run before training):
  1. Remove images OpenCV cannot decode (and their label files).
  2. Remove exact duplicate images by MD5 hash (and their label files).
  3. Drop outlier boxes whose normalised area (w * h) is outside [min_area, max_area]
     -- applied to the training split only, so validation stays untouched.

Usage:
  python src/data_quality.py --dataset path/to/dataset            # dry run: report only
  python src/data_quality.py --dataset path/to/dataset --apply    # actually delete / rewrite

Expected layout: <dataset>/<split>/images/*.jpg and <dataset>/<split>/labels/*.txt
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
from collections import Counter

import cv2

SPLITS = ("train", "valid", "test")


def _label_for(img_path: str, lbl_dir: str) -> str:
    return os.path.join(lbl_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")


def audit_split(dataset: str, split: str) -> dict:
    img_dir = os.path.join(dataset, split, "images")
    lbl_dir = os.path.join(dataset, split, "labels")
    imgs = glob.glob(os.path.join(img_dir, "*.jpg"))
    lbls = glob.glob(os.path.join(lbl_dir, "*.txt"))
    boxes, classes = 0, Counter()
    for lbl in lbls:
        with open(lbl) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 5:
                    boxes += 1
                    classes[int(parts[0])] += 1
    return {"split": split, "images": len(imgs), "labels": len(lbls), "boxes": boxes, "classes": dict(classes)}


def remove_corrupt_and_duplicates(dataset: str, split: str, apply: bool) -> tuple[int, int]:
    img_dir = os.path.join(dataset, split, "images")
    lbl_dir = os.path.join(dataset, split, "labels")
    seen: dict[str, str] = {}
    n_corrupt = n_dup = 0
    for p in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
        bad = cv2.imread(p) is None
        if not bad:
            with open(p, "rb") as f:
                h = hashlib.md5(f.read()).hexdigest()
            dup = h in seen
            seen.setdefault(h, p)
        if bad or dup:
            n_corrupt += bad
            n_dup += (not bad)
            if apply:
                os.remove(p)
                lbl = _label_for(p, lbl_dir)
                if os.path.exists(lbl):
                    os.remove(lbl)
    return n_corrupt, n_dup


def filter_outlier_boxes(dataset: str, split: str, min_area: float, max_area: float, apply: bool) -> tuple[int, int]:
    lbl_dir = os.path.join(dataset, split, "labels")
    kept = removed = 0
    for p in glob.glob(os.path.join(lbl_dir, "*.txt")):
        new_lines = []
        with open(p) as f:
            for line in f:
                parts = line.split()
                if len(parts) != 5:
                    continue
                w, h = float(parts[3]), float(parts[4])
                if min_area <= w * h <= max_area:
                    new_lines.append(line if line.endswith("\n") else line + "\n")
                    kept += 1
                else:
                    removed += 1
        if apply:
            with open(p, "w") as f:
                f.writelines(new_lines)
    return kept, removed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="dataset root containing train/valid/test")
    ap.add_argument("--apply", action="store_true", help="delete/rewrite files (default: report only)")
    ap.add_argument("--min-area", type=float, default=0.001)
    ap.add_argument("--max-area", type=float, default=0.9)
    args = ap.parse_args()

    splits = [s for s in SPLITS if os.path.isdir(os.path.join(args.dataset, s))]
    print(f"{'split':<8}{'images':>8}{'labels':>8}{'boxes':>8}  classes")
    for s in splits:
        a = audit_split(args.dataset, s)
        print(f"{s:<8}{a['images']:>8}{a['labels']:>8}{a['boxes']:>8}  {a['classes']}")

    mode = "APPLY" if args.apply else "DRY RUN"
    for s in splits:
        c, d = remove_corrupt_and_duplicates(args.dataset, s, args.apply)
        print(f"[{mode}] [{s}] corrupted: {c}, duplicates: {d}")
    if "train" in splits:
        k, r = filter_outlier_boxes(args.dataset, "train", args.min_area, args.max_area, args.apply)
        print(f"[{mode}] [train] boxes kept: {k}, outlier boxes: {r}")


if __name__ == "__main__":
    main()
