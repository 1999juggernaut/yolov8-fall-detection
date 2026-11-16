"""Train YOLOv8s for single-class fall detection with the settings used for the reported results.

Usage:
  python src/train.py --data path/to/data.yaml [--epochs 100] [--device 0] [--project runs]

Reported run: Tesla T4, 100 epochs, ~2.05 h. Best checkpoint: mAP@50 0.909, mAP@50-95 0.614.
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from ultralytics import YOLO

TRAIN_ARGS = dict(
    imgsz=640,
    batch=16,
    patience=20,          # early stopping
    optimizer="AdamW",
    lr0=0.001,
    cos_lr=True,
    weight_decay=0.0005,
    # augmentation (reduces overfitting on a ~3k image train split)
    mosaic=1.0,
    close_mosaic=10,      # last 10 epochs without mosaic -> visible loss drop at epoch 90
    mixup=0.1,
    degrees=10,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    seed=42,
    deterministic=True,
    save_period=1,        # checkpoint every epoch so a Colab disconnect does not lose the run
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--project", default="runs/fall")
    ap.add_argument("--name", default="exp_v8s")
    ap.add_argument("--resume", action="store_true", help="resume from <project>/<name>/weights/last.pt")
    args = ap.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if args.resume:
        YOLO(f"{args.project}/{args.name}/weights/last.pt").train(resume=True)
        return

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, device=args.device,
                project=args.project, name=args.name, **TRAIN_ARGS)
    metrics = model.val(data=args.data, split="val")
    print(f"mAP@50={metrics.box.map50:.3f}  mAP@50-95={metrics.box.map:.3f}  "
          f"P={metrics.box.mp:.3f}  R={metrics.box.mr:.3f}")


if __name__ == "__main__":
    main()
