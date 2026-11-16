"""Run the trained fall detector on an image, a folder, a video file or a webcam.

Usage:
  python src/detect.py --weights weights/best.pt --source path/to/image_or_video.mp4
  python src/detect.py --weights weights/best.pt --source 0            # webcam
  python src/detect.py --weights weights/best.pt --source clip.mp4 --alert-frames 5

--alert-frames N prints an ALERT once a fall is detected in N consecutive frames,
which suppresses one-frame false positives in video monitoring.
"""
from __future__ import annotations

import argparse
import time

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--source", required=True)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--alert-frames", type=int, default=5)
    ap.add_argument("--save", action="store_true", help="save annotated output under runs/detect")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(args.weights)

    streak, alerted, frames, t0 = 0, False, 0, time.perf_counter()
    for r in model.predict(source=source, conf=args.conf, stream=True, save=args.save, verbose=False):
        frames += 1
        falls = len(r.boxes)
        streak = streak + 1 if falls else 0
        if streak >= args.alert_frames and not alerted:
            print(f"ALERT: fall detected for {streak} consecutive frames (frame {frames})")
            alerted = True
        if not falls:
            alerted = False
        if frames <= 3 or falls:
            confs = ", ".join(f"{c:.2f}" for c in r.boxes.conf.tolist())
            print(f"frame {frames}: {falls} fall box(es) {('[' + confs + ']') if confs else ''}")
    dt = time.perf_counter() - t0
    print(f"processed {frames} frame(s) in {dt:.1f}s ({frames / dt:.1f} FPS end-to-end incl. I/O)")


if __name__ == "__main__":
    main()
