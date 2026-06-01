"""
Door & Window Detector — Inference
Run detection on an image, folder, or webcam feed.

Usage:
    python detect.py image.jpg
    python detect.py images/              # folder of images
    python detect.py image.jpg --weights runs/detect/door_window_v1/weights/best.pt
    python detect.py image.jpg --conf 0.5 --save
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

CLASSES = {0: "door", 1: "window"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run door/window detection on image(s)")
    parser.add_argument("source",          help="Image path, folder, or 0 for webcam")
    parser.add_argument("--weights", default="yolov8n.pt",
                        help="Model weights (default: yolov8n.pt pretrained, use best.pt after training)")
    parser.add_argument("--conf",    type=float, default=0.4,  help="Confidence threshold")
    parser.add_argument("--iou",     type=float, default=0.5,  help="NMS IoU threshold")
    parser.add_argument("--imgsz",   type=int,   default=640,  help="Inference image size")
    parser.add_argument("--save",    action="store_true",      help="Save annotated images to runs/")
    parser.add_argument("--show",    action="store_true",      help="Display results in window")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        save=args.save,
        show=args.show,
        device="cpu",
        verbose=False,
    )

    for i, r in enumerate(results):
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            print(f"[{i}] No detections.")
            continue

        counts: dict[str, int] = {}
        for box in boxes:
            name = r.names[int(box.cls)]
            counts[name] = counts.get(name, 0) + 1

        summary = ", ".join(f"{v} {k}{'s' if v > 1 else ''}" for k, v in counts.items())
        print(f"[{i}] Detected: {summary}")

    if args.save:
        print("\nAnnotated images saved to: runs/detect/predict/")


if __name__ == "__main__":
    main()
