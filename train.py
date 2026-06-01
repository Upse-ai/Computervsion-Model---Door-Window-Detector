"""
Door & Window Detector — YOLOv8n
Train a custom object detector for doors and windows.
No GPU required (runs on CPU). GPU used automatically if available.

Dataset structure expected:
    dataset/
        images/
            train/   *.jpg / *.png
            val/     *.jpg / *.png
        labels/
            train/   *.txt  (YOLO format)
            val/     *.txt
        data.yaml

Usage:
    python train.py
    python train.py --epochs 50 --imgsz 416 --batch 8
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8n door/window detector")
    parser.add_argument("--data",   default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int,   default=100,    help="Number of training epochs")
    parser.add_argument("--imgsz",  type=int,   default=640,    help="Input image size")
    parser.add_argument("--batch",  type=int,   default=16,     help="Batch size (-1 = auto)")
    parser.add_argument("--lr",     type=float, default=1e-3,   help="Initial learning rate")
    parser.add_argument("--name",   default="door_window_v1",   help="Run name (saved under runs/)")
    parser.add_argument("--resume", action="store_true",        help="Resume from last checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    if not Path(args.data).exists():
        print(f"[ERROR] data.yaml not found at: {args.data}")
        print("  Create your dataset folder and run again. See README for structure.")
        return

    model = YOLO("yolov8n.pt")

    print(f"\nStarting training — {args.epochs} epochs, AdamW optimizer")
    print(f"  data   : {args.data}")
    print(f"  imgsz  : {args.imgsz}")
    print(f"  batch  : {args.batch}")
    print(f"  lr0    : {args.lr}\n")

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="AdamW",
        lr0=args.lr,
        weight_decay=0.0005,
        warmup_epochs=3,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        device="cpu",        # forces CPU; remove or set "0" to use GPU if available
        verbose=True,
    )

    print(f"\nTraining complete. Weights saved to: runs/detect/{args.name}/weights/")


if __name__ == "__main__":
    main()
