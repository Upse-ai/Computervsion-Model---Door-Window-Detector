# Computer Vision — Door & Window Detector

A custom object detector built on **YOLOv8n** that identifies doors and windows in images. Designed to run on CPU — no GPU required.

---

## Model

| Setting | Value |
|---------|-------|
| Architecture | YOLOv8n |
| Optimizer | AdamW |
| Epochs | 100 (adjustable) |
| Input size | 640×640 |
| Classes | door, window |
| Device | CPU (GPU auto-detected if available) |

---

## Dataset Structure

Bring your own dataset. Place files like this before training:

```
dataset/
├── images/
│   ├── train/   ← training images (.jpg / .png)
│   └── val/     ← validation images
├── labels/
│   ├── train/   ← YOLO-format .txt labels
│   └── val/
└── data.yaml    ← already included
```

Each `.txt` label file follows YOLO format:
```
<class_id> <x_center> <y_center> <width> <height>
```
- `0` = door
- `1` = window

All values are normalized to `[0, 1]` relative to image size.

> **Tip:** [Roboflow Universe](https://universe.roboflow.com) has free annotated door/window datasets ready for YOLO.

---

## Requirements

```bash
pip install -r requirements.txt
```

```
ultralytics
torch
Pillow
```

---

## Training

```bash
# Default: 100 epochs, AdamW, 640px input
python train.py

# Custom settings
python train.py --epochs 50 --imgsz 416 --batch 8 --lr 0.0005

# Resume interrupted training
python train.py --resume
```

Trained weights are saved to:
```
runs/detect/door_window_v1/weights/
├── best.pt    ← best validation checkpoint
└── last.pt    ← last epoch checkpoint
```

---

## Inference

```bash
# Run on a single image
python detect.py image.jpg --weights runs/detect/door_window_v1/weights/best.pt

# Run on a folder of images and save results
python detect.py images/ --save

# Adjust confidence threshold
python detect.py image.jpg --conf 0.5
```

---

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | `dataset/data.yaml` | Path to dataset config |
| `--epochs` | `100` | Number of training epochs |
| `--imgsz` | `640` | Input image resolution |
| `--batch` | `16` | Batch size |
| `--lr` | `0.001` | Initial learning rate |
| `--name` | `door_window_v1` | Run folder name |
| `--resume` | off | Resume from last checkpoint |
