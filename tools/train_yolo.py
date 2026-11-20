#!/usr/bin/env python3
"""训练装甲板 YOLO 检测模型（数据集由 tools/label_frames 生成的伪标签构成）。

用法：
    python tools/train_yolo.py --data dataset/armor.yaml --epochs 80 --batch 8
"""
import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset/armor.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/armor")
    parser.add_argument("--name", default="yolov8n")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=25,
        workers=4,
        cache=False,
        exist_ok=True,
        plots=True,
        val=True,
    )


if __name__ == "__main__":
    main()
