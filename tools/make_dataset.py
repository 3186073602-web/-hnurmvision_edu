#!/usr/bin/env python3
"""把标注好的图片按 8:2 切成 YOLO 训练集/验证集，并写出 armor.yaml。"""
import argparse
import random
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="dataset/raw")
    parser.add_argument("--labels", default="dataset/labels_all")
    parser.add_argument("--out", default="dataset")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    images = sorted(Path(args.images).glob("*.png"))
    pairs = []
    for image in images:
        label = Path(args.labels) / f"{image.stem}.txt"
        if label.exists() and label.stat().st_size > 0:
            pairs.append((image, label))
    if not pairs:
        raise SystemExit("没有找到带标注的图片")

    random.Random(args.seed).shuffle(pairs)
    val_count = max(1, int(len(pairs) * args.val_ratio))
    splits = {"val": pairs[:val_count], "train": pairs[val_count:]}

    for split, items in splits.items():
        image_dir = out / "images" / split
        label_dir = out / "labels" / split
        for directory in (image_dir, label_dir):
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True)
        for image, label in items:
            shutil.copy2(image, image_dir / image.name)
            shutil.copy2(label, label_dir / label.name)

    yaml_path = out / "armor.yaml"
    yaml_path.write_text(
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: armor\n",
        encoding="utf-8",
    )
    print(f"train={len(splits['train'])} val={len(splits['val'])} -> {yaml_path}")


if __name__ == "__main__":
    main()
