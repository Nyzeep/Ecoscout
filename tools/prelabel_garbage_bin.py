from __future__ import annotations

import argparse
from pathlib import Path

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-label garbage-bin boxes with YOLO model.")
    parser.add_argument("--input-root", type=Path, required=True, help="Input image root (recursive scan).")
    parser.add_argument(
        "--output-label-root",
        type=Path,
        default=None,
        help="Output label root (YOLO txt). Default: <input-root>/labels_pre",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("D:/study_space_work/HuaLi_garbage_system/app/models/garbege.pt"),
        help="YOLO detection model path.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold.")
    parser.add_argument(
        "--source-class-id",
        type=int,
        default=0,
        help="Garbage-bin class id in the source detection model.",
    )
    parser.add_argument(
        "--output-class-id",
        type=int,
        default=0,
        help="Class id written to output labels.",
    )
    parser.add_argument(
        "--save-empty",
        action="store_true",
        help="Write empty txt for images with no bin detection.",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save visualization images with predicted boxes.",
    )
    parser.add_argument(
        "--vis-root",
        type=Path,
        default=None,
        help="Visualization output root. Default: <input-root>/vis_pre",
    )
    return parser.parse_args()


def list_images(root: Path) -> list[Path]:
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def to_yolo_line(class_id: int, x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> str:
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def clamp_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[float, float, float, float]:
    x1 = max(0.0, min(float(w - 1), x1))
    y1 = max(0.0, min(float(h - 1), y1))
    x2 = max(0.0, min(float(w), x2))
    y2 = max(0.0, min(float(h), y2))
    return x1, y1, x2, y2


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_label_root = (args.output_label_root or (input_root / "labels_pre")).resolve()
    vis_root = (args.vis_root or (input_root / "vis_pre")).resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"input-root not found: {input_root}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"model not found: {args.model_path}")

    from ultralytics import YOLO

    model = YOLO(str(args.model_path))
    images = list_images(input_root)
    if not images:
        print("No images found.")
        return

    total_boxes = 0
    has_box_images = 0
    no_box_images = 0
    output_label_root.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        vis_root.mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(images, start=1):
        rel = img_path.relative_to(input_root)
        txt_path = (output_label_root / rel).with_suffix(".txt")
        txt_path.parent.mkdir(parents=True, exist_ok=True)

        frame = cv2.imread(str(img_path))
        if frame is None:
            if args.save_empty:
                txt_path.write_text("", encoding="utf-8")
            no_box_images += 1
            continue
        h, w = frame.shape[:2]

        result = model(str(img_path), conf=args.conf, iou=args.iou, verbose=False)[0]
        lines: list[str] = []
        vis = frame.copy() if args.save_vis else None

        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id != args.source_class_id:
                continue
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
            if x2 <= x1 or y2 <= y1:
                continue
            lines.append(to_yolo_line(args.output_class_id, x1, y1, x2, y2, w, h))
            if vis is not None:
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        if lines:
            txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            total_boxes += len(lines)
            has_box_images += 1
        else:
            if args.save_empty:
                txt_path.write_text("", encoding="utf-8")
            no_box_images += 1

        if vis is not None:
            vis_path = vis_root / rel
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            ext = vis_path.suffix if vis_path.suffix else ".jpg"
            ok, buf = cv2.imencode(ext, vis)
            if ok:
                buf.tofile(str(vis_path))

        if idx % 50 == 0 or idx == len(images):
            print(f"[{idx}/{len(images)}] done")

    print("=== Prelabel Finished ===")
    print(f"input_root       : {input_root}")
    print(f"output_label_root: {output_label_root}")
    if args.save_vis:
        print(f"vis_root         : {vis_root}")
    print(f"images_total     : {len(images)}")
    print(f"images_with_box  : {has_box_images}")
    print(f"images_without   : {no_box_images}")
    print(f"total_boxes      : {total_boxes}")


if __name__ == "__main__":
    main()
