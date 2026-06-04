from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
    path: Path
    split: str
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick manual relabel tool for bin color dataset.")
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset root with train/val/color folders.")
    parser.add_argument("--colors", type=str, default="blue,green,gray,other", help="Class names, comma-separated.")
    parser.add_argument("--split", type=str, default="train,val", help="Splits to inspect, comma-separated.")
    parser.add_argument("--start", type=int, default=0, help="Start index in sample list.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle sample order.")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of samples (0 means all).")
    parser.add_argument(
        "--window-name",
        type=str,
        default="Bin Relabel Tool",
        help="OpenCV window name",
    )
    return parser.parse_args()


def list_samples(data_root: Path, splits: list[str], colors: list[str]) -> list[Sample]:
    samples: list[Sample] = []
    for split in splits:
        for color in colors:
            folder = data_root / split / color
            if not folder.exists():
                continue
            for p in sorted(folder.iterdir()):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    samples.append(Sample(path=p, split=split, label=color))
    return samples


def ensure_unique_path(dst: Path) -> Path:
    if not dst.exists():
        return dst
    stem = dst.stem
    suffix = dst.suffix
    idx = 1
    while True:
        candidate = dst.with_name(f"{stem}_r{idx}{suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def render_help(canvas: np.ndarray, colors: list[str]) -> None:
    keys = [str(i + 1) for i in range(min(len(colors), 9))]
    class_map = "  ".join([f"{k}={c}" for k, c in zip(keys, colors[:9])])
    lines = [
        "Keys:",
        "1..9 -> relabel to class",
        "d or Right -> next",
        "a or Left -> previous",
        "u -> undo last relabel",
        "q or ESC -> quit",
        "",
        f"Class map: {class_map}",
    ]
    x, y = 10, 24
    for line in lines:
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        y += 24


def make_view(image: np.ndarray, sample: Sample, idx: int, total: int, colors: list[str]) -> np.ndarray:
    panel_h = 130
    h, w = image.shape[:2]
    scale = min(980 / max(w, 1), 620 / max(h, 1), 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((new_h + panel_h, max(new_w, 980), 3), dtype=np.uint8)
    canvas[:new_h, :new_w] = resized

    info = [
        f"[{idx + 1}/{total}] split={sample.split} current={sample.label}",
        f"file={sample.path.name}",
    ]
    y0 = new_h + 28
    for line in info:
        cv2.putText(canvas, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (90, 255, 90), 2, cv2.LINE_AA)
        y0 += 28

    legend = " | ".join([f"{i + 1}:{c}" for i, c in enumerate(colors[:9])])
    cv2.putText(canvas, f"Target classes: {legend}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def read_image_unicode(path: Path) -> np.ndarray | None:
    try:
        arr = np.fromfile(path.as_posix(), dtype=np.uint8)
        if arr.size == 0:
            return None
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    colors = [x.strip() for x in args.colors.split(",") if x.strip()]
    splits = [x.strip() for x in args.split.split(",") if x.strip()]
    if len(colors) < 3:
        raise ValueError(f"--colors must have at least 3 classes, got {colors}")
    if len(colors) > 9:
        raise ValueError("At most 9 classes are supported by number keys 1..9.")

    samples = list_samples(data_root, splits, colors)
    if args.shuffle:
        rng = np.random.default_rng(42)
        rng.shuffle(samples)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        print("No samples found.")
        return

    idx = max(0, min(args.start, len(samples) - 1))
    changed = 0
    undo_stack: list[tuple[Path, Path, int]] = []
    log_path = data_root / f"relabel_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_file = log_path.open("w", newline="", encoding="utf-8-sig")
    writer = csv.writer(log_file)
    writer.writerow(["time", "split", "from_label", "to_label", "src_path", "dst_path"])

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.window_name, 1100, 800)

    try:
        while 0 <= idx < len(samples):
            sample = samples[idx]
            image = read_image_unicode(sample.path)
            if image is None:
                idx += 1
                continue

            view = make_view(image, sample, idx, len(samples), colors)
            overlay = np.zeros((230, 430, 3), dtype=np.uint8)
            render_help(overlay, colors)
            vh, vw = view.shape[:2]
            oh, ow = overlay.shape[:2]
            x0 = max(vw - ow - 10, 0)
            y0 = 10
            h_can = min(oh, max(0, vh - y0))
            w_can = min(ow, max(0, vw - x0))
            if h_can > 0 and w_can > 0:
                roi = view[y0 : y0 + h_can, x0 : x0 + w_can]
                ov = overlay[:h_can, :w_can]
                view[y0 : y0 + h_can, x0 : x0 + w_can] = cv2.addWeighted(roi, 0.35, ov, 0.65, 0)
            cv2.imshow(args.window_name, view)

            key = cv2.waitKeyEx(0)
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("d"), ord("D"), 2555904):
                idx = min(idx + 1, len(samples) - 1)
                continue
            if key in (ord("a"), ord("A"), 2424832):
                idx = max(idx - 1, 0)
                continue
            if key in (ord("u"), ord("U")) and undo_stack:
                src, dst, back_idx = undo_stack.pop()
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(src.as_posix(), dst.as_posix())
                samples[back_idx].path = dst
                samples[back_idx].label = dst.parent.name
                changed = max(0, changed - 1)
                idx = back_idx
                continue

            key_to_class = {ord(str(i + 1)): colors[i] for i in range(len(colors))}
            target_label = key_to_class.get(key)
            if target_label is None:
                continue
            if target_label == sample.label:
                idx = min(idx + 1, len(samples) - 1)
                continue

            dst_dir = data_root / sample.split / target_label
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = ensure_unique_path(dst_dir / sample.path.name)
            src_path = sample.path
            shutil.move(src_path.as_posix(), dst_path.as_posix())

            writer.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    sample.split,
                    sample.label,
                    target_label,
                    src_path.as_posix(),
                    dst_path.as_posix(),
                ],
            )
            log_file.flush()
            undo_stack.append((dst_path, src_path, idx))
            samples[idx].path = dst_path
            samples[idx].label = target_label
            changed += 1
            idx = min(idx + 1, len(samples) - 1)
    finally:
        log_file.close()
        cv2.destroyAllWindows()

    print(f"Done. relabeled={changed} log={log_path}")


if __name__ == "__main__":
    main()
