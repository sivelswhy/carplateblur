#!/usr/bin/env python3
"""
License plate (any country) and face anonymization for images and videos.

Detection:
  - plates: YOLOv11 fine-tuned on license plates from around the world
    (morsetechlab/yolov11-license-plate-detection)
  - faces: CenterFace, as used by deface (https://github.com/ORB-HD/deface, MIT license)
Masking: Gaussian blur, pixelation or a black box.

Runs fully offline once the plate model has been fetched with --download
(the face model ships inside the deface package).

Examples:
    python plate_anonymizer.py --download s                    # one-time, needs Internet
    python plate_anonymizer.py photo.jpg                       # plates + faces, blurred
    python plate_anonymizer.py photo.jpg --mode black          # black boxes
    python plate_anonymizer.py photo.jpg --targets plates      # plates only
    python plate_anonymizer.py video.mp4 --mode pixelate -o out.mp4
    python plate_anonymizer.py photos/ --mode blur --conf 0.2
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Fully offline: no telemetry, connectivity checks or update checks
# from Ultralytics / Hugging Face.
os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv", ".flv"}

HF_REPO = "morsetechlab/yolov11-license-plate-detection"
MODEL_SIZES = ("n", "s", "m", "l", "x")
MODELS_DIR = Path(__file__).resolve().parent / "models"

TARGETS = ("plates", "faces")


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def model_path(size: str) -> Path:
    return MODELS_DIR / f"license-plate-finetune-v1{size}.pt"


def download_models(sizes) -> None:
    """The only step that needs Internet: stores the plate weights in ./models."""
    os.environ["HF_HUB_OFFLINE"] = "0"
    import huggingface_hub.constants
    from huggingface_hub import hf_hub_download

    huggingface_hub.constants.HF_HUB_OFFLINE = False
    MODELS_DIR.mkdir(exist_ok=True)
    for size in sizes:
        print(f"Downloading plate model '{size}'…")
        hf_hub_download(HF_REPO, model_path(size).name, local_dir=MODELS_DIR)
    print(f"Models saved to {MODELS_DIR}")


def require(path: Path, size: str) -> Path:
    if not path.exists():
        sys.exit(
            f"Model not found: {path}\n"
            f"Download it once (Internet required) with:\n"
            f"  python {Path(__file__).name} --download {size}"
        )
    return path


class Detector:
    """Finds plates and/or faces and returns boxes already padded for masking."""

    def __init__(self, args):
        self.args = args
        self.plates = self.faces = None
        if "plates" in args.targets:
            from ultralytics import YOLO

            weights = Path(args.weights) if args.weights else model_path(args.model_size)
            self.plates = YOLO(str(require(weights, args.model_size)))
        if "faces" in args.targets:
            from deface.centerface import CenterFace

            self.faces = CenterFace(backend=args.face_backend)

    def __call__(self, frame: np.ndarray) -> tuple[list[tuple], int, int]:
        """Returns (padded boxes, plate count, face count)."""
        h, w = frame.shape[:2]
        plates = self._detect_plates(frame) if self.plates else []
        faces = self._detect_faces(frame) if self.faces else []
        boxes = ([expand_box(b, self.args.padding, w, h) for b in plates]
                 + [expand_box(b, self.args.face_padding, w, h) for b in faces])
        return boxes, len(plates), len(faces)

    def _detect_plates(self, frame):
        a = self.args
        results = self.plates.predict(frame, conf=a.conf, imgsz=a.imgsz, device=a.device, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        return [tuple(map(int, b)) for b in boxes.xyxy.cpu().numpy()]

    def _detect_faces(self, frame):
        # CenterFace expects RGB; OpenCV frames are BGR.
        dets, _ = self.faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=self.args.face_conf)
        return [tuple(map(int, d[:4])) for d in dets]


# --------------------------------------------------------------------------- #
# Masking
# --------------------------------------------------------------------------- #
def expand_box(box, padding: float, w: int, h: int):
    x1, y1, x2, y2 = box
    px, py = int((x2 - x1) * padding), int((y2 - y1) * padding)
    return max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)


def anonymize(frame: np.ndarray, boxes, mode: str, strength: int) -> np.ndarray:
    for x1, y1, x2, y2 in boxes:
        if x2 <= x1 or y2 <= y1:
            continue
        roi = frame[y1:y2, x1:x2]

        if mode == "black":
            frame[y1:y2, x1:x2] = 0
        elif mode == "pixelate":
            bw, bh = x2 - x1, y2 - y1
            # `strength` = number of blocks across the box width (lower = coarser)
            cells = max(2, min(strength, bw))
            small = cv2.resize(roi, (cells, max(1, int(cells * bh / bw))), interpolation=cv2.INTER_LINEAR)
            frame[y1:y2, x1:x2] = cv2.resize(small, (bw, bh), interpolation=cv2.INTER_NEAREST)
        else:  # blur
            # Kernel scales with the box size so the result is always unreadable
            k = max(strength, (max(x2 - x1, y2 - y1) // 2)) | 1
            blurred = cv2.GaussianBlur(roi, (k, k), 0)
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(blurred, (k, k), 0)
    return frame


def summary(plates: int, faces: int, targets) -> str:
    parts = []
    if "plates" in targets:
        parts.append(f"{plates} plate(s)")
    if "faces" in targets:
        parts.append(f"{faces} face(s)")
    return ", ".join(parts)


# --------------------------------------------------------------------------- #
# Image / video processing
# --------------------------------------------------------------------------- #
def process_image(src: Path, dst: Path, detector: Detector, args) -> tuple[int, int]:
    frame = cv2.imread(str(src))
    if frame is None:
        print(f"  ✗ Cannot read {src}", file=sys.stderr)
        return 0, 0
    boxes, n_plates, n_faces = detector(frame)
    anonymize(frame, boxes, args.mode, args.strength)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), frame)
    print(f"  ✓ {src.name} → {dst}  ({summary(n_plates, n_faces, args.targets)})")
    return n_plates, n_faces


def process_video(src: Path, dst: Path, detector: Detector, args) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"  ✗ Cannot open {src}", file=sys.stderr)
        return 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp()
    tmp_video = Path(tmp_dir) / "video_only.mp4"
    writer = cv2.VideoWriter(str(tmp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Keep recent detections so masks don't flicker when a detector
    # misses something for a few frames.
    history: list[list[tuple]] = []
    n_frames = n_plates = n_faces = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            boxes, plates, faces = detector(frame)
            n_plates += plates
            n_faces += faces
            history.append(boxes)
            history = history[-(args.persist + 1):]
            to_mask = [b for frame_boxes in history for b in frame_boxes]
            writer.write(anonymize(frame, to_mask, args.mode, args.strength))

            n_frames += 1
            if n_frames % 10 == 0 or n_frames == total:
                pct = f"{100 * n_frames / total:5.1f}%" if total else ""
                print(f"\r  … {src.name}: frame {n_frames}/{total or '?'} {pct}", end="", flush=True)
    finally:
        cap.release()
        writer.release()
    print()

    _finalize_video(src, tmp_video, dst, keep_audio=not args.no_audio)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"  ✓ {src.name} → {dst}  ({n_frames} frames; detections: {summary(n_plates, n_faces, args.targets)})")
    return n_plates, n_faces


def _finalize_video(src: Path, tmp_video: Path, dst: Path, keep_audio: bool = True) -> None:
    """Re-encodes to H.264 and, unless disabled, copies the original audio track when ffmpeg is available."""
    if not shutil.which("ffmpeg"):
        shutil.move(str(tmp_video), str(dst))
        print("  ⚠ ffmpeg not found: video saved without audio (mp4v codec).")
        return
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(tmp_video), "-i", str(src),
        "-map", "0:v:0", *(["-map", "1:a?"] if keep_audio else []),
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(dst),
    ]
    if subprocess.run(cmd).returncode != 0:
        shutil.move(str(tmp_video), str(dst))
        print("  ⚠ ffmpeg failed: raw video kept (no audio).")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def unique_path(path: Path) -> Path:
    """Never overwrite an existing file: "photo_anonymized (1).jpg", "(2)"… like Finder."""
    candidate, copy = path, 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem} ({copy}){path.suffix}")
        copy += 1
    return candidate


def output_path(src: Path, root_in: Path, args) -> Path:
    suffix = ".mp4" if src.suffix.lower() in VIDEO_EXTS else src.suffix
    if root_in.is_file():
        # An explicit -o file name is used as given.
        return Path(args.output) if args.output else unique_path(src.with_name(f"{src.stem}_anonymized{suffix}"))
    out_dir = Path(args.output) if args.output else root_in / "anonymized"
    return unique_path((out_dir / src.relative_to(root_in)).with_suffix(suffix))


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Detects and masks license plates (any country) and faces in images and videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", nargs="?", help="Image, video or folder to process")
    p.add_argument("--download", nargs="*", choices=MODEL_SIZES, metavar="SIZE",
                   help="Download models into ./models (the only online step) and exit. "
                        "Without a value: downloads all 5 plate model sizes")
    p.add_argument("-o", "--output", help="Output file (or folder when the input is a folder)")
    p.add_argument("-m", "--mode", choices=("blur", "black", "pixelate"), default="blur",
                   help="Masking style")
    p.add_argument("-t", "--targets", nargs="+", choices=TARGETS, default=list(TARGETS),
                   help="What to mask")
    p.add_argument("--conf", type=float, default=0.05, help="Plate detection confidence threshold (0-1)")
    p.add_argument("--face-conf", type=float, default=0.2,
                   help="Face detection confidence threshold (0-1), same default as deface")
    p.add_argument("--padding", type=float, default=0.15,
                   help="Margin added around each plate (fraction of its size)")
    p.add_argument("--face-padding", type=float, default=0.3,
                   help="Margin added on each side of a face (fraction of its size; 0.3 = deface's mask scale 1.3)")
    p.add_argument("--face-backend", choices=("auto", "onnxrt", "opencv"), default="auto",
                   help="Face inference backend (auto = onnxruntime when installed, else OpenCV)")
    p.add_argument("--strength", type=int, default=51,
                   help="Strength: minimum blur kernel size (blur) / blocks across the width (pixelate)")
    p.add_argument("--model-size", choices=MODEL_SIZES, default="s",
                   help="Plate model size: n=fastest … x=most accurate")
    p.add_argument("--weights", help="Path to custom YOLO plate weights (overrides --model-size)")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="Plate inference resolution (larger = better on small plates, slower)")
    p.add_argument("--no-audio", action="store_true", help="Video: remove the sound from the output")
    p.add_argument("--persist", type=int, default=3,
                   help="Video: number of frames a detection stays masked")
    p.add_argument("--device", default=None, help="cpu, mps, cuda, 0… (auto by default)")
    args = p.parse_args(argv)
    if args.mode == "pixelate" and args.strength == 51:
        args.strength = 6  # ~6 blocks across: faces become unrecognizable
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.download is not None:
        download_models(args.download or MODEL_SIZES)
        return 0
    if not args.input:
        print("Specify an image, a video or a folder (or --download).", file=sys.stderr)
        return 1
    root = Path(args.input).expanduser().resolve()
    if not root.exists():
        print(f"Not found: {root}", file=sys.stderr)
        return 1

    if root.is_dir():
        files = sorted(f for f in root.rglob("*")
                       if f.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS and "anonymized" not in f.parts)
    else:
        files = [root]
    files = [f for f in files if f.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS]
    if not files:
        print("No supported image or video found.", file=sys.stderr)
        return 1

    print(f"Loading models ({', '.join(args.targets)})…")
    detector = Detector(args)
    print(f"Mode: {args.mode} | {len(files)} file(s)")

    plates = faces = 0
    for f in files:
        dst = output_path(f, root, args)
        process = process_video if f.suffix.lower() in VIDEO_EXTS else process_image
        p, fc = process(f, dst, detector, args)
        plates += p
        faces += fc
    print(f"Done — {summary(plates, faces, args.targets)} detected in total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
