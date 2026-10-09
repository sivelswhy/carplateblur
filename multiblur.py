#!/usr/bin/env python3
"""
License plate (any country) and face anonymization for images and videos.

Detection:
  - plates: YOLOv11 fine-tuned on license plates from around the world
    (morsetechlab/yolov11-license-plate-detection)
  - faces: CenterFace, as used by deface (https://github.com/ORB-HD/deface, MIT license)
Masking: blur, mosaic, black box, a replacement image, or nothing (to only draw scores).
Videos: objects are tracked across frames, so masks follow them even when a frame is missed.

Runs fully offline once the plate model has been fetched with --download
(the face model ships inside the deface package).

Examples:
    python multiblur.py --download s                    # one-time, needs Internet
    python multiblur.py photo.jpg                       # plates + faces, blurred
    python multiblur.py photo.jpg --mode solid          # black boxes
    python multiblur.py photo.jpg --targets plates      # plates only
    python multiblur.py video.mp4 --mode mosaic -o out.mp4
    python multiblur.py photo.jpg --mode image --replace-img smiley.png
    python multiblur.py photos/ --draw-scores --mode none   # inspect detection scores
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
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
MODES = ("blur", "mosaic", "solid", "image", "none")
MODE_ALIASES = {"pixelate": "mosaic", "black": "solid"}  # names used by earlier versions


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


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    """A plate or face box (x1, y1, x2, y2) in pixels, top-left origin."""
    box: tuple[float, float, float, float]
    score: float
    is_face: bool


def area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a, b) -> float:
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def iou(a, b) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def enlarge(box, scale: float):
    """deface's mask scale: each side grows by (scale - 1) × the box size."""
    x1, y1, x2, y2 = box
    s = scale - 1
    w, h = x2 - x1, y2 - y1
    return x1 - w * s, y1 - h * s, x2 + w * s, y2 + h * s


def tiles(width: int, height: int, size: int) -> list[tuple[int, int, int, int]]:
    """Tiles (x, y, w, h) of `size` overlapping by 25%; none when the image is barely larger than a tile."""
    if max(width, height) <= size * 1.25:
        return []
    step = size * 0.75

    def starts(length: int) -> list[int]:
        if length <= size:
            return [0]
        count = math.ceil((length - size) / step) + 1
        return [int(min(i * step, length - size)) for i in range(count)]

    return [(x, y, min(size, width), min(size, height)) for y in starts(height) for x in starts(width)]


def suppress_duplicates(found: list[tuple[tuple, float]]) -> list[tuple[tuple, float]]:
    """Keeps the best of overlapping boxes, including a partial box cut by a tile edge inside a full one."""
    kept: list[tuple[tuple, float]] = []
    for box, score in sorted(found, key=lambda f: -f[1]):
        duplicate = any(
            iou(box, other) >= 0.4 or intersection(box, other) / max(min(area(box), area(other)), 1e-6) >= 0.7
            for other, _ in kept
        )
        if not duplicate:
            kept.append((box, score))
    return kept


# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #
class Tracker:
    """Follows plates and faces across video frames so each object keeps exactly one steady mask, and that
    mask keeps following the object for a few frames when the detector misses it."""

    SIZE_RESPONSE = 0.3           # how much a detection changes the mask size (box sizes flicker)
    MINIMUM_COVERAGE = 0.85       # a smoothed mask is never smaller than this share of the detection
    MISSED_VELOCITY_DECAY = 0.8   # share of the velocity kept per frame without a detection

    def __init__(self, max_missed: int):
        self.max_missed = max_missed
        self.tracks: list[dict] = []  # {"det": Detection, "velocity": (dx, dy), "missed": int}

    def update(self, detections: list[Detection]) -> list[Detection]:
        for t in self.tracks:  # predict where each tracked object is now
            dx, dy = t["velocity"]
            x1, y1, x2, y2 = t["det"].box
            t["det"] = replace(t["det"], box=(x1 + dx, y1 + dy, x2 + dx, y2 + dy))

        pairs = sorted(
            ((self._match_score(t["det"].box, d.box), ti, di)
             for ti, t in enumerate(self.tracks) for di, d in enumerate(detections)
             if d.is_face == t["det"].is_face),
            reverse=True,
        )
        matched_tracks, matched_detections = set(), set()
        for score, ti, di in pairs:
            if score <= 0 or ti in matched_tracks or di in matched_detections:
                continue
            matched_tracks.add(ti)
            matched_detections.add(di)
            self._update(self.tracks[ti], detections[di])

        for ti, t in enumerate(self.tracks):
            if ti not in matched_tracks:
                # Unseen: keep the size and let the motion fade out, so the mask can't fly off or zoom.
                t["missed"] += 1
                vx, vy = t["velocity"]
                t["velocity"] = (vx * self.MISSED_VELOCITY_DECAY, vy * self.MISSED_VELOCITY_DECAY)
        self.tracks = [t for t in self.tracks if t["missed"] <= self.max_missed]
        self.tracks += [{"det": d, "velocity": (0.0, 0.0), "missed": 0}
                        for di, d in enumerate(detections) if di not in matched_detections]
        return [t["det"] for t in self.tracks]

    def _update(self, track: dict, detection: Detection) -> None:
        px1, py1, px2, py2 = track["det"].box
        mx1, my1, mx2, my2 = detection.box
        # The center follows the detection exactly: smoothing it makes masks lag behind moving faces.
        cx, cy = (mx1 + mx2) / 2, (my1 + my2) / 2
        pw, ph, mw, mh = px2 - px1, py2 - py1, mx2 - mx1, my2 - my1
        w = max(pw + self.SIZE_RESPONSE * (mw - pw), mw * self.MINIMUM_COVERAGE)
        h = max(ph + self.SIZE_RESPONSE * (mh - ph), mh * self.MINIMUM_COVERAGE)

        # Velocity from the motion between detections, capped to a plausible speed per frame.
        vx, vy = track["velocity"]
        prev_cx, prev_cy = (px1 + px2) / 2 - vx, (py1 + py2) / 2 - vy
        limit = 0.3 * max(w, h)
        clamp = lambda v: min(max(v, -limit), limit)
        track["velocity"] = (clamp(0.5 * vx + 0.5 * (cx - prev_cx)), clamp(0.5 * vy + 0.5 * (cy - prev_cy)))
        track["det"] = replace(detection, box=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        track["missed"] = 0

    @staticmethod
    def _center(b):
        return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

    @staticmethod
    def _match_score(a, b) -> float:
        """IoU when the boxes overlap; otherwise a small score if the centers are close (fast motion)."""
        overlap = iou(a, b)
        if overlap >= 0.1:
            return overlap
        (ax, ay), (bx, by) = Tracker._center(a), Tracker._center(b)
        reach = 0.75 * max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
        distance = math.hypot(ax - bx, ay - by)
        return 0.05 * (1 - distance / reach) if distance < reach else 0.0


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
class Detector:
    """Finds plates and/or faces; boxes are enlarged by their mask scale."""

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

    def __call__(self, frame: np.ndarray, tiled: bool = False) -> tuple[list[Detection], int, int]:
        """Returns (detections, plate count, face count). `tiled` adds passes on overlapping tiles."""
        a = self.args
        plates = self._search(frame, a.imgsz if tiled else None, self._detect_plates) if self.plates else []
        face_tile = a.face_max_side if tiled and a.face_max_side else None
        faces = self._search(frame, face_tile, self._detect_faces) if self.faces else []
        detections = ([Detection(enlarge(b, a.plate_mask_scale), s, False) for b, s in plates]
                      + [Detection(enlarge(b, a.mask_scale), s, True) for b, s in faces])
        return detections, len(plates), len(faces)

    @staticmethod
    def _search(frame, tile_size, detect):
        found = detect(frame)
        if not tile_size:
            return found
        h, w = frame.shape[:2]
        for x, y, tw, th in tiles(w, h, tile_size):
            found += [((b[0] + x, b[1] + y, b[2] + x, b[3] + y), s) for b, s in detect(frame[y:y + th, x:x + tw])]
        return suppress_duplicates(found)

    def _detect_plates(self, img):
        a = self.args
        result = self.plates.predict(img, conf=a.conf, imgsz=a.imgsz, device=a.device, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        return [(tuple(map(float, b)), float(s))
                for b, s in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())]

    def _detect_faces(self, img):
        h, w = img.shape[:2]
        side = self.args.face_max_side
        scale = min(1.0, side / max(w, h)) if side else 1.0
        # deface --scale: CenterFace downscales internally and returns boxes in original pixels.
        self.faces.in_shape = (int(w * scale), int(h * scale)) if scale < 1 else None
        # CenterFace expects RGB; OpenCV frames are BGR.
        dets, _ = self.faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), threshold=self.args.face_conf)
        return [(tuple(map(float, d[:4])), float(d[4])) for d in dets]


# --------------------------------------------------------------------------- #
# Masking
# --------------------------------------------------------------------------- #
def load_replacement(path: str | None) -> np.ndarray | None:
    """Loads a replacement image as BGRA (deface --replaceimg)."""
    if not path:
        return None
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        sys.exit(f"Cannot read replacement image {path} (use PNG, JPEG, WebP or TIFF).")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    return image


def blurred(roi: np.ndarray, strength: int) -> np.ndarray:
    # Kernel scales with the box size so the result is always unreadable.
    k = max(strength, max(roi.shape[:2]) // 2) | 1
    return cv2.GaussianBlur(cv2.GaussianBlur(roi, (k, k), 0), (k, k), 0)


def mosaic(roi: np.ndarray, block: int) -> np.ndarray:
    """deface --mosaicsize: blocks of `block` pixels."""
    h, w = roi.shape[:2]
    small = cv2.resize(roi, (max(1, math.ceil(w / block)), max(1, math.ceil(h / block))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def anonymize(frame: np.ndarray, detections: list[Detection], args, replacement: np.ndarray | None = None) -> np.ndarray:
    fh, fw = frame.shape[:2]
    for det in detections:
        bx1, by1, bx2, by2 = det.box
        x1, y1 = max(0, int(round(bx1))), max(0, int(round(by1)))
        x2, y2 = min(fw, int(round(bx2))), min(fh, int(round(by2)))
        if x2 - x1 < 2 or y2 - y1 < 2 or args.mode == "none":
            continue
        roi = frame[y1:y2, x1:x2]

        # Patches only sample pixels inside the masked area, so nothing readable leaks back in.
        if args.mode == "solid":
            patch = np.zeros_like(roi)
        elif args.mode == "mosaic":
            patch = mosaic(roi, args.mosaic_size)
        elif args.mode == "image" and replacement is not None:
            # Stretched to the box like deface. Unlike deface, transparent areas show a blur,
            # never the original pixels, so the face can't be seen through them.
            full_w, full_h = max(1, int(round(bx2 - bx1))), max(1, int(round(by2 - by1)))
            img = cv2.resize(replacement, (full_w, full_h), interpolation=cv2.INTER_AREA)
            ox, oy = x1 - int(round(bx1)), y1 - int(round(by1))
            img = img[oy:oy + (y2 - y1), ox:ox + (x2 - x1)]
            if img.shape[:2] != roi.shape[:2]:
                img = cv2.resize(img, (x2 - x1, y2 - y1))
            alpha = img[:, :, 3:4].astype(np.float32) / 255
            patch = (img[:, :, :3] * alpha + blurred(roi, args.strength) * (1 - alpha)).astype(np.uint8)
        else:
            patch = blurred(roi, args.strength)

        if det.is_face and not args.boxes:
            # Ellipse inscribed in the (unclipped) detection box, like deface.
            mask = np.zeros(roi.shape[:2], np.uint8)
            center = (int(round((bx1 + bx2) / 2)) - x1, int(round((by1 + by2) / 2)) - y1)
            axes = (max(1, int(round((bx2 - bx1) / 2))), max(1, int(round((by2 - by1) / 2))))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            roi[mask > 0] = patch[mask > 0]
        else:
            roi[:] = patch

    if args.draw_scores:
        for det in detections:
            # Green score above the box, like deface --draw-scores.
            x, y = int(max(0, det.box[0])), int(det.box[1])
            scale = max(0.4, min(1.5, (det.box[3] - det.box[1]) / 80))
            cv2.putText(frame, f"{det.score:.2f}", (x, max(int(14 * scale), y - 6)),
                        cv2.FONT_HERSHEY_DUPLEX, scale, (0, 255, 0), max(1, int(scale * 1.5)), cv2.LINE_AA)
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
def save_image(frame: np.ndarray, src: Path, dst: Path, keep_metadata: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not keep_metadata:
        # OpenCV writes pixels only: EXIF and GPS metadata are dropped, which is the point here.
        cv2.imwrite(str(dst), frame)
        return
    # deface --keep-metadata: copy the original EXIF; pixels are already upright (OpenCV applied the orientation).
    from PIL import Image

    exif = Image.open(src).getexif()
    exif[0x0112] = 1  # orientation: normal
    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(dst, exif=exif.tobytes(), quality=95)


def process_image(src: Path, dst: Path, detector: Detector, args, replacement) -> tuple[int, int]:
    frame = cv2.imread(str(src))
    if frame is None:
        print(f"  ✗ Cannot read {src}", file=sys.stderr)
        return 0, 0
    detections, n_plates, n_faces = detector(frame, tiled=args.small_objects != "off")
    anonymize(frame, detections, args, replacement)
    save_image(frame, src, dst, args.keep_metadata)
    print(f"  ✓ {src.name} → {dst}  ({summary(n_plates, n_faces, args.targets)})")
    return n_plates, n_faces


def process_video(src: Path, dst: Path, detector: Detector, args, replacement) -> tuple[int, int]:
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

    tracker = Tracker(max_missed=max(3, round(fps * args.track_memory)))
    tiled = args.small_objects == "all"
    n_frames = n_plates = n_faces = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            detections, plates, faces = detector(frame, tiled=tiled)
            n_plates += plates
            n_faces += faces
            writer.write(anonymize(frame, tracker.update(detections), args, replacement))

            n_frames += 1
            if n_frames % 10 == 0 or n_frames == total:
                pct = f"{100 * n_frames / total:5.1f}%" if total else ""
                print(f"\r  … {src.name}: frame {n_frames}/{total or '?'} {pct}", end="", flush=True)
    finally:
        cap.release()
        writer.release()
    print()

    _finalize_video(src, tmp_video, dst, keep_audio=not args.no_audio, codec=args.codec)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"  ✓ {src.name} → {dst}  ({n_frames} frames; detections: {summary(n_plates, n_faces, args.targets)})")
    return n_plates, n_faces


def _finalize_video(src: Path, tmp_video: Path, dst: Path, keep_audio: bool = True, codec: str = "h264") -> None:
    """Re-encodes to H.264/HEVC and, unless disabled, copies the original audio track when ffmpeg is available."""
    if not shutil.which("ffmpeg"):
        shutil.move(str(tmp_video), str(dst))
        print("  ⚠ ffmpeg not found: video saved without audio (mp4v codec).")
        return
    video = ["-c:v", "libx264", "-preset", "medium", "-crf", "20"] if codec == "h264" else \
            ["-c:v", "libx265", "-preset", "medium", "-crf", "24", "-tag:v", "hvc1"]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(tmp_video), "-i", str(src),
        "-map", "0:v:0", *(["-map", "1:a?"] if keep_audio else []),
        *video, "-pix_fmt", "yuv420p",
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


def mode_name(value: str) -> str:
    value = MODE_ALIASES.get(value, value)
    if value not in MODES:
        raise argparse.ArgumentTypeError(f"invalid mode {value!r} (choose from {', '.join(MODES)})")
    return value


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Detects and masks license plates (any country) and faces in images and videos. "
                    "Options mirror deface's where an equivalent exists.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", nargs="?", help="Image, video or folder to process")
    p.add_argument("--download", nargs="*", choices=MODEL_SIZES, metavar="SIZE",
                   help="Download plate models into ./models (the only online step) and exit. "
                        "Without a value: downloads all 5 sizes")
    p.add_argument("-o", "--output", help="Output file (or folder when the input is a folder)")
    p.add_argument("-t", "--targets", nargs="+", choices=TARGETS, default=list(TARGETS), help="What to mask")

    masking = p.add_argument_group("masking")
    masking.add_argument("-m", "--mode", type=mode_name, default="blur", metavar="{" + ",".join(MODES) + "}",
                         help="How to hide detections (deface --replacewith; 'pixelate' and 'black' also work)")
    masking.add_argument("--mosaic-size", type=int, default=20, help="Mosaic block size in pixels (deface --mosaicsize)")
    masking.add_argument("--replace-img", help="Image for --mode image (deface --replaceimg)")
    masking.add_argument("--boxes", action="store_true", help="Use boxes instead of ellipses for faces (deface --boxes)")
    masking.add_argument("--mask-scale", type=float, default=1.3, help="Face mask scale (deface --mask-scale)")
    masking.add_argument("--plate-mask-scale", type=float, default=1.15, help="Plate mask scale")
    masking.add_argument("--strength", type=int, default=51, help="Minimum blur kernel size")
    masking.add_argument("--draw-scores", action="store_true", help="Draw detection scores (deface --draw-scores)")

    detection = p.add_argument_group("detection")
    detection.add_argument("--conf", type=float, default=0.05, help="Plate confidence threshold (0-1)")
    detection.add_argument("--face-conf", "--thresh", type=float, default=0.2,
                           help="Face confidence threshold (deface --thresh)")
    detection.add_argument("--small-objects", choices=("off", "photos", "all"), default="photos",
                           help="Also search overlapping tiles of large images for small, distant objects")
    detection.add_argument("--face-max-side", type=int, default=0,
                           help="Downscale images to this many pixels for face detection; 0 = full resolution (deface --scale)")
    detection.add_argument("--model-size", choices=MODEL_SIZES, default="s",
                           help="Plate model size: n=fastest … x=most accurate")
    detection.add_argument("--weights", help="Path to custom YOLO plate weights (overrides --model-size)")
    detection.add_argument("--imgsz", type=int, default=1280,
                           help="Plate inference resolution (also the tile size for --small-objects)")
    detection.add_argument("--track-memory", type=float, default=0.4,
                           help="Video: seconds a tracked object stays masked after it was last detected")

    output = p.add_argument_group("output")
    output.add_argument("--no-audio", action="store_true", help="Video: remove the sound from the output")
    output.add_argument("--codec", choices=("h264", "hevc"), default="h264", help="Video codec (needs ffmpeg)")
    output.add_argument("--keep-metadata", action="store_true", help="Keep image EXIF/GPS metadata (deface --keep-metadata)")

    runtime = p.add_argument_group("runtime")
    runtime.add_argument("--device", default=None, help="Plate model device: cpu, mps, cuda, 0… (auto by default)")
    runtime.add_argument("--face-backend", choices=("auto", "onnxrt", "opencv"), default="auto",
                         help="Face inference backend (deface --backend)")

    args = p.parse_args(argv)
    if args.mode == "image" and not args.replace_img:
        p.error("--mode image needs --replace-img")
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

    replacement = load_replacement(args.replace_img) if args.mode == "image" else None
    print(f"Loading models ({', '.join(args.targets)})…")
    detector = Detector(args)
    print(f"Mode: {args.mode} | {len(files)} file(s)")

    plates = faces = 0
    for f in files:
        dst = output_path(f, root, args)
        process = process_video if f.suffix.lower() in VIDEO_EXTS else process_image
        p, fc = process(f, dst, detector, args, replacement)
        plates += p
        faces += fc
    print(f"Done — {summary(plates, faces, args.targets)} detected in total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
