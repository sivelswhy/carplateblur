# MultiBlur

Detects license plates (any country) and faces in images, videos and folders, and masks them with a **blur**, a **mosaic**, a **black box** or a **replacement image**. In videos, plates and faces are **tracked** so masks follow them even when the detector misses a frame. Everything runs **fully offline**.

- **Plates**: YOLOv11 fine-tuned on license plates from many countries.
- **Faces**: CenterFace from [deface](https://github.com/ORB-HD/deface).

Two front-ends share the same detectors:

- `multiblur.py`: a cross-platform Python CLI.
- `macos/`: **MultiBlur**, a native macOS app (SwiftUI + CoreML + Vision + AVFoundation) with no Python dependency.

## Download

Grab the latest **MultiBlur-macOS.zip** from [Releases](https://github.com/sivelswhy/multiblur/releases/latest) (macOS 14+, Apple Silicon or Intel). Unzip it, move MultiBlur.app to Applications and open it once with right-click → Open (or System Settings → Privacy & Security → Open Anyway): the app is ad hoc signed, not notarized.

The detection models are included in the app, which never touches the network on its own. **Settings › Advanced › Check for Updates** compares the commit the app was built from with the latest commit on `main`; when they differ, **Update and Relaunch** downloads that commit's release, checks it's a validly signed MultiBlur, replaces the app and relaunches it.

Every push to `main` runs the Python tests, then builds a universal app and publishes it as a new release ([`.github/workflows/release.yml`](.github/workflows/release.yml)).

## Python CLI

### Setup

```bash
uv venv --python 3.12 .venv && uv pip install -r requirements.txt
# or: python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Then, **once** (this is the only step that needs Internet; the face model ships inside the `deface` package):

```bash
.venv/bin/python multiblur.py --download        # all 5 sizes (~250 MB)
.venv/bin/python multiblur.py --download n s    # or just some of them
```

The plate weights are stored in `./models/`. After that the script is **100% offline**: Ultralytics telemetry and Hugging Face calls are disabled (`YOLO_OFFLINE`, `HF_HUB_OFFLINE`). For an air-gapped machine, copy the project folder including `models/` and install dependencies from wheels (`pip download -r requirements.txt -d wheels`, then `pip install --no-index -f wheels -r requirements.txt`).

`ffmpeg` is optional but recommended: it keeps the audio track and encodes videos as H.264.

### Usage

```bash
.venv/bin/python multiblur.py photo.jpg                       # plates + faces, blurred → photo_anonymized.jpg
.venv/bin/python multiblur.py photo.jpg --mode solid          # black boxes
.venv/bin/python multiblur.py photo.jpg --targets faces       # faces only
.venv/bin/python multiblur.py video.mov --mode mosaic -o out.mp4
.venv/bin/python multiblur.py photo.jpg --mode image --replace-img smiley.png
.venv/bin/python multiblur.py photo.jpg --mode none --draw-scores   # inspect detection scores
.venv/bin/python multiblur.py folder/                         # → folder/anonymized/
```

Options mirror [deface](https://github.com/ORB-HD/deface)'s where an equivalent exists, and the macOS app's settings:

| Option | Default | Description |
|---|---|---|
| `-t, --targets` | `plates faces` | What to mask: `plates`, `faces` or both |
| `-m, --mode` | `blur` | `blur`, `mosaic`, `solid` (black box), `image` or `none` (deface `--replacewith`; `pixelate`/`black` also work) |
| `--mosaic-size` | `20` | Mosaic block size in pixels (deface `--mosaicsize`) |
| `--replace-img` | — | Image for `--mode image` (deface `--replaceimg`); transparent areas show a blur, never the original |
| `--boxes` | off | Boxes instead of ellipses for faces (deface `--boxes`) |
| `--mask-scale` | `1.3` | Face mask scale (deface `--mask-scale`) |
| `--plate-mask-scale` | `1.15` | Plate mask scale |
| `--draw-scores` | off | Draw detection scores (deface `--draw-scores`) |
| `--strength` | `51` | Minimum blur kernel |
| `--conf` | `0.05` | Plate confidence threshold. Defaults to the minimum: a false positive is harmless, a missed plate is not |
| `--face-conf`, `--thresh` | `0.2` | Face confidence threshold (deface `--thresh`) |
| `--small-objects` | `photos` | `off`, `photos` or `all`: also search overlapping tiles of large images for small, distant objects |
| `--face-max-side` | `0` | Downscale for face detection; `0` = full resolution (deface `--scale`) |
| `--track-memory` | `0.4` | Video: seconds a tracked object stays masked after it was last detected |
| `--model-size` | `s` | Plate model: `n` (fastest) → `x` (most accurate) |
| `--imgsz` | `1280` | Plate inference resolution, also the tile size |
| `--no-audio` | off | Video: remove the sound |
| `--codec` | `h264` | Video codec: `h264` or `hevc` (needs ffmpeg) |
| `--keep-metadata` | off | Keep image EXIF/GPS metadata (deface `--keep-metadata`) |
| `--device` | auto | Plate model device: `cpu`, `mps`, `cuda`, `0`… |
| `--face-backend` | `auto` | `onnxrt` (faster, uses CoreML/CUDA when available) or `opencv` (deface `--backend`) |
| `--weights` | — | Custom YOLO plate weights |
| `--download` | — | Download plate model weights and exit |

### Tests

```bash
.venv/bin/python -m pytest tests     # geometry, tracking, masking, file naming; no models needed
```

## macOS app (MultiBlur)

A single window: toggle **Plates** and **Faces**, pick a **Style** (Blur, Mosaic, Black box, Image, None) and its options right below it (face mask shape, mosaic block size, replacement image), then drop photos, videos or folders anywhere on the window (or click ＋). Each file shows a thumbnail of its result, what was hidden, or a readable error. Double-click a row to open the result, click ⏹ to stop a file being processed or × to remove it from the list, or right-click for more.

In Finder, select photos, videos or folders and right-click › **Services › Anonymize with MultiBlur** to send them to the app. The app follows the system language (English or French). Results get the `_anonymized` suffix and never overwrite an existing file (`(1)`, `(2)`… are added).

Every [deface](https://github.com/ORB-HD/deface) option is available: the masking ones in the main window, the others in **Settings** (⌘,), organized in Detection, Output and Advanced tabs. Settings are remembered between launches.

| deface | MultiBlur | Default |
|---|---|---|
| `--thresh` | Face threshold | 0.2 |
| `--replacewith blur/solid/mosaic/img/none` | Replace with: Blur / Black box / Mosaic / Image / None (applies to plates too) | Blur |
| `--replaceimg` | Replacement image: PNG, JPEG, SVG… (transparent areas show a blur, never the original pixels) | — |
| `--mosaicsize` | Mosaic size | 20 px |
| `--boxes` | Face mask: ◯ ellipse / ▢ box | Ellipse |
| `--mask-scale` | Faces › Mask scale | 1.3× |
| `--scale` | Faces › Detection resolution: Full / 1920 / 1280 / 640 px | Full |
| `--draw-scores` | Advanced › Show detection scores | Off |
| `--preview` | Live preview (in the main window) | Off |
| `--output` | Output › Save to: next to the originals or a chosen folder | Next to originals |
| `--keep-audio` | Keep audio | On |
| `--keep-metadata` | Keep image metadata (EXIF, GPS…) | Off |
| `--ffmpeg-config` | Video codec: H.264 / HEVC | H.264 |
| `--backend`, `--execution-provider` | Run models on: Neural Engine / GPU / CPU | Neural Engine |

Also in Settings › Output: **Sound when done** plays a macOS system sound (Glass, Ping, Hero…) after each export (off by default).

Plate-specific settings: plate confidence (default 0.05) and plate mask scale (1.15×). **Detection › Small objects** (Off / Photos / Photos & videos, default Photos) also searches overlapping tiles of large images for small, distant plates and faces. The webcam mode (`deface cam`) is not included.

- Detection runs on the Neural Engine / GPU through CoreML and Vision. Faces use deface's `centerface.onnx` converted to CoreML, with its decoding and NMS ported to Swift (`macos/Sources/FaceDetector.swift`); outputs match the Python version.
- Videos: plates and faces are tracked across frames (one mask per object that keeps following it for ~0.4 s when a frame is missed); audio is kept (re-encoded to AAC) and rotated iPhone videos are written upright.
- Image metadata (EXIF, GPS) is stripped from the output.
- Requires macOS 14 or later.

### Build

Only the Xcode Command Line Tools are needed (`xcode-select --install`), not Xcode.

```bash
scripts/export_coreml.sh        # once: converts the plate and face models to CoreML (models/*.mlpackage)
macos/build.sh                  # → macos/build/MultiBlur.app (icon: macos/Resources/AppIcon.icns, drawn by macos/tools/make_icon.swift)
open macos/build/MultiBlur.app
```

The CoreML models are versioned in `models/*.mlpackage` (22 MB): the copies converted and checked for this project, so builds never depend on a third-party download. `ARCHS="arm64 x86_64"` builds a universal app.

`export_coreml.sh` uses a separate `.venv-export` environment because `coremltools` requires torch 2.7. CenterFace is converted with `scripts/export_centerface.py` (ONNX → PyTorch via `onnx2torch` → CoreML, flexible input size up to 2048 px). Pass a size to export another model (`scripts/export_coreml.sh m`), then point `MODEL` in `macos/build.sh` to it.

The app is ad hoc signed. To run it on another Mac, right-click it and choose **Open** the first time, or sign it with a Developer ID certificate.

## Models & licenses

- [morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection): YOLOv11 trained on license plates from many countries. The model and Ultralytics are licensed under **AGPL-3.0**, which matters for commercial or distributed use.
- [deface](https://github.com/ORB-HD/deface) / CenterFace: **MIT**.

## Limitations

No detector is perfect. Plates or faces that are very small, heavily angled, blurred by motion or cut off at the frame edge can be missed (faces seen from behind or in profile are harder). Review the output before publishing anything sensitive, and lower the confidence threshold if plates slip through. At low thresholds the face detector also hides some face-like objects (round signs, traffic lights, textures): automatic filters tried for this (facial landmark checks, confirmation by Apple's person detector) also removed real faces in crowds, so they're not used.
