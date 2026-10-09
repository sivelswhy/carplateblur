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
| `--conf` | `0.4` | Plate confidence threshold. Lower catches more plates but also flags ordinary text (signs, shop fronts): on test photos, 0.4 halved signs flagged by mistake compared with 0.05 while still finding ~88% of plates |
| `--face-conf`, `--thresh` | `0.2` | Face confidence threshold (deface `--thresh`) |
| `--small-objects` | `photos` | `off`, `photos` or `all`: also search overlapping tiles of large images for small, distant objects |
| `--face-max-side` | `0` | Downscale for face detection; `0` = full resolution (deface `--scale`) |
| `--track-memory` | `0.4` | Video: seconds a tracked object stays masked after it was last detected |
| `--model-size` | `s` | Plate model: `n` (fastest) → `x` (most accurate) |
| `--imgsz` | `1280` | Plate inference resolution, also the tile size |
| `--no-audio` | off | Video: remove the sound |
| `--codec` | `h264` | Video codec: `h264` or `hevc` (needs ffmpeg) |
| `--voice` | `off` | Video: disguise voices: `lower`, `higher` (5 semitones), `robot`, or `whisper`, which can't be reversed (needs ffmpeg) |
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

A single window: toggle **Plates**, **Faces** and **Voices** (disguises voices in exported videos, with the effect chosen right below), pick a **Style** (Blur, Mosaic, Black box, Image, None) and its options right below it (face mask shape, mosaic block size, replacement image), then drop photos, videos or folders anywhere on the window (or click ＋). Each file shows a thumbnail of its result, what was hidden, or a readable error. Double-click a row to open the result, click ⏹ to stop a file being processed or × to remove it from the list, or right-click for more.

**Editor**: click ✏️ on an exported file (or right-click › Edit…) to review it. The preview shows the file as it will be exported, with an outline around every mask; scrub through videos frame by frame. Hover a mask and click it (×) to remove it, keeping that face or plate visible (everywhere in a video), e.g. to blur everyone except one person; the sidebar lists every face and plate with a thumbnail and a switch. Drag to mask something that was missed: in videos, the box then follows the object with Vision's object tracker. **Export** writes the file again exactly as previewed, replacing the previous result. The editor opens instantly: files are analyzed once, during the first export (in the same pass), and the editor reuses that analysis and your previous edits; it only analyzes again if detection settings changed since. People who leave and come back, or appear in several shots, are recognized by their face (SFace) and listed once with their number of appearances, so one click covers them all. Only clearly detected faces are compared (eyes at least 15 px apart), tracks visible at the same time are never joined, tracking compares detected boxes (not enlarged masks) so it doesn't jump between faces side by side, a track is split if its face stops matching, and tracking restarts at cuts between shots. Small faces in crowds are never joined: they're listed per appearance. Check the preview before exporting.

The list of exported files is kept across relaunches (in `~/Library/Application Support/MultiBlur/history.json`), with their export date; files from earlier sessions can be opened or shown in Finder, but not edited, since their analysis isn't kept. Results that were moved or deleted drop out of the list.

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
| — | Output › Voices: Unchanged / Lower / Higher / Robot / Whisper / Synthetic voice; also per video in the editor (Disguised voice). Whisper and synthetic voice are irreversible: a vocoder keeps only the speech envelope, discards pitch and timbre, and shifts formants by a random factor never stored. Words stay understandable | Unchanged |
| `--keep-metadata` | Keep image metadata (EXIF, GPS…) | Off |
| `--ffmpeg-config` | Video codec: H.264 / HEVC | H.264 |
| `--backend`, `--execution-provider` | Run models on: Neural Engine / GPU / CPU | Neural Engine |

Also in Settings › Output: **Sound when done** plays a macOS system sound (Glass, Ping, Hero…) after each export (off by default).

Plate-specific settings: plate confidence (default 0.4) and plate mask scale (1.15×). **Detection › Small objects** (Off / Photos / Photos & videos, default Photos) also searches overlapping tiles of large images for small, distant plates and faces. The webcam mode (`deface cam`) is not included.

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

The CoreML models are versioned in `models/*.mlpackage` (40 MB): the copies converted and checked for this project, so builds never depend on a third-party download. `ARCHS="arm64 x86_64"` builds a universal app.

`export_coreml.sh` uses a separate `.venv-export` environment because `coremltools` requires torch 2.7. CenterFace is converted with `scripts/export_centerface.py` (ONNX → PyTorch via `onnx2torch` → CoreML, flexible input size up to 2048 px). Pass a size to export another model (`scripts/export_coreml.sh m`), then point `MODEL` in `macos/build.sh` to it.

The app is ad hoc signed. To run it on another Mac, right-click it and choose **Open** the first time, or sign it with a Developer ID certificate.

## Models & licenses

- [morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection): YOLOv11 trained on license plates from many countries. The model and Ultralytics are licensed under **AGPL-3.0**, which matters for commercial or distributed use.
- [deface](https://github.com/ORB-HD/deface) / CenterFace: **MIT**.
- [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) (opencv_zoo), face recognition in the editor: **Apache 2.0**. Converted with `scripts/export_sface.py`. On photos of public figures, no two different people scored above 0.32 (threshold used: 0.45), while the same person scored 0.67 (median).

## Limitations

No detector is perfect. Plates or faces that are very small, heavily angled, blurred by motion or cut off at the frame edge can be missed (faces seen from behind or in profile are harder). Review the output before publishing anything sensitive, and lower the confidence threshold if plates slip through. At low thresholds the face detector also hides some face-like objects (round signs, traffic lights, textures): automatic filters tried for this (facial landmark checks, confirmation by Apple's person detector) also removed real faces in crowds, so they're not used.

## License

Copyright (C) 2026 sivelswhy

MultiBlur is free software under the **GNU Affero General Public License v3.0** ([LICENSE](LICENSE)): you may use, study, share and modify it, provided that distributed or network-served versions keep the same license and make their source code available. The AGPL is required by the YOLO license plate model and Ultralytics, which use it.

Third-party models and code (CenterFace, deface, SFace) keep their own MIT and Apache 2.0 licenses: see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

