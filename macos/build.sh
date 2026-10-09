#!/bin/bash
# Builds PlateBlur.app with the Command Line Tools only (no Xcode required).
set -euo pipefail
cd "$(dirname "$0")"

APP=build/PlateBlur.app
MODEL=../models/license-plate-finetune-v1s.mlpackage
FACE_MODEL=../models/CenterFace.mlpackage
ARCH=$(uname -m)

for m in "$MODEL" "$FACE_MODEL"; do
    if [ ! -d "$m" ]; then
        echo "Missing $m — run ../scripts/export_coreml.sh first." >&2
        exit 1
    fi
done

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

echo "→ Compiling CoreML models"
swift tools/compile_model.swift "$MODEL" "$APP/Contents/Resources/PlateDetector.mlmodelc"
swift tools/compile_model.swift "$FACE_MODEL" "$APP/Contents/Resources/CenterFace.mlmodelc"

echo "→ Compiling Swift sources"
swiftc -O -swift-version 5 -parse-as-library \
    -target "$ARCH-apple-macos14.0" \
    Sources/*.swift -o "$APP/Contents/MacOS/PlateBlur"

cp Info.plist "$APP/Contents/Info.plist"
[ -f Resources/AppIcon.icns ] && cp Resources/AppIcon.icns "$APP/Contents/Resources/"

echo "→ Signing (ad hoc)"
xattr -cr "$APP"  # extended attributes (e.g. Finder info) make codesign fail
codesign --force --deep --sign - "$APP"

echo "✓ Built $(pwd)/$APP"
