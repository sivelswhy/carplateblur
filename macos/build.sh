#!/bin/bash
# Builds MultiBlur.app with the Command Line Tools only (no Xcode required).
#
# Environment:
#   ARCHS         architectures to build, e.g. "arm64 x86_64" for a universal app (default: this Mac's)
#   BUILD_NUMBER  sets the bundle version to 1.0.<BUILD_NUMBER> (used by CI releases)
set -euo pipefail
cd "$(dirname "$0")"

APP=build/MultiBlur.app
MODEL=../models/license-plate-finetune-v1s.mlpackage
FACE_MODEL=../models/CenterFace.mlpackage
RECOGNITION_MODEL=../models/SFace.mlpackage
ARCHS=${ARCHS:-$(uname -m)}

for m in "$MODEL" "$FACE_MODEL" "$RECOGNITION_MODEL"; do
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
swift tools/compile_model.swift "$RECOGNITION_MODEL" "$APP/Contents/Resources/SFace.mlmodelc"

echo "→ Compiling Swift sources ($ARCHS)"
binaries=()
for arch in $ARCHS; do
    swiftc -O -swift-version 5 -parse-as-library \
        -target "$arch-apple-macos14.0" \
        Sources/*.swift -o "build/MultiBlur-$arch"
    binaries+=("build/MultiBlur-$arch")
done
lipo -create "${binaries[@]}" -output "$APP/Contents/MacOS/MultiBlur"
rm -f "${binaries[@]}"

cp Info.plist "$APP/Contents/Info.plist"
if [ -n "${BUILD_NUMBER:-}" ]; then
    plutil -replace CFBundleShortVersionString -string "1.0.$BUILD_NUMBER" "$APP/Contents/Info.plist"
    plutil -replace CFBundleVersion -string "$BUILD_NUMBER" "$APP/Contents/Info.plist"
fi
# The commit the app is built from, compared with GitHub by "Check for Updates".
if COMMIT=$(git rev-parse HEAD 2>/dev/null); then
    plutil -insert MultiBlurCommit -string "$COMMIT" "$APP/Contents/Info.plist"
fi
[ -f Resources/AppIcon.icns ] && cp Resources/AppIcon.icns "$APP/Contents/Resources/"
cp -R Resources/*.lproj "$APP/Contents/Resources/"  # translations

echo "→ Signing (ad hoc)"
xattr -cr "$APP"  # extended attributes (e.g. Finder info) make codesign fail
codesign --force --deep --sign - "$APP"

echo "✓ Built $(pwd)/$APP"
