#!/bin/bash
# Builds MultiBlur.app with the Command Line Tools only (no Xcode required).
#
# Environment:
#   ARCHS         architectures to build, e.g. "arm64 x86_64" for a universal app (default: this Mac's)
#   BUILD_NUMBER  sets the bundle version to 1.0.<BUILD_NUMBER> (used by CI releases)
#   BUNDLE_MODELS 0 = don't embed the CoreML models; the app then downloads them on first launch
#                 (default: embed them when ../models has them, e.g. after scripts/export_coreml.sh)
set -euo pipefail
cd "$(dirname "$0")"

APP=build/MultiBlur.app
MODEL=../models/license-plate-finetune-v1s.mlpackage
FACE_MODEL=../models/CenterFace.mlpackage
ARCHS=${ARCHS:-$(uname -m)}

if [ "${BUNDLE_MODELS:-1}" != 0 ] && [ -d "$MODEL" ] && [ -d "$FACE_MODEL" ]; then
    BUNDLE_MODELS=1
else
    BUNDLE_MODELS=0
fi

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

if [ "$BUNDLE_MODELS" = 1 ]; then
    echo "→ Compiling CoreML models"
    swift tools/compile_model.swift "$MODEL" "$APP/Contents/Resources/PlateDetector.mlmodelc"
    swift tools/compile_model.swift "$FACE_MODEL" "$APP/Contents/Resources/CenterFace.mlmodelc"
else
    echo "→ Models not embedded: the app downloads them on first launch"
fi

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
[ -f Resources/AppIcon.icns ] && cp Resources/AppIcon.icns "$APP/Contents/Resources/"

echo "→ Signing (ad hoc)"
xattr -cr "$APP"  # extended attributes (e.g. Finder info) make codesign fail
codesign --force --deep --sign - "$APP"

echo "✓ Built $(pwd)/$APP"
