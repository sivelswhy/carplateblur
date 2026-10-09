#!/bin/bash
# Packages the compiled CoreML models that the macOS app downloads on first launch,
# and prints the SHA-256 to put in macos/Sources/ModelStore.swift.
# Usage: scripts/package_models.sh [version]   (default: v1)
#   then: gh release create models-<version> MultiBlur-models-<version>.zip --latest=false
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION=${1:-v1}
OUT="MultiBlur-models-$VERSION.zip"
WORK=$(mktemp -d)

swift macos/tools/compile_model.swift models/license-plate-finetune-v1s.mlpackage "$WORK/PlateDetector.mlmodelc"
swift macos/tools/compile_model.swift models/CenterFace.mlpackage "$WORK/CenterFace.mlmodelc"
rm -f "$OUT"
(cd "$WORK" && zip -qr -X - PlateDetector.mlmodelc CenterFace.mlmodelc) > "$OUT"
rm -rf "$WORK"

echo "$OUT: $(stat -f%z "$OUT") bytes"
echo "sha256: $(shasum -a 256 "$OUT" | cut -d' ' -f1)"
