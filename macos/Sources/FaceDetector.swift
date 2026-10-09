import CoreImage
import CoreML

/// CenterFace face detector, ported from deface (https://github.com/ORB-HD/deface).
/// Runs the CoreML conversion of deface's centerface.onnx and decodes its heatmap in Swift.
final class FaceDetector: @unchecked Sendable {
    private static let nmsThreshold: Float = 0.3

    private let model: MLModel
    private let context = CIContext()
    private let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!

    init(url: URL, computeUnits: MLComputeUnits) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        model = try MLModel(contentsOf: url, configuration: config)
    }

    /// Returns face rectangles and scores in the image's pixel coordinates (origin bottom-left, like Core Image).
    /// `maxSide` downscales large images before inference, like deface's `--scale`.
    func detect(in image: CIImage, threshold: Float, maxSide: CGFloat) throws -> [(rect: CGRect, score: Float)] {
        let extent = image.extent
        let fit = min(1, maxSide / max(extent.width, extent.height))
        // Same preprocessing as deface: resize so both sides are multiples of 32.
        let width = Int((extent.width * fit / 32).rounded(.up)) * 32
        let height = Int((extent.height * fit / 32).rounded(.up)) * 32
        let scaleX = CGFloat(width) / extent.width, scaleY = CGFloat(height) / extent.height

        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA,
                            [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &buffer)
        guard let buffer else { return [] }
        context.render(image.transformed(by: .init(scaleX: scaleX, y: scaleY)), to: buffer,
                       bounds: CGRect(x: 0, y: 0, width: width, height: height), colorSpace: sRGB)

        let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": buffer]))
        func array(_ name: String) -> FeatureMap { FeatureMap(output.featureValue(for: name)!.multiArrayValue!) }
        let boxes = decode(heatmap: array("heatmap"), scale: array("scale"), offset: array("offset"),
                           inputWidth: Float(width), inputHeight: Float(height), threshold: threshold)

        // Back to original pixels, flipping from top-left to bottom-left origin.
        return boxes.map { box in
            let x1 = CGFloat(box.x1) / scaleX, x2 = CGFloat(box.x2) / scaleX
            let y1 = CGFloat(box.y1) / scaleY, y2 = CGFloat(box.y2) / scaleY
            return (CGRect(x: x1, y: extent.height - y2, width: x2 - x1, height: y2 - y1), box.score)
        }
    }

    private struct Box {
        var x1, y1, x2, y2, score: Float
        var area: Float { (x2 - x1) * (y2 - y1) }
    }

    /// A [1, C, H, W] model output copied into a dense Float array. Reads Float16 bits directly
    /// (the generic MLShapedArray conversion cost ~20 ms per frame, and Swift's Float16 type
    /// is unavailable on Intel Macs).
    private struct FeatureMap {
        let rows: Int, cols: Int
        let values: [Float]

        init(_ array: MLMultiArray) {
            let shape = array.shape.map(\.intValue), strides = array.strides.map(\.intValue)
            let channels = shape[1]
            rows = shape[2]
            cols = shape[3]
            var values = [Float](repeating: 0, count: channels * rows * cols)
            let isHalf = array.dataType == .float16
            array.withUnsafeBytes { raw in
                var i = 0
                for c in 0..<channels {
                    for r in 0..<shape[2] {
                        let row = c * strides[1] + r * strides[2]
                        for col in 0..<shape[3] {
                            let index = row + col * strides[3]
                            values[i] = isHalf
                                ? Self.float(fromHalf: raw.load(fromByteOffset: index * 2, as: UInt16.self))
                                : raw.load(fromByteOffset: index * 4, as: Float.self)
                            i += 1
                        }
                    }
                }
            }
            self.values = values
        }

        /// IEEE 754 half → single precision.
        @inline(__always) private static func float(fromHalf h: UInt16) -> Float {
            let sign = UInt32(h & 0x8000) << 16
            let exponent = UInt32(h >> 10) & 0x1F
            let mantissa = UInt32(h & 0x3FF)
            switch exponent {
            case 0:  // zero or subnormal
                let magnitude = Float(mantissa) * 0x1p-24
                return sign == 0 ? magnitude : -magnitude
            case 31:  // infinity or NaN
                return Float(bitPattern: sign | 0x7F80_0000 | (mantissa << 13))
            default:
                return Float(bitPattern: sign | ((exponent + 112) << 23) | (mantissa << 13))
            }
        }
    }

    /// Port of `CenterFace.decode` from deface (top-left origin, input-image pixels).
    private func decode(heatmap: FeatureMap, scale: FeatureMap, offset: FeatureMap,
                        inputWidth: Float, inputHeight: Float, threshold: Float) -> [Box] {
        let rows = heatmap.rows, cols = heatmap.cols, plane = rows * cols
        let heat = heatmap.values, scales = scale.values, offsets = offset.values

        var boxes: [Box] = []
        for r in 0..<rows {
            for c in 0..<cols {
                let i = r * cols + c
                let score = heat[i]
                guard score > threshold else { continue }
                let s0 = exp(scales[i]) * 4, s1 = exp(scales[plane + i]) * 4
                let o0 = offsets[i], o1 = offsets[plane + i]
                let x1 = min(max(0, (Float(c) + o1 + 0.5) * 4 - s1 / 2), inputWidth)
                let y1 = min(max(0, (Float(r) + o0 + 0.5) * 4 - s0 / 2), inputHeight)
                boxes.append(Box(x1: x1, y1: y1, x2: min(x1 + s1, inputWidth), y2: min(y1 + s0, inputHeight), score: score))
            }
        }
        return nonMaximumSuppression(boxes)
    }

    private func nonMaximumSuppression(_ boxes: [Box]) -> [Box] {
        var kept: [Box] = []
        for box in boxes.sorted(by: { $0.score > $1.score }) {
            let overlaps = kept.contains { other in
                let w = max(0, min(box.x2, other.x2) - max(box.x1, other.x1))
                let h = max(0, min(box.y2, other.y2) - max(box.y1, other.y1))
                let intersection = w * h
                return intersection / (box.area + other.area - intersection) >= Self.nmsThreshold
            }
            if !overlaps { kept.append(box) }
        }
        return kept
    }
}
