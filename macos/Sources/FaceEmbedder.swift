import CoreImage
import CoreML

/// Turns a face into a 128-number embedding with SFace (opencv_zoo, Apache 2.0): two views of the same
/// person give embeddings with a high cosine similarity. Used by the editor to recognize people who
/// leave and come back in a video.
final class FaceEmbedder: @unchecked Sendable {
    /// Where SFace expects the eyes, nose and mouth corners in its 112×112 input (ArcFace template),
    /// converted to Core Image's bottom-left origin.
    private static let template: [CGPoint] = [
        (38.2946, 51.6963), (73.5318, 51.5014), (56.0252, 71.7366), (41.5493, 92.3655), (70.7299, 92.2041),
    ].map { CGPoint(x: $0.0, y: 112 - $0.1) }

    private let model: MLModel
    private let context = CIContext()
    private let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!

    init(url: URL, computeUnits: MLComputeUnits) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        model = try MLModel(contentsOf: url, configuration: config)
    }

    /// Unit-length embedding of the face whose five landmarks are given (Core Image coordinates).
    func embedding(of image: CIImage, landmarks: [CGPoint]) throws -> [Float]? {
        guard landmarks.count == 5 else { return nil }
        let aligned = image.clampedToExtent().transformed(by: Self.similarity(from: landmarks, to: Self.template))
            .cropped(to: CGRect(x: 0, y: 0, width: 112, height: 112))

        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, 112, 112, kCVPixelFormatType_32BGRA, [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &buffer)
        guard let buffer else { return nil }
        context.render(aligned, to: buffer, bounds: CGRect(x: 0, y: 0, width: 112, height: 112), colorSpace: sRGB)

        let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": buffer]))
        guard let array = output.featureValue(for: "embedding")?.multiArrayValue else { return nil }
        var values = (0..<array.count).map { Float(truncating: array[$0]) }
        let norm = sqrt(values.reduce(0) { $0 + $1 * $1 })
        guard norm > 0 else { return nil }
        for i in values.indices { values[i] /= norm }
        return values
    }

    static func similarity(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
    }

    /// Least-squares rotation + uniform scale + translation mapping `source` points onto `target`
    /// (Umeyama's method, written with complex numbers).
    static func similarity(from source: [CGPoint], to target: [CGPoint]) -> CGAffineTransform {
        let n = CGFloat(source.count)
        let ms = CGPoint(x: source.map(\.x).reduce(0, +) / n, y: source.map(\.y).reduce(0, +) / n)
        let mt = CGPoint(x: target.map(\.x).reduce(0, +) / n, y: target.map(\.y).reduce(0, +) / n)
        var real: CGFloat = 0, imaginary: CGFloat = 0, energy: CGFloat = 0
        for (s, t) in zip(source, target) {
            let sx = s.x - ms.x, sy = s.y - ms.y, tx = t.x - mt.x, ty = t.y - mt.y
            real += sx * tx + sy * ty
            imaginary += sx * ty - sy * tx
            energy += sx * sx + sy * sy
        }
        let a = real / max(energy, 1e-9), b = imaginary / max(energy, 1e-9)
        // (x, y) ↦ (a·x − b·y, b·x + a·y) + translation
        return CGAffineTransform(a: a, b: b, c: -b, d: a,
                                 tx: mt.x - (a * ms.x - b * ms.y), ty: mt.y - (b * ms.x + a * ms.y))
    }
}
