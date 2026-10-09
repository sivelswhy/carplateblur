@preconcurrency import AVFoundation
import AppKit
import CoreImage
import CoreImage.CIFilterBuiltins
import CoreML
import CoreText
import ImageIO
import UniformTypeIdentifiers
import Vision

enum EngineError: LocalizedError {
    case modelMissing
    case unreadable(String)
    case unsupported
    case writeFailed(String)
    case replacementImageMissing
    case replacementImageUnreadable(String)

    var errorDescription: String? {
        switch self {
        case .modelMissing: String(localized: "Detection models not found in the app bundle.")
        case .unreadable(let name): String(localized: "Cannot read \(name).")
        case .unsupported: String(localized: "Unsupported file type.")
        case .writeFailed(let reason): String(localized: "Could not write output: \(reason)")
        case .replacementImageMissing: String(localized: "Choose a replacement image next to the Style menu.")
        case .replacementImageUnreadable(let name): String(localized: "Cannot read the replacement image \(name).")
        }
    }
}

struct DetectionCount: Sendable, Equatable {
    var plates = 0
    var faces = 0

    static func += (lhs: inout DetectionCount, rhs: DetectionCount) {
        lhs.plates += rhs.plates
        lhs.faces += rhs.faces
    }
}

/// A detected area, already enlarged by its mask scale, in Core Image pixel coordinates.
struct Detection {
    var rect: CGRect
    var score: Float
    var isFace: Bool
}

/// Detects license plates (YOLOv11 via Vision) and faces (CenterFace from deface), then masks them
/// with Core Image. Runs entirely on-device; no network access.
final class BlurEngine: @unchecked Sendable {
    /// Video: how long a tracked object stays masked after the detector last saw it.
    private static let trackMemory: Double = 0.4
    private static let previewInterval: TimeInterval = 0.15

    let compute: ComputeMode
    private let plateModel: VNCoreMLModel
    private let faces: FaceDetector
    private let imageContext = CIContext()
    // Video frames are masked in the decoder's native color space: no conversion round-trip.
    private let videoContext = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull()])

    init(compute: ComputeMode) throws {
        guard let plateURL = Bundle.main.url(forResource: "PlateDetector", withExtension: "mlmodelc"),
              let faceURL = Bundle.main.url(forResource: "CenterFace", withExtension: "mlmodelc") else {
            throw EngineError.modelMissing
        }
        self.compute = compute
        let config = MLModelConfiguration()
        config.computeUnits = compute.computeUnits
        plateModel = try VNCoreMLModel(for: MLModel(contentsOf: plateURL, configuration: config))
        // The model has built-in NMS; keep its threshold low and filter by the user's value instead.
        plateModel.featureProvider = try MLDictionaryFeatureProvider(dictionary: [
            "iouThreshold": 0.45,
            "confidenceThreshold": 0.05,
        ])
        faces = try FaceDetector(url: faceURL, computeUnits: compute.computeUnits)
    }

    static func outputURL(for url: URL, folder: URL?) -> URL {
        let isVideo = UTType(filenameExtension: url.pathExtension)?.conforms(to: .movie) ?? false
        var ext = url.pathExtension.lowercased()
        if isVideo, !["mov", "mp4", "m4v"].contains(ext) { ext = "mp4" }
        if !isVideo, !["jpg", "jpeg", "png", "heic", "heif", "tif", "tiff"].contains(ext) { ext = "png" }
        let name = url.deletingPathExtension().lastPathComponent + "_anonymized"
        let directory = folder ?? url.deletingLastPathComponent()
        // Never overwrite an existing file: "photo_anonymized (1).jpg", "(2)"… like Finder.
        var candidate = directory.appendingPathComponent(name).appendingPathExtension(ext)
        var copy = 1
        while FileManager.default.fileExists(atPath: candidate.path) {
            candidate = directory.appendingPathComponent("\(name) (\(copy))").appendingPathExtension(ext)
            copy += 1
        }
        return candidate
    }

    /// Processes an image or a video. `preview` receives downscaled processed frames when live preview is on.
    func process(_ src: URL, to dst: URL, options: Options,
                 progress: @escaping @Sendable (Double) -> Void,
                 preview: @escaping @Sendable (CGImage) -> Void) async throws -> DetectionCount {
        guard let type = UTType(filenameExtension: src.pathExtension) else { throw EngineError.unsupported }
        let replacement = try loadReplacementImage(options)
        if type.conforms(to: .movie) {
            return try await processVideo(src, to: dst, options: options, replacement: replacement,
                                          progress: progress, preview: preview)
        }
        if type.conforms(to: .image) {
            let count = try processImage(src, to: dst, options: options, replacement: replacement, preview: preview)
            progress(1)
            return count
        }
        throw EngineError.unsupported
    }

    private func loadReplacementImage(_ options: Options) throws -> CIImage? {
        guard options.mode == .image else { return nil }
        guard let path = options.replacementImagePath else { throw EngineError.replacementImageMissing }
        let url = URL(fileURLWithPath: path)
        if let image = CIImage(contentsOf: url, options: [.applyOrientationProperty: true]) { return image }
        // Core Image can't read vector formats such as SVG: rasterize them through NSImage instead.
        guard let image = NSImage(contentsOf: url), let raster = Self.rasterize(image) else {
            throw EngineError.replacementImageUnreadable(url.lastPathComponent)
        }
        return CIImage(cgImage: raster)
    }

    /// Renders an NSImage (e.g. an SVG) with transparency, 1024 px on its longest side so it stays sharp when stretched.
    private static func rasterize(_ image: NSImage) -> CGImage? {
        guard image.size.width > 0, image.size.height > 0 else { return nil }
        let scale = 1024 / max(image.size.width, image.size.height)
        let width = Int((image.size.width * scale).rounded()), height = Int((image.size.height * scale).rounded())
        guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(cgContext: context, flipped: false)
        image.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        NSGraphicsContext.restoreGraphicsState()
        return context.makeImage()
    }

    // MARK: - Detection

    /// `tiled` adds passes on overlapping tiles to find small or distant objects.
    func detect(in image: CIImage, options: Options, tiled: Bool = false) throws -> ([Detection], DetectionCount) {
        // Plates (Vision, Neural Engine) and faces (CenterFace, GPU) use different hardware: run them side by side.
        final class Results: @unchecked Sendable {
            var plates: [(rect: CGRect, score: Float)] = []
            var faces: [(rect: CGRect, score: Float)] = []
            var error: Error?
        }
        let results = Results()
        DispatchQueue.concurrentPerform(iterations: 2) { task in
            do {
                if task == 0, options.maskPlates {
                    results.plates = try Self.search(image, tileSize: tiled ? 960 : nil) {
                        try detectPlates(in: $0, minConfidence: Float(options.plateConfidence))
                    }
                } else if task == 1, options.maskFaces {
                    let maxSide = options.faceResolution.maxSide
                    // Faces are already analyzed at up to 2048 px: tiles only help larger images at full resolution.
                    let tile: CGFloat? = tiled && options.faceResolution == .full ? maxSide : nil
                    results.faces = try Self.search(image, tileSize: tile) {
                        try faces.detect(in: $0, threshold: Float(options.faceThreshold), maxSide: maxSide)
                    }
                }
            } catch {
                results.error = error
            }
        }
        if let error = results.error { throw error }

        // deface's mask scale enlarges each side by (scale - 1) × the box size.
        func enlarge(_ rect: CGRect, scale: Double) -> CGRect {
            let s = CGFloat(scale - 1)
            return rect.insetBy(dx: -rect.width * s, dy: -rect.height * s)
        }
        let detections = results.plates.map { Detection(rect: enlarge($0.rect, scale: options.plateMaskScale), score: $0.score, isFace: false) }
            + results.faces.map { Detection(rect: enlarge($0.rect, scale: options.faceMaskScale), score: $0.score, isFace: true) }
        return (detections, DetectionCount(plates: results.plates.count, faces: results.faces.count))
    }

    typealias Found = (rect: CGRect, score: Float)

    /// Runs `detector` on the whole image and, when `tileSize` is set and the image is larger,
    /// on overlapping tiles too; duplicates found in several passes are merged.
    static func search(_ image: CIImage, tileSize: CGFloat?, _ detector: (CIImage) throws -> [Found]) rethrows -> [Found] {
        var found = try detector(image)
        guard let tileSize else { return found }
        for tile in tiles(covering: image.extent, size: tileSize) {
            let crop = image.cropped(to: tile).transformed(by: .init(translationX: -tile.minX, y: -tile.minY))
            found += try detector(crop).map { ($0.rect.offsetBy(dx: tile.minX, dy: tile.minY), $0.score) }
        }
        return suppressDuplicates(found)
    }

    /// Tiles of `size` overlapping by 25%, covering `extent`; none when the image is barely larger than a tile.
    static func tiles(covering extent: CGRect, size: CGFloat) -> [CGRect] {
        guard max(extent.width, extent.height) > size * 1.25 else { return [] }
        let step = size * 0.75
        func starts(_ length: CGFloat) -> [CGFloat] {
            guard length > size else { return [0] }
            let count = Int(((length - size) / step).rounded(.up)) + 1
            return (0..<count).map { min(CGFloat($0) * step, length - size) }
        }
        return starts(extent.height).flatMap { y in
            starts(extent.width).map { x in
                CGRect(x: extent.minX + x, y: extent.minY + y, width: min(size, extent.width), height: min(size, extent.height))
            }
        }
    }

    /// Keeps the best of overlapping boxes, including a partial box cut by a tile edge inside a full one.
    static func suppressDuplicates(_ found: [Found]) -> [Found] {
        var kept: [Found] = []
        for candidate in found.sorted(by: { $0.score > $1.score }) {
            let duplicate = kept.contains { other in
                let overlap = other.rect.intersection(candidate.rect)
                guard !overlap.isNull else { return false }
                let intersection = overlap.width * overlap.height
                let a = other.rect.width * other.rect.height, b = candidate.rect.width * candidate.rect.height
                return intersection / (a + b - intersection) >= 0.4 || intersection / min(a, b) >= 0.7
            }
            if !duplicate { kept.append(candidate) }
        }
        return kept
    }

    /// Returns plate rectangles in the image's pixel coordinates (origin bottom-left, like Core Image).
    private func detectPlates(in image: CIImage, minConfidence: Float) throws -> [(rect: CGRect, score: Float)] {
        let request = VNCoreMLRequest(model: plateModel)
        request.imageCropAndScaleOption = .scaleFit
        try VNImageRequestHandler(ciImage: image).perform([request])
        let extent = image.extent
        return (request.results as? [VNRecognizedObjectObservation] ?? [])
            .filter { $0.confidence >= minConfidence }
            .map { (VNImageRectForNormalizedRect($0.boundingBox, Int(extent.width), Int(extent.height)), $0.confidence) }
    }

    // MARK: - Masking

    func mask(_ image: CIImage, detections: [Detection], options: Options, replacement: CIImage?) -> CIImage {
        var output = image
        for detection in detections {
            let rect = detection.rect.intersection(image.extent).integral
            guard !rect.isNull, rect.width >= 2, rect.height >= 2 else { continue }

            if let patch = patch(for: rect, in: image, mode: options.mode, mosaicSize: options.mosaicSize, replacement: replacement) {
                if detection.isFace, options.faceShape == .ellipse {
                    // Ellipse inscribed in the (unclipped) detection box, like deface. The blend is
                    // limited to the box: blending over the whole frame made videos ~15× slower.
                    let blended = patch.applyingFilter("CIBlendWithMask", parameters: [
                        kCIInputBackgroundImageKey: output.cropped(to: rect),
                        kCIInputMaskImageKey: Self.ellipseMask(in: detection.rect),
                    ]).cropped(to: rect)
                    output = blended.composited(over: output)
                } else {
                    output = patch.cropped(to: rect).composited(over: output)
                }
            }
        }
        if options.drawScores {
            for detection in detections {
                output = drawScore(detection.score, above: detection.rect, on: output)
            }
        }
        return output
    }

    private func patch(for rect: CGRect, in image: CIImage, mode: MaskMode, mosaicSize: Double, replacement: CIImage?) -> CIImage? {
        // Patches only sample pixels inside the masked area, so nothing readable leaks back in.
        let source = image.cropped(to: rect).clampedToExtent()
        switch mode {
        case .none:
            return nil
        case .solid:
            return CIImage(color: .black)
        case .blur:
            return source.applyingGaussianBlur(sigma: Double(max(rect.width, rect.height)) / 5).cropped(to: rect)
        case .mosaic:
            let filter = CIFilter.pixellate()
            filter.inputImage = source
            filter.scale = Float(max(mosaicSize, 2))
            filter.center = rect.origin
            return filter.outputImage?.cropped(to: rect)
        case .image:
            // Stretched to the box like deface. Unlike deface, transparent areas show a blurred version
            // of the area instead of the original pixels, so the face can't be seen through them.
            guard let replacement else { return CIImage(color: .black) }
            let blurred = source.applyingGaussianBlur(sigma: Double(max(rect.width, rect.height)) / 5).cropped(to: rect)
            let r = replacement.extent
            return replacement
                .transformed(by: CGAffineTransform(translationX: -r.minX, y: -r.minY)
                    .concatenating(CGAffineTransform(scaleX: rect.width / r.width, y: rect.height / r.height))
                    .concatenating(CGAffineTransform(translationX: rect.minX, y: rect.minY)))
                .composited(over: blurred)
        }
    }

    private static func ellipseMask(in rect: CGRect) -> CIImage {
        // A hard-edged white disk of radius 100 at the origin, stretched to the box.
        let disk = CIFilter.radialGradient()
        disk.center = .zero
        disk.radius0 = 99.5
        disk.radius1 = 100.5
        disk.color0 = .white
        disk.color1 = .black
        return disk.outputImage!.transformed(by: CGAffineTransform(scaleX: rect.width / 200, y: rect.height / 200)
            .concatenating(CGAffineTransform(translationX: rect.midX, y: rect.midY)))
    }

    /// Green score label above the box, like deface's `--draw-scores`.
    private func drawScore(_ score: Float, above rect: CGRect, on image: CIImage) -> CIImage {
        let fontSize = max(12, min(rect.height * 0.25, 48))
        let text = NSAttributedString(string: String(format: "%.2f", score), attributes: [
            NSAttributedString.Key(kCTFontAttributeName as String): CTFontCreateWithName("Helvetica-Bold" as CFString, fontSize, nil),
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(red: 0, green: 1, blue: 0, alpha: 1),
        ])
        let line = CTLineCreateWithAttributedString(text)
        let bounds = CTLineGetBoundsWithOptions(line, [])
        let width = Int(bounds.width.rounded(.up)) + 4, height = Int(bounds.height.rounded(.up)) + 4
        guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return image }
        context.textPosition = CGPoint(x: 2, y: 2 - bounds.minY)
        CTLineDraw(line, context)
        guard let label = context.makeImage() else { return image }

        let y = min(rect.maxY + 2, image.extent.maxY - CGFloat(height))
        let x = max(image.extent.minX, min(rect.minX, image.extent.maxX - CGFloat(width)))
        return CIImage(cgImage: label).transformed(by: .init(translationX: x, y: y)).composited(over: image)
    }

    private func makePreview(_ image: CIImage) -> CGImage? {
        let scale = min(1, 640 / max(image.extent.width, image.extent.height))
        let small = image.transformed(by: .init(scaleX: scale, y: scale))
        return imageContext.createCGImage(small, from: small.extent)
    }

    // MARK: - Images

    private func processImage(_ src: URL, to dst: URL, options: Options, replacement: CIImage?,
                              preview: @Sendable (CGImage) -> Void) throws -> DetectionCount {
        guard var image = CIImage(contentsOf: src, options: [.applyOrientationProperty: true]) else {
            throw EngineError.unreadable(src.lastPathComponent)
        }
        image = image.transformed(by: .init(translationX: -image.extent.minX, y: -image.extent.minY))

        let (detections, count) = try detect(in: image, options: options, tiled: options.smallObjects != .off)
        try Task.checkCancellation()
        let output = mask(image, detections: detections, options: options, replacement: replacement)
        if options.livePreview, let frame = makePreview(output) { preview(frame) }

        var colorSpace = image.colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB)!
        if colorSpace.model != .rgb { colorSpace = CGColorSpace(name: CGColorSpace.sRGB)! }
        guard let rendered = imageContext.createCGImage(output, from: image.extent, format: .RGBA8, colorSpace: colorSpace) else {
            throw EngineError.writeFailed("could not render image")
        }

        let ext = dst.pathExtension.lowercased()
        let type: UTType = switch ext {
        case "jpg", "jpeg": .jpeg
        case "heic", "heif": .heic
        case "tif", "tiff": .tiff
        default: .png
        }
        // Metadata (EXIF, GPS…) is dropped unless "Keep metadata" is on, like deface's --keep-metadata.
        var properties: [CFString: Any] = [:]
        if options.keepMetadata, let source = CGImageSourceCreateWithURL(src as CFURL, nil),
           let original = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any] {
            properties = original
            // Pixels are already upright.
            properties[kCGImagePropertyOrientation] = 1
            if var tiff = properties[kCGImagePropertyTIFFDictionary] as? [CFString: Any] {
                tiff[kCGImagePropertyTIFFOrientation] = 1
                properties[kCGImagePropertyTIFFDictionary] = tiff
            }
            properties[kCGImagePropertyPixelWidth] = nil
            properties[kCGImagePropertyPixelHeight] = nil
        }
        if type == .jpeg || type == .heic { properties[kCGImageDestinationLossyCompressionQuality] = 0.92 }

        try? FileManager.default.removeItem(at: dst)
        guard let destination = CGImageDestinationCreateWithURL(dst as CFURL, type.identifier as CFString, 1, nil) else {
            throw EngineError.writeFailed("unsupported output format")
        }
        CGImageDestinationAddImage(destination, rendered, properties as CFDictionary)
        guard CGImageDestinationFinalize(destination) else { throw EngineError.writeFailed("could not save image") }
        return count
    }

    // MARK: - Videos

    /// Number of frames whose detection runs ahead of encoding.
    private static let lookahead = 3

    private final class PendingFrame: @unchecked Sendable {
        let frame: CIImage
        let time: CMTime
        let sample: CMSampleBuffer  // keeps the decoded pixels alive until the frame is encoded
        var result: ([Detection], DetectionCount) = ([], DetectionCount())
        let done = DispatchSemaphore(value: 0)

        init(frame: CIImage, time: CMTime, sample: CMSampleBuffer) {
            self.frame = frame
            self.time = time
            self.sample = sample
        }
    }

    private final class VideoState: @unchecked Sendable {
        var pending: [PendingFrame] = []
        var readerFinished = false
        var cancelled = false
        let tracker: Tracker
        var detections = DetectionCount()

        init(tracker: Tracker) {
            self.tracker = tracker
        }
        var failure: Error?
        var lastPreview = Date.distantPast
    }

    private func processVideo(_ src: URL, to dst: URL, options: Options, replacement: CIImage?,
                              progress: @escaping @Sendable (Double) -> Void,
                              preview: @escaping @Sendable (CGImage) -> Void) async throws -> DetectionCount {
        let asset = AVURLAsset(url: src)
        guard let videoTrack = try await asset.loadTracks(withMediaType: .video).first else {
            throw EngineError.unreadable(src.lastPathComponent)
        }
        let audioTrack = options.keepAudio ? try await asset.loadTracks(withMediaType: .audio).first : nil
        let (naturalSize, transform, dataRate, frameRate) = try await videoTrack.load(
            .naturalSize, .preferredTransform, .estimatedDataRate, .nominalFrameRate)
        let duration = max(try await asset.load(.duration).seconds, 0.001)

        // Frames are rotated upright before detection, then written upright with an identity transform.
        let orientation = Self.orientation(for: transform)
        let upright = CGRect(origin: .zero, size: naturalSize).applying(transform).size
        let width = Int(abs(upright.width).rounded()), height = Int(abs(upright.height).rounded())

        let reader = try AVAssetReader(asset: asset)
        let videoOutput = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ])
        videoOutput.alwaysCopiesSampleData = false
        reader.add(videoOutput)

        try? FileManager.default.removeItem(at: dst)
        let writer = try AVAssetWriter(outputURL: dst, fileType: Self.fileType(for: dst))
        // H.264 encoders top out at 4096 px; larger videos always use HEVC.
        let codec: AVVideoCodecType = (options.videoCodec == .hevc || width > 4096 || height > 4096) ? .hevc : .h264
        let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: codec,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: max(Double(dataRate) * 1.2, 4_000_000)],
        ])
        videoInput.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: videoInput, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ])
        writer.add(videoInput)

        // Audio is decoded and re-encoded to AAC so any source codec fits an MP4/MOV container.
        // It is normalized to 44.1/48 kHz stereo at 192 kbps, a combination the AAC encoder always
        // accepts (e.g. 22.05 kHz mono HE-AAC from TikTok/Instagram cannot be encoded at 192 kbps).
        var audioOutput: AVAssetReaderTrackOutput?
        var audioInput: AVAssetWriterInput?
        if let audioTrack, let format = try await audioTrack.load(.formatDescriptions).first,
           let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(format)?.pointee {
            let channels = 2
            let sampleRate: Double = asbd.mSampleRate == 48_000 ? 48_000 : 44_100
            let output = AVAssetReaderTrackOutput(track: audioTrack, outputSettings: [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: sampleRate,
                AVNumberOfChannelsKey: channels,
            ])
            let input = AVAssetWriterInput(mediaType: .audio, outputSettings: [
                AVFormatIDKey: kAudioFormatMPEG4AAC,
                AVSampleRateKey: sampleRate,
                AVNumberOfChannelsKey: channels,
                AVEncoderBitRateKey: 192_000,
            ])
            input.expectsMediaDataInRealTime = false
            if reader.canAdd(output), writer.canAdd(input) {
                reader.add(output)
                writer.add(input)
                audioOutput = output
                audioInput = input
            }
        }

        guard reader.startReading() else { throw reader.error ?? EngineError.unreadable(src.lastPathComponent) }
        guard writer.startWriting() else { throw EngineError.writeFailed(writer.error?.localizedDescription ?? "unknown") }
        writer.startSession(atSourceTime: .zero)

        let state = VideoState(tracker: Tracker(maxMissed: max(3, Int((Double(frameRate) * Self.trackMemory).rounded()))))
        let tiled = options.smallObjects == .all
        let group = DispatchGroup()
        let detectionQueue = DispatchQueue(label: "multiblur.detect", qos: .userInitiated, attributes: .concurrent)

        group.enter()
        videoInput.requestMediaDataWhenReady(on: DispatchQueue(label: "multiblur.video", qos: .userInitiated)) { [self] in
            while videoInput.isReadyForMoreMediaData {
                // Keep a few frames detecting ahead of the one being encoded, so the GPU, Neural Engine
                // and video encoder all stay busy instead of waiting on each other.
                while !state.readerFinished, state.pending.count < Self.lookahead {
                    guard let sample = videoOutput.copyNextSampleBuffer(),
                          let pixels = CMSampleBufferGetImageBuffer(sample) else {
                        state.readerFinished = true
                        break
                    }
                    var frame = CIImage(cvPixelBuffer: pixels).oriented(orientation)
                    frame = frame.transformed(by: .init(translationX: -frame.extent.minX, y: -frame.extent.minY))
                    let pending = PendingFrame(frame: frame, time: CMSampleBufferGetPresentationTimeStamp(sample), sample: sample)
                    state.pending.append(pending)
                    detectionQueue.async {
                        pending.result = (try? self.detect(in: frame, options: options, tiled: tiled)) ?? ([], DetectionCount())
                        pending.done.signal()
                    }
                }
                if state.cancelled, state.failure == nil { state.failure = CancellationError() }
                guard state.failure == nil, !state.pending.isEmpty else {
                    // Let in-flight detections finish before tearing down.
                    state.pending.forEach { $0.done.wait() }
                    videoInput.markAsFinished()
                    group.leave()
                    return
                }

                let current = state.pending.removeFirst()
                current.done.wait()
                autoreleasepool {
                    let (detections, count) = current.result
                    state.detections += count
                    let tracked = state.tracker.update(with: detections)
                    let masked = mask(current.frame, detections: tracked, options: options, replacement: replacement)

                    var buffer: CVPixelBuffer?
                    guard let pool = adaptor.pixelBufferPool,
                          CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer) == kCVReturnSuccess,
                          let buffer else {
                        // The pool disappears when the writer has failed: report the writer's own error.
                        state.failure = writer.error ?? EngineError.writeFailed("could not allocate a frame buffer")
                        return
                    }
                    videoContext.render(masked, to: buffer, bounds: CGRect(x: 0, y: 0, width: width, height: height), colorSpace: nil)
                    if !adaptor.append(buffer, withPresentationTime: current.time) {
                        state.failure = writer.error ?? EngineError.writeFailed("frame rejected")
                    }
                    if options.livePreview, Date().timeIntervalSince(state.lastPreview) > Self.previewInterval,
                       let image = makePreview(CIImage(cvPixelBuffer: buffer)) {
                        state.lastPreview = Date()
                        preview(image)
                    }
                }
                progress(min(current.time.seconds / duration, 1))
            }
        }

        if let audioOutput, let audioInput {
            group.enter()
            audioInput.requestMediaDataWhenReady(on: DispatchQueue(label: "multiblur.audio")) {
                while audioInput.isReadyForMoreMediaData {
                    guard state.failure == nil, let sample = audioOutput.copyNextSampleBuffer() else {
                        audioInput.markAsFinished()
                        group.leave()
                        return
                    }
                    audioInput.append(sample)
                }
            }
        }

        await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                group.notify(queue: .global()) { continuation.resume() }
            }
        } onCancel: {
            state.cancelled = true
        }

        if let failure = state.failure ?? (reader.status == .failed ? reader.error : nil) {
            reader.cancelReading()
            writer.cancelWriting()
            try? FileManager.default.removeItem(at: dst)
            throw failure
        }
        await writer.finishWriting()
        if writer.status != .completed {
            throw EngineError.writeFailed(writer.error?.localizedDescription ?? "unknown")
        }
        return state.detections
    }

    private static func orientation(for t: CGAffineTransform) -> CGImagePropertyOrientation {
        switch (t.a, t.b, t.c, t.d) {
        case (0, 1, -1, 0): .right
        case (0, -1, 1, 0): .left
        case (-1, 0, 0, -1): .down
        default: .up
        }
    }

    private static func fileType(for url: URL) -> AVFileType {
        switch url.pathExtension.lowercased() {
        case "mov": .mov
        case "m4v": .m4v
        default: .mp4
        }
    }
}
