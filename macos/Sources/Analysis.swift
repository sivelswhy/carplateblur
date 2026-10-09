@preconcurrency import AVFoundation
import CoreImage
import Vision

/// Everything the editor needs: each frame's masks with their track identifiers, and the people
/// and plates found (tracks grouped when they're very likely the same object).
struct Analysis: @unchecked Sendable {
    /// Upright frame size, in pixels.
    let size: CGSize
    /// Presentation time of each video frame; a single zero for images.
    let frameTimes: [CMTime]
    let frames: [[Detection]]
    let groups: [TrackGroup]
    let frameRate: Double

    var isVideo: Bool { frameTimes.count > 1 }
}

/// What was changed in the editor, kept with the file so reopening the editor shows it again.
struct Edits: Sendable {
    var excludedGroups: Set<Int> = []
    var manual: [ManualMask] = []
}

/// A box drawn by hand; in videos it follows the object through the next frames.
struct ManualMask: Identifiable, Sendable {
    let id: Int
    var areas: [Int: CGRect]
}

/// One person or plate in the editor's list: one or more tracks, with a thumbnail of its best view.
struct TrackGroup: Identifiable, @unchecked Sendable {
    let id: Int
    let trackIDs: Set<Int>
    let isFace: Bool
    let thumbnail: CGImage?
    let firstFrame: Int
    let lastFrame: Int
    /// Separate stretches of the video in which this person or plate appears.
    var appearances = 1
}

extension BlurEngine {
    /// Analyzes a file once, then exports it from that analysis (no second detection pass). The analysis
    /// is returned so the editor can open instantly, showing exactly what was exported.
    func analyzeAndProcess(_ src: URL, to dst: URL, options: Options,
                           progress: @escaping @Sendable (Double) -> Void,
                           preview: @escaping @Sendable (CGImage) -> Void) async throws -> (DetectionCount, Analysis) {
        if UTType(filenameExtension: src.pathExtension)?.conforms(to: .movie) == true {
            // Videos: one pass. Export collects the analysis as it goes (no second decode or detection).
            let asset = AVURLAsset(url: src)
            guard let track = try await asset.loadTracks(withMediaType: .video).first else {
                throw EngineError.unreadable(src.lastPathComponent)
            }
            let collector = AnalysisCollector(engine: self, frameRate: Double(try await track.load(.nominalFrameRate)))
            let count = try await process(src, to: dst, options: options, collector: collector, progress: progress, preview: preview)
            return (count, collector.finish())
        }
        // Images: detection is quick; analyze, then export exactly that.
        let analysis = try await analyze(src, options: options) { _ in }
        let count = try await process(src, to: dst, options: options, plan: MaskPlan(frames: analysis.frames),
                                      progress: progress, preview: preview)
        return (count, analysis)
    }

    func analyze(_ src: URL, options: Options, progress: @escaping @Sendable (Double) -> Void) async throws -> Analysis {
        guard let type = UTType(filenameExtension: src.pathExtension) else { throw EngineError.unsupported }
        if type.conforms(to: .movie) { return try await analyzeVideo(src, options: options, progress: progress) }
        if type.conforms(to: .image) { return try analyzeImage(src, options: options) }
        throw EngineError.unsupported
    }

    private func analyzeImage(_ src: URL, options: Options) throws -> Analysis {
        guard var image = CIImage(contentsOf: src, options: [.applyOrientationProperty: true]) else {
            throw EngineError.unreadable(src.lastPathComponent)
        }
        image = image.transformed(by: .init(translationX: -image.extent.minX, y: -image.extent.minY))
        var (detections, _) = try detect(in: image, options: options, tiled: options.smallObjects != .off)
        for i in detections.indices { detections[i].id = i }
        let groups = detections.map {
            TrackGroup(id: $0.id, trackIDs: [$0.id], isFace: $0.isFace, thumbnail: thumbnail(of: $0.rect, in: image),
                       firstFrame: 0, lastFrame: 0)
        }
        return Analysis(size: image.extent.size, frameTimes: [.zero], frames: [detections], groups: groups, frameRate: 0)
    }

    private func analyzeVideo(_ src: URL, options: Options, progress: @escaping @Sendable (Double) -> Void) async throws -> Analysis {
        let asset = AVURLAsset(url: src)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw EngineError.unreadable(src.lastPathComponent)
        }
        let (naturalSize, transform, frameRate) = try await track.load(.naturalSize, .preferredTransform, .nominalFrameRate)
        let duration = max(try await asset.load(.duration).seconds, 0.001)
        let orientation = Self.orientation(for: transform)
        let upright = CGRect(origin: .zero, size: naturalSize).applying(transform).size

        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ])
        output.alwaysCopiesSampleData = false
        reader.add(output)
        guard reader.startReading() else { throw reader.error ?? EngineError.unreadable(src.lastPathComponent) }

        // Same tracking as export, so the editor shows exactly what export would produce.
        let tracker = Tracker(maxMissed: max(3, Int((Double(frameRate) * 0.4).rounded())))
        let cuts = CutDetector()
        let collector = AnalysisCollector(engine: self, frameRate: Double(frameRate))
        let tiled = options.smallObjects == .all

        var done = false
        while !done {
            try Task.checkCancellation()
            // Detect a few frames at once (in parallel), then track them in order.
            var batch: [(CIImage, CMTime, CMSampleBuffer)] = []
            while batch.count < 4, let sample = output.copyNextSampleBuffer() {
                guard let pixels = CMSampleBufferGetImageBuffer(sample) else { continue }
                var frame = CIImage(cvPixelBuffer: pixels).oriented(orientation)
                frame = frame.transformed(by: .init(translationX: -frame.extent.minX, y: -frame.extent.minY))
                batch.append((frame, CMSampleBufferGetPresentationTimeStamp(sample), sample))
            }
            done = batch.count < 4
            final class Results: @unchecked Sendable { var detections: [[Detection]] = [] }
            let results = Results()
            results.detections = Array(repeating: [], count: batch.count)
            DispatchQueue.concurrentPerform(iterations: batch.count) { i in
                results.detections[i] = (try? self.detect(in: batch[i].0, options: options, tiled: tiled).0) ?? []
            }
            for (i, (frame, time, _)) in batch.enumerated() {
                if cuts.isCut(frame) { tracker.reset() }
                _ = collector.add(tracker.update(with: results.detections[i]), frame: frame, time: time)
            }
            if let last = batch.last { progress(min(last.1.seconds / duration, 1)) }
        }
        if reader.status == .failed { throw reader.error ?? EngineError.unreadable(src.lastPathComponent) }
        _ = upright
        return collector.finish()
    }

    func thumbnail(of rect: CGRect, in image: CIImage) -> CGImage? {
        let area = rect.intersection(image.extent)
        guard !area.isNull, area.width > 1, area.height > 1 else { return nil }
        let scale = 96 / max(area.width, area.height)
        let crop = image.cropped(to: area)
            .transformed(by: CGAffineTransform(translationX: -area.minX, y: -area.minY).concatenating(.init(scaleX: scale, y: scale)))
        return imageContext.createCGImage(crop, from: crop.extent)
    }

    /// Follows a box drawn in the editor through the following frames with Vision's object tracker,
    /// until the object is lost. Returns its area per frame index.
    func trackBox(_ rect: CGRect, from start: Int, in src: URL, analysis: Analysis,
                  progress: @escaping @Sendable (Double) -> Void) async throws -> [Int: CGRect] {
        var areas = [start: rect]
        guard analysis.isVideo, start + 1 < analysis.frameTimes.count else { return areas }
        let asset = AVURLAsset(url: src)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { return areas }
        let orientation = Self.orientation(for: try await track.load(.preferredTransform))

        let reader = try AVAssetReader(asset: asset)
        reader.timeRange = CMTimeRange(start: analysis.frameTimes[start], end: .positiveInfinity)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ])
        reader.add(output)
        guard reader.startReading() else { return areas }

        let size = analysis.size
        var observation = VNDetectedObjectObservation(boundingBox: VNNormalizedRectForImageRect(rect, Int(size.width), Int(size.height)))
        let handler = VNSequenceRequestHandler()
        var index = start
        while let sample = output.copyNextSampleBuffer() {
            try Task.checkCancellation()
            guard let pixels = CMSampleBufferGetImageBuffer(sample) else { continue }
            // Skip the start frame itself (the reader may also return a frame just before it).
            let time = CMSampleBufferGetPresentationTimeStamp(sample)
            guard time > analysis.frameTimes[start] else { continue }
            index += 1
            guard index < analysis.frameTimes.count else { break }

            var frame = CIImage(cvPixelBuffer: pixels).oriented(orientation)
            frame = frame.transformed(by: .init(translationX: -frame.extent.minX, y: -frame.extent.minY))
            let request = VNTrackObjectRequest(detectedObjectObservation: observation)
            request.trackingLevel = .accurate
            try handler.perform([request], on: frame)
            guard let result = request.results?.first as? VNDetectedObjectObservation, result.confidence >= 0.3 else { break }
            observation = result
            areas[index] = VNImageRectForNormalizedRect(result.boundingBox, Int(size.width), Int(size.height))
            progress(Double(index - start) / Double(analysis.frameTimes.count - start))
        }
        reader.cancelReading()
        return areas
    }
}

private extension CGSize {
    var absolute: CGSize { CGSize(width: abs(width), height: abs(height)) }
}

/// Builds an `Analysis` frame by frame: from the export pipeline (so a single pass serves both export and
/// editor) or from the editor's own analysis. Records each track's span and best thumbnail, samples face
/// embeddings, splits tracks that jump to another person, and finally groups tracks of the same person.
final class AnalysisCollector: @unchecked Sendable {
    private let engine: BlurEngine
    private let frameRate: Double
    private var times: [CMTime] = [], frames: [[Detection]] = [], spans: [Int: Span] = [:]
    private var size: CGSize = .zero
    // Tracks split because they jumped to another person get new identifiers from here.
    private var renamed: [Int: Int] = [:], nextSplitID = 1_000_000

    init(engine: BlurEngine, frameRate: Double) {
        self.engine = engine
        self.frameRate = frameRate
    }

    /// Takes one frame's tracked areas, in frame order; returns them with their final track identifiers.
    func add(_ detections: [Detection], frame: CIImage, time: CMTime) -> [Detection] {
        let index = frames.count
        size = frame.extent.size
        var tracked = detections
        for k in tracked.indices {
            let trackerID = tracked[k].id
            tracked[k].id = renamed[trackerID] ?? trackerID
            let detection = tracked[k]
            var span = spans[detection.id] ?? Span(first: index, last: index, firstRect: detection.rect,
                                                   lastRect: detection.rect, isFace: detection.isFace)
            // Sample sharp, confidently detected faces about three times per second.
            if let embedder = engine.embedder, detection.isFace, detection.landmarks.count == 5,
               detection.score >= Self.minimumSampleScore,
               span.lastEmbedded.map({ index - $0 >= max(1, Int(frameRate / 3)) }) ?? true,
               hypot(detection.landmarks[1].x - detection.landmarks[0].x,
                     detection.landmarks[1].y - detection.landmarks[0].y) >= Self.minimumEyeDistance,
               let embedding = try? embedder.embedding(of: frame, landmarks: detection.landmarks) {
                if let mean = Self.mean(span.embeddings),
                   FaceEmbedder.similarity(embedding, mean) < Self.differentPersonSimilarity {
                    // The track jumped to someone else (e.g. faces side by side): from this frame on,
                    // it becomes a separate track, so the two people are never one group.
                    spans[detection.id] = span
                    let newID = nextSplitID
                    nextSplitID += 1
                    renamed[trackerID] = newID
                    tracked[k].id = newID
                    span = Span(first: index, last: index, firstRect: detection.rect, lastRect: detection.rect,
                                isFace: detection.isFace)
                }
                if span.embeddings.count < 12 { span.embeddings.append(embedding) }
                span.lastEmbedded = index
            }
            span.last = index
            span.lastRect = detection.rect
            let area = detection.rect.width * detection.rect.height
            if area > span.bestArea * 1.2 {
                span.bestArea = area
                span.thumbnail = engine.thumbnail(of: detection.rect, in: frame)
            }
            spans[tracked[k].id] = span
        }
        times.append(time)
        frames.append(tracked)
        return tracked
    }

    func finish() -> Analysis {
        Analysis(size: size, frameTimes: times, frames: frames, groups: Self.group(spans), frameRate: frameRate)
    }

    private struct Span {
        var first: Int, last: Int
        var firstRect: CGRect, lastRect: CGRect
        var isFace: Bool
        var bestArea: CGFloat = 0
        var thumbnail: CGImage?
        /// Face embeddings sampled along the track (sharp enough faces only).
        var embeddings: [[Float]] = []
        var lastEmbedded: Int?
    }

    /// Faces smaller than this eye distance (pixels), or less confidently detected, aren't used for
    /// recognition: on real videos, low-confidence "faces" (textures, blur) all look alike to the model.
    private static let minimumEyeDistance: CGFloat = 15
    private static let minimumSampleScore: Float = 0.6
    /// Below this similarity to its track's earlier samples, a face is someone else (different people
    /// measured up to 0.35; the same person rarely below 0.4).
    private static let differentPersonSimilarity: Float = 0.2
    /// Cosine similarity above which two people are considered the same. Calibrated on photos of
    /// public figures, also downscaled to 12–35 px between the eyes: different people stayed below 0.35,
    /// the same person had a median of 0.65.
    private static let sameFaceSimilarity: Float = 0.45

    /// Joins tracks of the same person, recognized by their face: someone who leaves and comes back,
    /// or appears in several shots. Only sharp faces are compared, and tracks visible at the same time
    /// are never joined. Position alone is never used: in crowds, whoever reappears where someone
    /// vanished is often someone else, and a wrong join would leave them unmasked.
    private static func group(_ spans: [Int: Span]) -> [TrackGroup] {
        var groupsByRoot = Dictionary(uniqueKeysWithValues: spans.keys.map { ($0, [$0]) })
        func embedding(_ ids: [Int]) -> [Float]? { mean(ids.flatMap { spans[$0]!.embeddings }) }
        func overlap(_ a: [Int], _ b: [Int]) -> Bool {
            a.contains { i in b.contains { j in spans[i]!.first <= spans[j]!.last && spans[j]!.first <= spans[i]!.last } }
        }
        while true {
            let faces = groupsByRoot.filter { spans[$0.value[0]]!.isFace }.compactMap { key, ids in embedding(ids).map { (key, ids, $0) } }
            var best: (Int, Int, Float)?
            for i in faces.indices {
                for j in faces.indices where j > i {
                    let similarity = FaceEmbedder.similarity(faces[i].2, faces[j].2)
                    if similarity >= Self.sameFaceSimilarity, similarity > (best?.2 ?? 0), !overlap(faces[i].1, faces[j].1) {
                        best = (faces[i].0, faces[j].0, similarity)
                    }
                }
            }
            guard let (a, b, _) = best else { break }
            groupsByRoot[a]! += groupsByRoot.removeValue(forKey: b)!
        }

        let members = groupsByRoot
        return members.map { groupID, ids in
            let longest = ids.max { (spans[$0]!.last - spans[$0]!.first) < (spans[$1]!.last - spans[$1]!.first) }!
            // Count separate stretches: tracks that don't touch in time.
            let ordered = ids.map { (spans[$0]!.first, spans[$0]!.last) }.sorted { $0.0 < $1.0 }
            var appearances = 0, end = Int.min
            for (first, last) in ordered {
                if first > end + 1 { appearances += 1 }
                end = max(end, last)
            }
            return TrackGroup(id: groupID, trackIDs: Set(ids), isFace: spans[longest]!.isFace,
                              thumbnail: spans[longest]!.thumbnail ?? ids.lazy.compactMap { spans[$0]!.thumbnail }.first,
                              firstFrame: ids.map { spans[$0]!.first }.min()!, lastFrame: ids.map { spans[$0]!.last }.max()!,
                              appearances: appearances)
        }
        .sorted { ($0.lastFrame - $0.firstFrame) > ($1.lastFrame - $1.firstFrame) }
    }

    /// Unit-length average of embeddings.
    private static func mean(_ embeddings: [[Float]]) -> [Float]? {
        guard let first = embeddings.first else { return nil }
        var mean = [Float](repeating: 0, count: first.count)
        for e in embeddings { for i in mean.indices { mean[i] += e[i] } }
        let norm = sqrt(mean.reduce(0) { $0 + $1 * $1 })
        return norm > 0 ? mean.map { $0 / norm } : nil
    }

}
