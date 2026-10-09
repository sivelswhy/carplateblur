import CoreGraphics

/// Follows plates and faces across video frames so each object keeps exactly one steady mask, and that
/// mask keeps following the object for a few frames when the detector misses it.
final class Tracker {
    private struct Track {
        let id: Int
        var detection: Detection
        /// The detected box (without the mask's margin), used for matching.
        var core: CGRect
        var velocity: CGVector = .zero
        var missed = 0
        /// Frames in which the object was actually detected.
        var hits = 1
    }

    /// Minimum overlap between a track's predicted box and a detection to continue the track.
    private static let minimumOverlap: CGFloat = 0.1

    /// How much a new detection changes the mask's size: detectors' box sizes flicker from frame to frame.
    private static let sizeResponse: CGFloat = 0.3
    /// A smoothed mask is never smaller than this share of the current detection, so lag can't uncover it.
    private static let minimumCoverage: CGFloat = 0.85
    /// Share of the velocity kept for each frame without a detection (total drift stays within ~1.5 mask sizes).
    private static let missedVelocityDecay: CGFloat = 0.8

    private var tracks: [Track] = []
    private var nextID = 0
    /// Track identifiers of the areas returned by the last update, in the same order.
    private(set) var lastIDs: [Int] = []
    /// Frames a track survives without a matching detection before it's dropped.
    private let maxMissed: Int

    init(maxMissed: Int) {
        self.maxMissed = maxMissed
    }

    /// Forgets every track, e.g. at a cut between shots; new tracks get new identifiers.
    func reset() {
        tracks = []
    }

    /// Feeds one frame's detections; returns the areas to mask in that frame.
    func update(with detections: [Detection]) -> [Detection] {
        // Predict where each tracked object is now.
        for i in tracks.indices {
            tracks[i].detection.rect = tracks[i].detection.rect.offsetBy(dx: tracks[i].velocity.dx, dy: tracks[i].velocity.dy)
            tracks[i].core = tracks[i].core.offsetBy(dx: tracks[i].velocity.dx, dy: tracks[i].velocity.dy)
        }

        // Candidate pairs, compared on detected boxes (not enlarged masks): clear overlap, or, for fast
        // movers, a nearby center when there's no other candidate on either side (never between neighbors).
        var overlapping: [(track: Int, detection: Int, score: CGFloat)] = []
        var nearby: [(track: Int, detection: Int, score: CGFloat, close: Bool)] = []
        for (t, track) in tracks.enumerated() {
            for (d, detection) in detections.enumerated() where detection.isFace == track.detection.isFace {
                let core = detection.core ?? detection.rect
                let iou = Self.iou(track.core, core)
                if iou >= Self.minimumOverlap {
                    overlapping.append((t, d, iou))
                } else {
                    let distance = hypot(track.core.midX - core.midX, track.core.midY - core.midY)
                    let size = max(track.core.width, track.core.height, core.width, core.height)
                    if distance < size { nearby.append((t, d, 0.05 * (1 - distance / size), distance < 0.5 * size)) }
                }
            }
        }
        // Close centers (under half a face) are accepted; farther ones (up to a face) only when there's
        // no other candidate on either side, so a track never reaches over to a neighbor.
        let candidates = overlapping.map { ($0.track, $0.detection) } + nearby.map { ($0.track, $0.detection) }
        let accepted = nearby.filter { pair in
            pair.close || (candidates.filter { $0.0 == pair.track }.count == 1 && candidates.filter { $0.1 == pair.detection }.count == 1)
        }
        let pairs = overlapping + accepted.map { ($0.track, $0.detection, $0.score) }
        var matchedTracks = Set<Int>(), matchedDetections = Set<Int>()
        for pair in pairs.sorted(by: { $0.score > $1.score })
        where !matchedTracks.contains(pair.track) && !matchedDetections.contains(pair.detection) {
            matchedTracks.insert(pair.track)
            matchedDetections.insert(pair.detection)
            update(&tracks[pair.track], with: detections[pair.detection])
        }

        for t in tracks.indices where !matchedTracks.contains(t) {
            // Unseen: keep the size and let the motion fade out, so the mask can't fly off or zoom.
            tracks[t].missed += 1
            tracks[t].detection.landmarks = []  // only actual detections carry landmarks
            tracks[t].velocity = CGVector(dx: tracks[t].velocity.dx * Self.missedVelocityDecay,
                                          dy: tracks[t].velocity.dy * Self.missedVelocityDecay)
        }
        // Objects seen only once or twice (crowds, flickering false detections) are kept briefly; well
        // established ones keep the full memory. Without this, a crowd video drew 4–5 masks per detected face.
        tracks.removeAll { $0.missed > min(maxMissed, 2 + 2 * $0.hits) }
        for (d, detection) in detections.enumerated() where !matchedDetections.contains(d) {
            tracks.append(Track(id: nextID, detection: detection, core: detection.core ?? detection.rect))
            nextID += 1
        }
        let shown = tracks
        lastIDs = shown.map(\.id)
        return shown.map { track in
            var detection = track.detection
            detection.id = track.id
            return detection
        }
    }

    private func update(_ track: inout Track, with detection: Detection) {
        let predicted = track.detection.rect, measured = detection.rect
        // The center follows the detection exactly: smoothing it makes masks lag behind moving faces.
        let center = CGPoint(x: measured.midX, y: measured.midY)
        let width = max(predicted.width + Self.sizeResponse * (measured.width - predicted.width),
                        measured.width * Self.minimumCoverage)
        let height = max(predicted.height + Self.sizeResponse * (measured.height - predicted.height),
                         measured.height * Self.minimumCoverage)
        let rect = CGRect(x: center.x - width / 2, y: center.y - height / 2, width: width, height: height)

        // Velocity from the motion between detections, capped to a plausible speed per frame.
        let previousCenter = CGPoint(x: predicted.midX - track.velocity.dx, y: predicted.midY - track.velocity.dy)
        let limit = 0.3 * max(width, height)
        func clamp(_ v: CGFloat) -> CGFloat { min(max(v, -limit), limit) }
        track.velocity = CGVector(dx: clamp(0.5 * track.velocity.dx + 0.5 * (center.x - previousCenter.x)),
                                  dy: clamp(0.5 * track.velocity.dy + 0.5 * (center.y - previousCenter.y)))
        track.detection = Detection(rect: rect, score: detection.score, isFace: detection.isFace, landmarks: detection.landmarks)
        track.core = detection.core ?? detection.rect
        track.missed = 0
        track.hits += 1
    }

    private static func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let overlap = a.intersection(b)
        guard !overlap.isNull, overlap.width > 0, overlap.height > 0 else { return 0 }
        let intersection = overlap.width * overlap.height
        return intersection / (a.width * a.height + b.width * b.height - intersection)
    }
}
