import CoreGraphics

/// Follows plates and faces across video frames so each object keeps exactly one steady mask, and that
/// mask keeps following the object for a few frames when the detector misses it.
final class Tracker {
    private struct Track {
        let id: Int
        var detection: Detection
        var velocity: CGVector = .zero
        var missed = 0
    }

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

    /// Feeds one frame's detections; returns the areas to mask in that frame.
    func update(with detections: [Detection]) -> [Detection] {
        // Predict where each tracked object is now.
        for i in tracks.indices {
            tracks[i].detection.rect = tracks[i].detection.rect.offsetBy(dx: tracks[i].velocity.dx, dy: tracks[i].velocity.dy)
        }

        // Greedy matching, best pairs first: overlap, or nearby centers for fast movers.
        var pairs: [(track: Int, detection: Int, score: CGFloat)] = []
        for (t, track) in tracks.enumerated() {
            for (d, detection) in detections.enumerated() where detection.isFace == track.detection.isFace {
                let score = Self.matchScore(track.detection.rect, detection.rect)
                if score > 0 { pairs.append((t, d, score)) }
            }
        }
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
            tracks[t].velocity = CGVector(dx: tracks[t].velocity.dx * Self.missedVelocityDecay,
                                          dy: tracks[t].velocity.dy * Self.missedVelocityDecay)
        }
        tracks.removeAll { $0.missed > maxMissed }
        for (d, detection) in detections.enumerated() where !matchedDetections.contains(d) {
            tracks.append(Track(id: nextID, detection: detection))
            nextID += 1
        }
        lastIDs = tracks.map(\.id)
        return tracks.map(\.detection)
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
        track.detection = Detection(rect: rect, score: detection.score, isFace: detection.isFace)
        track.missed = 0
    }

    /// IoU when the boxes overlap; otherwise a small score if the centers are close (fast motion).
    private static func matchScore(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let overlap = a.intersection(b)
        if !overlap.isNull, overlap.width > 0, overlap.height > 0 {
            let intersection = overlap.width * overlap.height
            let iou = intersection / (a.width * a.height + b.width * b.height - intersection)
            if iou >= 0.1 { return iou }
        }
        let distance = hypot(a.midX - b.midX, a.midY - b.midY)
        let reach = 0.75 * max(a.width, a.height, b.width, b.height)
        return distance < reach ? 0.05 * (1 - distance / reach) : 0
    }
}
