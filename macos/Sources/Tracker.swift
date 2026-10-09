import CoreGraphics

/// Follows plates and faces across video frames so each object keeps exactly one mask, and that mask
/// keeps moving along the object's path for a few frames when the detector misses it.
final class Tracker {
    private struct Track {
        var detection: Detection
        var velocity: CGVector = .zero
        var missed = 0
    }

    private var tracks: [Track] = []
    /// Frames a track survives without a matching detection before it's dropped.
    private let maxMissed: Int

    init(maxMissed: Int) {
        self.maxMissed = maxMissed
    }

    /// Feeds one frame's detections; returns the areas to mask in that frame.
    func update(with detections: [Detection]) -> [Detection] {
        let previousCenters = tracks.map { CGPoint(x: $0.detection.rect.midX, y: $0.detection.rect.midY) }
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
            let detection = detections[pair.detection]
            let moved = CGVector(dx: detection.rect.midX - previousCenters[pair.track].x,
                                 dy: detection.rect.midY - previousCenters[pair.track].y)
            let velocity = tracks[pair.track].velocity
            tracks[pair.track].velocity = CGVector(dx: 0.5 * velocity.dx + 0.5 * moved.dx, dy: 0.5 * velocity.dy + 0.5 * moved.dy)
            tracks[pair.track].detection = detection
            tracks[pair.track].missed = 0
        }

        for t in tracks.indices where !matchedTracks.contains(t) {
            tracks[t].missed += 1
            // Grow a little while unseen to cover the uncertainty of the prediction.
            let rect = tracks[t].detection.rect
            tracks[t].detection.rect = rect.insetBy(dx: -rect.width * 0.03, dy: -rect.height * 0.03)
        }
        tracks.removeAll { $0.missed > maxMissed }
        for (d, detection) in detections.enumerated() where !matchedDetections.contains(d) {
            tracks.append(Track(detection: detection))
        }
        return tracks.map(\.detection)
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
