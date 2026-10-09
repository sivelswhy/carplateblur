import CoreML
import Foundation

/// How detected areas are anonymized (deface's `--replacewith`).
enum MaskMode: String, CaseIterable, Identifiable, Codable {
    case blur, mosaic, solid, image, none

    var id: String { rawValue }

    var label: String {
        switch self {
        case .blur: String(localized: "Blur")
        case .mosaic: String(localized: "Mosaic")
        case .solid: String(localized: "Black box")
        case .image: String(localized: "Image")
        case .none: String(localized: "None")
        }
    }
}

/// Face mask shape (deface's `--boxes` switches from ellipse to box).
enum MaskShape: String, CaseIterable, Identifiable, Codable {
    case ellipse, box

    var id: String { rawValue }
    var label: String { self == .ellipse ? String(localized: "Ellipse") : String(localized: "Box") }
}

/// Resolution used for face detection (deface's `--scale`); `full` analyzes frames at native size.
enum DetectionResolution: Int, CaseIterable, Identifiable, Codable {
    case full = 0, p1920 = 1920, p1280 = 1280, p640 = 640

    var id: Int { rawValue }
    var label: String { self == .full ? String(localized: "Full resolution") : "\(rawValue) px" }
    /// The CoreML face model accepts up to 2048 px per side.
    var maxSide: CGFloat { self == .full ? 2048 : CGFloat(rawValue) }
}

/// Extra detection passes on overlapping tiles, to find small or distant plates and faces.
enum SmallObjectSearch: String, CaseIterable, Identifiable, Codable {
    case off, photos, all

    var id: String { rawValue }

    var label: String {
        switch self {
        case .off: String(localized: "Off")
        case .photos: String(localized: "Photos")
        case .all: String(localized: "Photos & videos")
        }
    }
}

/// Output video codec (stands in for deface's `--ffmpeg-config`).
enum VideoCodec: String, CaseIterable, Identifiable, Codable {
    case h264, hevc

    var id: String { rawValue }
    var label: String { self == .h264 ? "H.264" : "HEVC (H.265)" }
}

/// Where models run (stands in for deface's `--backend` / `--execution-provider`).
enum ComputeMode: String, CaseIterable, Identifiable, Codable {
    case all, gpu, cpu

    var id: String { rawValue }

    var label: String {
        switch self {
        case .all: String(localized: "Neural Engine (fastest)")
        case .gpu: String(localized: "GPU")
        case .cpu: String(localized: "CPU only")
        }
    }

    var computeUnits: MLComputeUnits {
        switch self {
        case .all: .all
        case .gpu: .cpuAndGPU
        case .cpu: .cpuOnly
        }
    }
}

enum CompletionSound {
    /// The built-in macOS alert sounds (Glass, Ping, Hero…), as listed in /System/Library/Sounds.
    static let available: [String] = {
        let files = (try? FileManager.default.contentsOfDirectory(atPath: "/System/Library/Sounds")) ?? []
        return files.filter { $0.hasSuffix(".aiff") }.map { String($0.dropLast(5)) }.sorted()
    }()
}

/// Every user-facing setting. Defaults mirror deface's where an equivalent exists.
/// Persisted as JSON in UserDefaults.
struct Options: Codable, Equatable {
    // What to detect
    var maskPlates = true
    var maskFaces = true

    // Anonymization (deface: --replacewith, --replaceimg, --mosaicsize, --boxes, --draw-scores)
    var mode: MaskMode = .blur
    var replacementImagePath: String?
    var mosaicSize: Double = 20
    var faceShape: MaskShape = .ellipse
    var drawScores = false

    // Detection (deface: --thresh, --mask-scale, --scale)
    var plateConfidence: Double = 0.05
    var faceThreshold: Double = 0.2
    var faceMaskScale: Double = 1.3
    var plateMaskScale: Double = 1.15
    var faceResolution: DetectionResolution = .full
    var smallObjects: SmallObjectSearch = .photos

    // Output (deface: --output, --keep-audio, --keep-metadata, --ffmpeg-config)
    var outputFolderPath: String?
    var keepAudio = true
    var keepMetadata = false
    /// macOS system sound played when each export finishes; nil = silent.
    var completionSound: String?
    var videoCodec: VideoCodec = .h264

    // Performance & display (deface: --backend / --execution-provider, --preview)
    var compute: ComputeMode = .all
    var livePreview = false

    /// Whether `other` finds and sizes the same masks, so an analysis made with it can be reused.
    func sameDetection(as other: Options) -> Bool {
        maskPlates == other.maskPlates && maskFaces == other.maskFaces
            && plateConfidence == other.plateConfidence && faceThreshold == other.faceThreshold
            && plateMaskScale == other.plateMaskScale && faceMaskScale == other.faceMaskScale
            && faceResolution == other.faceResolution && smallObjects == other.smallObjects
    }

    private static let key = "options"

    /// Settings saved before the app was renamed from PlateBlur to MultiBlur.
    private static let legacyDomain = "com.carplateblur.PlateBlur"

    static func load() -> Options {
        let saved = UserDefaults.standard.data(forKey: key)
            ?? UserDefaults(suiteName: legacyDomain)?.data(forKey: key)
        // Overlay saved values on the defaults, so settings added in newer versions don't
        // invalidate everything that was saved before.
        guard let data = saved,
              let defaults = try? JSONSerialization.jsonObject(with: JSONEncoder().encode(Options())) as? [String: Any],
              let stored = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let merged = try? JSONSerialization.data(withJSONObject: defaults.merging(stored) { _, saved in saved }),
              let options = try? JSONDecoder().decode(Options.self, from: merged) else { return Options() }
        return options
    }

    func save() {
        if let data = try? JSONEncoder().encode(self) { UserDefaults.standard.set(data, forKey: Self.key) }
    }
}
