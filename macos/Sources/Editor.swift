@preconcurrency import AVFoundation
import CoreImage
import SwiftUI

/// State of the editor window for one file: its analysis, which tracks stay visible, the boxes
/// added by hand, and a preview of the current frame as it will be exported.
@MainActor
final class EditorModel: ObservableObject {
    enum Phase: Equatable {
        case analyzing(Double)
        case ready
        case tracking(Double)
        case failed(String)
    }

    let job: Job
    private let options: Options
    private let engine: BlurEngine
    private let previewContext = CIContext()

    @Published private(set) var phase: Phase = .analyzing(0)
    @Published private(set) var analysis: Analysis?
    @Published private(set) var excludedGroups: Set<Int> = []
    @Published private(set) var manual: [ManualMask] = []
    /// Voice effect for this video's export (the editor's "Disguised voice" box); playback uses it too.
    @Published var voice: VoiceEffect = .off {
        didSet { if voice != oldValue { updateVoiceAudio() } }
    }
    /// The disguised sound is being prepared for playback.
    @Published private(set) var isPreparingVoice = false
    /// Processed sound per effect, so switching back is instant; deleted when the editor closes.
    private var voiceFiles: [VoiceEffect: URL] = [:]
    private var voiceTask: Task<Void, Never>?
    @Published private(set) var preview: CGImage?
    @Published var frameIndex = 0 {
        didSet { if !isPlaying { renderPreview() } }
    }
    @Published private(set) var isPlaying = false

    private var image: CIImage?
    private var generator: AVAssetImageGenerator?
    private var replacement: CIImage?
    private var renderTask: Task<Void, Never>?
    private var player: AVPlayer?
    private var playerOutput: AVPlayerItemVideoOutput?
    private var playTask: Task<Void, Never>?
    private var orientation: CGImagePropertyOrientation = .up
    private var nextManualID = 1

    /// Called with a new analysis made by the editor, so the file keeps it for next time.
    private let onAnalyzed: (Analysis, Options) -> Void

    init(job: Job, options: Options, engine: BlurEngine, onAnalyzed: @escaping (Analysis, Options) -> Void = { _, _ in }) {
        self.job = job
        self.options = options
        self.engine = engine
        self.onAnalyzed = onAnalyzed
    }

    /// The editor's changes, kept with the file on export.
    var edits: Edits { Edits(excludedGroups: excludedGroups, manual: manual, voice: voice) }

    // MARK: Loading

    func load() async {
        do {
            replacement = try? engine.replacementImage(options)
            voice = job.edits?.voice ?? options.voice
            let analysis: Analysis
            if let saved = job.analysis, let savedOptions = job.analysisOptions, savedOptions.sameDetection(as: options) {
                // Reuse the analysis made during export, with the previous edits.
                analysis = saved
                excludedGroups = job.edits?.excludedGroups ?? []
                manual = job.edits?.manual ?? []
                nextManualID = (manual.map(\.id).max() ?? 0) + 1
            } else {
                // Detection settings changed since the export (or it was never analyzed): analyze again.
                analysis = try await Task.detached(priority: .userInitiated) { [engine, job, options] in
                    try await engine.analyze(job.source, options: options) { fraction in
                        // Late progress updates must not bring back the progress view once loaded.
                        Task { @MainActor in if case .analyzing = self.phase { self.phase = .analyzing(fraction) } }
                    }
                }.value
                onAnalyzed(analysis, options)
            }
            if analysis.isVideo {
                let generator = AVAssetImageGenerator(asset: AVURLAsset(url: job.source))
                generator.appliesPreferredTrackTransform = true
                generator.requestedTimeToleranceBefore = .zero
                generator.requestedTimeToleranceAfter = .zero
                self.generator = generator
                // Playback: AVPlayer decodes in real time (with sound); frames are masked on the fly.
                let asset = AVURLAsset(url: job.source)
                if let track = try? await asset.loadTracks(withMediaType: .video).first,
                   let transform = try? await track.load(.preferredTransform) {
                    orientation = BlurEngine.orientation(for: transform)
                }
                await setPlayerItem(audio: nil)
            } else if let image = CIImage(contentsOf: job.source, options: [.applyOrientationProperty: true]) {
                self.image = image.transformed(by: .init(translationX: -image.extent.minX, y: -image.extent.minY))
            }
            self.analysis = analysis
            phase = .ready
            renderPreview()
            if voice != .off { updateVoiceAudio() }
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    // MARK: Editing

    private var excludedTracks: Set<Int> {
        Set((analysis?.groups ?? []).filter { excludedGroups.contains($0.id) }.flatMap(\.trackIDs))
    }

    func group(of trackID: Int) -> TrackGroup? {
        analysis?.groups.first { $0.trackIDs.contains(trackID) }
    }

    func isExcluded(_ group: TrackGroup) -> Bool { excludedGroups.contains(group.id) }

    func setExcluded(_ group: TrackGroup, _ excluded: Bool) {
        if excluded { excludedGroups.insert(group.id) } else { excludedGroups.remove(group.id) }
        renderPreview()
    }

    /// Detected areas in the current frame, with whether each one is masked.
    var detectionsInFrame: [(detection: Detection, masked: Bool)] {
        guard let analysis, frameIndex < analysis.frames.count else { return [] }
        let excluded = excludedTracks
        return analysis.frames[frameIndex].map { ($0, !excluded.contains($0.id)) }
    }

    var manualInFrame: [(id: Int, rect: CGRect)] {
        manual.compactMap { mask in mask.areas[frameIndex].map { (mask.id, $0) } }
    }

    /// Adds a hand-drawn box at the current frame (Core Image pixel coordinates) and, in videos,
    /// follows it through the next frames.
    func addBox(_ rect: CGRect) {
        guard let analysis, rect.width >= 4, rect.height >= 4 else { return }
        let id = nextManualID
        nextManualID += 1
        manual.append(ManualMask(id: id, areas: [frameIndex: rect]))
        renderPreview()
        guard analysis.isVideo else { return }

        phase = .tracking(0)
        let start = frameIndex
        Task {
            do {
                let areas = try await Task.detached(priority: .userInitiated) { [engine, job] in
                    try await engine.trackBox(rect, from: start, in: job.source, analysis: analysis) { fraction in
                        Task { @MainActor in if case .tracking = self.phase { self.phase = .tracking(fraction) } }
                    }
                }.value
                if let index = manual.firstIndex(where: { $0.id == id }) { manual[index].areas = areas }
                phase = .ready
                renderPreview()
            } catch {
                phase = .ready
            }
        }
    }

    func removeManual(_ id: Int) {
        manual.removeAll { $0.id == id }
        renderPreview()
    }

    func manualFrames(_ mask: ManualMask) -> ClosedRange<Int> {
        (mask.areas.keys.min() ?? 0)...(mask.areas.keys.max() ?? 0)
    }

    /// What export will mask, frame by frame.
    func plan() -> MaskPlan? {
        guard let analysis else { return nil }
        let excluded = excludedTracks
        let frames = analysis.frames.enumerated().map { index, detections in
            detections.filter { !excluded.contains($0.id) }
                + manual.compactMap { mask in
                    // Negative identifiers keep hand-drawn boxes apart from detected tracks.
                    mask.areas[index].map { Detection(rect: $0, score: 1, isFace: false, id: -mask.id) }
                }
        }
        return MaskPlan(frames: frames)
    }

    // MARK: Preview

    // MARK: Disguised voice in playback

    /// Prepares the sound with the chosen voice effect (same processing as export) and plays it with the video.
    private func updateVoiceAudio() {
        guard generator != nil else { return }  // videos only, once loaded
        pause()
        voiceTask?.cancel()
        let effect = voice
        voiceTask = Task {
            var audio: URL?
            if effect != .off {
                if let cached = voiceFiles[effect] {
                    audio = cached
                } else {
                    isPreparingVoice = true
                    defer { isPreparingVoice = false }
                    let source = AVURLAsset(url: job.source)
                    guard let track = try? await source.loadTracks(withMediaType: .audio).first,
                          let file = try? await Task.detached(priority: .userInitiated, operation: {
                              try await VoiceChanger.process(track, of: source, effect: effect)
                          }).value else { return }
                    guard !Task.isCancelled else {
                        try? FileManager.default.removeItem(at: file)
                        return
                    }
                    voiceFiles[effect] = file
                    audio = file
                }
            }
            await setPlayerItem(audio: audio)
        }
    }

    /// Plays the original video with either its own sound or a processed sound file.
    private func setPlayerItem(audio: URL?) async {
        let source = AVURLAsset(url: job.source)
        var asset: AVAsset = source
        // The processed sound's asset must stay alive while its track is inserted: a track doesn't
        // retain its asset, and inserting from a released one fails, leaving playback silent.
        let soundAsset = audio.map { AVURLAsset(url: $0) }
        if let soundAsset,
           let video = try? await source.loadTracks(withMediaType: .video).first,
           let sound = try? await soundAsset.loadTracks(withMediaType: .audio).first,
           let videoRange = try? await video.load(.timeRange),
           let soundRange = try? await sound.load(.timeRange) {
            let composition = AVMutableComposition()
            if let videoTrack = composition.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid),
               let soundTrack = composition.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid) {
                do {
                    try videoTrack.insertTimeRange(videoRange, of: video, at: videoRange.start)
                    videoTrack.preferredTransform = (try? await video.load(.preferredTransform)) ?? .identity
                    let length = CMTimeMinimum(soundRange.duration, videoRange.end)
                    try soundTrack.insertTimeRange(CMTimeRange(start: soundRange.start, duration: length), of: sound, at: .zero)
                    asset = composition
                } catch {
                    // Keep the original sound rather than playing silently.
                }
            }
        }
        let item = AVPlayerItem(asset: asset)
        let output = AVPlayerItemVideoOutput(pixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ])
        item.add(output)
        playerOutput = output
        if let player { player.replaceCurrentItem(with: item) } else { player = AVPlayer(playerItem: item) }
    }

    /// Deletes the processed sound files (when the editor closes).
    func cleanUp() {
        pause()
        voiceTask?.cancel()
        voiceFiles.values.forEach { try? FileManager.default.removeItem(at: $0) }
        voiceFiles = [:]
    }

    // MARK: Playback

    func togglePlayback() {
        isPlaying ? pause() : play()
    }

    func play() {
        guard let analysis, analysis.isVideo, let player, let playerOutput else { return }
        if frameIndex >= analysis.frameTimes.count - 1 { frameIndex = 0 }
        renderTask?.cancel()
        isPlaying = true
        let start = analysis.frameTimes[frameIndex]
        playTask = Task {
            await player.seek(to: start, toleranceBefore: .zero, toleranceAfter: .zero)
            player.play()
            // Poll the player's newest frame (about 60 times per second) and mask it like export would.
            while !Task.isCancelled, isPlaying {
                let time = playerOutput.itemTime(forHostTime: CACurrentMediaTime())
                if playerOutput.hasNewPixelBuffer(forItemTime: time),
                   let pixels = playerOutput.copyPixelBuffer(forItemTime: time, itemTimeForDisplay: nil) {
                    let index = frameIndex(at: time, in: analysis)
                    var frame = CIImage(cvPixelBuffer: pixels).oriented(orientation)
                    frame = frame.transformed(by: .init(translationX: -frame.extent.minX, y: -frame.extent.minY))
                    if let plan = plan(), index < plan.frames.count {
                        let masked = engine.mask(frame, detections: plan.frames[index], options: options, replacement: replacement)
                        preview = previewContext.createCGImage(masked, from: frame.extent)
                    }
                    frameIndex = index
                }
                if player.rate == 0, let item = player.currentItem, item.currentTime() >= item.duration { break }
                try? await Task.sleep(nanoseconds: 16_000_000)
            }
            pause()
        }
    }

    func pause() {
        guard isPlaying else { return }
        player?.pause()
        playTask?.cancel()
        isPlaying = false
        renderPreview()  // the exact frame, as it will be exported
    }

    private func frameIndex(at time: CMTime, in analysis: Analysis) -> Int {
        // Last frame shown at or before `time` (frame times are sorted).
        var low = 0, high = analysis.frameTimes.count - 1
        while low < high {
            let middle = (low + high + 1) / 2
            if analysis.frameTimes[middle] <= time { low = middle } else { high = middle - 1 }
        }
        return low
    }

    private func renderPreview() {
        guard let analysis, let plan = plan(), frameIndex < plan.frames.count else { return }
        let index = frameIndex
        let masks = plan.frames[index]
        renderTask?.cancel()
        renderTask = Task {
            let base: CIImage?
            if let generator {
                base = (try? await generator.image(at: analysis.frameTimes[index])).map { CIImage(cgImage: $0.image) }
            } else {
                base = image
            }
            guard !Task.isCancelled, let base else { return }
            let masked = engine.mask(base, detections: masks, options: options, replacement: replacement)
            let rendered = previewContext.createCGImage(masked, from: base.extent)
            guard !Task.isCancelled else { return }
            preview = rendered
        }
    }
}
