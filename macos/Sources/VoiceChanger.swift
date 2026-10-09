@preconcurrency import AVFoundation

/// How voices are disguised in exported videos.
enum VoiceEffect: String, CaseIterable, Identifiable, Codable {
    case off, lower, higher, robot, whisper, synthetic

    var id: String { rawValue }

    var label: String {
        switch self {
        case .off: String(localized: "Unchanged")
        case .lower: String(localized: "Lower")
        case .higher: String(localized: "Higher")
        case .robot: String(localized: "Robot")
        case .whisper: String(localized: "Whisper (irreversible)")
        case .synthetic: String(localized: "Synthetic voice (irreversible)")
        }
    }

    /// The original voice can't be recovered: its pitch and timbre are discarded, not transformed.
    var isIrreversible: Bool { self == .whisper || self == .synthetic }
}

/// Disguises voices in a video's sound track offline with AVAudioEngine effects, writing the result
/// to a temporary file that export then uses instead of the original sound.
enum VoiceChanger {
    /// Pitch shift in cents: 500 cents = 5 semitones, enough to change how a voice sounds while
    /// keeping speech clear.
    private static let shift: Float = 500

    /// Returns a temporary .caf file with the processed sound; the caller deletes it.
    static func process(_ track: AVAssetTrack, of asset: AVAsset, effect: VoiceEffect) async throws -> URL {
        let format = try await pcmFormat(of: track)
        let source = FileManager.default.temporaryDirectory.appendingPathComponent("MultiBlur-voice-in-\(UUID().uuidString).caf")
        let result = FileManager.default.temporaryDirectory.appendingPathComponent("MultiBlur-voice-\(UUID().uuidString).caf")
        defer { try? FileManager.default.removeItem(at: source) }

        try decode(track, of: asset, format: format, to: source)
        do {
            try render(source, effect: effect, format: format, to: result)
        } catch {
            try? FileManager.default.removeItem(at: result)
            throw error
        }
        return result
    }

    /// Float, non-interleaved PCM at the track's rate (44.1 or 48 kHz), mono or stereo.
    private static func pcmFormat(of track: AVAssetTrack) async throws -> AVAudioFormat {
        var sampleRate = 44_100.0, channels: AVAudioChannelCount = 2
        if let description = try await track.load(.formatDescriptions).first,
           let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(description)?.pointee {
            sampleRate = asbd.mSampleRate == 48_000 ? 48_000 : 44_100
            channels = asbd.mChannelsPerFrame == 1 ? 1 : 2
        }
        guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate,
                                         channels: channels, interleaved: false) else {
            throw EngineError.writeFailed("unsupported audio format")
        }
        return format
    }

    /// Decodes the sound track to a PCM file.
    private static func decode(_ track: AVAssetTrack, of asset: AVAsset, format: AVAudioFormat, to url: URL) throws {
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: format.sampleRate,
            AVNumberOfChannelsKey: format.channelCount,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsNonInterleaved: true,
            AVLinearPCMIsBigEndianKey: false,
        ])
        reader.add(output)
        guard reader.startReading() else { throw reader.error ?? EngineError.writeFailed("could not read the sound") }

        let file = try AVAudioFile(forWriting: url, settings: format.settings, commonFormat: .pcmFormatFloat32, interleaved: false)
        while let sample = output.copyNextSampleBuffer() {
            let frames = AVAudioFrameCount(CMSampleBufferGetNumSamples(sample))
            guard frames > 0, let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else { continue }
            buffer.frameLength = frames
            let status = CMSampleBufferCopyPCMDataIntoAudioBufferList(sample, at: 0, frameCount: Int32(frames),
                                                                       into: buffer.mutableAudioBufferList)
            guard status == noErr else { throw EngineError.writeFailed("could not decode the sound") }
            try file.write(from: buffer)
        }
        if reader.status == .failed { throw reader.error ?? EngineError.writeFailed("could not read the sound") }
    }

    /// Plays the PCM file through the effect chain in offline (faster than real time) rendering.
    private static func render(_ source: URL, effect: VoiceEffect, format: AVAudioFormat, to url: URL) throws {
        let input = try AVAudioFile(forReading: source, commonFormat: .pcmFormatFloat32, interleaved: false)
        if effect.isIrreversible {
            try Vocoder(source: effect == .whisper ? .whisper : .synthetic, sampleRate: format.sampleRate)
                .process(input, to: url)
            return
        }
        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        engine.attach(player)

        var chain: [AVAudioNode] = []
        switch effect {
        case .off, .whisper, .synthetic:
            break
        case .lower, .higher:
            let pitch = AVAudioUnitTimePitch()
            pitch.pitch = effect == .lower ? -shift : shift
            chain.append(pitch)
        case .robot:
            // A lowered pitch, then a ring-modulated "radio" distortion: metallic and hard to recognize.
            let pitch = AVAudioUnitTimePitch()
            pitch.pitch = -300
            let distortion = AVAudioUnitDistortion()
            distortion.loadFactoryPreset(.speechRadioTower)
            distortion.wetDryMix = 100
            chain += [pitch, distortion]
        }
        chain.forEach(engine.attach)
        var previous: AVAudioNode = player
        for node in chain {
            engine.connect(previous, to: node, format: format)
            previous = node
        }
        engine.connect(previous, to: engine.mainMixerNode, format: format)

        try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: 4096)
        try engine.start()
        defer { engine.stop() }
        player.scheduleFile(input, at: nil)
        player.play()

        let output = try AVAudioFile(forWriting: url, settings: format.settings, commonFormat: .pcmFormatFloat32, interleaved: false)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: engine.manualRenderingFormat, frameCapacity: engine.manualRenderingMaximumFrameCount) else {
            throw EngineError.writeFailed("could not process the sound")
        }
        // Same length as the original, so sound and picture stay in sync.
        while engine.manualRenderingSampleTime < input.length {
            let frames = min(buffer.frameCapacity, AVAudioFrameCount(input.length - engine.manualRenderingSampleTime))
            switch try engine.renderOffline(frames, to: buffer) {
            case .success:
                try output.write(from: buffer)
            case .error:
                throw EngineError.writeFailed("could not process the sound")
            default:
                continue
            }
        }
    }
}
