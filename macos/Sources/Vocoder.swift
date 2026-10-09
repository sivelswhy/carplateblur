import Accelerate
import AVFoundation

/// Irreversible voice anonymization: a channel vocoder.
///
/// Each short frame of sound is reduced to its spectral envelope over a few dozen bands (the vocal tract
/// shape that carries vowels and consonants, so speech stays understandable). Everything that identifies
/// a voice beyond that is discarded: pitch and intonation, harmonics, fine spectral detail, phase. The
/// sound is then rebuilt from a synthetic source (random noise, i.e. whispering, or a fixed-pitch buzz),
/// with formants shifted by a random factor that is never stored. Because the information is thrown
/// away rather than transformed, there is nothing to invert.
struct Vocoder {
    enum Source {
        /// Random noise: a whisper, with no pitch at all.
        case whisper
        /// A fixed-pitch pulse train: a flat, synthetic voice.
        case synthetic
    }

    private static let frameSize = 1024
    private static let hop = 256
    private static let bandCount = 28

    private let source: Source
    private let sampleRate: Double
    /// Formant shift, drawn at random for each export.
    private let warp: Float

    init(source: Source, sampleRate: Double) {
        self.source = source
        self.sampleRate = sampleRate
        // Shift formants up or down by 12–22%, at random: a different "vocal tract" on every export.
        let amount = Float.random(in: 1.12...1.22)
        warp = Bool.random() ? amount : 1 / amount
    }

    /// Reads `input` (any channel count), writes mono vocoded sound of the same length to `output`.
    func process(_ input: AVAudioFile, to output: URL) throws {
        let n = Self.frameSize, hop = Self.hop
        guard let monoFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false),
              let readBuffer = AVAudioPCMBuffer(pcmFormat: input.processingFormat, frameCapacity: AVAudioFrameCount(hop)),
              let writeBuffer = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: AVAudioFrameCount(hop)) else {
            throw EngineError.writeFailed("could not process the sound")
        }
        let out = try AVAudioFile(forWriting: output, settings: monoFormat.settings, commonFormat: .pcmFormatFloat32, interleaved: false)

        let log2n = vDSP_Length(log2(Double(n)))
        guard let fft = vDSP.FFT(log2n: log2n, radix: .radix2, ofType: DSPSplitComplex.self) else {
            throw EngineError.writeFailed("could not process the sound")
        }
        let window = vDSP.window(ofType: Float.self, usingSequence: .hanningDenormalized, count: n, isHalfWindow: false)
        let bands = Self.melBands(count: Self.bandCount, fftSize: n, sampleRate: Float(sampleRate))
        var carrier = Carrier(source: source, sampleRate: sampleRate)

        // Streaming STFT: `frame` holds the current analysis window, `overlap` accumulates the output.
        // The first frameSize − hop output samples belong to the zero padding before the sound and are skipped.
        var frame = [Float](repeating: 0, count: n)
        var overlap = [Float](repeating: 0, count: n)
        var toSkip = n - hop
        var remaining = Int(input.length)
        var inputLeft = Int(input.length)

        while remaining > 0 {
            // Next hop of input (mono mix), or silence after the end.
            var chunk = [Float](repeating: 0, count: hop)
            if inputLeft > 0 {
                readBuffer.frameLength = 0
                try input.read(into: readBuffer, frameCount: AVAudioFrameCount(min(hop, inputLeft)))
                let count = Int(readBuffer.frameLength)
                let channels = Int(readBuffer.format.channelCount)
                if let data = readBuffer.floatChannelData {
                    for c in 0..<channels {
                        for i in 0..<count { chunk[i] += data[c][i] / Float(channels) }
                    }
                }
                inputLeft -= max(count, 1)
            }
            frame.removeFirst(hop)
            frame += chunk

            let shaped = processFrame(frame, window: window, fft: fft, bands: bands, carrier: &carrier)
            vDSP.add(overlap, shaped, result: &overlap)

            // The first hop of the accumulator is final: write it (after the padding), then shift.
            let ready = Array(overlap[0..<hop])
            overlap.removeFirst(hop)
            overlap += [Float](repeating: 0, count: hop)
            if toSkip > 0 {
                toSkip -= hop
                continue
            }
            let count = min(hop, remaining)
            writeBuffer.frameLength = AVAudioFrameCount(count)
            for i in 0..<count { writeBuffer.floatChannelData![0][i] = ready[i] }
            try out.write(from: writeBuffer)
            remaining -= count
        }
    }

    /// One STFT frame: envelope of the speech × flattened synthetic source, at the speech frame's energy.
    private func processFrame(_ input: [Float], window: [Float], fft: vDSP.FFT<DSPSplitComplex>,
                              bands: [Band], carrier: inout Carrier) -> [Float] {
        let n = input.count, half = n / 2
        let speech = vDSP.multiply(input, window)
        let speechEnergy = vDSP.sumOfSquares(speech)
        guard speechEnergy > 1e-10 else { return [Float](repeating: 0, count: n) }

        let speechPower = Self.powerSpectrum(speech, fft: fft)
        let carrierFrame = vDSP.multiply(carrier.next(n), window)
        var (carrierReal, carrierImaginary) = Self.spectrum(carrierFrame, fft: fft)
        let carrierPower: [Float] = vDSP.add(vDSP.square(carrierReal), vDSP.square(carrierImaginary))

        // Band envelopes (amplitude per band) of the speech and of the source.
        let speechBands = bands.map { $0.amplitude(of: speechPower) }
        let carrierBands = bands.map { $0.amplitude(of: carrierPower) }

        // Each bin gets the speech envelope at its warped frequency, divided by the source's own envelope.
        let binWidth = Float(sampleRate) / Float(n)
        for k in 0..<half {
            let frequency = Float(k) * binWidth
            let target = Self.interpolate(speechBands, bands: bands, at: frequency / warp)
            let flat = max(Self.interpolate(carrierBands, bands: bands, at: frequency), 1e-6)
            carrierReal[k] *= target / flat
            carrierImaginary[k] *= target / flat
        }
        carrierImaginary[0] = 0  // packed Nyquist bin: dropped

        var output = Self.inverse(carrierReal, carrierImaginary, fft: fft, count: n)
        vDSP.multiply(output, window, result: &output)
        // Same loudness as the original frame; Hann² at 75% overlap sums to 1.5.
        let energy = vDSP.sumOfSquares(output)
        let gain = energy > 1e-12 ? sqrt(speechEnergy / energy) / 1.5 : 0
        return vDSP.multiply(gain, output)
    }

    // MARK: Spectra

    private static func spectrum(_ signal: [Float], fft: vDSP.FFT<DSPSplitComplex>) -> ([Float], [Float]) {
        let half = signal.count / 2
        var inReal = [Float](repeating: 0, count: half), inImaginary = [Float](repeating: 0, count: half)
        var outReal = [Float](repeating: 0, count: half), outImaginary = [Float](repeating: 0, count: half)
        inReal.withUnsafeMutableBufferPointer { ir in
            inImaginary.withUnsafeMutableBufferPointer { ii in
                outReal.withUnsafeMutableBufferPointer { or in
                    outImaginary.withUnsafeMutableBufferPointer { oi in
                        var input = DSPSplitComplex(realp: ir.baseAddress!, imagp: ii.baseAddress!)
                        var output = DSPSplitComplex(realp: or.baseAddress!, imagp: oi.baseAddress!)
                        signal.withUnsafeBytes { raw in
                            vDSP.convert(interleavedComplexVector: [DSPComplex](raw.bindMemory(to: DSPComplex.self)),
                                         toSplitComplexVector: &input)
                        }
                        fft.forward(input: input, output: &output)
                    }
                }
            }
        }
        return (outReal, outImaginary)
    }

    private static func powerSpectrum(_ signal: [Float], fft: vDSP.FFT<DSPSplitComplex>) -> [Float] {
        let (real, imaginary) = spectrum(signal, fft: fft)
        var power: [Float] = vDSP.add(vDSP.square(real), vDSP.square(imaginary))
        power[0] = real[0] * real[0]  // the packed Nyquist value isn't part of the DC bin
        return power
    }

    private static func inverse(_ real: [Float], _ imaginary: [Float], fft: vDSP.FFT<DSPSplitComplex>, count: Int) -> [Float] {
        let half = count / 2
        var inReal = real, inImaginary = imaginary
        var outReal = [Float](repeating: 0, count: half), outImaginary = [Float](repeating: 0, count: half)
        var result = [Float](repeating: 0, count: count)
        inReal.withUnsafeMutableBufferPointer { ir in
            inImaginary.withUnsafeMutableBufferPointer { ii in
                outReal.withUnsafeMutableBufferPointer { or in
                    outImaginary.withUnsafeMutableBufferPointer { oi in
                        let input = DSPSplitComplex(realp: ir.baseAddress!, imagp: ii.baseAddress!)
                        var output = DSPSplitComplex(realp: or.baseAddress!, imagp: oi.baseAddress!)
                        fft.inverse(input: input, output: &output)
                        var interleaved = [DSPComplex](repeating: DSPComplex(), count: half)
                        vDSP.convert(splitComplexVector: output, toInterleavedComplexVector: &interleaved)
                        for i in 0..<half {
                            result[2 * i] = interleaved[i].real
                            result[2 * i + 1] = interleaved[i].imag
                        }
                    }
                }
            }
        }
        return result  // scale doesn't matter: each frame is normalized to the speech frame's energy
    }

    // MARK: Bands

    /// A triangular mel-scale band over FFT bins.
    struct Band {
        let center: Float
        let bins: [(index: Int, weight: Float)]

        func amplitude(of power: [Float]) -> Float {
            var sum: Float = 0, weights: Float = 0
            for (index, weight) in bins where index < power.count {
                sum += power[index] * weight
                weights += weight
            }
            return weights > 0 ? sqrt(sum / weights) : 0
        }
    }

    private static func melBands(count: Int, fftSize: Int, sampleRate: Float) -> [Band] {
        func mel(_ f: Float) -> Float { 2595 * log10(1 + f / 700) }
        func hertz(_ m: Float) -> Float { 700 * (pow(10, m / 2595) - 1) }
        let low = mel(80), high = mel(min(7600, sampleRate / 2 - 100))
        let edges = (0...(count + 1)).map { hertz(low + (high - low) * Float($0) / Float(count + 1)) }
        let binWidth = sampleRate / Float(fftSize)
        return (1...count).map { b in
            let (left, center, right) = (edges[b - 1], edges[b], edges[b + 1])
            var bins: [(Int, Float)] = []
            for k in Int(left / binWidth)...Int(right / binWidth) + 1 {
                let f = Float(k) * binWidth
                let weight = f < center ? (f - left) / max(center - left, 1e-3) : (right - f) / max(right - center, 1e-3)
                if weight > 0 { bins.append((k, weight)) }
            }
            return Band(center: center, bins: bins)
        }
    }

    /// Linear interpolation of band values at a frequency; fades out above the last band.
    private static func interpolate(_ values: [Float], bands: [Band], at frequency: Float) -> Float {
        guard let first = bands.first, let last = bands.last else { return 0 }
        if frequency <= first.center { return values[0] * max(0, frequency / first.center) }
        if frequency >= last.center { return values[values.count - 1] * max(0, 1 - (frequency - last.center) / 1000) }
        var i = 0
        while i + 1 < bands.count, bands[i + 1].center < frequency { i += 1 }
        let t = (frequency - bands[i].center) / (bands[i + 1].center - bands[i].center)
        return values[i] * (1 - t) + values[i + 1] * t
    }
}

/// The synthetic source, continuous across frames.
private struct Carrier {
    let source: Vocoder.Source
    let period: Int
    var position = 0

    init(source: Vocoder.Source, sampleRate: Double) {
        self.source = source
        period = Int((sampleRate / 110).rounded())  // a flat 110 Hz voice
    }

    /// The next `count` samples, starting where the previous frame's window starts this time (hop-aligned).
    mutating func next(_ count: Int) -> [Float] {
        var samples = [Float](repeating: 0, count: count)
        switch source {
        case .whisper:
            for i in 0..<count { samples[i] = Float.random(in: -1...1) }
        case .synthetic:
            for i in 0..<count where (position + i) % period == 0 { samples[i] = 1 }
        }
        position += 256  // frames advance by one hop
        return samples
    }
}
