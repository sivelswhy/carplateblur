import CoreImage

/// Spots hard cuts between video shots, so tracking restarts instead of handing one person's
/// track to whoever stands at the same place in the next shot.
///
/// A cut is a sudden spike in how much a tiny grayscale version of the frame changes, compared with
/// the previous frames: measured cuts spiked ×12, while fast handheld motion stayed below ×3.5.
final class CutDetector {
    private static let size = (width: 32, height: 18)
    private static let spike = 6.0
    private static let minimumChange = 0.08
    private static let memory = 8

    private let context = CIContext(options: [.workingColorSpace: NSNull()])
    private var previous: [UInt8]?
    private var recentChanges: [Double] = []

    /// Feeds the next frame; true when it starts a new shot.
    func isCut(_ frame: CIImage) -> Bool {
        let extent = frame.extent
        let small = frame
            .transformed(by: CGAffineTransform(scaleX: CGFloat(Self.size.width) / extent.width,
                                               y: CGFloat(Self.size.height) / extent.height))
            .applyingFilter("CIColorControls", parameters: [kCIInputSaturationKey: 0])
        var pixels = [UInt8](repeating: 0, count: Self.size.width * Self.size.height * 4)
        context.render(small, toBitmap: &pixels, rowBytes: Self.size.width * 4,
                       bounds: CGRect(x: 0, y: 0, width: Self.size.width, height: Self.size.height),
                       format: .RGBA8, colorSpace: nil)
        let gray = stride(from: 0, to: pixels.count, by: 4).map { pixels[$0] }
        defer { previous = gray }
        guard let previous else { return false }

        let change = Double(zip(gray, previous).reduce(0) { $0 + abs(Int($1.0) - Int($1.1)) }) / Double(gray.count * 255)
        let usual = recentChanges.isEmpty ? 0 : recentChanges.reduce(0, +) / Double(recentChanges.count)
        recentChanges.append(change)
        if recentChanges.count > Self.memory { recentChanges.removeFirst() }
        let cut = change > Self.minimumChange && change > Self.spike * (usual + 0.01)
        if cut { recentChanges = [] }
        return cut
    }
}
