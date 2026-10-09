import CryptoKit
import Foundation

/// Locates the CoreML models and, when they aren't bundled with the app, downloads them once
/// from the project's GitHub release. This is the app's only network access.
enum ModelStore {
    static let names = ["PlateDetector", "CenterFace"]

    private static let version = "v1"
    static let downloadURL = URL(string: "https://github.com/sivelswhy/carplateblur/releases/download/models-\(version)/MultiBlur-models-\(version).zip")!
    /// Checked before installing, so a corrupted or tampered download is never used.
    private static let sha256 = "e7c2cff90d1fe0584c3b860cc933fe7bef483b7fb5a1d38adfea83b1a5d8072c"
    static let downloadSize = "21 MB"

    /// ~/Library/Application Support/MultiBlur/Models-v1
    static var directory: URL {
        URL.applicationSupportDirectory.appendingPathComponent("MultiBlur/Models-\(version)", isDirectory: true)
    }

    /// Models bundled in the app (local builds) win over downloaded ones.
    static func url(for name: String) -> URL? {
        if let bundled = Bundle.main.url(forResource: name, withExtension: "mlmodelc") { return bundled }
        let downloaded = directory.appendingPathComponent("\(name).mlmodelc")
        return FileManager.default.fileExists(atPath: downloaded.path) ? downloaded : nil
    }

    static var isInstalled: Bool { names.allSatisfy { url(for: $0) != nil } }

    enum InstallError: LocalizedError {
        case checksumMismatch
        case unzipFailed
        case incomplete

        var errorDescription: String? {
            switch self {
            case .checksumMismatch: "The download is corrupted (checksum mismatch). Please try again."
            case .unzipFailed: "The downloaded models could not be unpacked."
            case .incomplete: "The download doesn't contain the expected models."
            }
        }
    }

    /// Downloads, verifies and installs the models into `destination`.
    static func install(into destination: URL = directory, progress: @escaping @Sendable (Double) -> Void) async throws {
        let archive = try await download(progress: progress)
        defer { try? FileManager.default.removeItem(at: archive) }

        let digest = SHA256.hash(data: try Data(contentsOf: archive)).map { String(format: "%02x", $0) }.joined()
        guard digest == sha256 else { throw InstallError.checksumMismatch }

        let staging = FileManager.default.temporaryDirectory.appendingPathComponent("MultiBlur-models-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: staging) }
        let unzip = Process()
        unzip.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        unzip.arguments = ["-x", "-k", archive.path, staging.path]
        try unzip.run()
        unzip.waitUntilExit()
        guard unzip.terminationStatus == 0 else { throw InstallError.unzipFailed }
        guard names.allSatisfy({ FileManager.default.fileExists(atPath: staging.appendingPathComponent("\($0).mlmodelc").path) }) else {
            throw InstallError.incomplete
        }

        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        for name in names {
            let target = destination.appendingPathComponent("\(name).mlmodelc")
            try? FileManager.default.removeItem(at: target)
            try FileManager.default.moveItem(at: staging.appendingPathComponent("\(name).mlmodelc"), to: target)
        }
    }

    private static func download(progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        final class Observation: @unchecked Sendable { var token: NSKeyValueObservation? }
        let observation = Observation()
        return try await withCheckedThrowingContinuation { continuation in
            let task = URLSession.shared.downloadTask(with: downloadURL) { location, response, error in
                observation.token = nil
                if let error { return continuation.resume(throwing: error) }
                guard let location, let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                    return continuation.resume(throwing: URLError(.badServerResponse))
                }
                // The temporary file is deleted when this handler returns: move it somewhere we own.
                let kept = FileManager.default.temporaryDirectory.appendingPathComponent("MultiBlur-models-\(UUID().uuidString).zip")
                do {
                    try FileManager.default.moveItem(at: location, to: kept)
                    continuation.resume(returning: kept)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
            observation.token = task.progress.observe(\.fractionCompleted) { p, _ in progress(p.fractionCompleted) }
            task.resume()
        }
    }
}

/// Drives the first-launch download sheet.
@MainActor
final class ModelSetup: ObservableObject {
    enum State: Equatable {
        case needed
        case downloading(Double)
        case failed(String)
        case ready
    }

    @Published private(set) var state: State = ModelStore.isInstalled ? .ready : .needed

    func download() {
        state = .downloading(0)
        Task {
            do {
                try await ModelStore.install { fraction in
                    Task { @MainActor in
                        if case .downloading = self.state { self.state = .downloading(fraction) }
                    }
                }
                state = .ready
            } catch {
                state = .failed(error.localizedDescription)
            }
        }
    }
}
