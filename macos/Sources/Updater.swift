import AppKit
import Foundation

/// Compares the commit this app was built from with the latest commit on GitHub, and installs
/// the release CI built for it. Only runs when asked: the app stays offline otherwise.
@MainActor
final class Updater: ObservableObject {
    struct Release: Equatable {
        let version: String
        let commit: String
        let summary: String
        let download: URL
    }

    enum State: Equatable {
        case idle
        case checking
        case upToDate
        /// New commits exist on GitHub but CI hasn't published their build yet.
        case building(summary: String)
        case available(Release)
        case downloading(Double)
        case installing
        case failed(String)
    }

    static let repository = "sivelswhy/multiblur"
    nonisolated private static let bundleIdentifier = "com.multiblur.MultiBlur"

    @Published private(set) var state: State = .idle

    let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?"
    /// Commit embedded by build.sh; nil for builds made outside a git checkout.
    let commit: String?

    init(commit: String? = Bundle.main.infoDictionary?["MultiBlurCommit"] as? String) {
        self.commit = commit
    }

    var shortCommit: String? { commit.map { String($0.prefix(7)) } }

    // MARK: Check

    func check() {
        state = .checking
        Task {
            do {
                state = try await fetchState()
            } catch {
                state = .failed(String(localized: "Couldn't reach GitHub: \(error.localizedDescription)"))
            }
        }
    }

    private struct CommitResponse: Decodable {
        struct Commit: Decodable { let message: String }
        let sha: String
        let commit: Commit
    }

    private struct ReleaseResponse: Decodable {
        struct Asset: Decodable {
            let name: String
            let browser_download_url: URL
        }
        let name: String?
        let tag_name: String
        let target_commitish: String
        let assets: [Asset]
    }

    private func fetchState() async throws -> State {
        async let head: CommitResponse = get("commits/main")
        async let release: ReleaseResponse = get("releases/latest")
        let (latest, published) = try await (head, release)
        let summary = latest.commit.message.split(separator: "\n").first.map(String.init) ?? ""

        if latest.sha == commit { return .upToDate }
        guard published.target_commitish == latest.sha,
              let asset = published.assets.first(where: { $0.name == "MultiBlur-macOS.zip" }) else {
            return .building(summary: summary)
        }
        let version = (published.name ?? published.tag_name).replacingOccurrences(of: "MultiBlur ", with: "")
        return .available(Release(version: version, commit: String(latest.sha.prefix(7)), summary: summary,
                                  download: asset.browser_download_url))
    }

    private func get<T: Decodable>(_ path: String) async throws -> T {
        var request = URLRequest(url: URL(string: "https://api.github.com/repos/\(Self.repository)/\(path)")!)
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        let (data, response) = try await URLSession.shared.data(for: request)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
        return try JSONDecoder().decode(T.self, from: data)
    }

    // MARK: Update

    func install(_ release: Release) {
        state = .downloading(0)
        Task {
            do {
                let newApp = try await Self.downloadApp(from: release.download) { fraction in
                    Task { @MainActor in
                        if case .downloading = self.state { self.state = .downloading(fraction) }
                    }
                }
                state = .installing
                try Self.replace(Bundle.main.bundleURL, with: newApp, relaunch: true,
                                 afterExitOf: ProcessInfo.processInfo.processIdentifier)
                NSApp.terminate(nil)
            } catch {
                state = .failed(error.localizedDescription)
            }
        }
    }

    enum UpdateError: LocalizedError {
        case invalidDownload
        case notWritable(String)

        var errorDescription: String? {
            switch self {
            case .invalidDownload: String(localized: "The downloaded update isn't a valid MultiBlur app.")
            case .notWritable(let path): String(localized: "MultiBlur can't replace itself in \(path). Move it to a folder you can write to, such as Applications.")
            }
        }
    }

    /// Downloads and unzips a release, then checks it's a correctly signed MultiBlur app.
    nonisolated static func downloadApp(from url: URL, progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        let archive = try await download(url, progress: progress)
        defer { try? FileManager.default.removeItem(at: archive) }

        let staging = FileManager.default.temporaryDirectory.appendingPathComponent("MultiBlur-update-\(UUID().uuidString)")
        try run("/usr/bin/ditto", ["-x", "-k", archive.path, staging.path])
        let app = staging.appendingPathComponent("MultiBlur.app")
        guard Bundle(url: app)?.bundleIdentifier == bundleIdentifier,
              (try? run("/usr/bin/codesign", ["--verify", "--deep", app.path])) != nil else {
            throw UpdateError.invalidDownload
        }
        return app
    }

    /// Swaps the app bundle once the running app has quit, then optionally relaunches it.
    /// Runs as a detached shell script because an app can't replace itself while running.
    nonisolated static func replace(_ current: URL, with newApp: URL, relaunch: Bool, afterExitOf pid: Int32) throws {
        let folder = current.deletingLastPathComponent()
        guard FileManager.default.isWritableFile(atPath: folder.path) else { throw UpdateError.notWritable(folder.path) }

        let old = folder.appendingPathComponent(".MultiBlur-old-\(UUID().uuidString).app")
        let script = """
        while kill -0 "$PID" 2>/dev/null; do sleep 0.2; done
        if mv "$APP" "$OLD" && mv "$NEW" "$APP"; then rm -rf "$OLD"; else [ -d "$OLD" ] && mv "$OLD" "$APP"; fi
        rm -rf "$(dirname "$NEW")"
        [ "$RELAUNCH" = 1 ] && open "$APP"
        """
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/sh")
        process.arguments = ["-c", script]
        process.environment = ["PID": String(pid), "APP": current.path, "NEW": newApp.path, "OLD": old.path,
                               "RELAUNCH": relaunch ? "1" : "0", "PATH": "/usr/bin:/bin"]
        try process.run()
    }

    @discardableResult
    private nonisolated static func run(_ tool: String, _ arguments: [String]) throws -> Int32 {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: tool)
        process.arguments = arguments
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { throw UpdateError.invalidDownload }
        return process.terminationStatus
    }

    private nonisolated static func download(_ url: URL, progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        final class Observation: @unchecked Sendable { var token: NSKeyValueObservation? }
        let observation = Observation()
        return try await withCheckedThrowingContinuation { continuation in
            let task = URLSession.shared.downloadTask(with: url) { location, response, error in
                observation.token = nil
                if let error { return continuation.resume(throwing: error) }
                guard let location, (response as? HTTPURLResponse)?.statusCode == 200 else {
                    return continuation.resume(throwing: URLError(.badServerResponse))
                }
                // The temporary file is deleted when this handler returns: keep our own copy.
                let kept = FileManager.default.temporaryDirectory.appendingPathComponent("MultiBlur-\(UUID().uuidString).zip")
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
