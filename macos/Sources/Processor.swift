import AppKit
import CoreGraphics
import Foundation
import UniformTypeIdentifiers

struct Job: Identifiable {
    enum Status: Equatable {
        case waiting
        case running(Double)
        case done(DetectionCount)
        case failed(String)
    }

    let id = UUID()
    let source: URL
    var output: URL
    var status: Status = .waiting

    var isVideo: Bool { UTType(filenameExtension: source.pathExtension)?.conforms(to: .movie) ?? false }

    var isRunning: Bool {
        if case .running = status { return true }
        return false
    }
}

/// Queue of files to anonymize; processes them one at a time in the background.
@MainActor
final class Processor: ObservableObject {
    @Published var options = Options.load() {
        didSet { options.save() }
    }
    @Published private(set) var jobs: [Job] = []
    @Published private(set) var engineError: String?
    @Published private(set) var preview: CGImage?

    private var engine: BlurEngine?
    private var isRunning = false

    var hasFinishedJobs: Bool {
        jobs.contains {
            switch $0.status {
            case .done, .failed: true
            default: false
            }
        }
    }

    var isProcessing: Bool { jobs.contains(where: \.isRunning) }

    var outputFolder: URL? { options.outputFolderPath.map { URL(fileURLWithPath: $0, isDirectory: true) } }

    func add(_ urls: [URL]) {
        let files = urls.flatMap(Self.mediaFiles(in:))
        jobs += files.map { Job(source: $0, output: $0) }
        startIfNeeded()
    }

    func clearFinished() {
        jobs.removeAll {
            switch $0.status {
            case .done, .failed: true
            default: false
            }
        }
        preview = nil
    }

    func remove(_ id: UUID) {
        jobs.removeAll { $0.id == id && !$0.isRunning }
    }

    private func startIfNeeded() {
        guard !isRunning else { return }
        isRunning = true
        Task {
            await runQueue()
            isRunning = false
        }
    }

    private func runQueue() async {
        while let index = jobs.firstIndex(where: { $0.status == .waiting }) {
            // Options are read per job, so changes apply from the next file in the queue.
            let options = self.options
            if engine?.compute != options.compute {
                do {
                    engine = try BlurEngine(compute: options.compute)
                } catch {
                    engineError = error.localizedDescription
                    return
                }
            }
            guard let engine else { return }

            jobs[index].output = BlurEngine.outputURL(for: jobs[index].source, folder: outputFolder)
            let job = jobs[index]
            setStatus(.running(0), for: job.id)
            do {
                let count = try await Task.detached(priority: .userInitiated) {
                    try await engine.process(job.source, to: job.output, options: options) { fraction in
                        Task { @MainActor in self.setStatus(.running(fraction), for: job.id) }
                    } preview: { image in
                        Task { @MainActor in self.preview = image }
                    }
                }.value
                setStatus(.done(count), for: job.id)
                if let sound = options.completionSound { NSSound(named: sound)?.play() }
            } catch {
                setStatus(.failed(error.localizedDescription), for: job.id)
            }
        }
    }

    private func setStatus(_ status: Job.Status, for id: UUID) {
        guard let index = jobs.firstIndex(where: { $0.id == id }) else { return }
        // Ignore late progress updates arriving after the job has completed.
        if case .running = status, case .done = jobs[index].status { return }
        if case .running = status, case .failed = jobs[index].status { return }
        jobs[index].status = status
    }

    /// Expands folders recursively and keeps only images and videos (skipping previous outputs).
    private static func mediaFiles(in url: URL) -> [URL] {
        let isMedia: (URL) -> Bool = { file in
            guard let type = UTType(filenameExtension: file.pathExtension) else { return false }
            return (type.conforms(to: .image) || type.conforms(to: .movie))
                && file.deletingPathExtension().lastPathComponent
                    .range(of: #"_anonymized( \(\d+\))?$"#, options: .regularExpression) == nil
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else { return [] }
        guard isDirectory.boolValue else { return isMedia(url) ? [url] : [] }

        let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: nil,
                                                        options: [.skipsHiddenFiles])
        let files = (enumerator?.allObjects as? [URL] ?? []).filter(isMedia)
        return files.sorted { $0.path.localizedStandardCompare($1.path) == .orderedAscending }
    }
}
