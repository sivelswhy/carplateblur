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
        case cancelled
    }

    var id = UUID()
    let source: URL
    var output: URL
    var status: Status = .waiting
    /// Exported in an earlier session: shown for reference, but can't be edited (its analysis is gone).
    var fromHistory = false
    /// When the export finished.
    var exportedAt: Date?
    /// Set when the file was reviewed in the editor: export masks exactly this.
    var plan: MaskPlan?
    /// The analysis made during export (and the detection settings it used), reused by the editor.
    var analysis: Analysis?
    var analysisOptions: Options?
    /// The editor's changes, shown again when it's reopened.
    var edits: Edits?

    var isVideo: Bool { UTType(filenameExtension: source.pathExtension)?.conforms(to: .movie) ?? false }

    var isRunning: Bool {
        if case .running = status { return true }
        return false
    }

    var isDone: Bool {
        if case .done = status { return true }
        return false
    }
}

/// A finished export, saved so the list survives relaunches.
private struct HistoryEntry: Codable {
    let id: UUID
    let source: URL
    let output: URL
    let plates: Int
    let faces: Int
    let exportedAt: Date
}

/// Queue of files to anonymize; processes them one at a time in the background.
@MainActor
final class Processor: ObservableObject {
    @Published var options = Options.load() {
        didSet { options.save() }
    }
    @Published private(set) var jobs: [Job] = Processor.loadHistory()
    @Published private(set) var engineError: String?
    @Published private(set) var preview: CGImage?

    private var engine: BlurEngine?
    /// The job being processed and its task, so it can be cancelled.
    private var current: (id: UUID, task: Task<(DetectionCount, Analysis?), Error>)?
    private var isRunning = false

    var hasFinishedJobs: Bool {
        jobs.contains {
            switch $0.status {
            case .done, .failed, .cancelled: true
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
            case .done, .failed, .cancelled: true
            default: false
            }
        }
        preview = nil
        saveHistory()
    }

    // MARK: History

    /// ~/Library/Application Support/MultiBlur/history.json
    private static var historyURL: URL {
        URL.applicationSupportDirectory.appendingPathComponent("MultiBlur/history.json")
    }

    /// Exports from earlier sessions whose result still exists.
    private static func loadHistory() -> [Job] {
        guard let data = try? Data(contentsOf: historyURL),
              let entries = try? JSONDecoder().decode([HistoryEntry].self, from: data) else { return [] }
        return entries
            .filter { FileManager.default.fileExists(atPath: $0.output.path) }
            .map { entry in
                var job = Job(source: entry.source, output: entry.output)
                job.id = entry.id
                job.status = .done(DetectionCount(plates: entry.plates, faces: entry.faces))
                job.fromHistory = true
                job.exportedAt = entry.exportedAt
                return job
            }
    }

    private func saveHistory() {
        let entries = jobs.compactMap { job -> HistoryEntry? in
            guard case .done(let count) = job.status else { return nil }
            return HistoryEntry(id: job.id, source: job.source, output: job.output, plates: count.plates,
                                faces: count.faces, exportedAt: job.exportedAt ?? Date())
        }
        do {
            try FileManager.default.createDirectory(at: Self.historyURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONEncoder().encode(entries).write(to: Self.historyURL, options: .atomic)
        } catch {
            // The history is a convenience; failing to save it must not disturb exports.
        }
    }

    /// Stops the job being processed; its partial output is deleted.
    func cancel(_ id: UUID) {
        guard current?.id == id else { return }
        current?.task.cancel()
    }

    /// Moves a finished file's exported result (never the original) to the Trash, then removes it
    /// from the list. Returns false if the file couldn't be moved.
    @discardableResult
    func trashOutput(_ id: UUID) -> Bool {
        guard let job = job(id), case .done = job.status, !job.isRunning else { return false }
        do {
            if FileManager.default.fileExists(atPath: job.output.path) {
                try FileManager.default.trashItem(at: job.output, resultingItemURL: nil)
            }
        } catch {
            return false
        }
        remove(id)
        return true
    }

    func remove(_ id: UUID) {
        jobs.removeAll { $0.id == id && !$0.isRunning }
        saveHistory()
    }

    /// The engine shared with editor windows (models load once).
    func sharedEngine() throws -> BlurEngine {
        if let engine, engine.compute == options.compute { return engine }
        let engine = try BlurEngine(compute: options.compute)
        self.engine = engine
        return engine
    }

    func job(_ id: UUID) -> Job? { jobs.first { $0.id == id } }

    /// Keeps an analysis made by the editor (after detection settings changed) for next time.
    func store(_ analysis: Analysis, options: Options, for id: UUID) {
        guard let index = jobs.firstIndex(where: { $0.id == id }) else { return }
        jobs[index].analysis = analysis
        jobs[index].analysisOptions = options
        jobs[index].edits = nil
    }

    /// Exports a file again as reviewed in the editor, replacing its previous result.
    func reexport(_ id: UUID, plan: MaskPlan, edits: Edits) {
        guard let index = jobs.firstIndex(where: { $0.id == id }), !jobs[index].isRunning, !jobs[index].fromHistory else { return }
        jobs[index].plan = plan
        jobs[index].edits = edits
        jobs[index].status = .waiting
        startIfNeeded()
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

            // A reviewed file replaces its own previous result instead of getting a new name.
            if jobs[index].plan == nil {
                jobs[index].output = BlurEngine.outputURL(for: jobs[index].source, folder: outputFolder)
            }
            let job = jobs[index]
            setStatus(.running(0), for: job.id)
            let task = Task.detached(priority: .userInitiated) { () -> (DetectionCount, Analysis?) in
                let progress: @Sendable (Double) -> Void = { fraction in
                    Task { @MainActor in self.setStatus(.running(fraction), for: job.id) }
                }
                let preview: @Sendable (CGImage) -> Void = { image in
                    Task { @MainActor in self.preview = image }
                }
                // Reviewed files are exported from their plan; new ones are analyzed once and exported
                // from that analysis, which the editor then reuses.
                if let plan = job.plan {
                    return (try await engine.process(job.source, to: job.output, options: options, plan: plan,
                                                     progress: progress, preview: preview), nil)
                }
                let (count, analysis) = try await engine.analyzeAndProcess(job.source, to: job.output, options: options,
                                                                           progress: progress, preview: preview)
                return (count, analysis)
            }
            current = (job.id, task)
            do {
                let (count, analysis) = try await task.value
                if let analysis, let index = jobs.firstIndex(where: { $0.id == job.id }) {
                    jobs[index].analysis = analysis
                    jobs[index].analysisOptions = options
                    jobs[index].edits = nil
                }
                setStatus(.done(count), for: job.id)
                if let index = jobs.firstIndex(where: { $0.id == job.id }) { jobs[index].exportedAt = Date() }
                saveHistory()
                if let sound = options.completionSound { NSSound(named: sound)?.play() }
            } catch is CancellationError {
                setStatus(.cancelled, for: job.id)
            } catch {
                setStatus(.failed(error.localizedDescription), for: job.id)
            }
            current = nil
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
