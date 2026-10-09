import AppKit
import QuickLookThumbnailing
import SwiftUI
import UniformTypeIdentifiers

/// Shows an open panel; used instead of `.fileImporter` so no extra view state is needed.
@MainActor
func choose(_ types: [UTType], directories: Bool = false, multiple: Bool = false) -> [URL] {
    let panel = NSOpenPanel()
    panel.allowedContentTypes = types
    panel.canChooseDirectories = directories
    panel.canChooseFiles = !types.isEmpty
    panel.allowsMultipleSelection = multiple
    panel.canCreateDirectories = directories
    return panel.runModal() == .OK ? panel.urls : []
}

@MainActor
func chooseMedia(into processor: Processor) {
    processor.add(choose([.image, .movie, .folder], directories: true, multiple: true))
}

/// Local view state. Kept in an ObservableObject rather than `@State`, whose macro plugin
/// ships only with Xcode, so the app builds with the Command Line Tools alone.
@MainActor
final class ViewState: ObservableObject {
    @Published var isTargeted = false
}

extension MaskMode {
    var symbol: String {
        switch self {
        case .blur: "drop.fill"
        case .mosaic: "squareshape.split.3x3"
        case .solid: "rectangle.fill"
        case .image: "photo"
        case .none: "eye"
        }
    }
}

// MARK: - Main window

struct ContentView: View {
    @EnvironmentObject private var processor: Processor
    @StateObject private var ui = ViewState()

    var body: some View {
        VStack(spacing: 0) {
            ControlBar()
            Divider()
            Group {
                if processor.jobs.isEmpty {
                    EmptyState()
                } else {
                    VStack(spacing: 0) {
                        if processor.options.livePreview, processor.isProcessing, let preview = processor.preview {
                            LivePreview(image: preview)
                        }
                        JobList()
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            Divider()
            StatusBar()
        }
        .overlay {
            if ui.isTargeted {
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(Color.accentColor, lineWidth: 3)
                    .background(RoundedRectangle(cornerRadius: 10).fill(Color.accentColor.opacity(0.06)))
                    .padding(6)
                    .allowsHitTesting(false)
            }
        }
        .dropDestination(for: URL.self) { urls, _ in
            processor.add(urls)
            return true
        } isTargeted: { ui.isTargeted = $0 }
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button { chooseMedia(into: processor) } label: { Label("Add Files", systemImage: "plus") }
                    .help("Add photos, videos or folders")
                SettingsLink { Label("Settings", systemImage: "gearshape") }
                    .help("Settings")
            }
        }
        .alert("Model unavailable", isPresented: .constant(processor.engineError != nil)) {
            Button("Quit") { NSApp.terminate(nil) }
        } message: {
            Text(processor.engineError ?? "")
        }
    }
}

/// What to detect and how to hide it: the settings people change most often.
struct ControlBar: View {
    @EnvironmentObject private var processor: Processor

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                // At least one target stays enabled.
                Toggle(isOn: Binding(
                    get: { processor.options.maskPlates },
                    set: { if $0 || processor.options.maskFaces { processor.options.maskPlates = $0 } })
                ) { Label("Plates", systemImage: "car.fill") }
                Toggle(isOn: Binding(
                    get: { processor.options.maskFaces },
                    set: { if $0 || processor.options.maskPlates { processor.options.maskFaces = $0 } })
                ) { Label("Faces", systemImage: "face.smiling") }

                Spacer()

                Picker("Style", selection: $processor.options.mode) {
                    ForEach(MaskMode.allCases) { mode in
                        Label(mode.label, systemImage: mode.symbol).tag(mode)
                    }
                }
                .help("deface --replacewith")
                .fixedSize()
            }
            .toggleStyle(ChipToggleStyle())
            .controlSize(.large)

            MaskingOptions()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }
}

/// Second row of the control bar: options for the selected style.
/// Drops its text labels when the window is too narrow for them.
struct MaskingOptions: View {
    @EnvironmentObject private var processor: Processor

    var body: some View {
        ViewThatFits(in: .horizontal) {
            row(labels: true)
            row(labels: false)
        }
        .font(.callout)
        .foregroundStyle(.secondary)
        .controlSize(.small)
    }

    private func row(labels: Bool) -> some View {
        HStack(spacing: 14) {
            HStack(spacing: 6) {
                if labels { Text("Face mask").fixedSize() }
                Picker("Face mask", selection: $processor.options.faceShape) {
                    Image(systemName: "circle").help("Ellipse").tag(MaskShape.ellipse)
                    Image(systemName: "square").help("Box").tag(MaskShape.box)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
            }
            .disabled(!processor.options.maskFaces)
            .help("Face mask shape: ellipse or box (deface --boxes)")

            switch processor.options.mode {
            case .mosaic:
                HStack(spacing: 6) {
                    if labels { Text("Block size").fixedSize() }
                    Slider(value: $processor.options.mosaicSize, in: 4...60)
                        .frame(width: labels ? 120 : 90)
                    Text("\(Int(processor.options.mosaicSize)) px")
                        .monospacedDigit()
                        .fixedSize()
                }
                .help("Mosaic block size (deface --mosaicsize)")
            case .image:
                let path = processor.options.replacementImagePath
                Button {
                    if let url = choose([.image]).first { processor.options.replacementImagePath = url.path }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: path == nil ? "exclamationmark.circle.fill" : "photo")
                            .foregroundStyle(path == nil ? Color.red : Color.secondary)
                        Text(path.map { URL(fileURLWithPath: $0).lastPathComponent } ?? "Choose Image…")
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                    .frame(maxWidth: labels ? 180 : 120)
                }
                .help("Replacement image (deface --replaceimg): PNG, JPEG, SVG… Transparent areas show a blur, never the original.")
            default:
                EmptyView()
            }

            Spacer(minLength: 0)
        }
    }
}

/// A capsule that is filled with the accent color when on and outlined when off,
/// so the state reads clearly even when the window is inactive.
struct ChipToggleStyle: ToggleStyle {
    func makeBody(configuration: Configuration) -> some View {
        Button { configuration.isOn.toggle() } label: {
            configuration.label
                .font(.callout.weight(.medium))
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .foregroundStyle(configuration.isOn ? Color.white : Color.secondary)
                .background(Capsule().fill(configuration.isOn ? Color.accentColor : Color.clear))
                .overlay(Capsule().strokeBorder(configuration.isOn ? Color.clear : Color.secondary.opacity(0.35)))
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityValue(configuration.isOn ? "On" : "Off")
    }
}

struct EmptyState: View {
    @EnvironmentObject private var processor: Processor

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: "eye.slash.circle")
                .font(.system(size: 56, weight: .thin))
                .foregroundStyle(.tertiary)
            VStack(spacing: 4) {
                Text("Drop Photos or Videos")
                    .font(.title2.weight(.semibold))
                Text("License plates and faces are hidden on this Mac.\nNothing is uploaded.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            Button("Choose Files…") { chooseMedia(into: processor) }
                .controlSize(.large)
                .keyboardShortcut("o")
        }
        .padding(40)
    }
}

struct LivePreview: View {
    let image: CGImage

    var body: some View {
        Image(decorative: image, scale: 1)
            .resizable()
            .aspectRatio(contentMode: .fit)
            .frame(maxWidth: .infinity, maxHeight: 220)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .padding(12)
            .background(.quaternary.opacity(0.5))
    }
}

struct JobList: View {
    @EnvironmentObject private var processor: Processor

    var body: some View {
        List(processor.jobs) { job in
            JobRow(job: job)
                .contextMenu {
                    if case .done = job.status {
                        Button("Open") { NSWorkspace.shared.open(job.output) }
                        Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([job.output]) }
                        Divider()
                    }
                    Button("Remove from List") { processor.remove(job.id) }
                        .disabled(job.isRunning)
                }
        }
        .listStyle(.inset)
        .scrollContentBackground(.hidden)
    }
}

/// Loads a Finder-style thumbnail; shows the anonymized output once it exists.
@MainActor
final class ThumbnailLoader: ObservableObject {
    @Published var image: CGImage?

    func load(_ url: URL) async {
        let request = QLThumbnailGenerator.Request(fileAt: url, size: CGSize(width: 40, height: 40),
                                                   scale: NSScreen.main?.backingScaleFactor ?? 2,
                                                   representationTypes: .thumbnail)
        image = try? await QLThumbnailGenerator.shared.generateBestRepresentation(for: request).cgImage
    }
}

struct JobRow: View {
    let job: Job
    @EnvironmentObject private var processor: Processor
    @StateObject private var thumbnail = ThumbnailLoader()

    private var thumbnailURL: URL {
        if case .done = job.status { return job.output }
        return job.source
    }

    var body: some View {
        HStack(spacing: 12) {
            Group {
                if let image = thumbnail.image {
                    Image(decorative: image, scale: 1).resizable().aspectRatio(contentMode: .fill)
                } else {
                    Image(systemName: job.isVideo ? "film" : "photo").foregroundStyle(.secondary)
                }
            }
            .frame(width: 40, height: 40)
            .background(.quaternary)
            .clipShape(RoundedRectangle(cornerRadius: 6))

            VStack(alignment: .leading, spacing: 3) {
                Text(job.source.lastPathComponent)
                    .lineLimit(1)
                    .truncationMode(.middle)
                detail
            }
            Spacer(minLength: 8)
            trailing
            // Running jobs can't be removed; keep the space so rows stay aligned.
            Button {
                processor.remove(job.id)
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.title3)
                    .foregroundStyle(.tertiary)
            }
            .buttonStyle(.plain)
            .help("Remove from list")
            .opacity(job.isRunning ? 0 : 1)
            .disabled(job.isRunning)
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
        .onTapGesture(count: 2) {
            if case .done = job.status { NSWorkspace.shared.open(job.output) }
        }
        .task(id: thumbnailURL) { await thumbnail.load(thumbnailURL) }
    }

    @ViewBuilder private var detail: some View {
        switch job.status {
        case .waiting:
            Text("Waiting…").font(.caption).foregroundStyle(.secondary)
        case .running(let fraction):
            ProgressView(value: fraction)
                .progressViewStyle(.linear)
                .controlSize(.small)
                .frame(maxWidth: 220)
        case .done(let count):
            Text(Self.describe(count)).font(.caption).foregroundStyle(.secondary)
        case .failed(let message):
            Text(message).font(.caption).foregroundStyle(.red).lineLimit(2)
        }
    }

    @ViewBuilder private var trailing: some View {
        switch job.status {
        case .done:
            Button {
                NSWorkspace.shared.activateFileViewerSelecting([job.output])
            } label: {
                Image(systemName: "magnifyingglass.circle.fill")
                    .font(.title2)
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .help("Show in Finder")
        case .failed:
            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.red)
        case .running(let fraction):
            Text(fraction.formatted(.percent.precision(.fractionLength(0))))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        case .waiting:
            EmptyView()
        }
    }

    private static func describe(_ count: DetectionCount) -> String {
        var parts: [String] = []
        if count.plates > 0 { parts.append(count.plates == 1 ? "1 plate" : "\(count.plates) plates") }
        if count.faces > 0 { parts.append(count.faces == 1 ? "1 face" : "\(count.faces) faces") }
        return parts.isEmpty ? "Nothing detected" : parts.joined(separator: " · ") + " hidden"
    }
}

struct StatusBar: View {
    @EnvironmentObject private var processor: Processor

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "folder")
            Text(processor.outputFolder.map { "Saving to \($0.lastPathComponent)" } ?? "Saving next to the originals")
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer()
            if processor.hasFinishedJobs {
                Button("Clear Finished") { processor.clearFinished() }
                    .buttonStyle(.link)
            }
        }
        .font(.callout)
        .foregroundStyle(.secondary)
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }
}

// MARK: - Settings window

struct LabeledSlider: View {
    @Binding var value: Double
    let range: ClosedRange<Double>
    var fractionDigits = 2
    var suffix = ""

    var body: some View {
        HStack {
            Slider(value: $value, in: range)
            Text(value.formatted(.number.precision(.fractionLength(fractionDigits))) + suffix)
                .monospacedDigit()
                .foregroundStyle(.secondary)
                .frame(width: 52, alignment: .trailing)
        }
    }
}

/// The remaining deface options (masking ones live in the main window), as a standard tabbed settings window.
struct SettingsView: View {
    var body: some View {
        TabView {
            DetectionSettings().tabItem { Label("Detection", systemImage: "viewfinder") }
            OutputSettings().tabItem { Label("Output", systemImage: "square.and.arrow.down") }
            AdvancedSettings().tabItem { Label("Advanced", systemImage: "gearshape.2") }
        }
        .frame(width: 500)
    }
}

struct DetectionSettings: View {
    @EnvironmentObject private var processor: Processor

    var body: some View {
        Form {
            Section {
                LabeledContent("Confidence") {
                    LabeledSlider(value: $processor.options.plateConfidence, range: 0.05...0.8)
                }
                LabeledContent("Mask scale") {
                    LabeledSlider(value: $processor.options.plateMaskScale, range: 1.0...1.6, suffix: "×")
                }
            } header: {
                Label("License Plates", systemImage: "car.fill")
            } footer: {
                Text("Lower confidence catches more plates but may also hide other objects.")
            }

            Section {
                LabeledContent("Threshold") {
                    LabeledSlider(value: $processor.options.faceThreshold, range: 0.05...0.9)
                }
                .help("deface --thresh")
                LabeledContent("Mask scale") {
                    LabeledSlider(value: $processor.options.faceMaskScale, range: 1.0...2.0, suffix: "×")
                }
                .help("deface --mask-scale")
                Picker("Resolution", selection: $processor.options.faceResolution) {
                    ForEach(DetectionResolution.allCases) { Text($0.label).tag($0) }
                }
                .help("deface --scale")
            } header: {
                Label("Faces", systemImage: "face.smiling")
            } footer: {
                Text("Lower resolutions are faster but miss small, distant faces.")
            }
        }
        .formStyle(.grouped)
    }
}

struct OutputSettings: View {
    @EnvironmentObject private var processor: Processor

    var body: some View {
        Form {
            Section {
                LabeledContent("Save to") {
                    HStack {
                        Text(processor.outputFolder?.lastPathComponent ?? "Next to originals")
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        if processor.options.outputFolderPath != nil {
                            Button("Reset") { processor.options.outputFolderPath = nil }
                        }
                        Button("Choose…") {
                            if let url = choose([], directories: true).first { processor.options.outputFolderPath = url.path }
                        }
                    }
                }
                .help("deface --output")
            } footer: {
                Text("Files get the “_anonymized” suffix; existing files are never overwritten.")
            }

            Section {
                Picker("Sound when done", selection: Binding(
                    get: { processor.options.completionSound },
                    set: { sound in
                        processor.options.completionSound = sound
                        // Let people hear their pick right away.
                        if let sound { NSSound(named: sound)?.play() }
                    })
                ) {
                    Text("None").tag(String?.none)
                    Divider()
                    ForEach(CompletionSound.available, id: \.self) { Text($0).tag(String?.some($0)) }
                }
            } footer: {
                Text("Plays after each file is exported.")
            }

            Section("Videos") {
                Toggle("Keep audio", isOn: $processor.options.keepAudio)
                    .help("deface --keep-audio")
                Picker("Codec", selection: $processor.options.videoCodec) {
                    ForEach(VideoCodec.allCases) { Text($0.label).tag($0) }
                }
                .help("Stands in for deface --ffmpeg-config")
            }

            Section {
                Toggle("Keep metadata (EXIF, GPS…)", isOn: $processor.options.keepMetadata)
                    .help("deface --keep-metadata")
            } header: {
                Text("Photos")
            } footer: {
                Text("Off by default: metadata can reveal where and with which device a photo was taken.")
            }
        }
        .formStyle(.grouped)
    }
}

struct AdvancedSettings: View {
    @EnvironmentObject private var processor: Processor

    var body: some View {
        Form {
            Section {
                Picker("Run models on", selection: $processor.options.compute) {
                    ForEach(ComputeMode.allCases) { Text($0.label).tag($0) }
                }
                .help("Stands in for deface --backend / --execution-provider")
                Toggle("Live preview", isOn: $processor.options.livePreview)
                    .help("deface --preview")
            } footer: {
                Text("Live preview shows processed frames in the main window and slightly slows exports.")
            }

            Section {
                Toggle("Show detection scores", isOn: $processor.options.drawScores)
                    .help("deface --draw-scores")
            } footer: {
                Text("Writes each detection's confidence score above its mask, to tune the thresholds in Detection.")
            }

            Section {
                HStack {
                    Spacer()
                    Button("Restore Defaults") { processor.options = Options() }
                }
            }
        }
        .formStyle(.grouped)
    }
}
