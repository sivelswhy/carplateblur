import AppKit
import SwiftUI

/// Editor window for one exported file: review every mask, keep chosen faces or plates visible,
/// mask anything that was missed, then export again.
struct EditorWindow: View {
    let jobID: UUID
    @EnvironmentObject private var processor: Processor

    var body: some View {
        if let job = processor.job(jobID), job.fromHistory {
            Text("Exported in an earlier session: it can be opened, but not edited.")
                .foregroundStyle(.secondary)
                .frame(width: 420, height: 200)
        } else if let job = processor.job(jobID), let engine = try? processor.sharedEngine() {
            EditorView(model: EditorModel(job: job, options: processor.options, engine: engine) { analysis, options in
                processor.store(analysis, options: options, for: jobID)
            })
        } else {
            Text("This file is no longer in the list.")
                .foregroundStyle(.secondary)
                .frame(width: 360, height: 200)
        }
    }
}

struct EditorView: View {
    @StateObject private var model: EditorModel
    @EnvironmentObject private var processor: Processor
    @Environment(\.dismiss) private var dismiss

    init(model: EditorModel) {
        _model = StateObject(wrappedValue: model)
    }

    var body: some View {
        HStack(spacing: 0) {
            VStack(spacing: 0) {
                canvas
                if let analysis = model.analysis, analysis.isVideo { scrubber(analysis) }
            }
            // Full screen is for watching the video: the list makes room for it.
            if !ui.isFullScreen {
                Divider()
                sidebar.frame(width: 240)
            }
        }
        .safeAreaInset(edge: .bottom) { bottomBar }
        .background(WindowReader { ui.window = $0 })
        .onReceive(NotificationCenter.default.publisher(for: NSWindow.didEnterFullScreenNotification)) { note in
            if note.object as? NSWindow === ui.window { ui.isFullScreen = true }
        }
        .onReceive(NotificationCenter.default.publisher(for: NSWindow.didExitFullScreenNotification)) { note in
            if note.object as? NSWindow === ui.window { ui.isFullScreen = false }
        }
        .frame(minWidth: 820, minHeight: 560)
        .navigationTitle(model.job.source.lastPathComponent)
        .task { await model.load() }
        .onDisappear { model.pause() }
    }

    // MARK: Canvas

    private var canvas: some View {
        ZStack {
            Color.black.opacity(0.85)
            if let preview = model.preview, let analysis = model.analysis {
                GeometryReader { geometry in
                    let layout = CanvasLayout(imageSize: analysis.size, in: geometry.size)
                    ZStack(alignment: .topLeading) {
                        Image(decorative: preview, scale: 1)
                            .resizable()
                            .frame(width: layout.drawn.width, height: layout.drawn.height)
                            .offset(x: layout.drawn.minX, y: layout.drawn.minY)
                        MaskOutlines(model: model, layout: layout, hovered: ui.hover.flatMap { target(at: $0, layout: layout) })
                        if let draft = ui.draft {
                            Rectangle()
                                .strokeBorder(Color.orange, style: StrokeStyle(lineWidth: 2, dash: [5, 4]))
                                .frame(width: draft.width, height: draft.height)
                                .offset(x: draft.minX, y: draft.minY)
                                .allowsHitTesting(false)
                        }
                    }
                    // The image is drawn with an offset: make the whole canvas the hit area, in the
                    // same coordinates, or clicks outside the unshifted image size are lost.
                    .frame(width: geometry.size.width, height: geometry.size.height, alignment: .topLeading)
                    .contentShape(Rectangle())
                    .gesture(pointerGesture(layout))
                    .onContinuousHover(coordinateSpace: .local) { phase in
                        if case .active(let location) = phase { ui.hover = location } else { ui.hover = nil }
                    }
                }
            }
            switch model.phase {
            case .analyzing(let fraction):
                progress(String(localized: "Finding plates and faces…"), fraction)
            case .tracking(let fraction):
                progress(String(localized: "Following the box through the video…"), fraction)
            case .failed(let message):
                Label(message, systemImage: "exclamationmark.triangle.fill").foregroundStyle(.white)
            case .ready:
                EmptyView()
            }
        }
    }

    private func progress(_ title: String, _ fraction: Double) -> some View {
        VStack(spacing: 8) {
            ProgressView(value: fraction).frame(width: 220)
            Text(title).font(.callout)
        }
        .padding(16)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
    }

    /// One gesture for clicks and drags: on macOS a separate tap gesture never fires next to a drag
    /// gesture on the same view. Moving less than a few points is a click; more draws a box.
    private func pointerGesture(_ layout: CanvasLayout) -> some Gesture {
        DragGesture(minimumDistance: 0, coordinateSpace: .local)
            .onChanged { value in
                guard Self.isDrag(value) else { return }
                model.pause()
                ui.draft = CGRect(origin: value.startLocation, size: .zero).union(CGRect(origin: value.location, size: .zero))
            }
            .onEnded { value in
                ui.draft = nil
                if Self.isDrag(value) {
                    let viewRect = CGRect(origin: value.startLocation, size: .zero).union(CGRect(origin: value.location, size: .zero))
                    model.addBox(layout.imageRect(fromView: viewRect))
                } else {
                    tap(at: value.startLocation, layout: layout)
                }
            }
    }

    private static func isDrag(_ value: DragGesture.Value) -> Bool {
        hypot(value.translation.width, value.translation.height) >= 4
    }

    /// The mask under a point: hand-drawn boxes first, then the smallest detected area.
    private func target(at point: CGPoint, layout: CanvasLayout) -> EditorTarget? {
        let location = layout.imagePoint(fromView: point)
        if let manual = model.manualInFrame.filter({ $0.rect.contains(location) }).min(by: { $0.rect.width < $1.rect.width }) {
            return .manual(id: manual.id)
        }
        return model.detectionsInFrame.filter { $0.detection.rect.contains(location) }
            .min { $0.detection.rect.width < $1.detection.rect.width }
            .map { .detection(id: $0.detection.id, masked: $0.masked) }
    }

    /// Clicking a mask removes it, keeping that face or plate visible (everywhere in a video); clicking
    /// it again masks it. Clicking a hand-drawn box deletes it.
    private func tap(at point: CGPoint, layout: CanvasLayout) {
        model.pause()
        switch target(at: point, layout: layout) {
        case .manual(let id):
            model.removeManual(id)
        case .detection(let id, let masked):
            if let group = model.group(of: id) { model.setExcluded(group, masked) }
        case nil:
            break
        }
    }

    @StateObject private var ui = EditorUIState()

    // MARK: Scrubber

    private func scrubber(_ analysis: Analysis) -> some View {
        HStack(spacing: 10) {
            Button { model.togglePlayback() } label: {
                Image(systemName: model.isPlaying ? "pause.fill" : "play.fill").frame(width: 16)
            }
            .keyboardShortcut(.space, modifiers: [])
            .help(model.isPlaying ? "Pause" : "Play")
            Button { model.pause(); model.frameIndex = max(0, model.frameIndex - 1) } label: { Image(systemName: "backward.frame.fill") }
                .keyboardShortcut(.leftArrow, modifiers: [])
            Button { model.pause(); model.frameIndex = min(analysis.frameTimes.count - 1, model.frameIndex + 1) } label: { Image(systemName: "forward.frame.fill") }
                .keyboardShortcut(.rightArrow, modifiers: [])
            Slider(value: Binding(
                get: { Double(model.frameIndex) },
                set: {
                    model.pause()
                    model.frameIndex = Int($0.rounded())
                }
            ), in: 0...Double(max(1, analysis.frameTimes.count - 1)))
            Text(Self.timecode(analysis.frameTimes[model.frameIndex].seconds))
                .monospacedDigit()
                .foregroundStyle(.secondary)
                .frame(width: 56, alignment: .trailing)
            fullScreenButton
        }
        .buttonStyle(.borderless)
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }

    private static func timecode(_ seconds: Double) -> String {
        let total = max(0, seconds)
        return String(format: "%d:%05.2f", Int(total) / 60, total.truncatingRemainder(dividingBy: 60))
    }

    // MARK: Sidebar

    private var sidebar: some View {
        List {
            if let analysis = model.analysis {
                // Tracks shorter than half a second (crowds, brief false detections) are folded away.
                let minimumFrames = analysis.isVideo ? max(1, Int(analysis.frameRate / 2)) : 0
                let lasting = analysis.groups.filter { $0.lastFrame - $0.firstFrame >= minimumFrames }
                let brief = analysis.groups.filter { $0.lastFrame - $0.firstFrame < minimumFrames }
                let faces = lasting.filter(\.isFace), plates = lasting.filter { !$0.isFace }
                if !faces.isEmpty {
                    Section("Faces") { ForEach(faces) { GroupRow(group: $0, model: model) } }
                }
                if !plates.isEmpty {
                    Section("License Plates") { ForEach(plates) { GroupRow(group: $0, model: model) } }
                }
                if !brief.isEmpty {
                    Section {
                        DisclosureGroup("Brief detections (\(brief.count))") {
                            ForEach(brief) { GroupRow(group: $0, model: model) }
                        }
                    }
                }
                if !model.manual.isEmpty {
                    Section("Added by Hand") {
                        ForEach(model.manual) { mask in
                            HStack {
                                Image(systemName: "rectangle.dashed").foregroundStyle(.orange)
                                Text(analysis.isVideo ? "Frames \(model.manualFrames(mask).lowerBound + 1)–\(model.manualFrames(mask).upperBound + 1)" : "Box")
                                Spacer()
                                Button { model.removeManual(mask.id) } label: { Image(systemName: "trash") }
                                    .buttonStyle(.borderless)
                                    .help("Remove")
                            }
                        }
                    }
                }
                if analysis.groups.isEmpty, model.manual.isEmpty, model.phase == .ready {
                    Text("Nothing detected. Drag on the image to mask an area.").foregroundStyle(.secondary)
                }
            }
        }
        .listStyle(.sidebar)
    }

    private var fullScreenButton: some View {
        Button { ui.window?.toggleFullScreen(nil) } label: {
            Image(systemName: ui.isFullScreen ? "arrow.down.right.and.arrow.up.left" : "arrow.up.left.and.arrow.down.right")
        }
        .keyboardShortcut("f", modifiers: [.control, .command])
        .help(ui.isFullScreen ? "Exit Full Screen" : "Full Screen")
    }

    // MARK: Bottom bar

    private var bottomBar: some View {
        HStack(spacing: 12) {
            Text("Click a mask (×) to remove it. Drag to mask something that was missed.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .lineLimit(2)
            Spacer()
            if model.analysis?.isVideo == false { fullScreenButton.buttonStyle(.borderless) }
            if ui.isFullScreen {
                // Escape leaves full screen instead of closing the editor.
                Button("Cancel") { dismiss() }
                Button("") { ui.window?.toggleFullScreen(nil) }
                    .keyboardShortcut(.cancelAction)
                    .frame(width: 0, height: 0)
                    .opacity(0)
            } else {
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)
            }
            Button("Export") {
                if let plan = model.plan() { processor.reexport(model.job.id, plan: plan, edits: model.edits) }
                dismiss()
            }
            .keyboardShortcut(.defaultAction)
            .disabled(model.phase != .ready)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.bar)
    }
}

/// Drag state kept outside the view (no `@State`, whose macro needs Xcode).
@MainActor
final class EditorUIState: ObservableObject {
    @Published var draft: CGRect?
    @Published var isFullScreen = false
    /// The editor's own window, to toggle full screen and recognize its notifications.
    weak var window: NSWindow?
    /// Pointer position over the canvas, to highlight the mask a click would act on.
    @Published var hover: CGPoint?
}

/// The mask under the pointer.
enum EditorTarget: Equatable {
    case detection(id: Int, masked: Bool)
    case manual(id: Int)
}

/// Maps between image pixels (Core Image coordinates, origin bottom-left) and the fitted view.
struct CanvasLayout {
    let imageSize: CGSize
    let drawn: CGRect
    let scale: CGFloat

    init(imageSize: CGSize, in container: CGSize) {
        self.imageSize = imageSize
        scale = min(container.width / max(imageSize.width, 1), container.height / max(imageSize.height, 1))
        let size = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
        drawn = CGRect(x: (container.width - size.width) / 2, y: (container.height - size.height) / 2,
                       width: size.width, height: size.height)
    }

    func viewRect(fromImage rect: CGRect) -> CGRect {
        let visible = rect.intersection(CGRect(origin: .zero, size: imageSize))
        guard !visible.isNull else { return .zero }
        return CGRect(x: drawn.minX + visible.minX * scale, y: drawn.minY + (imageSize.height - visible.maxY) * scale,
                      width: visible.width * scale, height: visible.height * scale)
    }

    func imagePoint(fromView point: CGPoint) -> CGPoint {
        CGPoint(x: (point.x - drawn.minX) / scale, y: imageSize.height - (point.y - drawn.minY) / scale)
    }

    func imageRect(fromView rect: CGRect) -> CGRect {
        let a = imagePoint(fromView: CGPoint(x: rect.minX, y: rect.maxY))
        let b = imagePoint(fromView: CGPoint(x: rect.maxX, y: rect.minY))
        return CGRect(x: a.x, y: a.y, width: b.x - a.x, height: b.y - a.y)
            .intersection(CGRect(origin: .zero, size: imageSize))
    }
}

/// Outlines over the preview: masked areas in the accent color, visible ones dashed, hand-drawn ones orange.
struct MaskOutlines: View {
    @ObservedObject var model: EditorModel
    let layout: CanvasLayout
    let hovered: EditorTarget?

    var body: some View {
        ZStack(alignment: .topLeading) {
            // Areas predicted entirely outside the frame have nothing to outline.
            ForEach(Array(model.detectionsInFrame.filter { layout.viewRect(fromImage: $0.detection.rect).width > 0 }.enumerated()),
                    id: \.offset) { _, item in
                let rect = layout.viewRect(fromImage: item.detection.rect)
                let isHovered = hovered == .detection(id: item.detection.id, masked: item.masked)
                outline(rect, color: item.masked ? .accentColor : .white, dashed: !item.masked, highlighted: isHovered)
                if isHovered {
                    badge(item.masked ? "xmark" : "plus", color: item.masked ? .red : .accentColor, at: rect)
                } else if !item.masked {
                    Image(systemName: "eye.fill")
                        .font(.caption)
                        .padding(3)
                        .background(.black.opacity(0.6), in: Circle())
                        .foregroundStyle(.white)
                        .offset(x: rect.minX + 2, y: rect.minY + 2)
                }
            }
            ForEach(model.manualInFrame, id: \.id) { item in
                let rect = layout.viewRect(fromImage: item.rect)
                let isHovered = hovered == .manual(id: item.id)
                outline(rect, color: .orange, dashed: false, highlighted: isHovered)
                if isHovered { badge("xmark", color: .red, at: rect) }
            }
        }
        .allowsHitTesting(false)
    }

    private func outline(_ rect: CGRect, color: Color, dashed: Bool, highlighted: Bool) -> some View {
        Rectangle()
            .strokeBorder(color, style: StrokeStyle(lineWidth: highlighted ? 3 : 2, dash: dashed ? [5, 4] : []))
            .background(Rectangle().fill(color.opacity(highlighted ? 0.18 : 0)))
            .frame(width: max(rect.width, 4), height: max(rect.height, 4))
            .offset(x: rect.minX, y: rect.minY)
    }

    /// Round button-like badge on the top-right corner of the hovered mask: what a click will do.
    private func badge(_ symbol: String, color: Color, at rect: CGRect) -> some View {
        Image(systemName: symbol)
            .font(.system(size: 10, weight: .bold))
            .foregroundStyle(.white)
            .frame(width: 18, height: 18)
            .background(color, in: Circle())
            .overlay(Circle().strokeBorder(.white, lineWidth: 1.5))
            .offset(x: rect.maxX - 11, y: rect.minY - 7)
    }
}

struct GroupRow: View {
    let group: TrackGroup
    @ObservedObject var model: EditorModel

    var body: some View {
        HStack(spacing: 10) {
            Group {
                if let thumbnail = group.thumbnail {
                    Image(decorative: thumbnail, scale: 1).resizable().aspectRatio(contentMode: .fill)
                } else {
                    Image(systemName: group.isFace ? "face.smiling" : "car.fill").foregroundStyle(.secondary)
                }
            }
            .frame(width: 36, height: 36)
            .background(.quaternary)
            .clipShape(RoundedRectangle(cornerRadius: 5))
            .onTapGesture { model.frameIndex = group.firstFrame }

            Toggle(isOn: Binding(
                get: { !model.isExcluded(group) },
                set: { model.setExcluded(group, !$0) }
            )) {
                VStack(alignment: .leading, spacing: 1) {
                    Text(model.isExcluded(group) ? "Visible" : "Masked")
                        .foregroundStyle(model.isExcluded(group) ? .secondary : .primary)
                    if group.appearances > 1 {
                        Text("\(group.appearances) appearances").font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            .toggleStyle(.switch)
            .controlSize(.small)
        }
        .help("Click the thumbnail to jump to it")
    }
}

/// Hands over the NSWindow hosting a SwiftUI view.
struct WindowReader: NSViewRepresentable {
    let onWindow: (NSWindow?) -> Void

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async { onWindow(view.window) }
        return view
    }

    func updateNSView(_ view: NSView, context: Context) {
        DispatchQueue.main.async { onWindow(view.window) }
    }
}
