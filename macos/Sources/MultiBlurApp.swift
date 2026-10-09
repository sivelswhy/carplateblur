import AppKit
import SwiftUI
import UniformTypeIdentifiers

@main
struct MultiBlurApp: App {
    @NSApplicationDelegateAdaptor private var appDelegate: AppDelegate
    @StateObject private var processor = Processor()
    @StateObject private var updater = Updater()

    var body: some Scene {
        Window("MultiBlur", id: "main") {
            ContentView()
                .environmentObject(processor)
                .onAppear { appDelegate.processor = processor }
                .frame(minWidth: 480, idealWidth: 540, minHeight: 420, idealHeight: 560)
        }
        .windowResizability(.contentMinSize)
        .windowToolbarStyle(.unified(showsTitle: true))

        WindowGroup("Edit", id: "editor", for: UUID.self) { $jobID in
            if let jobID {
                EditorWindow(jobID: jobID)
                    .environmentObject(processor)
            }
        }
        .windowResizability(.contentMinSize)

        Settings {
            SettingsView()
                .environmentObject(processor)
                .environmentObject(updater)
        }
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    /// Files received from Finder before the window, and its processor, exist.
    private var pendingFiles: [URL] = []

    var processor: Processor? {
        didSet {
            guard let processor, !pendingFiles.isEmpty else { return }
            processor.add(pendingFiles)
            pendingFiles = []
        }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.servicesProvider = self
        NSUpdateDynamicServices()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    /// The "Anonymize with MultiBlur" Finder service (declared under NSServices in Info.plist).
    @objc func anonymizeFiles(_ pasteboard: NSPasteboard, userData: String?, error: AutoreleasingUnsafeMutablePointer<NSString?>) {
        let urls = pasteboard.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL] ?? []
        if let processor { processor.add(urls) } else { pendingFiles += urls }
        NSApp.activate(ignoringOtherApps: true)
    }
}
