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
                .frame(minWidth: 480, idealWidth: 540, minHeight: 420, idealHeight: 560)
        }
        .windowResizability(.contentMinSize)
        .windowToolbarStyle(.unified(showsTitle: true))

        Settings {
            SettingsView()
                .environmentObject(processor)
                .environmentObject(updater)
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}
