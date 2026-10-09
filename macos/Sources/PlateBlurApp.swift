import AppKit
import SwiftUI
import UniformTypeIdentifiers

@main
struct PlateBlurApp: App {
    @NSApplicationDelegateAdaptor private var appDelegate: AppDelegate
    @StateObject private var processor = Processor()

    var body: some Scene {
        Window("PlateBlur", id: "main") {
            ContentView()
                .environmentObject(processor)
                .frame(minWidth: 480, idealWidth: 540, minHeight: 420, idealHeight: 560)
        }
        .windowResizability(.contentMinSize)
        .windowToolbarStyle(.unified(showsTitle: true))

        Settings {
            SettingsView()
                .environmentObject(processor)
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}
