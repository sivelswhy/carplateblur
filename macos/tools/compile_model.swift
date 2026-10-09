// Compiles a .mlpackage into a .mlmodelc without needing Xcode.
// Usage: swift compile_model.swift <input.mlpackage> <output.mlmodelc>
import CoreML
import Foundation

let args = CommandLine.arguments
guard args.count == 3 else {
    print("usage: compile_model.swift <input.mlpackage> <output.mlmodelc>")
    exit(1)
}
let compiled = try MLModel.compileModel(at: URL(fileURLWithPath: args[1]))
let destination = URL(fileURLWithPath: args[2])
try? FileManager.default.removeItem(at: destination)
try FileManager.default.copyItem(at: compiled, to: destination)
print("Compiled model → \(destination.path)")
