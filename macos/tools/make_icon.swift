// Draws the MultiBlur app icon and writes Resources/AppIcon.icns.
// Usage: swift tools/make_icon.swift <output.icns> [preview.png]
import AppKit
import CoreGraphics

let canvas: CGFloat = 1024

func color(_ hex: UInt32, _ alpha: CGFloat = 1) -> CGColor {
    CGColor(srgbRed: CGFloat((hex >> 16) & 0xFF) / 255, green: CGFloat((hex >> 8) & 0xFF) / 255,
            blue: CGFloat(hex & 0xFF) / 255, alpha: alpha)
}

func drawIcon(in ctx: CGContext) {
    // macOS icon grid: 824 pt rounded square centered on a 1024 canvas.
    let tile = CGRect(x: 100, y: 100, width: 824, height: 824)
    let tilePath = CGPath(roundedRect: tile, cornerWidth: 185, cornerHeight: 185, transform: nil)

    ctx.saveGState()
    ctx.setShadow(offset: CGSize(width: 0, height: -12), blur: 28, color: color(0x000000, 0.35))
    ctx.addPath(tilePath)
    ctx.setFillColor(color(0x3D5BEA))
    ctx.fillPath()
    ctx.restoreGState()

    ctx.saveGState()
    ctx.addPath(tilePath)
    ctx.clip()
    let background = CGGradient(colorsSpace: CGColorSpace(name: CGColorSpace.sRGB),
                                colors: [color(0x5B9BFF), color(0x4A3FE0)] as CFArray, locations: [0, 1])!
    ctx.drawLinearGradient(background, start: CGPoint(x: 512, y: 924), end: CGPoint(x: 512, y: 100), options: [])
    // Soft highlight on the upper half.
    let glow = CGGradient(colorsSpace: CGColorSpace(name: CGColorSpace.sRGB),
                          colors: [color(0xFFFFFF, 0.18), color(0xFFFFFF, 0)] as CFArray, locations: [0, 1])!
    ctx.drawRadialGradient(glow, startCenter: CGPoint(x: 512, y: 900), startRadius: 0,
                           endCenter: CGPoint(x: 512, y: 900), endRadius: 620, options: [])
    ctx.restoreGState()

    // License plate.
    let plate = CGRect(x: 222, y: 382, width: 580, height: 250)
    let platePath = CGPath(roundedRect: plate, cornerWidth: 44, cornerHeight: 44, transform: nil)
    ctx.saveGState()
    ctx.setShadow(offset: CGSize(width: 0, height: -10), blur: 24, color: color(0x1A1060, 0.45))
    ctx.addPath(platePath)
    ctx.setFillColor(color(0xFFFFFF))
    ctx.fillPath()
    ctx.restoreGState()

    let rim = plate.insetBy(dx: 18, dy: 18)
    ctx.addPath(CGPath(roundedRect: rim, cornerWidth: 28, cornerHeight: 28, transform: nil))
    ctx.setStrokeColor(color(0x2B2F45))
    ctx.setLineWidth(9)
    ctx.strokePath()

    // Pixelated characters: a mosaic of gray blocks, the app's signature effect.
    let cols = 9, rows = 3
    let inner = rim.insetBy(dx: 30, dy: 32)
    let cell = CGSize(width: inner.width / CGFloat(cols), height: inner.height / CGFloat(rows))
    let shades: [UInt32] = [0x3A3F5C, 0x6B7194, 0x9AA0BF, 0x4E5578, 0xC3C7DC, 0x586084]
    var seed: UInt32 = 7
    for r in 0..<rows {
        for c in 0..<cols {
            seed = seed &* 1_103_515_245 &+ 12_345
            let shade = shades[Int((seed >> 16) % UInt32(shades.count))]
            let block = CGRect(x: inner.minX + CGFloat(c) * cell.width, y: inner.minY + CGFloat(r) * cell.height,
                               width: cell.width + 0.5, height: cell.height + 0.5)
            ctx.setFillColor(color(shade))
            ctx.fill(block)
        }
    }

    // Viewfinder corners around the plate: detection.
    let frame = plate.insetBy(dx: -58, dy: -62)
    let arm: CGFloat = 92
    ctx.setStrokeColor(color(0xFFFFFF, 0.95))
    ctx.setLineWidth(26)
    ctx.setLineCap(.round)
    ctx.setLineJoin(.round)
    for (x, y, dx, dy) in [(frame.minX, frame.minY, 1.0, 1.0), (frame.maxX, frame.minY, -1.0, 1.0),
                           (frame.minX, frame.maxY, 1.0, -1.0), (frame.maxX, frame.maxY, -1.0, -1.0)] {
        ctx.move(to: CGPoint(x: x, y: y + dy * arm))
        ctx.addLine(to: CGPoint(x: x, y: y))
        ctx.addLine(to: CGPoint(x: x + dx * arm, y: y))
    }
    ctx.strokePath()
}

func render(_ size: Int) -> CGImage {
    let ctx = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: 0,
                        space: CGColorSpace(name: CGColorSpace.sRGB)!,
                        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.interpolationQuality = .high
    ctx.scaleBy(x: CGFloat(size) / canvas, y: CGFloat(size) / canvas)
    drawIcon(in: ctx)
    return ctx.makeImage()!
}

func writePNG(_ image: CGImage, to url: URL) {
    let rep = NSBitmapImageRep(cgImage: image)
    try! rep.representation(using: .png, properties: [:])!.write(to: url)
}

let args = CommandLine.arguments
guard args.count >= 2 else {
    print("usage: make_icon.swift <output.icns> [preview.png]")
    exit(1)
}
let iconset = FileManager.default.temporaryDirectory.appendingPathComponent("AppIcon-\(UUID().uuidString).iconset")
try FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)
for points in [16, 32, 128, 256, 512] {
    writePNG(render(points), to: iconset.appendingPathComponent("icon_\(points)x\(points).png"))
    writePNG(render(points * 2), to: iconset.appendingPathComponent("icon_\(points)x\(points)@2x.png"))
}
if args.count >= 3 { writePNG(render(1024), to: URL(fileURLWithPath: args[2])) }

let iconutil = Process()
iconutil.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
iconutil.arguments = ["-c", "icns", iconset.path, "-o", args[1]]
try iconutil.run()
iconutil.waitUntilExit()
try? FileManager.default.removeItem(at: iconset)
print(iconutil.terminationStatus == 0 ? "Icon written to \(args[1])" : "iconutil failed")
