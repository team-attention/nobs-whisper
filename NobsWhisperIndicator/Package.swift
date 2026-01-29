// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "NobsWhisperIndicator",
    platforms: [.macOS(.v10_15)],
    targets: [
        .executableTarget(
            name: "NobsWhisperIndicator",
            path: "Sources"
        )
    ]
)
