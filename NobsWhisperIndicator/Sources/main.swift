import AppKit
import Foundation

// MARK: - Indicator View

class IndicatorView: NSView {
    enum Status: String {
        case recording
        case processing
        case copied
    }

    var status: Status = .recording {
        didSet {
            needsDisplay = true
            updateBlinkTimer()
        }
    }

    // Blink state for animation
    private var blinkOn: Bool = true
    private var blinkTimer: Timer?

    private func updateBlinkTimer() {
        // Stop existing timer
        blinkTimer?.invalidate()
        blinkTimer = nil

        // Only blink for recording and processing states
        if status == .recording || status == .processing {
            blinkTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
                self?.blinkOn.toggle()
                self?.needsDisplay = true
            }
        } else {
            blinkOn = true
            needsDisplay = true
        }
    }

    func stopBlinking() {
        blinkTimer?.invalidate()
        blinkTimer = nil
    }

    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)

        // Background with rounded corners
        let bgColor = NSColor(white: 0.1, alpha: 0.9)
        let bgPath = NSBezierPath(roundedRect: bounds, xRadius: 12, yRadius: 12)
        bgColor.setFill()
        bgPath.fill()

        // Status indicator circle
        let circleSize: CGFloat = 12
        let circleRect = NSRect(
            x: 16,
            y: (bounds.height - circleSize) / 2,
            width: circleSize,
            height: circleSize
        )

        let circleColor: NSColor
        let text: String

        switch status {
        case .recording:
            circleColor = NSColor(red: 0.9, green: 0.3, blue: 0.3, alpha: blinkOn ? 1.0 : 0.3)
            text = "Recording..."
        case .processing:
            circleColor = NSColor(red: 0.3, green: 0.6, blue: 0.9, alpha: blinkOn ? 1.0 : 0.3)
            text = "Processing..."
        case .copied:
            circleColor = NSColor(red: 0.3, green: 0.8, blue: 0.4, alpha: 1.0)
            text = "Copied to clipboard"
        }

        let circlePath = NSBezierPath(ovalIn: circleRect)
        circleColor.setFill()
        circlePath.fill()

        // Status text
        let attributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 13, weight: .medium),
            .foregroundColor: NSColor.white
        ]
        let textSize = text.size(withAttributes: attributes)
        let textRect = NSRect(
            x: circleRect.maxX + 10,
            y: (bounds.height - textSize.height) / 2,
            width: textSize.width,
            height: textSize.height
        )
        text.draw(in: textRect, withAttributes: attributes)
    }
}

// MARK: - Indicator Panel

class IndicatorPanel: NSPanel {
    private let indicatorView = IndicatorView()

    init() {
        let panelWidth: CGFloat = 200
        let panelHeight: CGFloat = 40

        // Get screen size for positioning
        let screenFrame = NSScreen.main?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1920, height: 1080)
        let x = screenFrame.midX - panelWidth / 2
        let y = screenFrame.minY + 60

        let contentRect = NSRect(x: x, y: y, width: panelWidth, height: panelHeight)

        super.init(
            contentRect: contentRect,
            styleMask: [.nonactivatingPanel, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )

        // Configure panel for floating above everything including fullscreen
        self.isFloatingPanel = true
        self.level = .screenSaver + 1
        self.collectionBehavior = [
            .canJoinAllSpaces,
            .fullScreenAuxiliary,
            .ignoresCycle
        ]
        self.isOpaque = false
        self.backgroundColor = .clear
        self.hasShadow = true
        self.hidesOnDeactivate = false
        self.worksWhenModal = true
        self.becomesKeyOnlyIfNeeded = true
        self.isMovableByWindowBackground = false

        // Set up content view
        indicatorView.frame = NSRect(x: 0, y: 0, width: panelWidth, height: panelHeight)
        self.contentView = indicatorView
    }

    func setStatus(_ status: IndicatorView.Status) {
        indicatorView.status = status
    }

    func updatePosition() {
        guard let screen = NSScreen.main else { return }
        let screenFrame = screen.visibleFrame
        let panelWidth = frame.width
        let x = screenFrame.midX - panelWidth / 2
        let y = screenFrame.minY + 60
        setFrameOrigin(NSPoint(x: x, y: y))
    }
}

// MARK: - App Delegate

class AppDelegate: NSObject, NSApplicationDelegate {
    private var panel: IndicatorPanel?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Create the indicator panel
        panel = IndicatorPanel()

        // Register for distributed notifications from Tauri app
        let center = DistributedNotificationCenter.default()

        center.addObserver(
            self,
            selector: #selector(showIndicator(_:)),
            name: NSNotification.Name("com.nobswhisper.indicator.show"),
            object: nil
        )

        center.addObserver(
            self,
            selector: #selector(hideIndicator(_:)),
            name: NSNotification.Name("com.nobswhisper.indicator.hide"),
            object: nil
        )

        center.addObserver(
            self,
            selector: #selector(setStatus(_:)),
            name: NSNotification.Name("com.nobswhisper.indicator.status"),
            object: nil
        )

        center.addObserver(
            self,
            selector: #selector(terminateApp(_:)),
            name: NSNotification.Name("com.nobswhisper.indicator.terminate"),
            object: nil
        )

        NSLog("NobsWhisperIndicator: Ready and listening for notifications")
    }

    @objc func showIndicator(_ notification: Notification) {
        DispatchQueue.main.async { [weak self] in
            guard let panel = self?.panel else { return }
            panel.updatePosition()
            panel.orderFront(nil)
            NSLog("NobsWhisperIndicator: Shown")
        }
    }

    @objc func hideIndicator(_ notification: Notification) {
        DispatchQueue.main.async { [weak self] in
            self?.panel?.orderOut(nil)
            NSLog("NobsWhisperIndicator: Hidden")
        }
    }

    @objc func setStatus(_ notification: Notification) {
        DispatchQueue.main.async { [weak self] in
            if let statusString = notification.userInfo?["status"] as? String,
               let status = IndicatorView.Status(rawValue: statusString) {
                self?.panel?.setStatus(status)
                NSLog("NobsWhisperIndicator: Status set to \(statusString)")
            }
        }
    }

    @objc func terminateApp(_ notification: Notification) {
        NSLog("NobsWhisperIndicator: Terminating")
        NSApplication.shared.terminate(nil)
    }

    func applicationWillTerminate(_ notification: Notification) {
        DistributedNotificationCenter.default().removeObserver(self)
    }
}

// MARK: - Main

let app = NSApplication.shared
app.setActivationPolicy(.accessory) // No dock icon, no menu bar
let delegate = AppDelegate()
app.delegate = delegate
app.run()
