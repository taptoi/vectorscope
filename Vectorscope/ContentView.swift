import SwiftUI
import MetalKit
import AVFoundation
#if os(macOS)
import AppKit
import UniformTypeIdentifiers
#endif

// MARK: - SwiftUI wrapper hosting the Metal vectorscope view (macOS)
#if os(macOS)
struct VectorscopeView: NSViewRepresentable {
    enum Source: Equatable {
        case microphone
        case file(URL)
    }

    @Binding var source: Source
    @Binding var gain: Float
    @Binding var pointSize: Float
    @Binding var brightness: Float
    @Binding var samplesToDraw: Int

    final class Coordinator {
        var view: VectorscopeMTKView?
        var lastSource: Source?
    }
    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> MTKView {
        let metalDevice = MTLCreateSystemDefaultDevice()
        let v = VectorscopeMTKView(frame: .zero, device: metalDevice)
        context.coordinator.view = v
        // Start once with initial source
        startIfNeeded(context: context, view: v)
        applyParams(view: v)
        return v
    }

    func updateNSView(_ nsView: MTKView, context: Context) {
        guard let v = context.coordinator.view else { return }
        startIfNeeded(context: context, view: v)
        applyParams(view: v)
    }

    private func startIfNeeded(context: Context, view: VectorscopeMTKView) {
        if context.coordinator.lastSource != source {
            switch source {
            case .microphone:
                try? view.start(source: .microphone)
            case .file(let url):
                try? view.start(source: .file(url))
            }
            context.coordinator.lastSource = source
        }
    }

    private func applyParams(view: VectorscopeMTKView) {
        view.gain = gain
        view.pointSize = pointSize
        view.brightness = brightness
        view.samplesToDraw = samplesToDraw
    }
}
#endif

struct ContentView: View {
    #if os(macOS)
    @State private var source: VectorscopeView.Source = .microphone
    #endif
    @State private var gain: Float = 1.2
    @State private var pointSize: Float = 2.0
    @State private var brightness: Float = 0.9
    @State private var samplesToDraw: Int = 65536

    var body: some View {
        VStack(spacing: 8) {
            #if os(macOS)
            HStack(spacing: 12) {
                Button("Microphone") { source = .microphone }
                Button("Open Audio File…") { pickAudioFile() }
                Spacer()
                HStack(spacing: 6) {
                    Text("Gain").font(.caption)
                    Slider(value: Binding(get: { Double(gain) }, set: { gain = Float($0) }), in: 0.1...20.0)
                        .frame(width: 140)
                }
                HStack(spacing: 6) {
                    Text("Point").font(.caption)
                    Slider(value: Binding(get: { Double(pointSize) }, set: { pointSize = Float($0) }), in: 1.0...8.0)
                        .frame(width: 120)
                }
                HStack(spacing: 6) {
                    Text("Alpha").font(.caption)
                    Slider(value: Binding(get: { Double(brightness) }, set: { brightness = Float($0) }), in: 0.05...1.0)
                        .frame(width: 120)
                }
                HStack(spacing: 6) {
                    Text("Samples").font(.caption)
                    Slider(value: Binding(get: { Double(samplesToDraw) }, set: { samplesToDraw = Int($0) }), in: 1024...131072)
                        .frame(width: 200)
                }
            }
            .padding(.horizontal)

            VectorscopeView(source: $source, gain: $gain, pointSize: $pointSize, brightness: $brightness, samplesToDraw: $samplesToDraw)
                .background(Color.black)
            #else
            Text("This demo currently targets macOS.")
            #endif
        }
        .padding(.vertical, 8)
    }

    #if os(macOS)
    private func pickAudioFile() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        let exts = ["wav", "aiff", "aif", "mp3", "m4a", "aac", "caf"]
        panel.allowedContentTypes = exts.compactMap { UTType(filenameExtension: $0) }
        panel.begin { resp in
            if resp == .OK, let url = panel.url {
                source = .file(url)
            }
        }
    }
    #endif
}

#Preview {
    ContentView()
}
