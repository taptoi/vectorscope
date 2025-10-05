import Foundation
import Metal
import MetalKit
import simd

// MARK: - Uniforms (shared with .metal)
struct VectorscopeUniforms {
    var gain: Float          // amplitude scale
    var sampleCount: UInt32  // number of vertices to draw
    var pointSize: Float     // in pixels
    var aspectScaleY: Float  // width/height for undistorted aspect
    var brightness: Float    // alpha for points
}

final class VectorscopeRenderer: NSObject, MTKViewDelegate {
    // GPU
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var pipeline: MTLRenderPipelineState!

    // Buffers (double-buffering to avoid races)
    private let maxSamples: Int
    private var leftBuffers: [MTLBuffer] = []
    private var rightBuffers: [MTLBuffer] = []
    private var uniformsBuffers: [MTLBuffer] = []
    private var writeBufferIndex = 0

    // State
    var gain: Float = 1.0
    var pointSize: Float = 2.0
    var brightness: Float = 0.9
    private(set) var currentSampleCount: Int = 0
    private var aspectScaleY: Float = 1.0
    
    // How many recent samples to draw per frame (clamped to maxSamples)
    var samplesToDraw: Int = 16384

    // Audio
    private let audio: StereoAudioSource
    
    // Debug logging properties
    private var lastDrawTime: CFTimeInterval = 0
    private var frameCount = 0
    private var totalSamples = 0

    // MARK: - Init
    init(mtkView: MTKView, audio: StereoAudioSource, maxSamples: Int = 16384) {
        guard let dev = mtkView.device ?? MTLCreateSystemDefaultDevice() else {
            fatalError("Metal device not available")
        }
        self.device = dev
        self.commandQueue = dev.makeCommandQueue()!
        self.audio = audio
        self.maxSamples = maxSamples

        super.init()

        configureView(mtkView)
        buildPipeline(mtkView: mtkView)
        allocateBuffers()
    }

    private func configureView(_ view: MTKView) {
        view.device = device
        view.colorPixelFormat = .bgra8Unorm
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        view.sampleCount = 1
        view.framebufferOnly = true
        view.isPaused = false
        view.enableSetNeedsDisplay = false
        view.preferredFramesPerSecond = 60
        view.delegate = self
    }

    private func buildPipeline(mtkView: MTKView) {
        let library = try! device.makeDefaultLibrary(bundle: .main)
        let vfn = library.makeFunction(name: "vectorscope_vertex")!
        let ffn = library.makeFunction(name: "vectorscope_fragment")!

        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = vfn
        desc.fragmentFunction = ffn
        desc.colorAttachments[0].pixelFormat = mtkView.colorPixelFormat
        // Optional: slightly additive look
        desc.colorAttachments[0].isBlendingEnabled = true
        desc.colorAttachments[0].rgbBlendOperation = .add
        desc.colorAttachments[0].alphaBlendOperation = .add
        desc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        desc.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        desc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        desc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha

        self.pipeline = try! device.makeRenderPipelineState(descriptor: desc)
    }

    private func allocateBuffers() {
        func makeBuffer(byteCount: Int) -> MTLBuffer {
            device.makeBuffer(length: byteCount, options: [.storageModeShared])!
        }
        let sampleBytes = maxSamples * MemoryLayout<Float>.stride
        for _ in 0..<2 {
            leftBuffers.append(makeBuffer(byteCount: sampleBytes))
            rightBuffers.append(makeBuffer(byteCount: sampleBytes))
            uniformsBuffers.append(makeBuffer(byteCount: MemoryLayout<VectorscopeUniforms>.stride))
        }
    }

    // MARK: - MTKViewDelegate
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // scale Y to keep circles undistorted in non-square viewports
        aspectScaleY = size.width > 0 ? Float(size.width / size.height) : 1.0
    }

    func draw(in view: MTKView) {
        guard let rpd = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable else { return }

        // Add debug logging for frame rate analysis
        let currentTime = CACurrentMediaTime()
        
        frameCount += 1

        // 1) Pull latest audio samples into the *write* buffers
        let lb = leftBuffers[writeBufferIndex]
        let rb = rightBuffers[writeBufferIndex]
        let lPtr = lb.contents().bindMemory(to: Float.self, capacity: maxSamples)
        let rPtr = rb.contents().bindMemory(to: Float.self, capacity: maxSamples)
        
        // Draw a larger recent window for denser visuals
        let requested = min(maxSamples, max(0, samplesToDraw))
        let got = audio.copyLatest(into: lPtr, rightOut: rPtr, maxOut: requested)
        currentSampleCount = got
        totalSamples += got
        
        // Log statistics every second
        if currentTime - lastDrawTime > 1.0 {
            print("=== Vectorscope Debug Stats ===")
            print("Draw calls per second: \(frameCount)")
            print("Average samples per frame: \(totalSamples / frameCount)")
            print("Current sample count: \(got)")
            print("Point size: \(pointSize), Gain: \(gain)")
            print("===============================")
            frameCount = 0
            totalSamples = 0
            lastDrawTime = currentTime
        }

        // 2) Update uniforms for current frame
        var u = VectorscopeUniforms(
            gain: gain,
            sampleCount: UInt32(got),
            pointSize: pointSize,
            aspectScaleY: aspectScaleY,
            brightness: brightness
        )
        let ub = uniformsBuffers[writeBufferIndex]
        memcpy(ub.contents(), &u, MemoryLayout<VectorscopeUniforms>.stride)

        // 3) Encode draw (point list)
        let cb = commandQueue.makeCommandBuffer()!
        let enc = cb.makeRenderCommandEncoder(descriptor: rpd)!
        enc.setRenderPipelineState(pipeline)
        enc.setVertexBuffer(lb, offset: 0, index: 0)  // left channel
        enc.setVertexBuffer(rb, offset: 0, index: 1)  // right channel
        enc.setVertexBuffer(ub, offset: 0, index: 2)  // uniforms

        if got > 0 {
            enc.drawPrimitives(type: .point, vertexStart: 0, vertexCount: got)
        }

        enc.endEncoding()
        cb.present(drawable)
        cb.commit()

        // 4) Swap write buffer for next frame
        writeBufferIndex = 1 - writeBufferIndex
    }
}

// MARK: - Minimal helper to wire up MTKView + renderer
public final class VectorscopeMTKView: MTKView {
    private var renderer: VectorscopeRenderer?
    private let audio = StereoAudioSource(maxSamples: 65536)

    /// Start with microphone or a file URL
    public func start(source: StereoAudioSource.Source) throws {
        // Stop any previous session
        stop()
        isPaused = false
        try audio.start(source)
        renderer = VectorscopeRenderer(mtkView: self, audio: audio, maxSamples: 163840)
        // Render a larger set of points by default (clamped by ring size)
        renderer?.samplesToDraw = min(audio.maxSamplesInRing, 131072)
    }

    public func stop() {
        audio.stop()
        isPaused = true
        delegate = nil
        renderer = nil
    }

    // Expose some controls
    public var gain: Float {
        get { renderer?.gain ?? 1.0 }
        set { renderer?.gain = newValue }
    }

    public var pointSize: Float {
        get { renderer?.pointSize ?? 2.0 }
        set { renderer?.pointSize = newValue }
    }

    public var brightness: Float {
        get { renderer?.brightness ?? 0.9 }
        set { renderer?.brightness = newValue }
    }

    public var samplesToDraw: Int {
        get { renderer?.samplesToDraw ?? 16384 }
        set { renderer?.samplesToDraw = newValue }
    }
}
