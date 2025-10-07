import Foundation
import Metal
import MetalKit
import simd

// MARK: - Uniforms (shared with .metal)
struct VectorscopeUniforms {
    var viewProjection: simd_float4x4
    var misc: SIMD4<Float> // x: brightness, y: point size, z: sampleCount, w: unused
}

struct AudioLiftParams {
    var sampleCount: UInt32
    var tauSamples: UInt32
    var scaleXY: SIMD2<Float>
    var scaleZ: Float
    var pad0: Float = 0
    var offset: SIMD3<Float>
    var pad1: Float = 0
}

struct LineExpandUniforms {
    var texelSize: SIMD2<Float>
    var lineWidth: Float
    var intensityScale: Float
}

final class VectorscopeRenderer: NSObject, MTKViewDelegate {
    // GPU
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var pipeline: MTLRenderPipelineState!
    private var computePipeline: MTLComputePipelineState!
    private var postPipeline: MTLRenderPipelineState!

    // Buffers (double-buffering to avoid races)
    private let maxSamples: Int
    private var leftBuffers: [MTLBuffer] = []
    private var rightBuffers: [MTLBuffer] = []
    private var uniformsBuffers: [MTLBuffer] = []
    private var positionsBuffers: [MTLBuffer] = []
    private var liftParamsBuffers: [MTLBuffer] = []
    private var postUniformBuffers: [MTLBuffer] = []
    private var writeBufferIndex = 0

    // State
    var gain: Float = 1.0
    var pointSize: Float = 2.0
    var brightness: Float = 0.9
    private(set) var currentSampleCount: Int = 0
    var zGain: Float = 1.0
    var timeLagMilliseconds: Float = 2.0
    var liftOffset = SIMD3<Float>(repeating: 0)
    private var viewProjectionMatrix = matrix_identity_float4x4
    private var lineTexture: MTLTexture?

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
        updateViewProjection(for: view.drawableSize)
        ensureLineTexture(for: view.drawableSize)
    }

    private func buildPipeline(mtkView: MTKView) {
        let library = try! device.makeDefaultLibrary(bundle: .main)
        let vfn = library.makeFunction(name: "vectorscope_lift_vertex")!
        let ffn = library.makeFunction(name: "vectorscope_fragment")!
        let cfn = library.makeFunction(name: "liftStereoTimeLag")!
        let postVfn = library.makeFunction(name: "line_expand_vertex")!
        let postFfn = library.makeFunction(name: "line_expand_fragment")!

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
        self.computePipeline = try! device.makeComputePipelineState(function: cfn)

        let postDesc = MTLRenderPipelineDescriptor()
        postDesc.vertexFunction = postVfn
        postDesc.fragmentFunction = postFfn
        postDesc.colorAttachments[0].pixelFormat = mtkView.colorPixelFormat
        postDesc.colorAttachments[0].isBlendingEnabled = false
        self.postPipeline = try! device.makeRenderPipelineState(descriptor: postDesc)
    }

    private func allocateBuffers() {
        func makeBuffer(byteCount: Int) -> MTLBuffer {
            device.makeBuffer(length: byteCount, options: [.storageModeShared])!
        }
        let sampleBytes = maxSamples * MemoryLayout<Float>.stride
        let positionBytes = maxSamples * MemoryLayout<SIMD3<Float>>.stride
        for _ in 0..<2 {
            leftBuffers.append(makeBuffer(byteCount: sampleBytes))
            rightBuffers.append(makeBuffer(byteCount: sampleBytes))
            uniformsBuffers.append(makeBuffer(byteCount: MemoryLayout<VectorscopeUniforms>.stride))
            positionsBuffers.append(device.makeBuffer(length: positionBytes, options: [.storageModePrivate])!)
            liftParamsBuffers.append(makeBuffer(byteCount: MemoryLayout<AudioLiftParams>.stride))
            postUniformBuffers.append(makeBuffer(byteCount: MemoryLayout<LineExpandUniforms>.stride))
        }
    }

    private func updateViewProjection(for size: CGSize) {
        let aspect = size.height > 0 ? Float(size.width / size.height) : 1.0
        let projection = simd_float4x4.perspective(fovY: .pi / 3, aspect: aspect, nearZ: 0.01, farZ: 20.0)
        let rotateX = simd_float4x4(rotation: -.pi / 3.2, axis: SIMD3<Float>(1, 0, 0), frame: frameCount)
        let rotateZ = simd_float4x4(rotation: .pi / 6, axis: SIMD3<Float>(0, 0, 1), frame: frameCount)
        let translate = simd_float4x4(translation: SIMD3<Float>(0, 0, -3.0), frame: frameCount)
        viewProjectionMatrix = projection * translate * rotateX * rotateZ
    }

    private func ensureLineTexture(for size: CGSize) {
        let width = max(Int(size.width), 1)
        let height = max(Int(size.height), 1)
        guard width > 0 && height > 0 else {
            lineTexture = nil
            return
        }

        if let tex = lineTexture, tex.width == width && tex.height == height {
            return
        }

        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm,
                                                            width: width,
                                                            height: height,
                                                            mipmapped: false)
        desc.usage = [.renderTarget, .shaderRead]
        desc.storageMode = .private
        lineTexture = device.makeTexture(descriptor: desc)
    }

    // MARK: - MTKViewDelegate
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        updateViewProjection(for: size)
        ensureLineTexture(for: size)
    }

    func draw(in view: MTKView) {
        guard let rpd = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable else { return }

        updateViewProjection(for: view.drawableSize)
        ensureLineTexture(for: view.drawableSize)
        guard let lineTexture = lineTexture else { return }

        // Add debug logging for frame rate analysis
        let currentTime = CACurrentMediaTime()

        frameCount += 1

        // 1) Pull latest audio samples into the *write* buffers
        let lb = leftBuffers[writeBufferIndex]
        let rb = rightBuffers[writeBufferIndex]
        let lPtr = lb.contents().bindMemory(to: Float.self, capacity: maxSamples)
        let rPtr = rb.contents().bindMemory(to: Float.self, capacity: maxSamples)

        let requested = min(maxSamples, max(0, samplesToDraw))
        let got = audio.copyLatest(into: lPtr, rightOut: rPtr, maxOut: requested)
        currentSampleCount = got
        totalSamples += got

        let sampleRate = max(Float(audio.currentSampleRate), 1.0)
        let tauSamples = UInt32(max(1, Int((timeLagMilliseconds / 1000.0) * sampleRate)))

        // Log statistics every second
        if currentTime - lastDrawTime > 1.0 {
            print("=== Vectorscope Debug Stats ===")
            print("Draw calls per second: \(frameCount)")
            print("Average samples per frame: \(frameCount > 0 ? totalSamples / frameCount : 0)")
            print("Current sample count: \(got)")
            print("Gain XY: \(gain), Gain Z: \(zGain), τ(samples): \(tauSamples)")
            print("===============================")
            totalSamples = 0
            lastDrawTime = currentTime
        }

        // 2) Update lift params + uniforms for current frame
        let paramsBuffer = liftParamsBuffers[writeBufferIndex]
        if got > 0 {
            var params = AudioLiftParams(
                sampleCount: UInt32(got),
                tauSamples: tauSamples,
                scaleXY: SIMD2<Float>(gain, gain),
                scaleZ: zGain,
                offset: liftOffset
            )
            memcpy(paramsBuffer.contents(), &params, MemoryLayout<AudioLiftParams>.stride)
        }

        var uniforms = VectorscopeUniforms(
            viewProjection: viewProjectionMatrix,
            misc: SIMD4<Float>(brightness, pointSize, Float(got), Float(frameCount))
        )
        let uniformsBuffer = uniformsBuffers[writeBufferIndex]
        memcpy(uniformsBuffer.contents(), &uniforms, MemoryLayout<VectorscopeUniforms>.stride)

        // 3) Encode compute + render
        let cb = commandQueue.makeCommandBuffer()!

        if got > 0 {
            let compute = cb.makeComputeCommandEncoder()!
            compute.setComputePipelineState(computePipeline)
            compute.setBuffer(lb, offset: 0, index: 0)
            compute.setBuffer(rb, offset: 0, index: 1)
            compute.setBuffer(positionsBuffers[writeBufferIndex], offset: 0, index: 2)
            compute.setBuffer(paramsBuffer, offset: 0, index: 3)
            compute.setBuffer(uniformsBuffer, offset: 0, index: 4)

            let threadWidth = computePipeline.threadExecutionWidth
            let threadsPerGroup = MTLSize(width: min(threadWidth, max(1, got)), height: 1, depth: 1)
            let threads = MTLSize(width: got, height: 1, depth: 1)
            compute.dispatchThreads(threads, threadsPerThreadgroup: threadsPerGroup)
            compute.endEncoding()
        }

        let offscreenPass = MTLRenderPassDescriptor()
        offscreenPass.colorAttachments[0].texture = lineTexture
        offscreenPass.colorAttachments[0].loadAction = .clear
        offscreenPass.colorAttachments[0].storeAction = .store
        offscreenPass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)

        if let lineEncoder = cb.makeRenderCommandEncoder(descriptor: offscreenPass) {
            lineEncoder.setRenderPipelineState(pipeline)
            lineEncoder.setVertexBuffer(positionsBuffers[writeBufferIndex], offset: 0, index: 0)
            lineEncoder.setVertexBuffer(uniformsBuffer, offset: 0, index: 1)

            if got >= 2 {
                lineEncoder.drawPrimitives(type: .lineStrip, vertexStart: 0, vertexCount: got)
            } else if got == 1 {
                lineEncoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: 1)
            }

            lineEncoder.endEncoding()
        }

        var postUniforms = LineExpandUniforms(
            texelSize: SIMD2<Float>(1.0 / max(Float(lineTexture.width), 1.0),
                                    1.0 / max(Float(lineTexture.height), 1.0)),
            lineWidth: max(pointSize, 0.1),
            intensityScale: brightness
        )
        memcpy(postUniformBuffers[writeBufferIndex].contents(), &postUniforms, MemoryLayout<LineExpandUniforms>.stride)

        if let postEncoder = cb.makeRenderCommandEncoder(descriptor: rpd) {
            postEncoder.setRenderPipelineState(postPipeline)
            postEncoder.setFragmentTexture(lineTexture, index: 0)
            postEncoder.setFragmentBuffer(postUniformBuffers[writeBufferIndex], offset: 0, index: 0)
            postEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            postEncoder.endEncoding()
        }

        cb.present(drawable)
        cb.commit()

        // 4) Swap write buffer for next frame
        writeBufferIndex = 1 - writeBufferIndex
    }
}

// MARK: - Minimal helper to wire up MTKView + renderer
public final class VectorscopeMTKView: MTKView {
    private var renderer: VectorscopeRenderer?
    private let audio = StereoAudioSource(maxSamples: 262144)

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
        set {
            renderer?.gain = newValue
            renderer?.zGain = newValue
        }
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

private extension simd_float4x4 {
    static func perspective(fovY: Float, aspect: Float, nearZ: Float, farZ: Float) -> simd_float4x4 {
        let yScale = 1 / tan(fovY * 0.1)
        let xScale = yScale / max(aspect, 0.001)
        let zRange = farZ - nearZ
        let zScale = -(farZ + nearZ) / zRange
        let wz = -2 * farZ * nearZ / zRange
        return simd_float4x4(columns: (
            SIMD4<Float>(xScale, 0, 0, 0),
            SIMD4<Float>(0, yScale, 0, 0),
            SIMD4<Float>(0, 0, zScale, -1),
            SIMD4<Float>(0, 0, wz, 0)
        ))
    }

    init(rotation angle: Float, axis: SIMD3<Float>, frame: Int) {
        let a = simd_normalize(axis)
        let f = Float(frame)
        let c = cos(angle)
        let s = sin(angle)
        let t = 1 - c
        let x = a.x, y = a.y, z = a.z
        self.init(columns: (
            SIMD4<Float>(t * x * x + c,     t * x * y + s * z, t * x * z - s * y, 0),
            SIMD4<Float>(t * x * y - s * z, t * y * y + c,     t * y * z + s * x, 0),
            SIMD4<Float>(t * x * z + s * y, t * y * z - s * x, t * z * z + c,     0),
            SIMD4<Float>(0,                 0,                 0,                 1)
        ))
    }

    init(translation t: SIMD3<Float>, frame: Int) {
        self = matrix_identity_float4x4
        let f = Float(frame)
        self.columns.3 = SIMD4<Float>(t.x, t.y, t.z, 1)
    }
}
