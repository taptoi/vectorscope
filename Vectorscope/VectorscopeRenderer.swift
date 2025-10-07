import Foundation
import Metal
import MetalKit
import simd

// MARK: - Uniforms (shared with .metal)
struct VectorscopeUniforms {
    var viewProjection: simd_float4x4
    var misc: SIMD4<Float> // x: brightness, y: point size, z: particleCount, w: unused
    var audioBands: SIMD4<Float>
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

struct Particle {
    var posLife: SIMD4<Float>
    var velLife: SIMD4<Float>
    var misc: SIMD4<Float>
}

struct ParticleUpdateUniforms {
    var counts: SIMD2<UInt32>
    var deltaTime: Float
    var emissionRate: Float
    var baseLifetime: Float
    var damping: Float
    var transient: Float
    var spawnJitter: Float
    var velocityScale: Float
    var bandLowMid: SIMD2<Float>
    var bandHighTime: SIMD2<Float>
}

final class VectorscopeRenderer: NSObject, MTKViewDelegate {
    // GPU
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var renderPipeline: MTLRenderPipelineState!
    private var liftPipeline: MTLComputePipelineState!
    private var particlePipeline: MTLComputePipelineState!

    // Buffers (double-buffering to avoid races)
    private let maxSamples: Int
    private var leftBuffers: [MTLBuffer] = []
    private var rightBuffers: [MTLBuffer] = []
    private var uniformsBuffers: [MTLBuffer] = []
    private var emitterBuffers: [MTLBuffer] = []
    private var liftParamsBuffers: [MTLBuffer] = []
    private var particleParamsBuffers: [MTLBuffer] = []
    private var particleBuffer: MTLBuffer!
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

    // Particles
    private let maxParticles: Int = 65536
    private var particleLifetime: Float = 5.0
    private var particleEmissionRate: Float = 1.2
    private var particleVelocityScale: Float = 2.8
    private var particleDamping: Float = 0.35
    private var particleJitter: Float = 0.35

    // Audio dynamics
    private var smoothedPower: Float = 0
    private var lowState: Float = 0
    private var prevSample: Float = 0
    private var lastFrameTimestamp: CFTimeInterval?

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
    }

    private func buildPipeline(mtkView: MTKView) {
        let library = try! device.makeDefaultLibrary(bundle: .main)
        let vfn = library.makeFunction(name: "particle_vertex")!
        let ffn = library.makeFunction(name: "particle_fragment")!
        let liftFn = library.makeFunction(name: "liftStereoTimeLag")!
        let particleFn = library.makeFunction(name: "updateParticles")!

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

        self.renderPipeline = try! device.makeRenderPipelineState(descriptor: desc)
        self.liftPipeline = try! device.makeComputePipelineState(function: liftFn)
        self.particlePipeline = try! device.makeComputePipelineState(function: particleFn)
    }

    private func allocateBuffers() {
        func makeBuffer(byteCount: Int) -> MTLBuffer {
            device.makeBuffer(length: byteCount, options: [.storageModeShared])!
        }
        let sampleBytes = maxSamples * MemoryLayout<Float>.stride
        let emitterBytes = maxSamples * MemoryLayout<SIMD4<Float>>.stride
        for _ in 0..<2 {
            leftBuffers.append(makeBuffer(byteCount: sampleBytes))
            rightBuffers.append(makeBuffer(byteCount: sampleBytes))
            uniformsBuffers.append(makeBuffer(byteCount: MemoryLayout<VectorscopeUniforms>.stride))
            emitterBuffers.append(device.makeBuffer(length: emitterBytes, options: [.storageModePrivate])!)
            liftParamsBuffers.append(makeBuffer(byteCount: MemoryLayout<AudioLiftParams>.stride))
            particleParamsBuffers.append(makeBuffer(byteCount: MemoryLayout<ParticleUpdateUniforms>.stride))
        }

        let particleBytes = maxParticles * MemoryLayout<Particle>.stride
        guard let pBuffer = device.makeBuffer(length: particleBytes, options: [.storageModeShared]) else {
            fatalError("Unable to allocate particle buffer")
        }
        particleBuffer = pBuffer

        let basePtr = particleBuffer.contents().bindMemory(to: Particle.self, capacity: maxParticles)
        for i in 0..<maxParticles {
            let seed = Float(i) * 0.1234
            basePtr[i] = Particle(
                posLife: SIMD4<Float>(0, 0, 0, 0),
                velLife: SIMD4<Float>(0, 0, 0, particleLifetime),
                misc: SIMD4<Float>(seed, 0, 0, 0)
            )
        }
    }

    private func updateViewProjection(for size: CGSize) {
        let aspect = size.height > 0 ? Float(size.width / size.height) : 1.0
        let projection = simd_float4x4.perspective(fovY: .pi / 3, aspect: aspect, nearZ: 0.01, farZ: 20.0)
        let rotateX = simd_float4x4(rotation: -.pi / 3.2, axis: SIMD3<Float>(1, 0, 0))
        let rotateZ = simd_float4x4(rotation: .pi / 6, axis: SIMD3<Float>(0, 0, 1))
        let translate = simd_float4x4(translation: SIMD3<Float>(0, 0, -3.0))
        viewProjectionMatrix = projection * translate * rotateX * rotateZ
    }

    // MARK: - MTKViewDelegate
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        updateViewProjection(for: size)
    }

    func draw(in view: MTKView) {
        guard let rpd = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable else { return }

        updateViewProjection(for: view.drawableSize)

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

        let deltaTime: Float
        if let last = lastFrameTimestamp {
            deltaTime = Float(min(max(currentTime - last, 0.0), 0.25))
        } else {
            deltaTime = 1.0 / 60.0
        }
        lastFrameTimestamp = currentTime

        // Log statistics every second
        if currentTime - lastDrawTime > 1.0 {
            print("=== Vectorscope Debug Stats ===")
            print("Draw calls per second: \(frameCount)")
            print("Average samples per frame: \(frameCount > 0 ? totalSamples / frameCount : 0)")
            print("Current sample count: \(got)")
            print("Gain XY: \(gain), Gain Z: \(zGain), τ(samples): \(tauSamples)")
            print("===============================")
            frameCount = 0
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

        // 2b) Analyse audio bands for artistic modulation
        var lowEnergy: Float = 0
        var midEnergy: Float = 0
        var highEnergy: Float = 0
        var totalPower: Float = 0
        var lowStateLocal = lowState
        var prevMono = prevSample

        if got > 0 {
            let alphaLow: Float = 0.025
            for i in 0..<got {
                let mono = 0.5 * (lPtr[i] + rPtr[i])
                lowStateLocal += (mono - lowStateLocal) * alphaLow
                let lowVal = lowStateLocal
                let midVal = mono - lowStateLocal
                let highVal = mono - prevMono
                prevMono = mono

                lowEnergy += abs(lowVal)
                midEnergy += abs(midVal)
                highEnergy += abs(highVal)
                totalPower += abs(mono)
            }
            lowState = lowStateLocal
            prevSample = prevMono
        } else {
            lowState *= 0.98
            prevSample *= 0.98
        }

        let invCount = got > 0 ? 1.0 / Float(got) : 0
        var bands = SIMD3<Float>(lowEnergy * invCount, midEnergy * invCount, highEnergy * invCount)
        var power = totalPower * invCount

        if got == 0 {
            power = smoothedPower * 0.95
            bands = SIMD3<Float>(repeating: 0)
        }

        let smoothing: Float = 0.12
        smoothedPower += (power - smoothedPower) * smoothing
        let transient = max(0, power - smoothedPower) * 6.0
        let clampedTransient = min(transient, 4.0)

        let maxBand = max(max(bands.x, bands.y), max(bands.z, 1e-6))
        if maxBand > 0 {
            bands /= maxBand
        }
        bands = simd_clamp(bands, SIMD3<Float>(repeating: 0), SIMD3<Float>(repeating: 1))
        bands.x = powf(bands.x, 0.8)
        bands.y = powf(bands.y, 0.8)
        bands.z = powf(bands.z, 0.8)

        var uniforms = VectorscopeUniforms(
            viewProjection: viewProjectionMatrix,
            misc: SIMD4<Float>(brightness, pointSize, Float(maxParticles), 0),
            audioBands: SIMD4<Float>(bands.x, bands.y, bands.z, 0)
        )
        let uniformsBuffer = uniformsBuffers[writeBufferIndex]
        memcpy(uniformsBuffer.contents(), &uniforms, MemoryLayout<VectorscopeUniforms>.stride)

        var particleUniforms = ParticleUpdateUniforms(
            counts: SIMD2<UInt32>(UInt32(maxParticles), UInt32(got)),
            deltaTime: deltaTime,
            emissionRate: particleEmissionRate,
            baseLifetime: particleLifetime,
            damping: particleDamping,
            transient: clampedTransient,
            spawnJitter: particleJitter,
            velocityScale: particleVelocityScale,
            bandLowMid: SIMD2<Float>(bands.x, bands.y),
            bandHighTime: SIMD2<Float>(bands.z, Float(currentTime))
        )
        let particleParamsBuffer = particleParamsBuffers[writeBufferIndex]
        memcpy(particleParamsBuffer.contents(), &particleUniforms, MemoryLayout<ParticleUpdateUniforms>.stride)

        // 3) Encode compute + render
        let cb = commandQueue.makeCommandBuffer()!

        if got > 0 {
            let liftEncoder = cb.makeComputeCommandEncoder()!
            liftEncoder.setComputePipelineState(liftPipeline)
            liftEncoder.setBuffer(lb, offset: 0, index: 0)
            liftEncoder.setBuffer(rb, offset: 0, index: 1)
            liftEncoder.setBuffer(emitterBuffers[writeBufferIndex], offset: 0, index: 2)
            liftEncoder.setBuffer(paramsBuffer, offset: 0, index: 3)

            let threadWidth = liftPipeline.threadExecutionWidth
            let threadsPerGroup = MTLSize(width: min(threadWidth, max(1, got)), height: 1, depth: 1)
            let threads = MTLSize(width: got, height: 1, depth: 1)
            liftEncoder.dispatchThreads(threads, threadsPerThreadgroup: threadsPerGroup)
            liftEncoder.endEncoding()
        }

        let particleEncoder = cb.makeComputeCommandEncoder()!
        particleEncoder.setComputePipelineState(particlePipeline)
        particleEncoder.setBuffer(particleBuffer, offset: 0, index: 0)
        particleEncoder.setBuffer(emitterBuffers[writeBufferIndex], offset: 0, index: 1)
        particleEncoder.setBuffer(particleParamsBuffer, offset: 0, index: 2)

        let particleWidth = max(1, particlePipeline.threadExecutionWidth)
        let particleThreads = MTLSize(width: maxParticles, height: 1, depth: 1)
        let particleGroup = MTLSize(width: min(particleWidth, maxParticles), height: 1, depth: 1)
        particleEncoder.dispatchThreads(particleThreads, threadsPerThreadgroup: particleGroup)
        particleEncoder.endEncoding()

        let enc = cb.makeRenderCommandEncoder(descriptor: rpd)!
        enc.setRenderPipelineState(renderPipeline)
        enc.setVertexBuffer(particleBuffer, offset: 0, index: 0)
        enc.setVertexBuffer(uniformsBuffer, offset: 0, index: 1)
        enc.drawPrimitives(type: .point, vertexStart: 0, vertexCount: maxParticles)
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
        let yScale = 1 / tan(fovY * 0.5)
        let xScale = yScale / max(aspect, 0.0001)
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

    init(rotation angle: Float, axis: SIMD3<Float>) {
        let a = simd_normalize(axis)
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

    init(translation t: SIMD3<Float>) {
        self = matrix_identity_float4x4
        self.columns.3 = SIMD4<Float>(t.x, t.y, t.z, 1)
    }
}
