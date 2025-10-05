import Foundation
import AVFoundation
import CoreAudio
import AudioToolbox

enum CAError: Error { case property(String, OSStatus) }

private func osstatusString(_ s: OSStatus) -> String {
    let n = UInt32(bitPattern: s)
    let b1 = UInt8((n >> 24) & 0xFF)
    let b2 = UInt8((n >> 16) & 0xFF)
    let b3 = UInt8((n >> 8) & 0xFF)
    let b4 = UInt8(n & 0xFF)
    let bytes: [UInt8] = [b1, b2, b3, b4]
    let isPrintable = bytes.allSatisfy { $0 >= 32 && $0 < 127 }
    if isPrintable, let s = String(bytes: bytes, encoding: .ascii) {
        return "'\(s)'"
    } else {
        return String(s)
    }
}

/// Helper to build a property address
private func prop(_ sel: AudioObjectPropertySelector,
                  _ scope: AudioObjectPropertyScope,
                  _ elem: AudioObjectPropertyElement = kAudioObjectPropertyElementMain)
-> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: sel, mScope: scope, mElement: elem)
}

/// Get the default output or input device (macOS has no AVAudioSession)
func defaultDeviceID(isInput: Bool) throws -> AudioObjectID {
    var deviceID = kAudioObjectUnknown
    var size = UInt32(MemoryLayout.size(ofValue: deviceID))
    let sel: AudioObjectPropertySelector = isInput
        ? kAudioHardwarePropertyDefaultInputDevice
        : kAudioHardwarePropertyDefaultOutputDevice
    var addr = prop(sel, kAudioObjectPropertyScopeGlobal)

    let status = AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject),
                                            &addr, 0, nil, &size, &deviceID)
    guard status == noErr, deviceID != kAudioObjectUnknown else {
        throw CAError.property("Default \(isInput ? "input" : "output") device", status)
    }
    return deviceID
}

/// Query current buffer size and allowed range (min/max) for a device/scope
func queryBufferFrames(deviceID: AudioObjectID,
                       scope: AudioObjectPropertyScope) throws -> (current: UInt32, min: UInt32, max: UInt32) {
    // Current
    var currentFrames = UInt32(0)
    var size = UInt32(MemoryLayout.size(ofValue: currentFrames))
    var addr = prop(kAudioDevicePropertyBufferFrameSize, scope)

    var status = AudioObjectGetPropertyData(deviceID, &addr, 0, nil, &size, &currentFrames)
    guard status == noErr else { throw CAError.property("Get BufferFrameSize", status) }

    // Range
    var range = AudioValueRange()
    size = UInt32(MemoryLayout.size(ofValue: range))
    addr = prop(kAudioDevicePropertyBufferFrameSizeRange, scope)

    status = AudioObjectGetPropertyData(deviceID, &addr, 0, nil, &size, &range)
    guard status == noErr else { throw CAError.property("Get BufferFrameSizeRange", status) }

    return (currentFrames, UInt32(range.mMinimum), UInt32(range.mMaximum))
}

/// Attempt to set the buffer frame size (frames) for a device+scope.
@discardableResult
func setBufferFrames(deviceID: AudioObjectID,
                     scope: AudioObjectPropertyScope,
                     frames desired: UInt32) throws -> UInt32 {
    // Is property settable?
    var addr = prop(kAudioDevicePropertyBufferFrameSize, scope)
    var settable: DarwinBoolean = false
    var status = AudioObjectIsPropertySettable(deviceID, &addr, &settable)
    guard status == noErr else { throw CAError.property("IsPropertySettable", status) }
    guard settable.boolValue else { throw CAError.property("BufferFrameSize not settable", kAudioHardwareUnsupportedOperationError) }

    var frames = desired
    print("Setting buffer frames to \(frames) for device \(deviceID) scope \(scope)")
    var size = UInt32(MemoryLayout.size(ofValue: frames))
    status = AudioObjectSetPropertyData(deviceID, &addr, 0, nil, size, &frames)
    guard status == noErr else { throw CAError.property("Set BufferFrameSize", status) }

    // Read back the actual value (device may clamp)
    var applied = UInt32(0)
    size = UInt32(MemoryLayout.size(ofValue: applied))
    status = AudioObjectGetPropertyData(deviceID, &addr, 0, nil, &size, &applied)
    guard status == noErr else { throw CAError.property("Get BufferFrameSize (after set)", status) }
    return applied
}

/// A simple stereo audio source that can pull from microphone or a file and exposes a lock-free ring of recent samples.
public final class StereoAudioSource {
    public enum Source {
        case microphone
        case file(URL)
    }

    // Public settings
    public var desiredSampleRate: Double = 48000.0  // Default, overridden by hardware
    public var maxSamplesInRing: Int = 65536  // Much larger ring buffer for proper cycling

    // Ring buffers (interleaving avoided for simplicity)
    private var leftRing: [Float]
    private var rightRing: [Float]
    private var writeIndex: Int = 0
    private var count: Int = 0

    // Single-producer/single-consumer: protect indices with atomic semantics (here a fast serial queue suffices)
    private let indexQueue = DispatchQueue(label: "stereo.ring.index", qos: .userInteractive)

    // AVAudioEngine graph
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private var currentFormat: AVAudioFormat!
    private var tapInstalled = false

    // Tapped node for input or output
    private var tappedNode: AVAudioNode?

    // Sink node for low-latency render-quantum capture
    private var sinkNode: AVAudioSinkNode?

    // Debug logging properties
    private var audioCallbackCount = 0
    private var lastAudioLogTime: CFTimeInterval = 0
    private var totalFramesProcessed = 0

    public init(maxSamples: Int = 65536) {
        self.maxSamplesInRing = max(2048, maxSamples)  // minimum safety
        self.leftRing = Array(repeating: 0, count: self.maxSamplesInRing)
        self.rightRing = Array(repeating: 0, count: self.maxSamplesInRing)
    }

    /// Start the audio chain. If switching sources at runtime, call stop() first.
    public func start(_ source: Source) throws {
        // macOS microphone permission preflight
        #if os(macOS)
        if case .microphone = source {
            let status = AVCaptureDevice.authorizationStatus(for: .audio)
            switch status {
            case .authorized:
                break
            case .notDetermined:
                AVCaptureDevice.requestAccess(for: .audio) { [weak self] granted in
                    guard let self = self else { return }
                    if granted {
                        DispatchQueue.main.async {
                            _ = try? self.start(.microphone)
                        }
                    } else {
                        print("Microphone access was not granted.")
                    }
                }
                return
            case .denied, .restricted:
                print("Microphone access denied or restricted. Enable it in System Settings > Privacy & Security > Microphone.")
                return
            @unknown default:
                break
            }
        }
        #endif

        stop()

        // 1) First, force smaller hardware buffer sizes using CoreAudio directly
        do {
            let outID = try defaultDeviceID(isInput: false)
            let inID = try defaultDeviceID(isInput: true)

            // Query current buffer sizes
            let (curOut, minOut, maxOut) = try queryBufferFrames(deviceID: outID, scope: kAudioObjectPropertyScopeOutput)
            let (curIn, minIn, maxIn) = try queryBufferFrames(deviceID: inID, scope: kAudioObjectPropertyScopeInput)

            print("### Hardware Buffer Info ###")
            print("Output device: current=\(curOut), range=[\(minOut)...\(maxOut)]")
            print("Input device: current=\(curIn), range=[\(minIn)...\(maxIn)]")

            // Request very small buffer sizes (64 frames at 48kHz = ~1.3ms)
            let desiredFrames: UInt32 = 64
            let appliedOut = try setBufferFrames(deviceID: outID, scope: kAudioObjectPropertyScopeOutput, frames: desiredFrames)
            let appliedIn = try setBufferFrames(deviceID: inID, scope: kAudioObjectPropertyScopeInput, frames: desiredFrames)

            print("Requested \(desiredFrames) frames: Output got \(appliedOut), Input got \(appliedIn)")
            print("##############################")
        } catch {
            if let ca = error as? CAError, case let .property(name, status) = ca {
                print("CoreAudio buffer setup failed: \(name) OSStatus=\(osstatusString(status))")
            } else {
                print("CoreAudio buffer setup failed: \(error)")
            }
        }

        // 2) Now configure AVAudioEngine as usual
        #if os(iOS) || os(tvOS) || os(watchOS)
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.playAndRecord, options: [.defaultToSpeaker, .mixWithOthers, .allowBluetooth, .allowAirPlay])
        try? session.setPreferredIOBufferDuration(0.005) // 5ms buffer for lower latency
        try? session.setActive(true)
        let hwSampleRate = session.sampleRate
        #else
        let hwSampleRate = engine.outputNode.outputFormat(forBus: 0).sampleRate
        #endif
        let rate = hwSampleRate > 0 ? hwSampleRate : desiredSampleRate
        let stereo = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: 2, interleaved: false)!
        currentFormat = stereo

        // Build graph
        engine.reset()

        switch source {
        case .microphone:
            let input = engine.inputNode
            let inputCh = max(2, Int(input.inputFormat(forBus: 0).channelCount))
            let desired = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: AVAudioChannelCount(inputCh), interleaved: false)!
            engine.connect(input, to: engine.mainMixerNode, format: desired)
        case .file:
            engine.attach(player)
            engine.connect(player, to: engine.mainMixerNode, format: nil)
        }

        // Create and attach a sink node to receive render-quantum audio from the main mixer
        if sinkNode == nil {
            if #available(macOS 10.15, *) {
                let sink = AVAudioSinkNode { [weak self] (ts, frameCount, inputData) -> OSStatus in
                    self?.ingest(abl: inputData, frames: Int(frameCount))
                    return noErr
                }
                sinkNode = sink
                engine.attach(sink)
                // Connect main mixer to sink with current stereo float format
                engine.connect(engine.mainMixerNode, to: sink, format: currentFormat)
            }
        }

        try engine.start()

        if case let .file(url) = source {
            let file = try AVAudioFile(forReading: url)
            player.stop()
            player.scheduleFile(file, at: nil, completionHandler: nil)
            player.play()
        }
    }

    public func stop() {
        if let node = tappedNode {
            node.removeTap(onBus: 0)
            tappedNode = nil
        }
        if let sn = sinkNode {
            engine.disconnectNodeInput(sn)
            engine.detach(sn)
            sinkNode = nil
        }
        if engine.isRunning {
            player.stop()
            engine.stop()
        }
        // keep tap; it gets re-used across starts
    }

    // MARK: - Ingest (Audio Thread)
    private func ingest(abl: UnsafePointer<AudioBufferList>, frames: Int) {
        let mabl = UnsafeMutablePointer(mutating: abl)
        let ablp = UnsafeMutableAudioBufferListPointer(mabl)
        if ablp.count == 0 || frames == 0 { return }

        // Add debug logging for audio callback analysis (sink-based path)
        audioCallbackCount += 1
        totalFramesProcessed += frames
        let currentTime = CACurrentMediaTime()
        if currentTime - lastAudioLogTime > 1.0 {
            print("### Audio Debug Stats ###")
            print("Audio callbacks per second: \(audioCallbackCount)")
            print("Frames per callback: \(totalFramesProcessed / audioCallbackCount)")
            print("Total frames this second: \(totalFramesProcessed)")
            print("Buffer format: \(currentFormat.sampleRate)Hz, \(currentFormat.channelCount)ch")
            print("Ring buffer count: \(count)/\(maxSamplesInRing)")
            print("#########################")
            audioCallbackCount = 0
            totalFramesProcessed = 0
            lastAudioLogTime = currentTime
        }

        let p0 = ablp[0].mData?.assumingMemoryBound(to: Float.self)
        let p1 = (ablp.count >= 2 ? ablp[1].mData : ablp[0].mData)?.assumingMemoryBound(to: Float.self)
        guard let c0 = p0, let c1 = p1 else { return }

        indexQueue.sync {
            var w = writeIndex
            for i in 0..<frames {
                leftRing[w] = c0[i]
                rightRing[w] = c1[i]
                w += 1
                if w == maxSamplesInRing { w = 0 }
            }
            writeIndex = w
            count = min(maxSamplesInRing, count + frames)
        }
    }

    private func ingest(buffer: AVAudioPCMBuffer) {
        guard let ch = buffer.floatChannelData else { return }
        let frames = Int(buffer.frameLength)
        if frames == 0 { return }

        // Add debug logging for audio callback analysis
        audioCallbackCount += 1
        totalFramesProcessed += frames

        let currentTime = CACurrentMediaTime()
        if currentTime - lastAudioLogTime > 1.0 {
            print("### Audio Debug Stats ###")
            print("Audio callbacks per second: \(audioCallbackCount)")
            print("Frames per callback: \(totalFramesProcessed / audioCallbackCount)")
            print("Total frames this second: \(totalFramesProcessed)")
            print("Buffer format: \(buffer.format.sampleRate)Hz, \(buffer.format.channelCount)ch")
            print("Ring buffer count: \(count)/\(maxSamplesInRing)")
            print("#########################")
            audioCallbackCount = 0
            totalFramesProcessed = 0
            lastAudioLogTime = currentTime
        }

        // Ensure we have at least two channels (duplicate if mono)
        let c0 = ch[0]
        let c1: UnsafeMutablePointer<Float> = (buffer.format.channelCount >= 2) ? ch[1] : ch[0]

        // Copy into ring
        indexQueue.sync {
            var w = writeIndex
            for i in 0..<frames {
                leftRing[w] = c0[i]
                rightRing[w] = c1[i]
                w += 1
                if w == maxSamplesInRing { w = 0 }
            }
            writeIndex = w
            count = min(maxSamplesInRing, count + frames)
        }
    }

    // MARK: - Consumer API (Render Thread)
    /// Copy the most recent `maxOut` samples into provided raw pointers. Returns the number of samples written.
    @discardableResult
    public func copyLatest(into leftOut: UnsafeMutablePointer<Float>,
                    rightOut: UnsafeMutablePointer<Float>,
                    maxOut: Int) -> Int {
        var n = 0
        var start = 0

        indexQueue.sync {
            n = min(count, maxOut)
            if n == 0 { return }
            // Oldest index we want: writeIndex - n (mod ring size)
            start = writeIndex - n
            if start < 0 { start += maxSamplesInRing }
        }
        if n == 0 { return 0 }

        // Two copy spans if wrapping
        let firstSpan = min(n, maxSamplesInRing - start)
        let secondSpan = n - firstSpan

        // Copy span 1
        leftRing.withUnsafeBufferPointer { lbuf in
            rightRing.withUnsafeBufferPointer { rbuf in
                memcpy(leftOut, lbuf.baseAddress! + start, firstSpan * MemoryLayout<Float>.size)
                memcpy(rightOut, rbuf.baseAddress! + start, firstSpan * MemoryLayout<Float>.size)
            }
        }
        // Copy span 2 (wrapped)
        if secondSpan > 0 {
            leftRing.withUnsafeBufferPointer { lbuf in
                rightRing.withUnsafeBufferPointer { rbuf in
                    memcpy(leftOut.advanced(by: firstSpan), lbuf.baseAddress!, secondSpan * MemoryLayout<Float>.size)
                    memcpy(rightOut.advanced(by: firstSpan), rbuf.baseAddress!, secondSpan * MemoryLayout<Float>.size)
                }
            }
        }
        return n
    }

    public var currentSampleRate: Double {
        indexQueue.sync {
            currentFormat?.sampleRate ?? desiredSampleRate
        }
    }
}
