#include <metal_stdlib>
using namespace metal;

struct AudioLiftParams {
    uint   sampleCount;
    uint   tauSamples;
    float2 scaleXY;
    float  scaleZ;
    float  pad0;
    float3 offset;
    float  pad1;
};

struct LiftRenderUniforms {
    float4x4 viewProjection;
    float4   misc;       // x: brightness, y: point size, z: particle count, w: unused
    float4   audioBands; // xyz: normalized bands, w: unused
};

struct Particle {
    float4 posLife; // xyz: position, w: remaining life (seconds)
    float4 velLife; // xyz: velocity, w: total lifetime (seconds)
    float4 misc;    // x: seed, yzw: color accent
};

struct ParticleUpdateUniforms {
    uint2 counts;           // x: particle count, y: emitter count
    float deltaTime;
    float emissionRate;
    float baseLifetime;
    float damping;
    float transient;
    float spawnJitter;
    float velocityScale;
    float2 bandLowMid;
    float2 bandHighTime;    // x: high band, y: time
};

struct VSOut {
    float4 position [[position]];
    float4 color;
    float  pointSize [[point_size]];
};

inline float rand(float seed, float offset) {
    return fract(sin(seed * 12.9898 + offset * 78.233) * 43758.5453);
}

kernel void liftStereoTimeLag(device const float* leftSamples   [[buffer(0)]],
                              device const float* rightSamples  [[buffer(1)]],
                              device float4* outEmitters        [[buffer(2)]],
                              constant AudioLiftParams& params  [[buffer(3)]],
                              uint tid                          [[thread_position_in_grid]]) {
    if (tid >= params.sampleCount) {
        return;
    }

    uint lagIndex = (tid >= params.tauSamples) ? (tid - params.tauSamples) : 0;

    float x = leftSamples[tid];
    float y = rightSamples[tid];
    float z = leftSamples[lagIndex];

    float3 lifted = float3(x * params.scaleXY.x,
                           y * params.scaleXY.y,
                           z * params.scaleZ) + params.offset;

    float amplitude = length(float2(x, y));
    outEmitters[tid] = float4(lifted, amplitude);
}

kernel void updateParticles(device Particle* particles           [[buffer(0)]],
                            device const float4* emitters        [[buffer(1)]],
                            constant ParticleUpdateUniforms& params [[buffer(2)]],
                            uint tid                             [[thread_position_in_grid]]) {
    uint particleCount = params.counts.x;
    if (tid >= particleCount) {
        return;
    }

    Particle p = particles[tid];
    float dt = max(params.deltaTime, 0.0001);
    float decay = dt * max(params.emissionRate, 0.01);
    float lifetime = max(p.velLife.w, 0.0001);
    float remaining = max(p.posLife.w - decay, 0.0);

    float dampingFactor = clamp(1.0 - params.damping * dt, 0.0, 1.0);
    p.velLife.xyz *= dampingFactor;

    float3 audioBands = float3(params.bandLowMid.x,
                               params.bandLowMid.y,
                               params.bandHighTime.x);

    // Cymatic-like swirl influenced by mid frequencies
    float3 swirlAxis = float3(0.0, 0.0, 1.0);
    float3 swirl = cross(swirlAxis, p.posLife.xyz) * (0.4 * audioBands.y);
    p.velLife.xyz += swirl * dt;
    p.misc.x += dt * 0.17;

    p.posLife.xyz += p.velLife.xyz * dt;
    p.posLife.w = remaining;

    bool respawn = (remaining <= 0.0001);
    uint emitterCount = params.counts.y;
    if (respawn && emitterCount > 0) {
        float seed = p.misc.x + params.bandHighTime.y;
        float choose = rand(seed, 1.0);
        uint emitterIndex = min(emitterCount - 1, uint(choose * float(emitterCount)));
        float4 emitter = emitters[emitterIndex];

        float3 basePos = emitter.xyz;
        float amplitude = emitter.w;
        float3 outward = normalize(basePos);
        if (!isfinite(outward.x) || !isfinite(outward.y) || !isfinite(outward.z) || length(outward) < 1e-5) {
            outward = float3(0.0, 0.0, 1.0);
        }

        float jitterAngle = rand(seed, 2.0) * 6.2831853;
        float jitterRadius = (rand(seed, 3.0) - 0.5) * params.spawnJitter;
        float jitterZ = (rand(seed, 4.0) - 0.5) * params.spawnJitter * 0.5;
        float3 jitter = float3(cos(jitterAngle) * jitterRadius,
                               sin(jitterAngle) * jitterRadius,
                               jitterZ);

        float transientBoost = 1.0 + params.transient * 2.5;
        float amplitudeFactor = clamp(amplitude * 4.0, 0.2, 6.0);
        float bandDrive = 0.6 + dot(audioBands, float3(0.5, 0.7, 0.9));
        float speed = params.velocityScale * transientBoost * bandDrive * amplitudeFactor;

        float3 velocity = outward * speed + jitter;

        float lifetimeNew = params.baseLifetime * (0.6 + 0.8 * rand(seed, 5.0));
        p.posLife = float4(basePos + jitter, lifetimeNew);
        p.velLife = float4(velocity, lifetimeNew);

        float3 colorBias = normalize(audioBands + 0.0001);
        p.misc = float4(seed + 13.37, colorBias);
    } else if (respawn) {
        // Keep seed evolving even if no emitters available
        p.misc.x += dt;
    }

    particles[tid] = p;
}

vertex VSOut particle_vertex(uint vid [[vertex_id]],
                             const device Particle* particles [[buffer(0)]],
                             constant LiftRenderUniforms& uniforms [[buffer(1)]]) {
    VSOut out;
    Particle particle = particles[vid];
    float lifetime = max(particle.velLife.w, 0.0001);
    float life = clamp(particle.posLife.w / lifetime, 0.0, 1.0);

    float4 world = float4(particle.posLife.xyz, 1.0);
    out.position = uniforms.viewProjection * world;

    float3 bandColor = float3(particle.misc.y, particle.misc.z, particle.misc.w);
    if (all(bandColor == float3(0.0))) {
        bandColor = normalize(uniforms.audioBands.xyz + 0.0001);
    }

    float velocityMag = length(particle.velLife.xyz);
    float glow = clamp(velocityMag * 0.3, 0.0, 1.0);
    float3 cool = float3(0.2, 0.7, 1.0);
    float3 warm = float3(1.0, 0.35, 0.6);
    float tone = clamp(dot(bandColor, float3(0.33, 0.45, 0.22)) + glow * 0.4, 0.0, 1.0);
    float3 baseColor = mix(cool, warm, tone);
    baseColor *= mix(float3(0.6), float3(1.3), bandColor);

    float alpha = uniforms.misc.x * pow(life, 1.6);
    out.color = float4(baseColor, alpha);
    out.pointSize = max(1.0, uniforms.misc.y * (0.6 + 1.4 * sqrt(life)));
    return out;
}

fragment float4 particle_fragment(VSOut in [[stage_in]]) {
    return in.color;
}
