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

struct EmitterSample {
    float3 position;
    float  power;
};

struct Particle {
    float3 position;
    float  age;
    float3 velocity;
    float  lifetime;
    float3 color;
    float  brightness;
};

struct ParticleUpdateUniforms {
    uint2  counts;              // x: emitter count, y: particle count
    float2 dt_time;             // x: delta time, y: accumulated time
    float2 emission_lifetime;   // x: base speed, y: base lifetime
    float2 damping_transient;   // x: damping, y: transient boost
    float3 bandEnergy;          // low, mid, high
    float  spawnJitter;
    uint   frameIndex;
    uint3  pad;
};

struct LiftRenderUniforms {
    float4x4 viewProjection;
    float4   misc;     // x: brightness, y: point size, z: transient, w: unused
    float4   bandInfo; // xyz: low/mid/high energy, w: time
};

struct VSOut {
    float4 position [[position]];
    float4 color;
    float  pointSize [[point_size]];
};

static inline float rand11(uint n) {
    return fract(sin(float(n) * 12.9898) * 43758.5453);
}

static inline float rand21(uint2 n) {
    float2 fn = float2(n);
    return fract(sin(dot(fn, float2(127.1, 311.7))) * 43758.5453);
}

static inline float3 safeNormalize(float3 v) {
    float len = length(v);
    return (len > 1e-4f) ? (v / len) : float3(0.0, 0.0, 1.0);
}

kernel void liftStereoTimeLag(device const float* leftSamples   [[buffer(0)]],
                              device const float* rightSamples  [[buffer(1)]],
                              device EmitterSample* outSamples  [[buffer(2)]],
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

    float power = clamp(length(float2(x, y)), 0.0, 8.0);

    outSamples[tid] = EmitterSample{ lifted, power };
}

kernel void updateParticles(device const Particle* inParticles [[buffer(0)]],
                           device Particle* outParticles      [[buffer(1)]],
                           device const EmitterSample* emitters [[buffer(2)]],
                           constant ParticleUpdateUniforms& uniforms [[buffer(3)]],
                           uint tid                           [[thread_position_in_grid]]) {
    if (tid >= uniforms.counts.y) {
        return;
    }

    Particle particle = inParticles[tid];
    uint emitterCount = uniforms.counts.x;
    float dt = uniforms.dt_time.x;
    float timeAccum = uniforms.dt_time.y;
    float damping = clamp(uniforms.damping_transient.x, 0.0, 0.99);
    float transient = uniforms.damping_transient.y;

    float dampingFactor = exp(-damping * dt);
    bool shouldRespawn = (particle.age >= particle.lifetime) || (particle.lifetime <= 0.0001f);

    if (!shouldRespawn) {
        particle.velocity *= dampingFactor;

        float3 radial = safeNormalize(particle.position);
        float cymatic = sin(timeAccum * (2.0 + uniforms.bandEnergy.y * 6.0) + float(tid) * 0.025);
        float swirl = sin(timeAccum * (3.5 + uniforms.bandEnergy.z * 8.0) + float(tid) * 0.013);
        float3 cymaticOffset = radial * cymatic * (0.015 + 0.035 * uniforms.bandEnergy.y);
        float3 twistAxis = safeNormalize(float3(0.32, 0.87, 0.25));
        float3 swirlOffset = cross(twistAxis, particle.position) * swirl * (0.01 + 0.02 * uniforms.bandEnergy.z);

        particle.position += (particle.velocity * dt) + cymaticOffset + swirlOffset;
        particle.age += dt;

        if (particle.age >= particle.lifetime) {
            shouldRespawn = true;
        }
    }

    if (shouldRespawn && emitterCount > 0) {
        uint2 hashSeed = uint2(tid, uniforms.frameIndex);
        uint emitterIndex = uint(rand21(hashSeed) * float(emitterCount - 1)) % max(emitterCount, 1u);
        EmitterSample emitter = emitters[emitterIndex];

        float3 dir = safeNormalize(emitter.position);
        float randomAngle = rand11(tid * 9283u + uniforms.frameIndex * 193u) * 6.28318f;
        float3 basis = safeNormalize(cross(dir, float3(0.23, 0.97, 0.41)));
        float3 tangent = safeNormalize(cross(dir, basis));

        float baseSpeed = uniforms.emission_lifetime.x * (0.35 + emitter.power);
        float speed = baseSpeed * (1.0 + transient * 2.0);
        float swirlAmount = (0.2 + uniforms.bandEnergy.z * 1.8) * emitter.power;
        float3 swirlVec = (cos(randomAngle) * basis + sin(randomAngle) * tangent) * swirlAmount;

        particle.position = emitter.position;
        particle.velocity = dir * speed + swirlVec;
        particle.age = 0.0f;

        float jitter = 0.6f + uniforms.spawnJitter * rand11(tid * 6131u + uniforms.frameIndex * 11u);
        particle.lifetime = uniforms.emission_lifetime.y * jitter;

        float3 lowColor = float3(0.2, 0.5, 1.0);
        float3 midColor = float3(0.8, 0.3, 1.0);
        float3 highColor = float3(1.0, 0.85, 0.25);
        float totalEnergy = max(0.0001f, uniforms.bandEnergy.x + uniforms.bandEnergy.y + uniforms.bandEnergy.z);
        float choice = rand11(tid * 19391u + uniforms.frameIndex * 733u) * totalEnergy;

        if (choice < uniforms.bandEnergy.x) {
            particle.color = lowColor;
        } else if (choice < uniforms.bandEnergy.x + uniforms.bandEnergy.y) {
            particle.color = midColor;
        } else {
            particle.color = highColor;
        }

        float brightnessBase = clamp(emitter.power * (0.6 + uniforms.bandEnergy.y * 1.8), 0.1, 4.0);
        particle.brightness = brightnessBase;
    } else if (shouldRespawn) {
        particle.brightness = 0.0f;
    }

    outParticles[tid] = particle;
}

vertex VSOut vectorscope_lift_vertex(uint vid [[vertex_id]],
                                     const device Particle* particles [[buffer(0)]],
                                     constant LiftRenderUniforms& uniforms [[buffer(1)]]) {
    VSOut out;
    Particle particle = particles[vid];
    float3 p = particle.position;
    out.position = uniforms.viewProjection * float4(p, 1.0);

    float lifeRatio = (particle.lifetime > 0.0001f)
        ? clamp(particle.age / particle.lifetime, 0.0f, 1.0f)
        : 1.0f;
    float fade = pow(max(0.0f, 1.0f - lifeRatio), 2.2f);
    float transientBoost = 1.0f + uniforms.misc.z * 1.5f;
    float brightness = uniforms.misc.x * particle.brightness * fade * transientBoost;

    float3 freqWeight = normalize(uniforms.bandInfo.xyz + 0.0001f);
    float spectralPulse = 0.5f + 0.5f * sin(uniforms.bandInfo.w * 1.8f + float(vid) * 0.01f);
    float spectralBoost = mix(0.7f, 1.4f, spectralPulse);
    float3 color = particle.color * spectralBoost * (0.6f + 0.4f * dot(freqWeight, particle.color));

    float alpha = clamp(brightness, 0.0f, 1.0f);
    out.color = float4(color * brightness, alpha);
    out.pointSize = max(1.0f, uniforms.misc.y * (0.8f + sqrt(particle.brightness)));
    return out;
}

fragment float4 vectorscope_fragment(VSOut in [[stage_in]]) {
    return in.color;
}
