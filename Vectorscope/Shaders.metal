#include <metal_stdlib>
using namespace metal;


inline float fade(float t) {
    // Quintic smoothing: 6t^5 - 15t^4 + 10t^3
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

inline float lerp(float a, float b, float t) { return a + t * (b - a); }

// Simple 32-bit FNV-1a style hash (good enough for gradient selection)
inline uint hash_u32(uint x) {
    uint h = 2166136261u;
    h = (h ^ x) * 16777619u;
    return h;
}

inline uint hash4(uint xi, uint yi, uint zi, uint wi) {
    // Mix coordinates with different large odd primes to reduce collisions
    uint h = 2166136261u;
    h = (h ^ (xi * 374761393u)) * 16777619u;
    h = (h ^ (yi * 668265263u)) * 16777619u;
    h = (h ^ (zi * 2246822519u)) * 16777619u;
    h = (h ^ (wi * 3266489917u)) * 16777619u;
    // final avalanche
    h ^= h >> 16;
    h *= 2246822519u;
    h ^= h >> 13;
    h *= 3266489917u;
    h ^= h >> 16;
    return h;
}

// Build a pseudo-random unit gradient vector in 4D from a lattice corner
inline float4 grad4(uint xi, uint yi, uint zi, uint wi) {
    uint h = hash4(xi, yi, zi, wi);

    // Split into 4 bytes -> [-1, 1], then normalize
    float4 g = float4(
        ((float)((h      ) & 0xFF) / 127.5f) - 1.0f,
        ((float)((h >>  8) & 0xFF) / 127.5f) - 1.0f,
        ((float)((h >> 16) & 0xFF) / 127.5f) - 1.0f,
        ((float)((h >> 24) & 0xFF) / 127.5f) - 1.0f
    );

    // Avoid zero-length; normalize to unit length
    float len2 = max(dot(g, g), 1e-8f);
    return g * rsqrt(len2);
}

// Dot of gradient at integer corner (pi + corner) with distance vector (pf - corner)
inline float gradDotAtCorner(int4 pi, float4 pf, int cx, int cy, int cz, int cw) {
    uint xi = (uint)(pi.x + cx);
    uint yi = (uint)(pi.y + cy);
    uint zi = (uint)(pi.z + cz);
    uint wi = (uint)(pi.w + cw);
    float4 g = grad4(xi, yi, zi, wi);
    float4 d = pf - float4((float)cx, (float)cy, (float)cz, (float)cw);
    return dot(g, d);
}

// Scalar 4D Perlin noise at point p (range approx [-1, 1])
inline float perlin4d(float4 p) {
    float4 pf = fract(p);                  // fractional part in [0,1)
    int4   pi = int4(floor(p));            // integer lattice coord

    // Quintic fade per axis
    float4 u = float4(fade(pf.x), fade(pf.y), fade(pf.z), fade(pf.w));

    // Evaluate all 16 hypercube corners
    // w = 0 "slice"
    float n0000 = gradDotAtCorner(pi, pf, 0,0,0,0);
    float n1000 = gradDotAtCorner(pi, pf, 1,0,0,0);
    float n0100 = gradDotAtCorner(pi, pf, 0,1,0,0);
    float n1100 = gradDotAtCorner(pi, pf, 1,1,0,0);
    float n0010 = gradDotAtCorner(pi, pf, 0,0,1,0);
    float n1010 = gradDotAtCorner(pi, pf, 1,0,1,0);
    float n0110 = gradDotAtCorner(pi, pf, 0,1,1,0);
    float n1110 = gradDotAtCorner(pi, pf, 1,1,1,0);

    // w = 1 "slice"
    float n0001 = gradDotAtCorner(pi, pf, 0,0,0,1);
    float n1001 = gradDotAtCorner(pi, pf, 1,0,0,1);
    float n0101 = gradDotAtCorner(pi, pf, 0,1,0,1);
    float n1101 = gradDotAtCorner(pi, pf, 1,1,0,1);
    float n0011 = gradDotAtCorner(pi, pf, 0,0,1,1);
    float n1011 = gradDotAtCorner(pi, pf, 1,0,1,1);
    float n0111 = gradDotAtCorner(pi, pf, 0,1,1,1);
    float n1111 = gradDotAtCorner(pi, pf, 1,1,1,1);

    // Interpolate along x
    float nx000 = lerp(n0000, n1000, u.x);
    float nx010 = lerp(n0100, n1100, u.x);
    float nx001 = lerp(n0010, n1010, u.x);
    float nx011 = lerp(n0110, n1110, u.x);

    float nx001_w = lerp(n0001, n1001, u.x);
    float nx011_w = lerp(n0101, n1101, u.x);
    float nx101_w = lerp(n0011, n1011, u.x);
    float nx111_w = lerp(n0111, n1111, u.x);

    // Interpolate along y
    float nxy00 = lerp(nx000,  nx010,  u.y);
    float nxy01 = lerp(nx001,  nx011,  u.y);
    float nxy00_w = lerp(nx001_w, nx011_w, u.y);
    float nxy01_w = lerp(nx101_w, nx111_w, u.y);

    // Interpolate along z
    float nxyz0 = lerp(nxy00,   nxy01,   u.z);
    float nxyz1 = lerp(nxy00_w, nxy01_w, u.z);

    // Interpolate along w
    float n = lerp(nxyz0, nxyz1, u.w);

    // Scale to roughly [-1,1]; gradients are unit, so the raw range is already ~[-1,1]
    // Optionally scale by ~0.87 to better confine range—left as-is here.
    return n;
}

// Public: 4-channel noise from one input point.
// Each channel is the same 4D Perlin evaluated at a fixed offset to decorrelate.
float4 perlinNoise4(float4 p) {
    const float4 o0 = float4( 0.0f,   0.0f,   0.0f,   0.0f);
    const float4 o1 = float4(19.19f, 47.11f,  3.71f,  7.13f);
    const float4 o2 = float4(101.3f, 13.07f, 59.73f, 23.91f);
    const float4 o3 = float4(17.0f,  89.2f, 271.0f,  11.0f);

    return float4(
        perlin4d(p + o0),
        perlin4d(p + o1),
        perlin4d(p + o2),
        perlin4d(p + o3)
    );
}




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
    float4   misc; // x: brightness, y: line width, z: sample count, w: render flag (>0 for lines)
    float2   viewportSize;
    float2   pad;
};

struct VSOut {
    float4 position [[position]];
    float4 color;
    float  pointSize [[point_size]];
};

kernel void liftStereoTimeLag(device const float* leftSamples   [[buffer(0)]],
                              device const float* rightSamples  [[buffer(1)]],
                              device float3* outXYZ             [[buffer(2)]],
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

    outXYZ[tid] = lifted;
}

struct EvaluatedPoint {
    float3 original;
    float4 clip;
};

inline EvaluatedPoint evaluatePoint(uint index,
                                    constant LiftRenderUniforms& uniforms,
                                    const device float3* positions,
                                    float totalSamples) {
    EvaluatedPoint result;
    float3 p = positions[index];
    float4 p_in = float4(p, totalSamples);
    float4 perlin = perlinNoise4(p_in) * 10.0f;
    float4 clip = uniforms.viewProjection * float4(perlin.xyz, 1.0);
    result.original = p;
    result.clip = clip;
    return result;
}

vertex VSOut vectorscope_lift_vertex(uint vid [[vertex_id]],
                                     const device float3* positions [[buffer(0)]],
                                     constant LiftRenderUniforms& uniforms [[buffer(1)]]) {
    VSOut out;
    uint sampleCount = (uint)max(uniforms.misc.z, 0.0);
    bool renderStrip = (uniforms.misc.w > 0.5f) && (sampleCount >= 2u);

    uint sampleIndex = renderStrip ? (vid >> 1) : vid;
    sampleIndex = min(sampleIndex, sampleCount > 0u ? (sampleCount - 1u) : 0u);

    uint prevIndex = (sampleIndex > 0u) ? (sampleIndex - 1u) : sampleIndex;
    uint nextIndex = (sampleIndex + 1u < sampleCount) ? (sampleIndex + 1u) : sampleIndex;

    float total = max(1.0f, uniforms.misc.z - 1.0f);

    EvaluatedPoint prevEval = evaluatePoint(prevIndex, uniforms, positions, total);
    EvaluatedPoint currEval = evaluatePoint(sampleIndex, uniforms, positions, total);
    EvaluatedPoint nextEval = evaluatePoint(nextIndex, uniforms, positions, total);

    float4 clipPrev = prevEval.clip;
    float4 clipCurr = currEval.clip;
    float4 clipNext = nextEval.clip;

    float invPrevW = clipPrev.w != 0.0f ? 1.0f / clipPrev.w : 0.0f;
    float invCurrW = clipCurr.w != 0.0f ? 1.0f / clipCurr.w : 0.0f;
    float invNextW = clipNext.w != 0.0f ? 1.0f / clipNext.w : 0.0f;

    float2 ndcPrev = clipPrev.xy * invPrevW;
    float2 ndcCurr = clipCurr.xy * invCurrW;
    float2 ndcNext = clipNext.xy * invNextW;

    float2 tangent = ndcNext - ndcPrev;
    float len = length(tangent);
    if (len > 1e-6f) {
        tangent /= len;
    } else {
        tangent = float2(0.0f, 1.0f);
    }
    float2 normal = float2(-tangent.y, tangent.x);

    float halfWidth = max(0.5f, uniforms.misc.y * 0.5f);
    float2 viewport = uniforms.viewportSize;
    float2 pixelToNDC = float2(viewport.x > 0.0f ? (2.0f / viewport.x) : 0.0f,
                               viewport.y > 0.0f ? (2.0f / viewport.y) : 0.0f);
    float2 offsetNDC = normal * (halfWidth * pixelToNDC);

    if (renderStrip) {
        offsetNDC = (vid & 1u) == 0u ? -offsetNDC : offsetNDC;
    } else {
        offsetNDC = float2(0.0f);
    }

    float2 extrudedNDC = ndcCurr + offsetNDC;
    float4 extrudedClip = clipCurr;
    extrudedClip.x = extrudedNDC.x * clipCurr.w;
    extrudedClip.y = extrudedNDC.y * clipCurr.w;

    float3 wrapped = fract(extrudedClip.xyz) - float3(0.5f);
    float4 finalPosition = float4(wrapped, clipCurr.w);
    out.position = finalPosition;

    float totalForAge = max(1.0f, uniforms.misc.z - 1.0f);
    float age = (totalForAge > 0.0f) ? float(sampleIndex) / totalForAge : 0.0f;
    float alpha = clamp(uniforms.misc.x * age, 0.0f, 1.0f);

    float depth = clamp(0.5f + 0.5f * tanh(currEval.original.z), 0.0f, 1.0f);
    float3 cold = float3(0.2f, 0.6f, 1.0f);
    float3 warm = float3(1.0f, 0.4f, 0.6f);
    float3 baseColor = mix(cold, warm, depth);

    out.color = float4(baseColor, alpha);
    out.pointSize = max(1.0f, uniforms.misc.y);
    return out;
}

fragment float4 vectorscope_fragment(VSOut in [[stage_in]]) {
    return in.color * 1.3;
}
