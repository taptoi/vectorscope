#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    float  gain;
    uint   sampleCount;
    float  pointSize;
    float  aspectScaleY;
    float  brightness;
};

struct VSOut {
    float4 position [[position]];
    float4 color;
    float  ptSize [[point_size]];
};

vertex VSOut vectorscope_vertex(uint vid [[vertex_id]],
                                const device float* left  [[buffer(0)]],
                                const device float* right [[buffer(1)]],
                                constant Uniforms& u      [[buffer(2)]]) {
    VSOut out;

    // Fetch normalized stereo samples and scale
    float x = clamp(left[vid]  * u.gain, -1.0, 1.0);
    float y = clamp(right[vid] * u.gain, -1.0, 1.0);

    // Correct aspect so circles look circular on non-square viewports
    float2 p = float2(x, y * u.aspectScaleY);

    out.position = float4(p, 0.0, 1.0);
    out.ptSize   = max(1.0, u.pointSize);

    // Linear fade by age: oldest (vid=0) is dim, newest (vid=sampleCount-1) is full brightness
    // Handle small sampleCount to avoid div-by-zero.
    float denom = max(1.0, float(max(1u, u.sampleCount) - 1u));
    float ageFactor = clamp(float(vid) / denom, 0.0, 1.0);
    float alpha = clamp(u.brightness * ageFactor, 0.0, 1.0);

    out.color = float4(1.0, 1.0, 1.0, alpha);
    return out;
}

fragment float4 vectorscope_fragment(VSOut in [[stage_in]]) {
    return in.color;
}
