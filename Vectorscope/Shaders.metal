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
    float4   misc; // x: brightness, y: point size, z: sample count, w: unused
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

vertex VSOut vectorscope_lift_vertex(uint vid [[vertex_id]],
                                     const device float3* positions [[buffer(0)]],
                                     constant LiftRenderUniforms& uniforms [[buffer(1)]]) {
    VSOut out;
    float3 p = positions[vid];
    out.position = uniforms.viewProjection * float4(p, 1.0);

    float total = max(1.0, uniforms.misc.z - 1.0);
    float age = (total > 0.0) ? float(vid) / total : 0.0;
    float alpha = clamp(uniforms.misc.x * age, 0.0, 1.0);

    float depth = clamp(0.5 + 0.5 * tanh(p.z), 0.0, 1.0);
    float3 cold = float3(0.2, 0.6, 1.0);
    float3 warm = float3(1.0, 0.4, 0.6);
    float3 baseColor = mix(cold, warm, depth);

    out.color = float4(baseColor, alpha);
    out.pointSize = max(1.0, uniforms.misc.y);
    return out;
}

fragment float4 vectorscope_fragment(VSOut in [[stage_in]]) {
    return in.color;
}
