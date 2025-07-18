@group(0) @binding(0) var texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> time: f32;

fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3<f32>(0.0), vec3<f32>(1.0)), c.y);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(texture);
    if any(id.xy >= size) {
        return;
    }
    let sizef = vec2<f32>(size);
    let sizef_min = min(sizef.x, sizef.y);
    let uv = (vec2<f32>(id.xy) + (sizef_min - sizef) * 0.5) / sizef_min;

    let center = uv - 0.5;
    let r = length(center);
    let wave = 0.5 + 0.5 * sin((r * 20.0) - (time * 0.5));
    let fade = smoothstep(0.5, 0.0, r); // fade toward edge
    let hue = fract(time * 0.1 + r * 0.2);
    let sat = 0.8;
    let val = wave * fade;
    let color = hsv2rgb(vec3<f32>(hue, sat, val));
    textureStore(texture, id.xy, vec4<f32>(color, 1.0));
}
