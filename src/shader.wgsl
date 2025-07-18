@group(0) @binding(0) var texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> time: f32;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(texture);
    if any(id.xy >= size) {
        return;
    }
    let sizef = vec2<f32>(size);
    let sizef_min = min(sizef.x, sizef.y);
    let uv = (vec2<f32>(id.xy) + (sizef_min - sizef) * 0.5) / sizef_min;

    let color = 0.5 + 0.5 * cos(time + uv.xyx + vec3<f32>(0.0, 2.0, 4.0));
    textureStore(texture, id.xy, vec4<f32>(color, 1.0));
}
