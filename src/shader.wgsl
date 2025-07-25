struct UniformBufferObject {
    time: f32,
}

struct StorageBufferObject {
    nodes_length: u32,
    edges_length: u32,
    nodes: array<u32>,
    edges: array<u32>,
}

@group(0) @binding(0) var texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> u: UniformBufferObject;
@group(0) @binding(2) var<storage, read_write> s: StorageBufferObject;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // if all(id.xy == vec2<u32>(0)) {
    //     // TODO: Simulate physics
    // }
    // workgroupBarrier();

    let size = textureDimensions(texture);
    if any(id.xy >= size) {
        return;
    }

    let sizef = vec2<f32>(size);
    let sizef_min = min(sizef.x, sizef.y);
    let uv = (vec2<f32>(id.xy) + (sizef_min - sizef) * 0.5) / sizef_min;

    // TODO: Render
    let color = 0.5 + 0.5 * cos(u.time + uv.xyx + vec3<f32>(0.0, 2.0, 4.0));

    textureStore(texture, id.xy, vec4<f32>(color, 1.0));
}
