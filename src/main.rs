use std::{
    io,
    io::Error as IoError,
    panic,
    time::{Duration, Instant},
};

use crossterm::{
    event,
    event::{DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyModifiers},
};
use image::{DynamicImage, RgbaImage};
use ratatui::Frame;
use ratatui_image::{Resize, StatefulImage, errors::Errors as ImageError, picker::Picker};
// use ratatui_image::protocol::StatefulProtocol;
use thiserror::Error;
use tracy_client::{Client as TracyClient, frame_mark, span};
use wgpu::{
    BackendOptions,
    Backends,
    BindGroup,
    BindGroupDescriptor,
    BindGroupEntry,
    BindGroupLayout,
    BindGroupLayoutDescriptor,
    BindGroupLayoutEntry,
    BindingResource,
    BindingType,
    Buffer,
    BufferBindingType,
    BufferDescriptor,
    BufferSize,
    BufferUsages,
    CommandEncoderDescriptor,
    ComputePassDescriptor,
    ComputePipeline,
    ComputePipelineDescriptor,
    Device,
    DeviceDescriptor,
    Extent3d,
    Features,
    Instance,
    InstanceDescriptor,
    InstanceFlags,
    Limits,
    MapMode,
    MemoryBudgetThresholds,
    MemoryHints,
    PipelineCacheDescriptor,
    PipelineCompilationOptions,
    PipelineLayoutDescriptor,
    PollType,
    PowerPreference,
    Queue,
    RequestAdapterError,
    RequestAdapterOptions,
    RequestDeviceError,
    ShaderStages,
    StorageTextureAccess,
    TexelCopyBufferInfo,
    TexelCopyBufferLayout,
    Texture,
    TextureAspect,
    TextureDescriptor,
    TextureDimension,
    TextureFormat,
    TextureUsages,
    TextureViewDescriptor,
    TextureViewDimension,
    Trace,
};

#[derive(Error, Debug)]
#[non_exhaustive]
enum ThisError {
    #[error("IO error: {0}.")]
    Io(#[from] IoError),
    #[error("Image error: {0}.")]
    Image(#[from] ImageError),
    #[error("No suitable `wgpu::Adapter` found: {0}.")]
    AdapterNotFound(#[from] RequestAdapterError),
    #[error("No suitable `wgpu::Device` found: {0}.")]
    DeviceNotFound(#[from] RequestDeviceError),
    // TODO: PollError desn't implement `std::error::Error`.
    // #[error("Poll error: {0}.")]
    // Poll(#[from] PollError),
    #[error("Poll error.")]
    Poll,
    #[error("Image generation error.")]
    ImageGen,
}

type ThisResult<T> = Result<T, ThisError>;

#[derive(Clone, Copy)]
struct Size {
    width: u32,
    height: u32,
}

struct WgpuInstance {
    device: Device,
    queue: Queue,
    bind_group_layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

struct WgpuResources {
    texture: Texture,
    uniform_buffer: Buffer,
    bind_group: BindGroup,
    buffer_bytes_per_row: u32,
    buffer: Buffer,
}

impl WgpuInstance {
    async fn new() -> ThisResult<Self> {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::PRIMARY,
            flags: InstanceFlags::from_build_config(),
            memory_budget_thresholds: MemoryBudgetThresholds::default(),
            backend_options: BackendOptions::from_env_or_default(),
        });
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await?;
        let use_pipeline_cache = adapter.features().contains(Features::PIPELINE_CACHE);
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: None,
                required_features: if use_pipeline_cache { Features::PIPELINE_CACHE } else { Features::empty() },
                required_limits: Limits::downlevel_defaults()
                    .using_resolution(adapter.limits())
                    .using_alignment(adapter.limits()),
                memory_hints: MemoryHints::Performance,
                trace: Trace::Off,
            })
            .await?;
        let shader_module = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(size_of::<f32>() as u64),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let maybe_pipeline_cache = if use_pipeline_cache {
            #[allow(unsafe_code)]
            Some(&unsafe {
                device.create_pipeline_cache(&PipelineCacheDescriptor {
                    label: None,
                    data: None,
                    fallback: true,
                })
            })
        } else {
            None
        };
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions {
                constants: Default::default(),
                zero_initialize_workgroup_memory: false,
            },
            cache: maybe_pipeline_cache,
        });
        Ok(Self {
            device,
            queue,
            bind_group_layout,
            pipeline,
        })
    }

    fn resources(&self, size: Size) -> WgpuResources {
        let texture = self.device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&TextureViewDescriptor {
            label: None,
            format: None,
            dimension: None,
            usage: None,
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });
        let uniform_buffer = self.device.create_buffer(&BufferDescriptor {
            label: None,
            size: size_of::<f32>() as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM, // TODO
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let buffer_bytes_per_row = (size.width * 4).div_ceil(256) * 256; // TODO: Verify!
        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: None,
            size: u64::from(buffer_bytes_per_row * size.height),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        WgpuResources {
            texture,
            uniform_buffer,
            bind_group,
            buffer_bytes_per_row,
            buffer,
        }
    }
}

struct WgpuState {
    size: Size,
    instance: WgpuInstance,
    resources: WgpuResources,
    instant: Instant,
}

impl WgpuState {
    fn new(size: Size) -> ThisResult<Self> {
        let _zone = span!("WGPU state init");
        let instance = pollster::block_on(WgpuInstance::new())?;
        let resources = instance.resources(size);
        let instant = Instant::now();
        Ok(Self {
            size,
            instance,
            resources,
            instant,
        })
    }

    fn resize(&mut self, size: Size) {
        self.size = size;
        self.resources = self.instance.resources(size);
    }

    pub fn draw_with<F>(&self, function: F) -> ThisResult<()>
    where
        F: FnOnce(DynamicImage) -> ThisResult<()>, // TODO: Reference?
    {
        let _zone = span!("WGPU draw");
        self.instance.queue.write_buffer(
            &self.resources.uniform_buffer,
            0,
            &self.instant.elapsed().as_secs_f32().to_ne_bytes(),
        );
        let mut command_encoder = self.instance.device.create_command_encoder(&CommandEncoderDescriptor {
            label: None,
        });
        let _zone = span!("WGPU draw: compute pass");
        {
            let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.instance.pipeline);
            pass.set_bind_group(0, &self.resources.bind_group, &[]);
            pass.dispatch_workgroups(self.size.width.div_ceil(16), self.size.height.div_ceil(16), 1);
        }
        let _zone = span!("WGPU draw: texture to buffer");
        command_encoder.copy_texture_to_buffer(
            self.resources.texture.as_image_copy(),
            TexelCopyBufferInfo {
                buffer: &self.resources.buffer,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.resources.buffer_bytes_per_row),
                    rows_per_image: None,
                },
            },
            self.resources.texture.size(),
        );
        self.instance.queue.submit(Some(command_encoder.finish()));
        self.instance.device.poll(PollType::Wait).map_err(|_| ThisError::Poll)?; // TODO
        let _zone = span!("WGPU draw: buffer async map");
        self.resources.buffer.map_async(MapMode::Read, .., |_| {});
        self.instance.device.poll(PollType::Wait).map_err(|_| ThisError::Poll)?;
        let _zone = span!("WGPU draw: buffer alignment fix");
        let mut pixels = Vec::with_capacity((self.size.width * self.size.height * 4) as usize);
        {
            let buffer_view = self.resources.buffer.get_mapped_range(..);
            // TODO: Benchmark and optimize, possibly async.
            for row in 0..self.size.height {
                let start = (row * self.resources.buffer_bytes_per_row) as usize;
                let end = start + (self.size.width * 4) as usize;
                pixels.extend_from_slice(&buffer_view[start..end]);
            }
        }
        let _zone = span!("WGPU draw: unmap");
        self.resources.buffer.unmap();
        let _zone = span!("WGPU draw: callback");
        (function)(RgbaImage::from_raw(self.size.width, self.size.height, pixels).ok_or(ThisError::ImageGen)?.into())
    }
}

struct App {
    picker: Picker,
    // image: StatefulProtocol,
    wgpu_state: WgpuState,
}

impl App {
    fn new(size: Size) -> ThisResult<Self> {
        let _zone = span!("App state init");
        let picker = Picker::from_query_stdio()?;
        let size = Size {
            width: size.width * u32::from(picker.font_size().0),
            height: size.height * u32::from(picker.font_size().1),
        };
        let wgpu_state = WgpuState::new(size)?;
        Ok(Self {
            picker,
            wgpu_state,
        })
    }

    #[inline]
    fn resize(&mut self, size: Size) {
        let size = Size {
            width: size.width * u32::from(self.picker.font_size().0),
            height: size.height * u32::from(self.picker.font_size().1),
        };
        self.wgpu_state.resize(size);
    }

    fn draw(&self, frame: &mut Frame<'_>) -> ThisResult<()> {
        self.wgpu_state.draw_with(|image| {
            let mut image = self.picker.new_resize_protocol(image);
            frame.render_stateful_widget(
                // TODO: https://docs.rs/ratatui-image/8.0.1/ratatui_image/enum.FilterType.html selection or maybe sampling using a shader?
                StatefulImage::new().resize(Resize::Scale(None)),
                frame.area(),
                &mut image,
            );
            Ok(())
        })
    }
}

fn main() {
    TracyClient::start();
    let original_hook = panic::take_hook();
    panic::set_hook(Box::new(move |panic| {
        ratatui::restore();
        original_hook(panic);
    }));
    let mut terminal = ratatui::init();
    let mut stdout = io::stdout();
    crossterm::execute!(stdout, EnableMouseCapture).expect("Failed to enable mouse capture");
    let size = terminal.get_frame().area(); // TODO: `terminal.size()` might fail.
    let mut app = App::new(Size {
        width: u32::from(size.width),
        height: u32::from(size.height),
    })
    .expect("Failed to initialize app");
    let mut timeout = Duration::from_millis(16);
    'main: loop {
        let tick = Instant::now();
        while event::poll(Duration::from_millis(16).saturating_sub(timeout)).expect("Failed to poll event") {
            match event::read().expect("Failed to read event") {
                Event::Resize(width, height) => app.resize(Size {
                    width: u32::from(width),
                    height: u32::from(height),
                }),
                Event::Key(
                    KeyEvent {
                        code: KeyCode::Char('c'),
                        modifiers: KeyModifiers::CONTROL,
                        ..
                    }
                    | KeyEvent {
                        code: KeyCode::Char('q'),
                        modifiers: KeyModifiers::NONE,
                        ..
                    },
                ) => break 'main,
                _ => {}
            }
        }
        let _zone = span!("Main draw call");
        terminal.draw(|frame| app.draw(frame).expect("Failed to draw")).expect("Failed to draw");
        // app.image.last_encoding_result()?;
        frame_mark();
        timeout = tick.elapsed();
    }
    crossterm::execute!(stdout, DisableMouseCapture).expect("Failed to disable mouse capture");
    ratatui::restore();
}
