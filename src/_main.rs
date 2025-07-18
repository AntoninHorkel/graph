// use ratatui::{
//     style::Color,
//     symbols::Marker,
//     widgets::{
//         Block, BorderType,
//         canvas::{Canvas, Circle},
//     },
// };
// let area = frame.area();
// let block = Block::bordered()
//     .border_type(BorderType::Rounded)
//     .title("TODO");
// let inner_area = block.inner(area);
// let x_aspect_ratio =
//     (inner_area.width as f64 / inner_area.height as f64 * 0.5).max(1.0);
// let y_aspect_ratio =
//     (inner_area.height as f64 / inner_area.width as f64 * 0.5).max(1.0);
// let canvas = Canvas::default()
//     .x_bounds([-x_aspect_ratio, x_aspect_ratio])
//     .y_bounds([-y_aspect_ratio, y_aspect_ratio])
//     .block(block)
//     .marker(Marker::Braille)
//     .paint(|ctx| {
//         ctx.draw(&Circle {
//             x: 0.0,
//             y: 0.0,
//             radius: 0.9,
//             color: Color::LightBlue,
//         });
//     });
// frame.render_widget(canvas, area);

use std::{
    collections::HashSet,
    io,
    time::{Duration, Instant},
};

use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, MouseButton, MouseEvent, MouseEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use rand::{Rng, rng};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::Size,
    style::Color,
    symbols::Marker,
    widgets::canvas::{Canvas, Circle, Line},
};

const REPULSION_FORCE: f64 = 3000.0;
const EDGE_ATTRACTION: f64 = 5000.0;
const DAMPING: f64 = 0.8;
const NODE_RADIUS: f64 = 1.0;
const NUM_NODES: usize = 30;
const EDGE_PROBABILITY: f64 = 0.1;
const ASPECT_RATIO: f64 = 2.0; // Terminal cells are typically 2x taller than wide

struct Node {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
}

struct Edge {
    n1: usize,
    n2: usize,
}

struct Graph {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}

struct App {
    graph: Graph,
    drag_node: Option<(usize, (f64, f64))>, // (node_index, (offset_x, offset_y))
    world_width: f64,
    world_height: f64,
}

impl App {
    fn new(world_width: f64, world_height: f64) -> Self {
        let mut rng = rng();
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        // Create nodes at random positions within world dimensions
        for _ in 0..NUM_NODES {
            nodes.push(Node {
                x: rng.random_range(NODE_RADIUS..(world_width - NODE_RADIUS)),
                y: rng.random_range(NODE_RADIUS..(world_height - NODE_RADIUS)),
                vx: 0.0,
                vy: 0.0,
            });
        }

        // Create edges randomly
        let mut edge_set = HashSet::new();
        for i in 0..NUM_NODES {
            for j in (i + 1)..NUM_NODES {
                if rng.random_bool(EDGE_PROBABILITY) {
                    if !edge_set.contains(&(i, j)) {
                        edges.push(Edge { n1: i, n2: j });
                        edge_set.insert((i, j));
                    }
                }
            }
        }

        App {
            graph: Graph { nodes, edges },
            drag_node: None,
            world_width,
            world_height,
        }
    }

    fn handle_mouse(&mut self, mouse: &MouseEvent, area: &Size) {
        let col = mouse.column as f64;
        let row = mouse.row as f64;

        // Convert screen coordinates to world coordinates
        let world_x = col * self.world_width / area.width as f64;
        let world_y = row * self.world_height / area.height as f64;

        match mouse.kind {
            MouseEventKind::Down(MouseButton::Left) => {
                for (i, node) in self.graph.nodes.iter().enumerate() {
                    let dx = node.x - world_x;
                    let dy = node.y - world_y;
                    let distance_squared = dx * dx + dy * dy;
                    if distance_squared < (NODE_RADIUS * NODE_RADIUS * 4.0) {
                        self.drag_node = Some((i, (dx, dy)));
                        break;
                    }
                }
            }
            MouseEventKind::Up(MouseButton::Left) => {
                if let Some((i, _)) = self.drag_node {
                    // Reset velocity when releasing node
                    self.graph.nodes[i].vx = 0.0;
                    self.graph.nodes[i].vy = 0.0;
                }
                self.drag_node = None;
            }
            MouseEventKind::Drag(MouseButton::Left) => {
                if let Some((i, (offset_x, offset_y))) = self.drag_node {
                    let node = &mut self.graph.nodes[i];
                    node.x = world_x + offset_x;
                    node.y = world_y + offset_y;

                    // Keep node within bounds
                    node.x = node.x.max(NODE_RADIUS).min(self.world_width - NODE_RADIUS);
                    node.y = node.y.max(NODE_RADIUS).min(self.world_height - NODE_RADIUS);
                }
            }
            _ => {}
        }
    }

    fn update_physics(&mut self, delta: f64) {
        // Apply repulsion forces
        for i in 0..self.graph.nodes.len() {
            if self.drag_node.map(|(idx, _)| idx == i).unwrap_or(false) {
                continue;
            }

            let mut fx = 0.0;
            let mut fy = 0.0;

            // Node repulsion
            for j in 0..self.graph.nodes.len() {
                if i == j {
                    continue;
                }

                let dx = self.graph.nodes[i].x - self.graph.nodes[j].x;
                let dy = self.graph.nodes[i].y - self.graph.nodes[j].y;
                let dist_sq = dx * dx + dy * dy;

                // Avoid division by zero and extreme forces
                if dist_sq < 0.01 {
                    continue;
                }

                // Inverse distance force
                let force = REPULSION_FORCE / dist_sq.sqrt();
                fx += force * dx / dist_sq.sqrt();
                fy += force * dy / dist_sq.sqrt();
            }

            // Edge attraction
            for edge in &self.graph.edges {
                if edge.n1 == i || edge.n2 == i {
                    let other_idx = if edge.n1 == i { edge.n2 } else { edge.n1 };
                    let other = &self.graph.nodes[other_idx];

                    let dx = other.x - self.graph.nodes[i].x;
                    let dy = other.y - self.graph.nodes[i].y;
                    let dist_sq = dx * dx + dy * dy;

                    if dist_sq < 0.01 {
                        continue;
                    }

                    // Attraction force (like a spring)
                    let force = EDGE_ATTRACTION * dist_sq.sqrt() / 100.0;
                    fx += force * dx / dist_sq.sqrt();
                    fy += force * dy / dist_sq.sqrt();
                }
            }

            let node = &mut self.graph.nodes[i];
            node.vx = (node.vx + fx * delta) * DAMPING;
            node.vy = (node.vy + fy * delta) * DAMPING;
            node.x += node.vx * delta;
            node.y += node.vy * delta;

            // Boundary constraints
            node.x = node.x.clamp(NODE_RADIUS, self.world_width - NODE_RADIUS);
            node.y = node.y.clamp(NODE_RADIUS, self.world_height - NODE_RADIUS);
        }
    }
}

fn main() -> io::Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(
        stdout,
        EnterAlternateScreen,
        crossterm::event::EnableMouseCapture
    )?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Get terminal size for world dimensions
    let initial_size = terminal.size()?;
    let world_width = initial_size.width as f64;
    let world_height = initial_size.height as f64;

    let mut app = App::new(world_width, world_height);
    let mut last_update = Instant::now();

    // Main loop
    loop {
        let delta = last_update.elapsed().as_secs_f64();
        last_update = Instant::now();

        // Process all pending events
        while event::poll(Duration::from_millis(0))? {
            match event::read()? {
                Event::Key(KeyEvent {
                    code: KeyCode::Char('q'),
                    ..
                }) => {
                    // Cleanup before exit
                    disable_raw_mode()?;
                    execute!(
                        terminal.backend_mut(),
                        LeaveAlternateScreen,
                        crossterm::event::DisableMouseCapture
                    )?;
                    return Ok(());
                }
                Event::Resize(_, _) => {
                    // Handle terminal resize
                    let new_size = terminal.size()?;
                    app.world_width = new_size.width as f64;
                    app.world_height = new_size.height as f64;
                }
                Event::Mouse(mouse) => {
                    let area = terminal.size()?;
                    app.handle_mouse(&mouse, &area);
                }
                _ => {}
            }
        }

        // Update physics
        app.update_physics(delta);

        // Render
        terminal.draw(|f| {
            let area = f.area();
            let canvas = Canvas::default()
                .x_bounds([0.0, app.world_width])
                .y_bounds([0.0, app.world_height * ASPECT_RATIO]) // Account for aspect ratio
                .marker(Marker::Braille) // Use braille for better resolution
                .paint(|ctx| {
                    // Draw edges
                    for edge in &app.graph.edges {
                        let n1 = &app.graph.nodes[edge.n1];
                        let n2 = &app.graph.nodes[edge.n2];
                        ctx.draw(&Line {
                            x1: n1.x,
                            y1: n1.y * ASPECT_RATIO, // Adjust for aspect ratio
                            x2: n2.x,
                            y2: n2.y * ASPECT_RATIO, // Adjust for aspect ratio
                            color: Color::DarkGray,
                        });
                    }

                    // Draw nodes
                    for (i, node) in app.graph.nodes.iter().enumerate() {
                        let color = if app.drag_node.map(|(idx, _)| idx == i).unwrap_or(false) {
                            Color::LightRed
                        } else {
                            Color::LightBlue
                        };
                        ctx.draw(&Circle {
                            x: node.x,
                            y: node.y * ASPECT_RATIO, // Adjust for aspect ratio
                            radius: NODE_RADIUS,
                            color,
                        });
                    }
                });
            f.render_widget(canvas, area);
        })?;
    }
}
