extern crate zoom;
extern crate num;
extern crate nalgebra as na;
extern crate rand;
extern crate crossbeam;
extern crate glowygraph as gg;
extern crate glium;

use rand::{SeedableRng, Rng};
use zoom::*;
use num::{Zero, Float};

use na::{ToHomogeneous, Translation, Rotation};

mod bot;

type Vec3 = na::Vec3<f64>;

trait SpringPhysics: Particle<Vec3, f64> + Inertia<f64> {
    fn quanta(&self) -> f64;
}

impl Quanta<f64> for SpringPhysics {
    fn quanta(&self) -> f64 {
        self.quanta()
    }
}

impl PhysicsParticle<Vec3, f64> for SpringPhysics {}

trait GravityPhysics: Particle<Vec3, f64> + Inertia<f64> {
    fn quanta(&self) -> f64;
}

impl Quanta<f64> for GravityPhysics {
    fn quanta(&self) -> f64 {
        self.quanta()
    }
}

impl PhysicsParticle<Vec3, f64> for GravityPhysics {}

struct Thing {
    particle: BasicParticle<Vec3, f64>,
}

impl Thing {
    fn new(position: Vec3, velocity: Vec3) -> Self {
        Thing{
            particle: BasicParticle::new(1.0, position, velocity, 100000.0)
        }
    }
}

impl Ball<f64> for GravityPhysics {
    fn radius(&self) -> f64 {
        0.1
    }
}

impl Position<Vec3> for Thing {
    fn position(&self) -> Vec3 {
        self.particle.position()
    }
}

impl Velocity<Vec3> for Thing {
    fn velocity(&self) -> Vec3 {
        self.particle.velocity()
    }
}

impl Particle<Vec3, f64> for Thing {
    fn impulse(&self, vec: &Vec3) {
        self.particle.impulse(vec);
    }

    fn advance(&mut self, time: f64) {
        self.particle.advance(time);
    }
}

impl Inertia<f64> for Thing {
    fn inertia(&self) -> f64 {
        self.particle.inertia()
    }
}

impl GravityPhysics for Thing {
    fn quanta(&self) -> f64 {
        10.0
    }
}

impl SpringPhysics for Thing {
    fn quanta(&self) -> f64 {
        20.0
    }
}

fn main() {
    use glium::{DisplayBuild, Surface};
    let display = glium::glutin::WindowBuilder::new().build_glium().unwrap();
    let window = display.get_window().unwrap();
    let glowy = gg::Renderer::new(&display);

    //Set mouse cursor to middle
    {
        let (dimx, dimy) = display.get_framebuffer_dimensions();
        let (hdimx, hdimy) = (dimx/2, dimy/2);
        window.set_cursor_position(hdimx as i32, hdimy as i32).ok().unwrap();
    }

    let perspective = *na::Persp3::new(1.5, 1.0, 0.0, 500.0).to_mat().as_ref();
    let mut movement = na::Iso3::<f32>::new(
        na::Vec3::new(0.0, 0.0, 50.0),
        na::Vec3::new(0.0, 0.0, 0.0),
    );

    let mut upstate = glium::glutin::ElementState::Released;
    let mut dnstate = glium::glutin::ElementState::Released;
    let mut ltstate = glium::glutin::ElementState::Released;
    let mut rtstate = glium::glutin::ElementState::Released;
    let mut fdstate = glium::glutin::ElementState::Released;
    let mut bkstate = glium::glutin::ElementState::Released;

    struct SphereBall {
        color: [f32; 4],
        ball: Thing,
    }
    unsafe impl Sync for SphereBall {}
    unsafe impl Send for SphereBall {}
    let mut rng = rand::Isaac64Rng::from_seed(&[1, 3, 3, 4]);
    let mut sballs = (0..10000).map(|i| SphereBall{
        color: [
            (i as f32 * 0.134).sin()*0.8 + 0.2,
            (i as f32 * 0.17).sin()*0.8 + 0.2,
            (i as f32 * 0.2).sin()*0.8 + 0.2,
            1.0
        ],
        ball: Thing::new(Vec3::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5, rng.next_f64() - 0.5) * 10.0, Vec3::zero()),
    }).collect::<Vec<_>>();

    let thread_total = 4;
    let helix_order = 8;
    let len = sballs.len();

    struct TheCenter (());

    impl Position<Vec3> for TheCenter {
        fn position(&self) -> Vec3 {
            Vec3::new(0.0, 0.0, 0.0)
        }
    }

    impl Quanta<f64> for TheCenter {
        fn quanta(&self) -> f64 {
            1.0
        }
    }

    impl Ball<f64> for TheCenter {
        fn radius(&self) -> f64 {
            1.0
        }
    }

    loop {
        crossbeam::scope(|scope| {
            let sballs = &sballs;
            let handles = (0..thread_total).map(|t| {
                scope.spawn(move || {
                    for i in (len*t/thread_total)..(len*(t+1)/thread_total) {
                        SpringPhysics::hooke_to::<SpringPhysics>(&sballs[i].ball,
                            &sballs[((i + len - 1) % len)].ball, 2.0);
                        SpringPhysics::hooke_to::<SpringPhysics>(&sballs[i].ball,
                            &sballs[((i + len + 1) % len)].ball, 2.0);
                        SpringPhysics::hooke_to::<SpringPhysics>(&sballs[i].ball,
                            &sballs[((i + len + len/helix_order) % len)].ball, 2.0);
                        GravityPhysics::gravitate_radius_to::<GravityPhysics>(&sballs[i].ball,
                            &sballs[((i + len + len/helix_order) % len)].ball, -10.0);
                        GravityPhysics::gravitate_radius_to(&sballs[i].ball,
                            &TheCenter(()), -100000.0);
                    }
                })
            }).collect::<Vec<_>>();

            for handle in handles {
                handle.join();
            }
        });

        for sball in sballs.iter_mut() {
            GravityPhysics::drag(&mut sball.ball, 30000.0);
            sball.ball.advance(2.0);
        }

        let mut target = display.draw();
        target.clear_color(0.0, 0.0, 0.0, 1.0);

        //Render edges
        glowy.render_edges(
            &mut target,
            movement.to_homogeneous().as_ref(),
            &perspective,
            &(0..len).flat_map(|i| {
                    std::iter::once(gg::Node{
                        position: {
                            let Vec3{x, y, z} = sballs[i].ball.position();
                            [x as f32, y as f32, z as f32]
                        },
                        color: sballs[i].color,
                        falloff: 0.15,
                    }).chain(
                        std::iter::once(gg::Node{
                            position: {
                                let Vec3{x, y, z} = sballs[(i + 1) % len].ball.position();
                                [x as f32, y as f32, z as f32]
                            },
                            color: sballs[(i + 1) % len].color,
                            falloff: 0.15,
                        })
                    ).chain(
                        std::iter::once(gg::Node{
                            position: {
                                let Vec3{x, y, z} = sballs[i].ball.position();
                                [x as f32, y as f32, z as f32]
                            },
                            color: sballs[i].color,
                            falloff: 0.15,
                        })
                    ).chain(
                        std::iter::once(gg::Node{
                            position: {
                                let Vec3{x, y, z} = sballs[(i + len/helix_order) % len].ball.position();
                                [x as f32, y as f32, z as f32]
                            },
                            color: sballs[(i + len/helix_order) % len].color,
                            falloff: 0.15,
                        })
                    )
                }
            ).collect::<Vec<_>>()[..]
        );

        target.finish().unwrap();

        for ev in display.poll_events() {
            match ev {
                glium::glutin::Event::Closed => return,
                glium::glutin::Event::KeyboardInput(state, _, Some(glium::glutin::VirtualKeyCode::W)) => {
                    fdstate = state;
                },
                glium::glutin::Event::KeyboardInput(state, _, Some(glium::glutin::VirtualKeyCode::S)) => {
                    bkstate = state;
                },
                glium::glutin::Event::KeyboardInput(state, _, Some(glium::glutin::VirtualKeyCode::A)) => {
                    ltstate = state;
                },
                glium::glutin::Event::KeyboardInput(state, _, Some(glium::glutin::VirtualKeyCode::D)) => {
                    rtstate = state;
                },
                glium::glutin::Event::KeyboardInput(state, _, Some(glium::glutin::VirtualKeyCode::Q)) => {
                    dnstate = state;
                },
                glium::glutin::Event::KeyboardInput(state, _, Some(glium::glutin::VirtualKeyCode::E)) => {
                    upstate = state;
                },
                glium::glutin::Event::MouseMoved((x, y)) => {
                    let (dimx, dimy) = display.get_framebuffer_dimensions();
                    let (hdimx, hdimy) = (dimx/2, dimy/2);
                    movement.append_rotation_mut(&na::Vec3::new(-(y - hdimy as i32) as f32 / 128.0,
                        (x - hdimx as i32) as f32 / 128.0, 0.0));
                    window.set_cursor_position(hdimx as i32, hdimy as i32).ok().unwrap();
                },
                _ => ()
            }
        }

        if upstate == glium::glutin::ElementState::Pressed {
            movement.append_translation_mut(&na::Vec3::new(0.0, -0.2, 0.0));
        }
        if dnstate == glium::glutin::ElementState::Pressed {
            movement.append_translation_mut(&na::Vec3::new(0.0, 0.2, 0.0));
        }
        if ltstate == glium::glutin::ElementState::Pressed {
            movement.append_translation_mut(&na::Vec3::new(-0.2, 0.0, 0.0));
        }
        if rtstate == glium::glutin::ElementState::Pressed {
            movement.append_translation_mut(&na::Vec3::new(0.2, 0.0, 0.0));
        }
        if fdstate == glium::glutin::ElementState::Pressed {
            movement.append_translation_mut(&na::Vec3::new(0.0, 0.0, -0.2));
        }
        if bkstate == glium::glutin::ElementState::Pressed {
            movement.append_translation_mut(&na::Vec3::new(0.0, 0.0, 0.2));
        }
    }
}
