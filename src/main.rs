extern crate glowygraph as gg;
extern crate petgraph;
extern crate glium;
extern crate num;
extern crate nalgebra as na;
extern crate zoom;
extern crate rand;

const SEPARATION_MAGNITUDE: f64 = 0.1;

use na::{ToHomogeneous, Translation, Rotation};
use num::traits::One;

mod bot;
use bot::*;
mod node;
use node::*;
mod rank;
use rank::*;

pub type Vec3 = na::Vec3<f64>;

fn vec_to_spos(v: Vec3) -> [f32; 3] {
    match v {
        Vec3{x, y, z} => [x as f32, y as f32, z as f32]
    }
}

fn main() {
    use glium::DisplayBuild;
    use rand::{SeedableRng, Rng};
    let mut rng = rand::Isaac64Rng::from_seed(&[1, 2, 3, 4]);

    let display = glium::glutin::WindowBuilder::new().with_vsync().build_glium().unwrap();
    let window = display.get_window().unwrap();
    //window.set_cursor_state(glium::glutin::CursorState::Hide).ok().unwrap();
    let glowy = gg::Renderer::new(&display);

    let mut deps = petgraph::Graph::<Node, bool, petgraph::Undirected>::new_undirected();
    deps.add_node(Node::new(5000, zoom::BasicParticle::default()));

    //Set mouse cursor to middle
    {
        let (dimx, dimy) = display.get_framebuffer_dimensions();
        let (hdimx, hdimy) = (dimx/2, dimy/2);
        window.set_cursor_position(hdimx as i32, hdimy as i32).ok().unwrap();
    }

    let perspective = *na::Persp3::new(1.5, 1.0, 0.0, 500.0).to_mat().as_ref();
    let mut movement = na::Iso3::<f32>::one();

    let mut upstate = glium::glutin::ElementState::Released;
    let mut dnstate = glium::glutin::ElementState::Released;
    let mut ltstate = glium::glutin::ElementState::Released;
    let mut rtstate = glium::glutin::ElementState::Released;
    let mut fdstate = glium::glutin::ElementState::Released;
    let mut bkstate = glium::glutin::ElementState::Released;

    loop {
        use glium::Surface;

        let mut target = display.draw();
        target.clear_color(0.0, 0.0, 0.0, 1.0);

        let matr = movement.to_homogeneous() * 3.0;

        //Update forces between nodes
        for i in deps.edge_indices() {
            let node_indices = deps.edge_endpoints(i).unwrap();
            let nodes = deps.index_twice_mut(node_indices.0, node_indices.1);

            //Apply spring forces to keep them together
            zoom::hooke(&nodes.0.particle, &nodes.1.particle, 0.001);
        }

        {
            let nodes = deps.raw_nodes();
            for i in 0..nodes.len() {
                for j in (i+1)..nodes.len() {
                    //Apply repulsion forces to keep them from being too close
                    zoom::gravitate_radius(&nodes[i].weight.particle, &nodes[j].weight.particle, -0.001);
                }
            }
        }

        //Update nodes
        for i in deps.node_indices() {
            if deps.node_weight(i).unwrap().should_split() {
                use std::f64::consts::PI;
                use num::traits::Float;
                let theta = rng.gen_range(0.0, 2.0 * PI);
                let phi = rng.gen_range(-1.0, 1.0).acos();
                let rand_unit_dir = Vec3::new(
                    theta.cos() * phi.sin(),
                    theta.sin() * phi.sin(),
                    phi.cos(),
                );

                //Divide energy in half before splitting
                deps.node_weight_mut(i).unwrap().energy /= 2;

                //Destroy all bots in node
                deps.node_weight_mut(i).unwrap().bots.clear();

                let nnode = {
                    let nref = deps.node_weight(i).unwrap();
                    Node::new(
                        nref.energy,
                        nref.particle.p.clone(),
                    )
                };

                let newindex = deps.add_node(nnode);

                deps.add_edge(i, newindex, false);

                //Add a positive impulse to this particle
                deps.node_weight_mut(i).unwrap().particle.p.velocity =
                    deps.node_weight_mut(i).unwrap().particle.p.velocity +
                    rand_unit_dir * SEPARATION_MAGNITUDE;

                //Add a negative impulse to the other particle
                deps.node_weight_mut(newindex).unwrap().particle.p.velocity =
                    deps.node_weight_mut(newindex).unwrap().particle.p.velocity -
                    rand_unit_dir * SEPARATION_MAGNITUDE;
            }

            deps.node_weight_mut(i).unwrap().advance();
        }

        //Make arrays for bot brain inputs
        let mut node_inputs = [0i64; nodebrain::TOTAL_INPUTS];
        let mut bot_inputs = [0i64; botbrain::TOTAL_INPUTS];
        let mut final_inputs = [0i64; finalbrain::TOTAL_INPUTS];

        //Update bots in nodes
        for i in deps.node_indices() {
            let pnode = deps.node_weight(i).unwrap();
            //The current node is always 0; everything else comes after in no particular order
            let neighbors = std::iter::once(i).chain(deps.neighbors(i)).collect::<Vec<_>>();



            //Iterate through all bots (b) in the node being processed
            for (i, b) in pnode.bots.iter().enumerate() {
                use std::collections::BinaryHeap;
                //Create a BTree to rank the nodes and fill it with default nodes
                let mut map = BinaryHeap::from(
                    vec![Rank{rank: 0, data: [-1; nodebrain::TOTAL_OUTPUTS]}; finalbrain::TOTAL_NODE_INPUTS]
                );

                //Iterate through each node and produce the outputs
                for (i, &n) in neighbors.iter().enumerate() {
                    //Get the node reference
                    let n = deps.node_weight(n).unwrap();
                    //Set the inputs for the node brain
                    node_inputs[0] = n.energy;
                    node_inputs[1] = n.bots.len() as i64;
                    node_inputs[2] = pnode.bots.len() as i64;
                    node_inputs[3] = b.energy;
                }
            }
        }

        //Render nodes
        glowy.render_nodes(&mut target, matr.as_ref(), &perspective,
            &deps.node_weights_mut().map(|n|
                gg::Node{
                    position: vec_to_spos(n.particle.p.position),
                    color: n.color(),
                    falloff: 0.25,
                }
            ).collect::<Vec<_>>()[..]);

        //Render edges
        glowy.render_edges(
            &mut target,
            matr.as_ref(),
            &perspective,
            &deps.edge_indices().map(|e| deps.edge_endpoints(e)).flat_map(|n| {
                    let indices = n.unwrap().clone();
                    let nodes = (deps.node_weight(indices.0).unwrap(), deps.node_weight(indices.1).unwrap());
                    std::iter::once(gg::Node{
                        position: vec_to_spos(nodes.0.particle.p.position),
                        color: nodes.0.color(),
                        falloff: 0.25
                    }).chain(
                    std::iter::once(gg::Node{
                        position: vec_to_spos(nodes.1.particle.p.position),
                        color: nodes.1.color(),
                        falloff: 0.25
                    }))
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
            movement.append_translation_mut(&na::Vec3::new(0.0, -0.1, 0.0));
        }
        if dnstate == glium::glutin::ElementState::Pressed {
            movement.append_translation_mut(&na::Vec3::new(0.0, 0.1, 0.0));
        }
        if ltstate == glium::glutin::ElementState::Pressed {
            movement.append_translation_mut(&na::Vec3::new(-0.1, 0.0, 0.0));
        }
        if rtstate == glium::glutin::ElementState::Pressed {
            movement.append_translation_mut(&na::Vec3::new(0.1, 0.0, 0.0));
        }
        if fdstate == glium::glutin::ElementState::Pressed {
            movement.append_translation_mut(&na::Vec3::new(0.0, 0.0, -0.1));
        }
        if bkstate == glium::glutin::ElementState::Pressed {
            movement.append_translation_mut(&na::Vec3::new(0.0, 0.0, 0.1));
        }
    }
}
