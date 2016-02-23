extern crate glowygraph as gg;
extern crate petgraph;
extern crate glium;
extern crate num;
extern crate nalgebra as na;
extern crate zoom;
extern crate rand;
extern crate itertools;
extern crate mli;
use itertools::*;

//Magnitude of flinging apart of a node that split
const SEPARATION_MAGNITUDE: f64 = 0.02;
const SEPARATION_DELTA: f64 = 10.0;
//Magnitude of repulsion between all particles
const REPULSION_MAGNITUDE: f64 = 400.0;
const ATTRACTION_MAGNITUDE: f64 = 0.01;
//const BOT_GRAVITATION_MAGNITUDE: f64 = 0.0;
const PULL_CENTER_MAGNITUDE: f64 = 0.005;
//Probability of connecting after node is destroyed
const CONNECT_PROBABILITY: f64 = 0.0;
const CONNECT_AFTER: f64 = 35.0;
const CONNECT_MAX_LENGTH: f64 = 400.0;
//const CONNECT_MIN_LENGTH: f64 = 10.0;
//The length within which bots can connect their nodes together by choice
const BOT_COICE_CONNECT_LENGTH: f64 = 500.0;
const FRAME_PHYSICS_PERIOD: u64 = 1;
const BOT_PULL_MAGNITUDE: f64 = 150.0;

const STARTING_POSITION: f32 = 1000.0;
const MOVE_SPEED: f32 = 5.0;
const ROTATION_RATE: f32 = 0.005;

const START_SPAWNING_AT: i64 = 50000;
//Energy stops being generated after this many nodes exist
const ENERGY_CUTOFF_AT: usize = 1500;
const SPAWN_RATE: f64 = 1.0/(START_SPAWNING_AT as f64);
const NODE_STARTING_ENERGY: i64 = 200000;
//const FINAL_SPAWN_CYCLE: u64 = 0;
const NEW_NODE_SPAWNS: usize = 0;
//Cycle mutation rate; always mutates on division either way
const MUTATION_RATE: f64 = 0.0;
//The rate at which a bot will be spawned in empty nodes when the mesh is full
const EMPTY_NODE_FULL_MESH_SPAWN_RATE: f64 = 0.005;
//Minimum channel magnitude to connect
const CONNECT_SIGNAL_MIN: i64 = 16;

const EDGE_FALLOFF: f32 = 0.05;
const NODE_FALLOFF: f32 = 0.25;

use na::{ToHomogeneous, Translation, Rotation};

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
    use num::Zero;
    use rand::{SeedableRng, Rng};
    let mut rng = rand::Isaac64Rng::from_seed(&[51, 2, 2, 4]);

    let display = glium::glutin::WindowBuilder::new().with_vsync()
    .with_fullscreen(glium::glutin::get_available_monitors().next().unwrap())
    .build_glium().unwrap();
    let window = display.get_window().unwrap();
    /*match window.set_cursor_state(glium::glutin::CursorState::Hide) {
        Ok(_) => {},
        Err(_) => println!("Cursor hide not available on this platform; starting without it."),
    }*/
    let glowy = gg::Renderer::new(&display);
    let mut focus_state = true;

    let mut deps: petgraph::Graph<Node, (), petgraph::Undirected> = petgraph::Graph::new_undirected();

    let central = zoom::BasicParticle::new(1.0, Vec3::zero(), Vec3::zero(), 1.0);

    //Set mouse cursor to middle
    {
        let (dimx, dimy) = display.get_framebuffer_dimensions();
        let (hdimx, hdimy) = (dimx/2, dimy/2);
        window.set_cursor_position(hdimx as i32, hdimy as i32).ok().unwrap();
    }

    let mut movement = na::Iso3::<f32>::new(
        na::Vec3::new(0.0, 0.0, STARTING_POSITION),
        na::Vec3::new(0.0, 0.0, 0.0),
    );

    let mut upstate = glium::glutin::ElementState::Released;
    let mut dnstate = glium::glutin::ElementState::Released;
    let mut ltstate = glium::glutin::ElementState::Released;
    let mut rtstate = glium::glutin::ElementState::Released;
    let mut fdstate = glium::glutin::ElementState::Released;
    let mut bkstate = glium::glutin::ElementState::Released;

    let mut period = 0u64;

    loop {
        use glium::Surface;
        use std::collections::BinaryHeap;

        let mut target = display.draw();
        target.clear_color(0.0, 0.0, 0.0, 1.0);

        let matr = movement.to_homogeneous() * 3.0;

        //Print things out
        println!("Nodes: {}, Edges: {}", deps.node_count(), deps.edge_count());

        //Add node if none exist
        if deps.node_count() == 0 {
            deps.add_node(Node::new(NODE_STARTING_ENERGY, zoom::BasicParticle::default()));
        }

        //Update forces between nodes on the correct periods
        if period % FRAME_PHYSICS_PERIOD == 0 {
            for i in deps.edge_indices() {
                let node_indices = deps.edge_endpoints(i).unwrap();
                let nodes = deps.index_twice_mut(node_indices.0, node_indices.1);

                //Apply spring forces to keep them together
                zoom::hooke(&nodes.0.particle, &nodes.1.particle, ATTRACTION_MAGNITUDE /
                    (nodes.0.connections as f64 * nodes.1.connections as f64).sqrt());
                zoom::gravitate_radius(&nodes.0.particle, &nodes.1.particle, BOT_PULL_MAGNITUDE *
                (nodes.0.bots.len() as f64 *
                    (1.0/(1.0 + (nodes.0.pull as f64).exp()) - 0.5) +
                nodes.1.bots.len() as f64 *
                    (1.0/(1.0 + (nodes.1.pull as f64).exp()) - 0.5)));
            }

            let nc = deps.node_count();
            for n in deps.node_weights_mut() {
                if nc < ENERGY_CUTOFF_AT {
                    n.grow(false, &mut rng);
                } else {
                    n.grow(true, &mut rng);
                    if n.bots.len() == 0 {
                        if rng.gen_range(0.0, 1.0) < EMPTY_NODE_FULL_MESH_SPAWN_RATE {
                            n.bots.push(Box::new(Bot::new(&mut rng)));
                        }
                    }
                }
            }
        }

        //Determine how many nodes will spawn
        let spawners = rng.gen_range(0.0, (SPAWN_RATE * deps.node_count() as f64).powi(2)) as usize;
        let mut spawn_places = (0..spawners).map(|_| Rank{rank: rng.gen_range(0, deps.node_count() as i64), data: ()}).collect::<BinaryHeap<_>>();

        //Update nodes
        for (ix, i) in deps.node_indices().enumerate() {
            if deps[i].should_split() {
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
                deps[i].energy /= 2;

                let nnode = {
                    let nref = &deps[i];
                    Node::new(
                        nref.energy,
                        nref.particle.p.clone(),
                    )
                };

                let newindex = deps.add_node(nnode);
                //Add all of the old node's neighbors
                let it = deps.neighbors(i).collect_vec();
                for iin in it {
                    if rng.gen_range(0.0, 1.0) < 0.5 {
                        deps.add_edge(newindex, iin, ());
                        deps[newindex].connections += 1;
                        deps[i].connections -= 1;
                        let ed = deps.find_edge(iin, i).unwrap();
                        deps.remove_edge(ed);
                    }
                }

                for ib in (0..deps[i].bots.len()).rev() {
                    if rng.gen_range(0.0, 1.0) < 0.5 {
                        let b = deps[i].bots.swap_remove(ib);
                        deps[newindex].bots.push(b);
                    }
                }

                //Add the old node as a neighbor
                deps.add_edge(i, newindex, ());
                deps[i].connections += 1;
                deps[newindex].connections += 1;

                //Add a positive impulse to this particle
                deps[i].particle.p.velocity =
                    deps[i].particle.p.velocity +
                    rand_unit_dir * SEPARATION_MAGNITUDE;

                //Add a negative impulse to the other particle
                deps[newindex].particle.p.velocity =
                    deps[newindex].particle.p.velocity -
                    rand_unit_dir * SEPARATION_MAGNITUDE;

                //Move the particles far enough away from each other so they can stay connected
                deps[i].particle.p.position =
                    deps[i].particle.p.position + rand_unit_dir * SEPARATION_DELTA;
                deps[newindex].particle.p.position =
                    deps[newindex].particle.p.position - rand_unit_dir * SEPARATION_DELTA;

                //Add new bots to the nodes
                for _ in 0..NEW_NODE_SPAWNS {
                    deps[i].bots.push(Box::new(Bot::new(&mut rng)));
                    deps[newindex].bots.push(Box::new(Bot::new(&mut rng)));
                }
            }

            while let Some(&Rank{rank: ri, ..}) = spawn_places.peek() {
                if ri as usize == ix {
                    deps[i].bots.push(Box::new(Bot::new(&mut rng)));
                    spawn_places.pop();
                } else {
                    break;
                }
            }
        }

        //Update obliteration
        for i in deps.node_indices().rev() {
            deps[i].advance();
            if deps[i].should_obliterate() {
                let neighbors = deps.neighbors(i).collect_vec();
                for ix in 0..neighbors.len() {
                    for jx in (ix+1)..neighbors.len() {
                        if rng.gen_range(0.0, 1.0) < CONNECT_PROBABILITY {
                            deps.update_edge(neighbors[ix], neighbors[jx], ());
                        }
                    }
                }
                deps.remove_node(i);
            }
        }

        for i in deps.edge_indices().rev() {
            if let Some((i1, i2)) = deps.edge_endpoints(i) {
                use zoom::{Position, Vector};
                let mag = (deps[i1].particle.position() - deps[i2].particle.position()).displacement_squared();
                if mag > CONNECT_MAX_LENGTH.powi(2)/* || mag < CONNECT_MIN_LENGTH.powi(2)*/ {
                    deps.remove_edge(i);
                    deps[i1].connections -= 1;
                    deps[i2].connections -= 1;
                }
            }
        }

        //Make arrays for bot brain inputs
        let mut node_inputs = [0i64; nodebrain::TOTAL_INPUTS];
        let mut bot_inputs = [0i64; botbrain::TOTAL_INPUTS];
        let mut final_inputs = [0i64; finalbrain::TOTAL_INPUTS];

        //Make the static values
        let statics = [0, 1, 2, -1];
        //Assign static values to each of the input arrays
        node_inputs.iter_mut().set_from(statics.iter().cloned());
        bot_inputs.iter_mut().set_from(statics.iter().cloned());
        final_inputs.iter_mut().set_from(statics.iter().cloned());

        //Update bots in nodes
        for i in deps.node_indices() {
            use std::collections::BinaryHeap;
            //Set pull to 0 before accumulating
            deps[i].pull = 0;

            //Make rng value on a node basis to avoid insane clustering
            let rngval = rng.gen();
            node_inputs[4] = rngval;
            bot_inputs[4] = rngval;
            final_inputs[4] = rngval;
            //The current node is always 0; everything else comes after
            let neighbors = std::iter::once(i).chain(deps.neighbors(i)).collect_vec();

            let mut movers = BinaryHeap::<usize>::new();
            let mut maters = Vec::<usize>::new();

            //Iterate through all bots (b) in the node being processed
            for ib in 0..deps[i].bots.len() {
                use mli::SISO;
                {
                    let ref pnode = deps[i];
                    //Create a BTree to rank the nodes and fill it with default nodes
                    let mut node_heap = BinaryHeap::from(
                        vec![Rank{rank: i64::min_value(), data: [0; nodebrain::TOTAL_OUTPUTS]}; finalbrain::TOTAL_NODE_INPUTS]
                    );

                    //Create a BTree to rank the nodes and fill it with default bots
                    let mut bot_heap = BinaryHeap::from(
                        vec![Rank{rank: i64::min_value(), data: [0; botbrain::TOTAL_OUTPUTS]}; finalbrain::TOTAL_BOT_INPUTS]
                    );

                    //Iterate through each node and produce the outputs
                    for (i, &n) in neighbors.iter().enumerate() {
                        //Get the node reference
                        let n = &deps[n];
                        //Set the inputs for the node brain
                        node_inputs[5] = n.energy;
                        node_inputs[6] = n.bots.len() as i64;
                        node_inputs[7] = pnode.bots.len() as i64;
                        node_inputs[8] = pnode.bots[ib].energy;
                        node_inputs[9] = pnode.connections;
                        node_inputs[10] = n.connections;
                        node_inputs[nodebrain::STATIC_INPUTS..].iter_mut().set_from(pnode.bots[ib].memory.iter().cloned());

                        let mut compute = pnode.bots[ib].node_brain.compute(&node_inputs[..]);

                        let rank = Rank{
                            rank: compute.next().unwrap(),
                            data: {
                                let mut l = [-1; nodebrain::TOTAL_OUTPUTS];
                                l[0] = i as i64;
                                l[1..].iter_mut().set_from(compute);
                                l
                            },
                        };

                        //Add this rank to the heap
                        node_heap.push(rank);
                        //Remove the lowest rank from the heap to stay at the same amount
                        node_heap.pop();
                    }

                    //Iterate through each bot and produce the outputs
                    for (iob, ob) in pnode.bots.iter().enumerate() {
                        //Set the inputs for the bot brain
                        bot_inputs[5] = pnode.energy;
                        bot_inputs[6] = pnode.bots.len() as i64;
                        bot_inputs[7] = pnode.bots[ib].energy;
                        bot_inputs[8] = ob.energy;
                        bot_inputs[9] = ob.signal;
                        bot_inputs[10] = pnode.connections;
                        bot_inputs[botbrain::STATIC_INPUTS..].iter_mut().set_from(pnode.bots[ib].memory.iter().cloned());

                        let mut compute = pnode.bots[ib].bot_brain.compute(&bot_inputs[..]);

                        let rank = Rank{
                            rank: compute.next().unwrap(),
                            data: {
                                let mut l = [-1; botbrain::TOTAL_OUTPUTS];
                                l[0] = iob as i64;
                                l[1..].iter_mut().set_from(compute);
                                l
                            },
                        };

                        //Add this rank to the heap
                        bot_heap.push(rank);
                        //Remove the lowest rank from the heap to stay at the same amount
                        bot_heap.pop();
                    }

                    //Make the bot's final decision

                    //Provide static inputs
                    final_inputs[5] = pnode.energy;
                    final_inputs[6] = pnode.bots.len() as i64;
                    final_inputs[7] = pnode.bots[ib].energy;
                    final_inputs[8] = ib as i64;
                    final_inputs[9] = pnode.connections;
                    final_inputs[finalbrain::STATIC_INPUTS..].iter_mut().set_from(
                        pnode.bots[ib].memory.iter().cloned().chain(
                            //Provide the highest ranking node inputs
                            node_heap.iter().flat_map(|r| r.data.iter().cloned())
                        ).chain(
                            //Provide the highest ranking bot inputs
                            bot_heap.iter().flat_map(|r| r.data.iter().cloned())
                        )
                    );
                }


                {
                    let mb = &mut *deps[i].bots[ib];
                    let (brain, memory, decision) = (&mut mb.final_brain, &mut mb.memory, &mut mb.decision);
                    let mut compute = brain.compute(&final_inputs[..]);
                    decision.mate = compute.next().unwrap();
                    decision.node = compute.next().unwrap();
                    decision.rate = compute.next().unwrap();
                    decision.signal = compute.next().unwrap();
                    decision.connect_signal = compute.next().unwrap();
                    decision.sever_choice = compute.next().unwrap();
                    decision.pull = compute.next().unwrap();
                    memory.iter_mut().set_from(compute);
                }
                {
                    let mb = &*deps[i].bots[ib];
                    if mb.decision.mate >= 0 && mb.decision.mate < deps[i].bots.len() as i64 && mb.energy == MAX_ENERGY {
                        maters.push(ib);
                    }
                    //Node 0 is not included because that is the present node
                    if mb.decision.node > 0 && mb.decision.node < neighbors.len() as i64 {
                        movers.push(ib);
                    }
                }
                //Accumulate pull from bot (shift bots to avoid ordering issues)
                deps[i].pull = deps[i].pull.saturating_add(deps[i].bots[ib].decision.pull >> 8);

                let choice = deps[i].bots[ib].decision.sever_choice;
                if choice > 0 && choice < neighbors.len() as i64 {
                    match deps.find_edge(i, neighbors[choice as usize]) {
                        Some(e) => {deps.remove_edge(e);},
                        None => {},
                    }
                }
            }

            //Perform the matings on the node
            for ib in maters {
                if deps[i].bots[ib].decision.mate as usize == ib {
                    let nbot = Box::new(deps[i].bots[ib].divide(&mut rng));
                    deps[i].bots.push(nbot);
                } else {
                    let gn = &mut deps[i];
                    //Do this unsafely because we know the indices are in bounds and not the same
                    let nbot = Box::new(unsafe{
                        let bm = &mut *(gn.bots.get_unchecked_mut(ib) as *mut Box<Bot>);
                        let bo = gn.bots.get_unchecked_mut(bm.decision.mate as usize);
                        bm.mate(bo, &mut rng)
                    });
                    gn.bots.push(nbot);
                }
            }

            //Move bots to the node they desire starting from the end of the vector to avoid swaps
            while let Some(ib) = movers.pop() {
                let n = deps[i].bots[ib].decision.node;
                let b = deps[i].bots.swap_remove(ib);
                deps[neighbors[n as usize]].moved_bots.push(b);
            }
        }

        //Update particle forces between each node
        {
            let mut connect_plans: Vec<Vec<usize>> = Vec::new();
            {
                let nodes = deps.raw_nodes();
                for i in 0..nodes.len() {
                    use zoom::{PhysicsParticle, Position};
                    use na::Norm;
                    //Expand connect plans vector with new member
                    connect_plans.push(Vec::new());

                    nodes[i].weight.particle.hooke_to(&central, PULL_CENTER_MAGNITUDE);
                    for j in (i+1)..nodes.len() {
                        let mag_s = (nodes[i].weight.particle.position() - nodes[j].weight.particle.position()).sqnorm();
                        //Apply all gravitation forces
                        zoom::gravitate_radius(&nodes[i].weight.particle, &nodes[j].weight.particle,
                            //Repulse particles to keep them apart from each other
                            -REPULSION_MAGNITUDE
                            //Attract particles based on the amount of bots in them
                            //BOT_GRAVITATION_MAGNITUDE *
                            //((nodes[i].weight.bots.len() + nodes[j].weight.bots.len()) as f64) +
                        );

                        let mut acon = false;
                        //Do we consider a connection between these particles
                        if mag_s < BOT_COICE_CONNECT_LENGTH * BOT_COICE_CONNECT_LENGTH {
                            //If so do a search beteen their bots
                            'outer: for b1 in &nodes[i].weight.bots {
                                for b2 in &nodes[j].weight.bots {
                                    if b1.connect_signal.abs() >= CONNECT_SIGNAL_MIN &&
                                        b1.connect_signal == b2.connect_signal {
                                        connect_plans.last_mut().unwrap().push(j);
                                        acon = true;
                                        break 'outer;
                                    }
                                }
                            }
                        }

                        if mag_s < CONNECT_AFTER * CONNECT_AFTER && !acon {
                            connect_plans.last_mut().unwrap().push(j);
                        }
                    }
                }
            }

            //Now connect the nodes as necessary
            for (ix, v) in connect_plans.iter().enumerate() {
                for &jx in v {
                    deps.update_edge(petgraph::graph::NodeIndex::new(ix),
                        petgraph::graph::NodeIndex::new(jx), ());
                }
            }
        }

        //Update all nodes with bot movements and memory, etc
        for i in deps.node_indices() {
            let n = &mut deps[i];
            n.moves = n.moved_bots.len() as i64;
            while let Some(b) = n.moved_bots.pop() {
                n.bots.push(b);
            }
            n.deaths = 0;
            for ib in (0..n.bots.len()).rev() {
                n.bots[ib].cycle();
                if rng.gen_range(0.0, 1.0) < MUTATION_RATE {
                    n.bots[ib].mutate(&mut rng);
                }
                //Remove any dead bots
                if n.bots[ib].energy <= 0 {
                    n.bots.swap_remove(ib);
                    n.deaths += 1;
                }
            }
            //Consume energy after loosing some so bots can reach max
            for b in n.bots.iter_mut() {
                use num::Float;
                let asking = ((1.0/(1.0 + (b.decision.rate as f64).exp()) - 0.5) * ENERGY_EXCHANGE_MAGNITUDE as f64) as i64;
                b.energy = b.energy.saturating_add(asking);
                n.energy = n.energy.saturating_sub(asking);
                if b.energy > MAX_ENERGY {
                    b.energy = MAX_ENERGY;
                }
            }
        }

        //Update perspective every frame
        let fbdim = window.get_inner_size_pixels().unwrap();
        let perspective = *na::Persp3::new((fbdim.0 as f32)/(fbdim.1 as f32),
            1.0, 0.0, 500.0).to_mat().as_ref();

        //Render nodes
        glowy.render_nodes(&mut target, matr.as_ref(), &perspective,
            &deps.node_weights_mut().map(|n|
                gg::Node{
                    position: vec_to_spos(n.particle.p.position),
                    color: n.color(),
                    falloff: NODE_FALLOFF,
                    radius: n.radius(),
                }
            ).collect_vec()[..]);

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
                        falloff: EDGE_FALLOFF,
                        radius: nodes.0.radius(),
                    }).chain(
                    std::iter::once(gg::Node{
                        position: vec_to_spos(nodes.1.particle.p.position),
                        color: nodes.1.color(),
                        falloff: EDGE_FALLOFF,
                        radius: nodes.1.radius(),
                    }))
                }
            ).collect_vec()[..]
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
                    if focus_state {
                        movement.append_rotation_mut(&na::Vec3::new(-(y - hdimy as i32) as f32 * ROTATION_RATE,
                            (x - hdimx as i32) as f32 * ROTATION_RATE, 0.0));
                        window.set_cursor_position(hdimx as i32, hdimy as i32).ok().unwrap();
                    }
                },
                glium::glutin::Event::Focused(s) => {
                    focus_state = s;
                },
                _ => ()
            }
        }

        if upstate == glium::glutin::ElementState::Pressed {
            movement.append_translation_mut(&na::Vec3::new(0.0, -MOVE_SPEED, 0.0));
        }
        if dnstate == glium::glutin::ElementState::Pressed {
            movement.append_translation_mut(&na::Vec3::new(0.0, MOVE_SPEED, 0.0));
        }
        if ltstate == glium::glutin::ElementState::Pressed {
            movement.append_translation_mut(&na::Vec3::new(-MOVE_SPEED, 0.0, 0.0));
        }
        if rtstate == glium::glutin::ElementState::Pressed {
            movement.append_translation_mut(&na::Vec3::new(MOVE_SPEED, 0.0, 0.0));
        }
        if fdstate == glium::glutin::ElementState::Pressed {
            movement.append_translation_mut(&na::Vec3::new(0.0, 0.0, -MOVE_SPEED));
        }
        if bkstate == glium::glutin::ElementState::Pressed {
            movement.append_translation_mut(&na::Vec3::new(0.0, 0.0, MOVE_SPEED));
        }
        period += 1;
    }
}
