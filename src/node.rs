extern crate zoom;
extern crate rand;

use super::bot::*;
use super::Vec3;

static BOTS_RADIUS_MULTIPLIER: f32 = 2.5;
static RADIUS_STATIC: f32 = 5.0;
static ENERGY_RATIO: f64 = 0.002;
static ENERGY_VARIATION: f64 = 0.1;
pub static ENERGY_THRESHOLD: i64 = 500000;
static ENERGY_FULL_COST: i64 = 5000;

static DRAG: f64 = 0.1;

#[derive(Clone)]
pub struct RadParticle {
    pub p: zoom::BasicParticle<Vec3, f64>,
}

impl zoom::Position<Vec3> for RadParticle {
    fn position(&self) -> Vec3 {
        self.p.position()
    }
}

impl zoom::Velocity<Vec3> for RadParticle {
    fn velocity(&self) -> Vec3 {
        self.p.velocity()
    }
}

impl zoom::Particle<Vec3, f64> for RadParticle {
    fn impulse(&self, v: &Vec3) {
        self.p.impulse(v);
    }

    fn advance(&mut self, time: f64) {
        self.p.advance(time);
    }
}

impl zoom::Inertia<f64> for RadParticle {
    fn inertia(&self) -> f64 {
        self.p.inertia()
    }
}

impl zoom::Quanta<f64> for RadParticle {
    fn quanta(&self) -> f64 {
        self.p.quanta()
    }
}

impl zoom::PhysicsParticle<Vec3, f64> for RadParticle {

}

impl zoom::Ball<f64> for RadParticle {
    fn radius(&self) -> f64 {
        1.0
    }
}

pub struct Node {
    pub particle: RadParticle,
    pub energy: i64,
    pub bots: Vec<Box<Bot>>,
    pub moved_bots: Vec<Box<Bot>>,
    pub deaths: i64,
    pub moves: i64,
    pub connections: i64,
}

impl Node {
    pub fn new(energy: i64, particle: zoom::BasicParticle<Vec3, f64>) -> Self {
        Node{
            energy: energy,
            particle: RadParticle{p: particle},
            bots: Vec::new(),
            moved_bots: Vec::new(),
            deaths: 0,
            moves: 0,
            connections: 0,
        }
    }

    pub fn grow(&mut self, full: bool, rng: &mut rand::Isaac64Rng) {
        use rand::Rng;
        if full {
            if self.bots.is_empty() {
                self.energy = self.energy.saturating_sub((ENERGY_FULL_COST as f64 *
                    (1.0 + rng.gen_range(-ENERGY_VARIATION, ENERGY_VARIATION))) as i64);
            }
        } else {
            self.energy = self.energy.saturating_add((self.energy as f64 * ENERGY_RATIO *
                (1.0 + rng.gen_range(-ENERGY_VARIATION, ENERGY_VARIATION))) as i64);
        }
    }

    pub fn advance(&mut self) {
        use zoom::{Particle, PhysicsParticle};
        self.particle.drag(DRAG);
        self.particle.advance(1.0);
    }

    pub fn should_split(&self) -> bool {
        self.energy >= ENERGY_THRESHOLD
    }
    pub fn should_obliterate(&self) -> bool {
        use std::num::FpCategory;
        use zoom::Position;
        let clos = |x: f64| {
            match x.classify() {
                FpCategory::Nan | FpCategory::Infinite => true,
                _ => false,
            }
        };
        self.energy <= 0 || match self.particle.position() {
            Vec3{x, y, z} => {clos(x) || clos(y) || clos(z)}
        }
    }

    pub fn color(&self) -> [f32; 4] {
        [
            1.0 - self.energy as f32 / ENERGY_THRESHOLD as f32,
            self.energy as f32 / ENERGY_THRESHOLD as f32,
            if self.bots.len() == 0 {
                0.0
            } else {
                self.moves as f32 / self.bots.len() as f32
            },
            1.0,
        ]
    }
    pub fn radius(&self) -> f32 {
        RADIUS_STATIC + BOTS_RADIUS_MULTIPLIER * (self.bots.len() as f32)
    }
}
