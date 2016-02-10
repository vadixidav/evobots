extern crate zoom;

use super::bot::*;
use super::Vec3;
use self::zoom::*;

static ENERGY_RATIO: f64 = 0.003;
pub static ENERGY_THRESHOLD: i64 = 500000;

static DRAG: f64 = 0.1;

#[derive(Clone)]
pub struct RadParticle {
    pub p: zoom::BasicParticle<Vec3, f64>,
}

impl Position<Vec3> for RadParticle {
    fn position(&self) -> Vec3 {
        self.p.position()
    }
}

impl Velocity<Vec3> for RadParticle {
    fn velocity(&self) -> Vec3 {
        self.p.velocity()
    }
}

impl Particle<Vec3, f64> for RadParticle {
    fn impulse(&self, v: &Vec3) {
        self.p.impulse(v);
    }

    fn advance(&mut self, time: f64) {
        self.p.advance(time);
    }
}

impl Inertia<f64> for RadParticle {
    fn inertia(&self) -> f64 {
        self.p.inertia()
    }
}

impl Quanta<f64> for RadParticle {
    fn quanta(&self) -> f64 {
        self.p.quanta()
    }
}

impl PhysicsParticle<Vec3, f64> for RadParticle {

}

impl Ball<f64> for RadParticle {
    fn radius(&self) -> f64 {
        1.0
    }
}

pub struct Node {
    pub particle: RadParticle,
    pub energy: i64,
    pub bots: Vec<Box<Bot>>,
}

impl Node {
    pub fn new(energy: i64, particle: BasicParticle<Vec3, f64>) -> Self {
        Node{
            energy: energy,
            particle: RadParticle{p: particle},
            bots: Vec::new(),
        }
    }

    pub fn advance(&mut self) {
        use zoom::{Particle, PhysicsParticle};
        self.energy += (self.energy as f64 * ENERGY_RATIO) as i64;
        self.particle.drag(DRAG);
        self.particle.advance(1.0);
    }

    pub fn should_split(&self) -> bool {
        self.energy >= ENERGY_THRESHOLD
    }

    pub fn color(&self) -> [f32; 4] {
        [
            self.energy as f32 / ENERGY_THRESHOLD as f32,
            self.energy as f32 / ENERGY_THRESHOLD as f32,
            self.energy as f32 / ENERGY_THRESHOLD as f32,
            1.0,
        ]
    }
}
