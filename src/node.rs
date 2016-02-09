extern crate zoom;

use super::bot::*;
use super::Vec3;

static ENERGY_RATIO: f64 = 0.01;
pub static ENERGY_THRESHOLD: i64 = 500000;

pub struct Node {
    pub particle: zoom::BasicParticle<Vec3, f64>,
    pub energy: i64,
    pub bots: Vec<Box<Bot>>,
}

impl Node {
    pub fn new(energy: i64, particle: zoom::BasicParticle<Vec3, f64>) -> Self {
        Node{
            energy: energy,
            particle: particle,
            bots: Vec::new(),
        }
    }

    pub fn advance(&mut self) {
        self.energy += (self.energy as f64 * ENERGY_RATIO) as i64;
    }

    pub fn should_split(&self) -> bool {
        self.energy >= ENERGY_THRESHOLD
    }
}
