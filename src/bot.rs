extern crate mli;
extern crate rand;
use self::rand::Rng;

pub type R = rand::isaac::Isaac64Rng;

pub mod nodebrain {
    //0, 1, 2, -1, rand, node energy, bot count, present node bot count, self energy, present node connections, node connections, and memory are inputs.
    pub const STATIC_INPUTS: usize = 11;
    pub const TOTAL_INPUTS: usize = STATIC_INPUTS + super::finalbrain::TOTAL_MEMORY;
    pub const TOTAL_OUTPUTS: usize = 5;
    pub const DEFAULT_MUTATE_SIZE: usize = 30;
    pub const DEFAULT_CROSSOVER_POINTS: usize = 1;
    pub const DEFAULT_INSTRUCTIONS: usize = 64;
}

pub mod botbrain {
    //0, 1, 2, -1, rand, node energy, bot count, self energy, bot energy, bot signal, present node connections, and memory are inputs.
    pub const STATIC_INPUTS: usize = 11;
    pub const TOTAL_INPUTS: usize = STATIC_INPUTS + super::finalbrain::TOTAL_MEMORY;
    pub const TOTAL_OUTPUTS: usize = 5;
    pub const DEFAULT_MUTATE_SIZE: usize = 30;
    pub const DEFAULT_CROSSOVER_POINTS: usize = 1;
    pub const DEFAULT_INSTRUCTIONS: usize = 64;
}

pub mod finalbrain {
    pub const TOTAL_BOT_INPUTS: usize = 4;
    pub const TOTAL_NODE_INPUTS: usize = 4;
    pub const TOTAL_MEMORY: usize = 4;
    //0, 1, 2, -1, rand, present node energy, bot count, self energy, self index, present node connections, and memory are inputs
    pub const STATIC_INPUTS: usize = 10;
    pub const TOTAL_INPUTS: usize = STATIC_INPUTS + TOTAL_MEMORY +
        //Add inputs for all the node brains
        TOTAL_NODE_INPUTS * super::nodebrain::TOTAL_OUTPUTS +
        //Add inputs for all the bot brains
        TOTAL_BOT_INPUTS * super::botbrain::TOTAL_OUTPUTS;
    //Mate, Node, Energy Rate (as a sigmoid), Signal, Connect Signal, Pull
    pub const STATIC_OUTPUTS: usize = 7;
    pub const TOTAL_OUTPUTS: usize = STATIC_OUTPUTS + TOTAL_MEMORY;
    pub const DEFAULT_MUTATE_SIZE: usize = 30;
    pub const DEFAULT_CROSSOVER_POINTS: usize = 1;
    pub const DEFAULT_INSTRUCTIONS: usize = 256;
}

pub const ENERGY_EXCHANGE_MAGNITUDE: i64 = 500;
pub const EXISTENCE_COST: i64 = 50;
pub const MAX_ENERGY: i64 = 20000;
const DEFAULT_ENERGY: i64 = 4 * EXISTENCE_COST;
const MUTATE_PROBABILITY: f64 = 1.0;

#[derive(Clone)]
pub enum Ins {
    _NOP,
    _ADD,
    _SUB,
    _MUL,
    _DIV,
    _MOD,
    _GRT,
    _LES,
    _EQL,
    _NEQ,
    _AND,
    _OR,
    _EXP,
    _SIN,
    _COS,
    _SQT,
    MAX,
}

fn processor(ins: &Ins, a: i64, b: i64) -> i64 {
    match *ins {
        Ins::_NOP => a,
        Ins::_ADD => a + b,
        Ins::_SUB => a - b,
        Ins::_MUL => a * b,
        Ins::_DIV => {
            match a.checked_div(b) {
                Some(v) => v,
                None => 0,
            }
        },
        Ins::_MOD => {
            if b == 0 {
                0
            } else {
                a.wrapping_rem(b)
            }
        },
        Ins::_GRT => if a > b {
            1
        } else {
            0
        },
        Ins::_LES => if a < b {
            1
        } else {
            0
        },
        Ins::_EQL => if a == b {
            1
        } else {
            0
        },
        Ins::_NEQ => if a == b {
            1
        } else {
            0
        },
        Ins::_AND => if a != 0 && b != 0 {
            1
        } else {
            0
        },
        Ins::_OR => if a != 0 || b != 0 {
            1
        } else {
            0
        },
        Ins::_EXP => (a as f64).powf(b as f64) as i64,
        Ins::_SIN => ((a as f64 / b as f64).sin() * b as f64) as i64,
        Ins::_COS => ((a as f64 / b as f64).cos() * b as f64) as i64,
        Ins::_SQT => ((a as f64 / b as f64).sqrt() * b as f64) as i64,
        Ins::MAX => unreachable!(),
    }
}

fn mutator(ins: &mut Ins, rng: &mut R) {
    use std::mem;
    *ins = unsafe{mem::transmute(rng.gen_range::<u8>(0, Ins::MAX as u8))};
}

#[derive(Clone)]
pub struct Decision {
    pub mate: i64,
    pub node: i64,
    //This will be ran through a sigmoid
    pub rate: i64,
    pub signal: i64,
    pub connect_signal: i64,
    pub sever_choice: i64,
    pub pull: i64,
}

impl Default for Decision {
    fn default() -> Self {
        Decision{
            mate: -1,
            node: -1,
            rate: 0,
            signal: 0,
            connect_signal: 0,
            sever_choice: 0,
            pull: 0,
        }
    }
}

#[derive(Clone)]
pub struct Bot {
    pub bot_brain: mli::Mep<Ins, R, i64, fn(&mut Ins, &mut R), fn(&Ins, i64, i64) -> i64>,
    pub node_brain: mli::Mep<Ins, R, i64, fn(&mut Ins, &mut R), fn(&Ins, i64, i64) -> i64>,
    pub final_brain: mli::Mep<Ins, R, i64, fn(&mut Ins, &mut R), fn(&Ins, i64, i64) -> i64>,
    pub energy: i64,
    pub signal: i64,
    pub connect_signal: i64,
    pub memory: [i64; finalbrain::TOTAL_MEMORY],
    pub decision: Decision,
}

impl Bot {
    pub fn new(rng: &mut R) -> Self {
        let bvec = (0..botbrain::DEFAULT_INSTRUCTIONS).map(|_| {
                let mut ins = Ins::_NOP;
                mutator(&mut ins, rng);
                ins
            }).collect::<Vec<_>>();
        let nvec = (0..nodebrain::DEFAULT_INSTRUCTIONS).map(|_| {
                let mut ins = Ins::_NOP;
                mutator(&mut ins, rng);
                ins
            }).collect::<Vec<_>>();
        let fvec = (0..finalbrain::DEFAULT_INSTRUCTIONS).map(|_| {
                let mut ins = Ins::_NOP;
                mutator(&mut ins, rng);
                ins
            }).collect::<Vec<_>>();
        Bot {
            bot_brain: mli::Mep::new(botbrain::TOTAL_INPUTS, botbrain::TOTAL_OUTPUTS,
                botbrain::DEFAULT_MUTATE_SIZE, botbrain::DEFAULT_CROSSOVER_POINTS, rng,
                bvec.into_iter(),
                mutator, processor),

            node_brain: mli::Mep::new(nodebrain::TOTAL_INPUTS, nodebrain::TOTAL_OUTPUTS,
                nodebrain::DEFAULT_MUTATE_SIZE, nodebrain::DEFAULT_CROSSOVER_POINTS, rng,
                nvec.into_iter(),
                mutator, processor),

            final_brain: mli::Mep::new(finalbrain::TOTAL_INPUTS, finalbrain::TOTAL_OUTPUTS,
                finalbrain::DEFAULT_MUTATE_SIZE, finalbrain::DEFAULT_CROSSOVER_POINTS, rng,
                fvec.into_iter(),
                mutator, processor),

            energy: DEFAULT_ENERGY,

            signal: 0,
            connect_signal: 0,

            memory: [0; finalbrain::TOTAL_MEMORY],
            decision: Default::default(),
        }
    }

    pub fn mutate(&mut self, rng: &mut R) {
        use mli::Genetic;
        if rng.gen_range(0.0, 1.0) < MUTATE_PROBABILITY {
            self.bot_brain.mutate(rng);
            self.node_brain.mutate(rng);
            self.final_brain.mutate(rng);
        }
    }

    pub fn mate(&mut self, other: &Self, rng: &mut R) -> Self {
        use mli::Genetic;
        //Divide energy in half when mating for the mater
        self.energy /= 2;
        let mut b = Bot{
            bot_brain: mli::Genetic::mate((&self.bot_brain, &other.bot_brain), rng),
            node_brain: mli::Genetic::mate((&self.node_brain, &other.node_brain), rng),
            final_brain: mli::Genetic::mate((&self.final_brain, &other.final_brain), rng),
            energy: self.energy,
            signal: self.signal,
            connect_signal: 0,
            memory: self.memory,
            decision: self.decision.clone(),
        };
        //Perform unit mutations on offspring
        b.mutate(rng);
        b
    }

    pub fn divide(&mut self, rng: &mut R) -> Self {
        use mli::Genetic;
        //Divide energy in half when dividing
        self.energy /= 2;
        let mut b = Bot{
            bot_brain: self.bot_brain.clone(),
            node_brain: self.node_brain.clone(),
            final_brain: self.final_brain.clone(),
            energy: self.energy,
            signal: self.signal,
            connect_signal: 0,
            memory: self.memory,
            //Clone the rate of energy consumption in the decision
            decision: self.decision.clone(),
        };
        //Perform unit mutations on offspring
        b.mutate(rng);
        b
    }

    pub fn cycle(&mut self) {
        self.energy = self.energy.saturating_sub(EXISTENCE_COST);
        self.signal = self.decision.signal;
        self.connect_signal = self.decision.connect_signal;
    }
}
