extern crate mli;
extern crate rand;
use self::rand::Rng;

pub type R = rand::isaac::Isaac64Rng;

pub const ENERGY_EXCHANGE_MAGNITUDE: i64 = 100;

pub mod nodebrain {
    //0, 1, 2, -1, node energy, bot count, present node bot count, self energy, and memory are inputs.
    pub const STATIC_INPUTS: usize = 8;
    pub const TOTAL_INPUTS: usize = STATIC_INPUTS + super::finalbrain::TOTAL_MEMORY;
    pub const TOTAL_OUTPUTS: usize = 5;
    pub const DEFAULT_MUTATE_SIZE: usize = 30;
    pub const DEFAULT_CROSSOVER_POINTS: usize = 1;
    pub const DEFAULT_INSTRUCTIONS: usize = 64;
}

pub mod botbrain {
    //0, 1, 2, -1, node energy, bot count, self energy, bot energy, bot signal, and memory are inputs.
    pub const STATIC_INPUTS: usize = 9;
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
    //0, 1, 2, -1, present node energy, bot count, self energy, and memory are inputs
    pub const STATIC_INPUTS: usize = 7;
    pub const TOTAL_INPUTS: usize = STATIC_INPUTS + TOTAL_MEMORY +
        //Add inputs for all the node brains
        TOTAL_NODE_INPUTS * super::nodebrain::TOTAL_OUTPUTS +
        //Add inputs for all the bot brains
        TOTAL_BOT_INPUTS * super::botbrain::TOTAL_OUTPUTS;
    //Mate, Node, Energy Rate (as a sigmoid)
    pub const STATIC_OUTPUTS: usize = 3;
    pub const TOTAL_OUTPUTS: usize = STATIC_OUTPUTS + TOTAL_MEMORY;
    pub const DEFAULT_MUTATE_SIZE: usize = 30;
    pub const DEFAULT_CROSSOVER_POINTS: usize = 1;
    pub const DEFAULT_INSTRUCTIONS: usize = 64;
}

static DEFAULT_FOOD: i64 = 16384;

#[derive(Clone)]
pub enum Ins {
    ADD,
    SUB,
    MUL,
    DIV,
    GRT,
    LES,
    EQL,
    NEQ,
    SIN,
    COS,
    SQT,
    MAX,
}

fn processor(ins: &Ins, a: i64, b: i64) -> i64 {
    match *ins {
        Ins::ADD => a + b,
        Ins::SUB => a - b,
        Ins::MUL => a * b,
        Ins::DIV => if b == 0 {
            i64::max_value()
        } else {
            a / b
        },
        Ins::GRT => if a > b {
            1
        } else {
            0
        },
        Ins::LES => if a < b {
            1
        } else {
            0
        },
        Ins::EQL => if a == b {
            1
        } else {
            0
        },
        Ins::NEQ => if a == b {
            1
        } else {
            0
        },
        Ins::SIN => ((a as f64 / b as f64).sin() * b as f64) as i64,
        Ins::COS => ((a as f64 / b as f64).cos() * b as f64) as i64,
        Ins::SQT => ((a as f64 / b as f64).sqrt() * b as f64) as i64,
        Ins::MAX => unreachable!(),
    }
}

fn mutator(ins: &mut Ins, rng: &mut R) {
    use std::mem;
    *ins = unsafe{mem::transmute(rng.gen_range::<u8>(0, Ins::MAX as u8))};
}

#[derive(Clone, Default)]
pub struct Decision {
    pub mate: i64,
    pub node: i64,
    //This will be ran through a sigmoid
    pub rate: i64,
}

#[derive(Clone)]
pub struct Bot {
    pub bot_brain: mli::Mep<Ins, R, i64, fn(&mut Ins, &mut R), fn(&Ins, i64, i64) -> i64>,
    pub node_brain: mli::Mep<Ins, R, i64, fn(&mut Ins, &mut R), fn(&Ins, i64, i64) -> i64>,
    pub final_brain: mli::Mep<Ins, R, i64, fn(&mut Ins, &mut R), fn(&Ins, i64, i64) -> i64>,
    pub energy: i64,
    pub signal: i64,
    pub memory: [i64; finalbrain::TOTAL_MEMORY],
    pub decision: Decision,
}

impl Bot {
    pub fn new(rng: &mut R) -> Self {
        let bvec = (0..botbrain::DEFAULT_INSTRUCTIONS).map(|_| {
                let mut ins = Ins::ADD;
                mutator(&mut ins, rng);
                ins
            }).collect::<Vec<_>>();
        let nvec = (0..nodebrain::DEFAULT_INSTRUCTIONS).map(|_| {
                let mut ins = Ins::ADD;
                mutator(&mut ins, rng);
                ins
            }).collect::<Vec<_>>();
        let fvec = (0..finalbrain::DEFAULT_INSTRUCTIONS).map(|_| {
                let mut ins = Ins::ADD;
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

            energy: DEFAULT_FOOD,

            signal: 0,

            memory: [0; finalbrain::TOTAL_MEMORY],
            decision: Default::default(),
        }
    }
}
