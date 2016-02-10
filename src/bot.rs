extern crate mli;
extern crate rand;
use self::rand::Rng;

pub type R = rand::isaac::Isaac64Rng;

mod botbrain {
    pub static TOTAL_INPUTS: usize = 5;
    pub static TOTAL_OUTPUTS: usize = 5;
    pub static DEFAULT_MUTATE_SIZE: usize = 30;
    pub static DEFAULT_CROSSOVER_POINTS: usize = 1;
    pub static DEFAULT_INSTRUCTIONS: usize = 64;
}

mod nodebrain {
    pub static TOTAL_INPUTS: usize = 5;
    pub static TOTAL_OUTPUTS: usize = 5;
    pub static DEFAULT_MUTATE_SIZE: usize = 30;
    pub static DEFAULT_CROSSOVER_POINTS: usize = 1;
    pub static DEFAULT_INSTRUCTIONS: usize = 64;
}

mod finalbrain {
    pub static TOTAL_INPUTS: usize = 5;
    pub static TOTAL_OUTPUTS: usize = 5;
    pub static DEFAULT_MUTATE_SIZE: usize = 30;
    pub static DEFAULT_CROSSOVER_POINTS: usize = 1;
    pub static DEFAULT_INSTRUCTIONS: usize = 64;
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
    SIN,
    COS,
    SQT,
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
        Ins::SIN => ((a as f64 / b as f64).sin() * b as f64) as i64,
        Ins::COS => ((a as f64 / b as f64).cos() * b as f64) as i64,
        Ins::SQT => ((a as f64 / b as f64).sqrt() * b as f64) as i64,
    }
}

fn mutator(ins: &mut Ins, rng: &mut R) {
    use std::mem;
    *ins = unsafe{mem::transmute(rng.gen_range::<u8>(0, 9))};
}

#[derive(Clone)]
pub struct Bot {
    pub bot_brain: mli::Mep<Ins, R, i64, fn(&mut Ins, &mut R), fn(&Ins, i64, i64) -> i64>,
    pub node_brain: mli::Mep<Ins, R, i64, fn(&mut Ins, &mut R), fn(&Ins, i64, i64) -> i64>,
    pub final_brain: mli::Mep<Ins, R, i64, fn(&mut Ins, &mut R), fn(&Ins, i64, i64) -> i64>,
    pub energy: i64,
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
        }
    }
}
