extern crate mli;
extern crate rand;

type R = rand::isaac::Isaac64Rng;

#[derive(Clone)]
enum Ins {
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

}

#[derive(Clone)]
struct Bot {
    brain: mli::Mep<Ins, rand::isaac::Isaac64Rng, i64,
        fn(&mut Ins, &mut R), fn(&Ins, i64, i64)>,
}
