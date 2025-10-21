//! Example of scaffolding where function uses full `GateThreaderBuilder` instead of single `Context`
use std::result;

use clap::Parser;
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions};
use halo2_base::halo2_proofs::halo2curves::{bn256::Fr, ff::Field};
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::AssignedValue;
#[allow(unused_imports)]
use halo2_base::{
    Context,
    QuantumCell::{Constant, Existing, Witness},
};
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::{run_on_inputs, run_for_debug};
use nalgebra::DMatrix;
use rand::rngs::OsRng;
use halo2_graph::gadget::sdcd::*;
use halo2_graph::gadget::fixed_point::*;
use rand::Rng;

use std::env::{set_var, var};


// this algorithm takes a public input x, computes x^2 + 72, and outputs the result as public output
fn test_sdcd<F: BigPrimeField>(
    builder: &mut BaseCircuitBuilder<F>,
    x: F,
    make_public: &mut Vec<AssignedValue<F>>,
) {
    const PRECISION_BITS: u32 = 63;
    println!("build_lookup_bit: {:?}", builder.lookup_bits());
    let fixed_point_chip = FixedPointChip::<F, PRECISION_BITS>::default(builder);
    // can still get a Context via:
    let ctx = builder.main(0); // 0 means FirstPhase, don't worry about it
    // lookup bits must agree with the size of the lookup table, which is specified by an environmental variable

    // `Context` can roughly be thought of as a single-threaded execution trace of a program we want to ZK prove. We do some post-processing on `Context` to optimally divide the execution trace into multiple columns in a PLONKish arithmetization
    // More advanced usage with multi-threaded witness generation is possible, but we do not explain it here

    // first we load a number `x` into as system, as a "witness"
    let x = ctx.load_witness(x);
    // by default, all numbers in the system are private
    // we can make it public like so:
    make_public.push(x);

    // Create a sample matrix
    let mut x: Vec<Vec<AssignedValue<F>>> = vec![];
    let mut rng = rand::thread_rng();
    
    let num_attributes = 10;
    let num_samples = 10;
    for _ in 0..num_samples {
        let mut x_row: Vec<AssignedValue<F>> = vec![];
        for _ in 0..num_attributes {
            // fill in random number using random number generator
            x_row.push(ctx.load_witness(fixed_point_chip.quantization(rng.gen_range(-10.0..10.0))));
        }
        x.push(x_row);
    }

    let mut sdcd_chip = SDCDChip::<F,PRECISION_BITS>::new();
    let result = sdcd_chip.train_one_batch(ctx, &fixed_point_chip, x, 0.1);
    println!("result: {:?}", result);
    // make_public.push(result);










}

fn main() {
    env_logger::init();
    set_var("LOOKUP_BITS", 7.to_string());

    // let args = Cli::parse();

    //record the time
    let start = std::time::Instant::now();

    // let's say we don't want to run prover with inputs from file
    // instead we generate inputs here:
    let private_inputs = Fr::random(OsRng);
    run_for_debug(test_sdcd, private_inputs);

    // record time again
    let duration = start.elapsed();
    println!("Time elapsed in test_sdcd() is: {:?}", duration);
}
