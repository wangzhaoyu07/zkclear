#![allow(warnings)]
#[allow(unused_imports)]
use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
use halo2_base::utils::BigPrimeField;
use halo2_base::AssignedValue;
// use halo2_graph::gadget::linear_regression::LinearRegressionChip;
// use halo2_graph::gadget::linear_regression_closed_form::LinearRegressionChip;
use halo2_graph::gadget::dml::DMLChip;
use halo2_graph::gadget::dml::DMLChipNative;
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::halo2_proofs::halo2curves::{ff::Field};
use halo2_graph::scaffold::{run_on_inputs, run_for_debug, run};
// use halo2_graph::scaffold::{gen_key, prove_private};
use halo2_graph::gadget::fixed_point::*;

// use halo2_graph::scaffold::{mock, prove};
use log::warn;
use std::cmp::min;
use std::env::{var, set_var};
use std::cmp;
// use linfa::prelude::*;
// use linfa_linear::LinearRegression;
use ndarray::{Array, Axis};
use rand::rngs::OsRng;
use halo2_graph::scaffold::cmd::Cli;
use clap::Parser;
use serde::{Serialize, Deserialize, Serializer, Deserializer, ser::SerializeStruct};
use serde_json::json;
use num_bigint::BigUint;

// Circuit Input Structure
#[derive(Clone, Debug)]
pub struct CircuitInput {
    /// Public Inputs
    pub data_x: Vec<Vec<f64>>,
    pub data_y: Vec<f64>,
    pub data_t: Vec<f64>,
    pub t0: f64,
    pub t1: f64,
    pub test_x: Vec<Vec<f64>>,
    pub log_dir: String,
    pub output_dir: String,
}

// Implement Serialize manually
impl Serialize for CircuitInput {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("CircuitInput", 4)?;
        // Serialize arrays by converting them to Vecs
        state.serialize_field(
            "data_x", 
            &self.data_x
        )?;
        state.serialize_field(
            "data_y", 
            &self.data_y
        )?;
        state.serialize_field(
            "data_t", 
            &self.data_t
        )?;
        state.serialize_field(
            "t0", 
            &self.t0
        )?;
        state.serialize_field(
            "t1", 
            &self.t1
        )?;
        state.serialize_field(
            "test_x", 
            &self.test_x
        )?;
        state.serialize_field(
            "log_dir", 
            &self.log_dir
        )?;
        state.serialize_field(
            "output_dir",
            &self.output_dir
        )?;

        state.end()
    }
}

// Implement Deserialize manually
impl<'de> Deserialize<'de> for CircuitInput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct CircuitInputHelper {
            data_x: Vec<Vec<f64>>,
            data_y: Vec<f64>,
            data_t: Vec<f64>,
            t0: f64,
            t1: f64,
            test_x: Vec<Vec<f64>>,
            log_dir: String,
            output_dir: String,
        }

        let helper = CircuitInputHelper::deserialize(deserializer)?;

        Ok(CircuitInput {
            data_x: helper.data_x,
            data_y: helper.data_y,
            data_t: helper.data_t,
            t0: helper.t0,
            t1: helper.t1,
            test_x: helper.test_x,
            log_dir: helper.log_dir,
            output_dir: helper.output_dir,
        })
    }
}

pub fn train_and_inference<F: BigPrimeField>(
    builder: &mut BaseCircuitBuilder<F>,
    // _x: CircuitInput,
    _x: Fr,
    make_public: &mut Vec<AssignedValue<F>>,
) {
    // obtain dataset
    let t0: f64 = 0.0;
    let t1: f64 = 1.0;
    let dataset = linfa_datasets::diabetes();

    let mut train_x: Vec<Vec<f64>> = vec![];
    let mut train_y: Vec<f64> = vec![];
    let mut train_t: Vec<f64> = vec![];
    let mut num_samples = 0;
    let mut target_num_samples = 1000;
    for (sample_x, sample_y) in dataset.sample_iter() {
        // if num_samples >= target_num_samples {
        //     break;
        // }
        train_x.push(sample_x.iter().map(|xi| *xi).collect::<Vec<f64>>());
        train_y.push(*sample_y.iter().peekable().next().unwrap());
        train_t.push(1.0);
    }

    let in_dim = train_x[0].len();
    let n_samples = train_x.len();


    // Initialize the chip
    const PRECISION_BITS: u32 = 63;
    const EPS_ERR: f64 = 1e-5; //TODO: change to 1e-10
    
    println!("build_lookup_bit: {:?}", builder.lookup_bits());
    let fpchip = FixedPointChip::<F, PRECISION_BITS>::default(builder);
    let ctx = builder.main(0); // 0 means FirstPhase, don't worry about it
    let init_rand: AssignedValue<F> = ctx.load_witness(F::from(1));

    // let err_svd_scale = LinearRegressionChip::<F, PRECISION_BITS>::cal_scaled_error(n_samples, in_dim);
    let max_dim = cmp::max(n_samples, in_dim);
    let dml_chip = DMLChip::<F, PRECISION_BITS>::new(ctx, &fpchip, in_dim, max_dim, EPS_ERR, &init_rand);

    // Convert the dataset to AssignedValue
    let t0_witness: AssignedValue<F> = ctx.load_witness(fpchip.quantization(t0));
    let t1_witness: AssignedValue<F> = ctx.load_witness(fpchip.quantization(t1));
    let mut train_x_witness: Vec<Vec<AssignedValue<F>>> = vec![];
    for xi in &train_x {
        train_x_witness.push(xi.iter().map(|xij| ctx.load_witness(fpchip.quantization(*xij))).collect::<Vec<AssignedValue<F>>>());
    }
    let mut test_x_witness: Vec<Vec<AssignedValue<F>>> = vec![];
    test_x_witness.push(train_x_witness[0].clone());

    let train_y_witness: Vec<AssignedValue<F>> = train_y.iter().map(|yi| ctx.load_witness(fpchip.quantization(*yi))).collect();
    let train_t_witness: Vec<AssignedValue<F>> = train_t.iter().map(|ti| ctx.load_witness(fpchip.quantization(*ti))).collect();

    // Train the model
    let model = dml_chip.fit(ctx, &fpchip, train_x_witness.clone(), train_t_witness.clone(), train_y_witness.clone());
    // let ate = model.ate_estimate(ctx, &fpchip, test_x_witness.clone(), train_t_witness.clone()[0], train_y_witness.clone()[1]);
    let ate = model.ate_estimate(ctx, &fpchip, test_x_witness.clone(), t0_witness, t1_witness);

    let dequantized_ate: f64 = fpchip.dequantization(*ate.value());
    println!("ate: {:?}", dequantized_ate);
}

fn main() {
    set_var("RUST_LOG", "warn");
    set_var("RUST_BACKTRACE", "full");
    env_logger::init();
    set_var("LOOKUP_BITS", 7.to_string());

    // let args = Cli::parse();

    //record the time
    let start = std::time::Instant::now();

    // let's say we don't want to run prover with inputs from file
    // instead we generate inputs here:
    let private_inputs = Fr::random(OsRng);
    let args = Cli::parse();
    run_on_inputs(train_and_inference, args, private_inputs);

    
    // run_for_debug(train_and_inference, private_inputs);
    

    // record time again
    let duration = start.elapsed();
    println!("Time elapsed in test_sdcd() is: {:?}", duration);
}
