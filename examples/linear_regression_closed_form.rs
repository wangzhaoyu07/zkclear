#![allow(warnings)]
#[allow(unused_imports)]
use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
use halo2_base::utils::BigPrimeField;
use halo2_base::AssignedValue;
// use halo2_graph::gadget::linear_regression::LinearRegressionChip;
use halo2_graph::gadget::linear_regression_closed_form::LinearRegressionChip;
use halo2_graph::gadget::linear_regression_closed_form::LinearRegressionNative;
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::halo2_proofs::halo2curves::{ff::Field};
use halo2_graph::scaffold::{run_on_inputs, run_for_debug, run};
// use halo2_graph::scaffold::{gen_key, prove_private};
use halo2_graph::gadget::fixed_point::*;
#[allow(unused_imports)]
// use halo2_graph::scaffold::{mock, prove};
use log::warn;
use std::cmp::min;
use std::env::{var, set_var};
use std::cmp;
use linfa::prelude::*;
// use linfa_linear::LinearRegression;
use ndarray::{Array, Axis};
use nalgebra::{DMatrix, DVector};
use rand::rngs::OsRng;
use halo2_graph::scaffold::cmd::Cli;
use clap::Parser;
use serde::{Serialize, Deserialize, Serializer, Deserializer, ser::SerializeStruct};
use num_bigint::BigUint;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub x: String, // field element, but easier to deserialize as a string
}



pub fn train_native( 
    train_x: Vec<Vec<f64>>, train_y: Vec<f64>
) {
    //closed form solution from scratch
    let mut model_native = LinearRegressionNative::new(train_x[0].len());
    // let X = vec_to_dmatrix(train_x);
    // let y = DVector::from(train_y.clone());
    model_native.fit_native(&train_x, &train_y);

}

pub fn train_and_inference<F: BigPrimeField>(
    builder: &mut BaseCircuitBuilder<F>,
    // _x: CircuitInput,
    _x: Fr,
    make_public: &mut Vec<AssignedValue<F>>,
) {
    // obtain dataset
    let dataset = linfa_datasets::diabetes();
    let dataset_attr_num = dataset.nfeatures();

    let mut train_x: Vec<Vec<f64>> = vec![];
    let mut train_y: Vec<f64> = vec![];
    let mut num_samples = 0;
    let args = Cli::parse();
    let mut target_num_samples = 1;//args.num_sample.unwrap_or(10);
    let num_attributes = args.num_attr.unwrap_or(10);
    let in_dim = num_attributes;
    println!("num_of_nodes: {:?}", num_attributes);

    // exmaple data
    while num_samples < target_num_samples {
        for (sample_x, sample_y) in dataset.sample_iter() {
            if num_samples >= target_num_samples {
                break;
            }
            let mut input_x:Vec<f64> = vec![];
            if dataset_attr_num >= num_attributes {
                // println!("sample_x: {:?}", sample_x);
                input_x = sample_x.iter().map(|xi| *xi).collect::<Vec<f64>>();
                // println!("sample_x: {:?}", sample_x);
                input_x.truncate(num_attributes);
                // println!("sample_x: {:?}", sample_x);
            } else {
                input_x = sample_x.iter().map(|xi| *xi).collect::<Vec<f64>>();
                // println!("sample_x.len(): {:?}", sample_x.len());
                // println!("num_attributes: {:?}", num_attributes);
                for i in sample_x.len()..num_attributes {
                    if i == num_samples {
                        input_x.push(1.0);
                    } else {
                        input_x.push(0.0);
                    }
                }
            }

            train_x.push(input_x);
            train_y.push(*sample_y.iter().peekable().next().unwrap());
            num_samples += 1;
        }
    }
    // println!("train_x: {:?}", train_x.len());

    

    // train_native(train_x.clone(), train_y.clone());

    // Initialize the chip
    const PRECISION_BITS: u32 = 63;
    const EPS_ERR: f64 = 1e-5;
    
    // println!("build_lookup_bit: {:?}", builder.lookup_bits());
    let fpchip = FixedPointChip::<F, PRECISION_BITS>::default(builder);
    let ctx = builder.main(0); // 0 means FirstPhase, don't worry about it
    let init_rand: AssignedValue<F> = ctx.load_witness(F::from(2));

    // let err_svd_scale = LinearRegressionChip::<F, PRECISION_BITS>::cal_scaled_error(n_samples, in_dim);
    let max_dim = cmp::max(target_num_samples, in_dim);
    let lrchip = LinearRegressionChip::<F, PRECISION_BITS>::new(ctx, &fpchip, in_dim, max_dim, EPS_ERR, &init_rand);

    // Convert the dataset to AssignedValue
    let mut train_x_witness: Vec<Vec<AssignedValue<F>>> = vec![];
    for xi in train_x {
        train_x_witness.push(xi.iter().map(|xij| ctx.load_witness(fpchip.quantization(*xij))).collect::<Vec<AssignedValue<F>>>());
    }
    let train_y: Vec<AssignedValue<F>> = train_y.iter().map(|yi| ctx.load_witness(fpchip.quantization(*yi))).collect();

    // Train the model
    let model = lrchip.fit(ctx, &fpchip, &train_x_witness, &train_y);
    let final_weight = model.get_weight();
    let final_bias = model.get_bias();

    // Make the final weight and bias public
    for wi in final_weight.iter() {
        make_public.push(*wi);
    }
    make_public.push(final_bias);
    
    let dequantized_weight: Vec<f64> = final_weight.iter().map(|wi| fpchip.dequantization(*wi.value())).collect();
    let dequantized_bias: f64 = fpchip.dequantization(*final_bias.value());
    // println!("params: {:?}", dequantized_weight);
    // println!("bias: {:?}", dequantized_bias);

    // // Inference
    // let predicted_y = lrchip.inference(ctx, &fpchip, train_x_witness);


    // train_x_witness: remove the bias term
    // let test_x_witness: Vec<Vec<AssignedValue<F>>> = train_x_witness;//iter().map(|xi| xi[..in_dim].to_vec()).collect();
    // let prediction = LinearRegressionChip::inference_one_sample(ctx, &fpchip, final_weight.clone(), test_x_witness[0].clone(), final_bias);
    // let dequantized_prediction = fpchip.dequantization(*prediction.value());
    // println!("prediction: {:?}", dequantized_prediction);
    // let prediction_batch = model.inference(ctx, &fpchip, test_x_witness);
    // let dequantized_prediction_batch: Vec<f64> = prediction_batch.iter().map(|yi| fpchip.dequantization(*yi.value())).collect();
    // println!("prediction_batch: {:?}", dequantized_prediction_batch);
}

fn main() {
    set_var("RUST_LOG", "warn");
    env_logger::init();
    set_var("LOOKUP_BITS", 7.to_string());

    let args = Cli::parse();

    //record the time
    let start = std::time::Instant::now();

    // let's say we don't want to run prover with inputs from file
    // instead we generate inputs here:
    // let args = Cli::parse();
    // run(train_and_inference, args);

    let private_inputs = Fr::random(OsRng);
    // run_for_debug(train_and_inference, private_inputs);
    run_on_inputs(train_and_inference, args, private_inputs);
    

    // record time again
    let duration = start.elapsed();
    // println!("Time elapsed in test_sdcd() is: {:?}", duration);
}
