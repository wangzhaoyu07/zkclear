//! Example of scaffolding where function uses full `GateThreaderBuilder` instead of single `Context`
#![allow(warnings)]
#[allow(unused_imports)]
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
use halo2_graph::scaffold::{run_on_inputs, run_for_debug, run};
use nalgebra::DMatrix;
use rand::rngs::OsRng;
use halo2_graph::gadget::graph::*;
use halo2_graph::gadget::fixed_point::*;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo::toposort;
use petgraph::visit::Topo;
use std::env::{set_var, var};
use serde::{Serialize, Deserialize, Serializer, Deserializer, ser::SerializeStruct};
use std::fs::File;
use serde_json::json;

const NODE_SIZE: usize = 24;

/// Circuit Input Structure
#[derive(Clone, Debug)]
pub struct CircuitInput {
    /// Public Inputs
    /// Private Witnesses
    pub adj_matrix: [[f64; NODE_SIZE]; NODE_SIZE], // Adjacency matrix
    pub output_dir: String,
}

// Implement Serialize manually
impl Serialize for CircuitInput {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("CircuitInput", 1)?;

        // Serialize arrays by converting them to Vecs
        state.serialize_field(
            "adj_matrix", 
            &self.adj_matrix.iter().map(|row| row.to_vec()).collect::<Vec<_>>()
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
            adj_matrix: Vec<Vec<f64>>,
            output_dir: String,
        }

        let helper = CircuitInputHelper::deserialize(deserializer)?;

        // Convert Vecs back into fixed-size arrays
        let mut adj_matrix = [[0f64; NODE_SIZE]; NODE_SIZE];
        for (i, row) in helper.adj_matrix.into_iter().enumerate() {
            adj_matrix[i].copy_from_slice(&row);
        }

        Ok(CircuitInput {
            adj_matrix,
            output_dir: helper.output_dir,
        })
    }
}


// this algorithm takes a public input x, computes x^2 + 72, and outputs the result as public output
fn test_matrix_to_digraph<F: BigPrimeField>(
    builder: &mut BaseCircuitBuilder<F>,
    x: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) {
    const PRECISION_BITS: u32 = 63;
    println!("build_lookup_bit: {:?}", builder.lookup_bits());
    let fixed_point_chip = FixedPointChip::<F, PRECISION_BITS>::default(builder);
    // can still get a Context via:
    let ctx = builder.main(0); // 0 means FirstPhase, don't worry about it

    let num_of_graphs = 10;
    let args = Cli::parse();
    let num_of_nodes: usize = args.num_attr.unwrap_or(10);
    println!("num_of_nodes: {:?}", num_of_nodes);
    let mut graph_f64 = vec![vec![0.0; num_of_nodes]; num_of_nodes];
    for _ in 0..num_of_graphs {
        let mut d_matrix: DMatrix<f64> = DMatrix::from_vec(num_of_nodes, num_of_nodes, graph_f64.iter().flatten().cloned().collect());
        for i in 0..num_of_nodes {
            for j in 0..num_of_nodes {
                // random add 0.0 or 1.0
                if i < j {
                    d_matrix[(i,j)] = (rand::random::<f64>() * 2.0).floor();
                }
            }
        }
        
        let result_graph = CausalGraph::<F,PRECISION_BITS>::ancestors_identification(ctx, &fixed_point_chip, &d_matrix);
        let graph_dmatrix = CausalGraph::<F,PRECISION_BITS>::zkmatrix_to_dmatrix(&fixed_point_chip, &result_graph);
    }
}

fn main() {
    env_logger::init();
    set_var("LOOKUP_BITS", 7.to_string());

    let args = Cli::parse();

    // let's say we don't want to run prover with inputs from file
    // instead we generate inputs here:
    // let private_inputs = Fr::random(OsRng);


    //record the time
    // let start = std::time::Instant::now();
    run(test_matrix_to_digraph, args);

    // // record time again
    // let duration = start.elapsed();
    // println!("Time elapsed in test_matrix_to_digraph() is: {:?}", duration);
}
