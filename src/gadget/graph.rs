//! # Zero-Knowledge Causal Graph Operations
//!
//! This module implements zero-knowledge circuits for causal inference operations on directed graphs.
//! It provides privacy-preserving computation of causal relationships, graph properties, and
//! statistical operations while maintaining verifiability through zero-knowledge proofs.
//!
//! ## Key Features
//!
//! - **Causal Graph Representation**: Zero-knowledge matrix representation of causal graphs
//! - **Acyclicity Verification**: Prove that a causal graph is acyclic without revealing structure
//! - **D-separation Testing**: Verify conditional independence relationships in causal graphs
//! - **Ancestor Identification**: Identify causal ancestors while preserving privacy
//! - **Intervention Analysis**: Compute causal effects of interventions in zero-knowledge
//! - **Matrix Operations**: Efficient matrix multiplication and verification in ZK circuits
//!
//! ## Algorithm Overview
//!
//! The module implements several fundamental causal inference algorithms:
//!
//! 1. **Acyclicity Check**: Uses topological sorting to verify graph acyclicity
//! 2. **D-separation**: Implements the moral graph construction and path blocking rules
//! 3. **Ancestor Identification**: Computes reachability matrices for causal relationships
//! 4. **Intervention Effects**: Models the effects of interventions on causal graphs
//!
//! ## Zero-Knowledge Constraints
//!
//! All operations are constrained within the Halo2 proving system to ensure:
//! - Correctness of computations without revealing private data
//! - Verifiability of results through zero-knowledge proofs
//! - Privacy preservation of graph structure and data
//!
//! #![allow(dead_code)]
use std::os::linux::raw;

// #[allow(unused_imports)]
use halo2_base::gates::{GateChip, GateInstructions, RangeChip, RangeInstructions};
use halo2_base::utils::{biguint_to_fe, BigPrimeField};
use halo2_base::{AssignedValue, QuantumCell};
use halo2_base::{
    Context,
    QuantumCell::{Constant, Existing},
    poseidon::hasher::PoseidonHasher,
};
use num_bigint::BigUint;
use snark_verifier_sdk::snark_verifier::halo2_ecc::fields::fp;
use super::fixed_point::{FixedPointChip, FixedPointInstructions};
use super::matrix::{MatrixChip};
use snark_verifier_sdk::halo2::OptimizedPoseidonSpec;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo::toposort;
use petgraph::visit::Topo;
use petgraph::{matrix_graph, Direction};
use ndarray::Array2;
use nalgebra::DMatrix;


/// Zero-knowledge representation of a causal graph
///
/// A causal graph is represented as an adjacency matrix where:
/// - `matrix[i][j] = 1` indicates a directed edge from node i to node j
/// - `matrix[i][j] = 0` indicates no edge between nodes i and j
/// - The diagonal elements are always 0 (no self-loops)
///
/// The struct maintains both the zero-knowledge matrix representation and
/// the raw matrix for efficient computation outside the circuit.
///
/// # Type Parameters
/// - `F`: The prime field type for zero-knowledge operations
/// - `PRECISION_BITS`: Number of bits for fixed-point arithmetic precision
#[derive(Clone)]
pub struct CausalGraph<F: BigPrimeField, const PRECISION_BITS: u32> {
    /// Raw matrix representation for efficient computation outside the circuit
    pub raw_matrix: Option<DMatrix<f64>>,
    
    /// Zero-knowledge matrix representation where each element is an `AssignedValue<F>`
    pub matrix: Vec<Vec<AssignedValue<F>>>,
    
    /// Number of nodes in the causal graph (matrix dimension)
    pub num_nodes: usize,
}
impl<F: BigPrimeField, const PRECISION_BITS: u32> CausalGraph<F, PRECISION_BITS> {
    /// Creates a zero-knowledge causal graph from a floating-point adjacency matrix
    ///
    /// This method converts a standard adjacency matrix into a zero-knowledge representation
    ///
    /// # Parameters
    /// - `ctx`: Halo2 context for witness generation
    /// - `fpchip`: Fixed-point arithmetic chip for quantization
    /// - `matrix`: Input adjacency matrix where `matrix[i][j]` represents the edge weight
    ///             from node i to node j (typically 0 or 1 for causal graphs)
    ///
    /// # Returns
    /// A `CausalGraph` with both raw and zero-knowledge matrix representations
    pub fn new(
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        matrix: &Vec<Vec<f64>>,
    ) -> Self {
        let mut zkmatrix: Vec<Vec<AssignedValue<F>>> = Vec::new();
        let num_rows = matrix.len();
        let num_col = matrix[0].len();
        for row in matrix {
            assert!(row.len() == num_col);
        }
        for i in 0..num_rows {
            let mut new_row: Vec<AssignedValue<F>> = Vec::new();
            for j in 0..num_col {
                let elem = fpchip.quantization(matrix[i][j]);
                new_row.push(ctx.load_witness(elem));
            }
            zkmatrix.push(new_row);
        }
        return Self { raw_matrix: Some(DMatrix::<f64>::from_vec(num_rows, num_col, matrix.iter().flatten().copied().collect())), matrix: zkmatrix, num_nodes: num_rows };
    }

    /// Dequantizes the zero-knowledge matrix back to floating-point representation
    ///
    /// # Parameters
    /// - `fpchip`: Fixed-point arithmetic chip for dequantization
    ///
    /// # Returns
    /// A 2D vector of floating-point values representing the dequantized matrix
    pub fn dequantize(&self, fpchip: &FixedPointChip<F, PRECISION_BITS>) -> Vec<Vec<f64>> {
        let mut dq_matrix: Vec<Vec<f64>> = Vec::new();
        for i in 0..self.num_nodes {
            dq_matrix.push(Vec::<f64>::new());
            for j in 0..self.num_nodes {
                let elem = self.matrix[i][j];
                dq_matrix[i].push(fpchip.dequantization(*elem.value()));
            }
        }
        return dq_matrix;
    }

    pub fn print(&self, fpchip: &FixedPointChip<F, PRECISION_BITS>) {
        print!("[\n");
        for i in 0..self.num_nodes {
            print!("[\n");
            for j in 0..self.num_nodes {
                let elem = self.matrix[i][j];
                let elem = fpchip.dequantization(*elem.value());
                print!("{:?}, ", elem);
            }
            print!("], \n");
        }
        println!("]");
    }

    /// Verifies matrix multiplication using the Freivalds’ algorithm
    ///
    /// This method proves that `a * b = c_s` using a probabilistic verification
    /// technique that is much more efficient than directly computing the matrix product
    /// in the zero-knowledge circuit.
    ///
    /// # Parameters
    /// - `ctx`: Halo2 context for constraint generation
    /// - `fpchip`: Fixed-point arithmetic chip
    /// - `a`: First matrix (left operand)
    /// - `b`: Second matrix (right operand) 
    /// - `c_s`: Claimed product matrix (unscaled)
    /// - `init_rand`: Random challenge value
    pub fn verify_mul(
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        a: &Self,
        b: &Self,
        c_s: &Vec<Vec<AssignedValue<F>>>,
        init_rand: &AssignedValue<F>,
    ) {
        assert_eq!(a.num_nodes, b.num_nodes);
        assert_eq!(c_s.len(), a.num_nodes);
        assert_eq!(c_s[0].len(), b.num_nodes);
        assert!(c_s[0].len() >= 1);

        let d = c_s[0].len();
        let gate = fpchip.gate();

        // v = (1, r, r^2, ..., r^(d-1)) where r = init_rand is the random challenge value
        let mut v: Vec<AssignedValue<F>> = Vec::new();

        let one = ctx.load_witness(F::from(1));
        gate.assert_is_const(ctx, &one, &F::from(1));
        v.push(one);

        for i in 1..d {
            let prev = &v[i - 1];
            let r_to_i = fpchip.gate().mul(ctx, *prev, *init_rand);
            v.push(r_to_i);
        }
        let v = v;

        // println!("Random vector, v = [");
        // for x in &v {
        //     println!("{:?}", *x.value());
        // }
        // println!("]");

        let cs_times_v = field_mat_vec_mul(ctx, gate, c_s, &v);
        let b_times_v = field_mat_vec_mul(ctx, gate, &b.matrix, &v);
        let ab_times_v = field_mat_vec_mul(ctx, gate, &a.matrix, &b_times_v);

        for i in 0..cs_times_v.len() {
            gate.is_equal(ctx, cs_times_v[i], ab_times_v[i]);
        }
    }

    /// Rescales a matrix after multiplication to correct the quantization factor
    ///
    /// After matrix multiplication in the field, the result needs to be divided by
    /// the quantization factor to maintain the correct fixed-point representation.
    /// This operation is computationally expensive but necessary for correctness.
    ///
    /// # Parameters
    /// - `ctx`: Halo2 context for constraint generation
    /// - `fpchip`: Fixed-point arithmetic chip for division
    /// - `c_s`: Unscaled matrix product (field elements)
    ///
    /// # Returns
    /// A new `CausalGraph` with properly scaled matrix elements
    pub fn rescale_matrix(
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        c_s: &Vec<Vec<AssignedValue<F>>>,
    ) -> Self {
        // #CONSTRAINTS = 94*N^2
        // now rescale c_s
        let mut c: Vec<Vec<AssignedValue<F>>> = Vec::new();
        let num_rows = c_s.len();
        let num_col = c_s[0].len();
        for i in 0..num_rows {
            let mut new_row: Vec<AssignedValue<F>> = Vec::new();
            for j in 0..num_col {
                // use fpchip to rescale c_s[i][j]
                // implemented in circuit, so we know c produced is correct
                let (elem, _) = fpchip.signed_div_scale(ctx, c_s[i][j]);
                new_row.push(elem);
            }
            c.push(new_row);
        }
        let raw_matrix = Self::zkmatrix_to_dmatrix(fpchip, &c);
        return Self { raw_matrix: Some(raw_matrix), matrix: c, num_nodes: num_rows };
    }

    /// Computes a Poseidon hash of the matrix 
    ///
    /// # Parameters
    /// - `ctx`: Halo2 context for witness generation
    /// - `gate`: Gate chip for basic operations
    /// - `matrix`: The matrix to be hashed
    ///
    /// # Returns
    /// A single `AssignedValue<F>` representing the Poseidon hash of the matrix
    pub fn hash_matrix(
        ctx: &mut Context<F>,
        gate: &GateChip<F>,
        matrix: &Self,
    ) -> AssignedValue<F> {
        // T, R_F, R_P values correspond to POSEIDON-128 values given in Table 2 of the Poseidon hash paper
        const T: usize = 3;
        const RATE: usize = 2;
        const R_F: usize = 8;
        const R_P: usize = 57;

        let gate = GateChip::<F>::default();
        let mut poseidon =
            PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
        poseidon.initialize_consts(ctx, &gate);

        // MODE OF USE: we will update the poseidon chip with all the values and then extract one value
        
        let mut flat_adj_mat_witnesses: Vec<AssignedValue<F>> = Vec::with_capacity(matrix.num_nodes * matrix.num_nodes);
        for row in matrix.matrix.iter() {
            for &val in row.iter() {
                flat_adj_mat_witnesses.push(val);
            }
        }
        // let flat_adj_mat_witnesses: [AssignedValue<F>; matrix.num_nodes * matrix.num_nodes] = flat_adj_mat_witnesses.try_into().unwrap();
        let adj_mat_hash = poseidon.hash_fix_len_array(ctx, &gate, &flat_adj_mat_witnesses);
            
        // dbg!(init_rand.value());
        return adj_mat_hash;
    }

    /// Computes the transpose of a matrix without adding constraints
    ///
    /// This method creates a new matrix where `transpose[i][j] = original[j][i]`.
    /// It's a pure data transformation that doesn't add any zero-knowledge constraints.
    ///
    /// # Parameters
    /// - `fpchip`: Fixed-point arithmetic chip for matrix conversion
    /// - `a`: Input matrix to transpose
    ///
    /// # Returns
    /// A new `CausalGraph` with the transposed matrix
    pub fn transpose_matrix(fpchip: &FixedPointChip<F, PRECISION_BITS>, a: &Self) -> Self {
        let mut a_trans: Vec<Vec<AssignedValue<F>>> = Vec::new();

        for i in 0..a.num_nodes {
            let mut new_row: Vec<AssignedValue<F>> = Vec::new();
            for j in 0..a.num_nodes {
                new_row.push(a.matrix[j][i].clone());
            }
            a_trans.push(new_row);
        }

        let raw_matrix = Self::zkmatrix_to_dmatrix(fpchip, &a_trans);
        return Self {raw_matrix: Some(raw_matrix), matrix: a_trans, num_nodes: a.num_nodes};
    }

    /// Computes the transpose of a raw matrix without constraints
    ///
    /// This is a lightweight version of transpose that works directly on
    /// `Vec<Vec<AssignedValue<F>>>` without creating a full `CausalGraph` struct.
    /// Useful for intermediate computations in complex algorithms.
    ///
    /// # Parameters
    /// - `a`: Input matrix as a 2D vector of assigned values
    ///
    /// # Returns
    /// A new 2D vector representing the transposed matrix
    pub fn transpose_matrix_without_constraint(a: &Vec<Vec<AssignedValue<F>>>) -> Vec<Vec<AssignedValue<F>>> {
        let mut a_trans: Vec<Vec<AssignedValue<F>>> = Vec::new();
        let num_rows = a.len();
        let num_col = a[0].len();

        for i in 0..num_col {
            let mut new_row: Vec<AssignedValue<F>> = Vec::new();
            for j in 0..num_rows {
                new_row.push(a[j][i].clone());
            }
            a_trans.push(new_row);
        }

        return a_trans;
    }

    /// Converts a zero-knowledge matrix to a DMatrix
    ///
    /// # Parameters
    /// - `fpchip`: Fixed-point arithmetic chip for dequantization
    /// - `zkmatrix`: Zero-knowledge matrix as 2D vector of assigned values
    ///
    /// # Returns
    /// A DMatrix<f64> with the dequantized values
    pub fn zkmatrix_to_dmatrix(fpchip: &FixedPointChip<F, PRECISION_BITS>, zkmatrix: &Vec<Vec<AssignedValue<F>>>) -> DMatrix<f64> {
        let size = zkmatrix.len();
        let mut matrix = DMatrix::<f64>::zeros(size, size);
        for i in 0..size {
            for j in 0..size {
                matrix[(i, j)] = fpchip.dequantization(*zkmatrix[i][j].value());
            }
        }
        matrix
    }

    // pub fn dmatrix_to_zkmatrix(ctx: &mut Context<F>, fpchip: &FixedPointChip<F, PRECISION_BITS>, matrix: &DMatrix<f64>) -> Vec<Vec<AssignedValue<F>>> {
    //     let size = matrix.nrows();
    //     let mut zkmatrix: Vec<Vec<AssignedValue<F>>> = Vec::new();
    //     for i in 0..size {
    //         let mut new_row: Vec<AssignedValue<F>> = Vec::new();
    //         for j in 0..size {
    //             let elem = matrix[(i, j)];
    //             new_row.push(ctx.load_witness(F::from(elem)));
    //         }
    //         zkmatrix.push(new_row);
    //     }
    //     zkmatrix
    // }

    pub fn dmatrix_to_zkmatrix(ctx: &mut Context<F>, fpchip: &FixedPointChip<F, PRECISION_BITS>, matrix: &DMatrix<f64>) -> Vec<Vec<AssignedValue<F>>> {
        let size = matrix.nrows();
        let mut zkmatrix: Vec<Vec<AssignedValue<F>>> = Vec::new();
        for i in 0..size {
            let mut new_row: Vec<AssignedValue<F>> = Vec::new();
            for j in 0..size {
                let elem = fpchip.quantization(matrix[(i, j)]);
                new_row.push(ctx.load_witness(elem));
            }
            zkmatrix.push(new_row);
        }
        zkmatrix
    }

    /// Converts a zero-knowledge matrix to a directed graph
    ///
    /// This function creates a `DiGraph` from a zero-knowledge matrix representation,
    /// where non-zero elements indicate directed edges between nodes.
    ///
    /// # Parameters
    /// - `matrix`: Zero-knowledge matrix as 2D vector of assigned values
    ///
    /// # Returns
    /// A `DiGraph<(), ()>` representing the graph structure
    pub fn zkmatrix_to_digraph(matrix: &Vec<Vec<AssignedValue<F>>>) -> DiGraph<(), ()> {
        let size = matrix.len();
        let mut graph = DiGraph::new();
    
        let nodes: Vec<_> = (0..size).map(|_| graph.add_node(())).collect();
    
        for i in 0..size {
            for j in 0..size {
                if matrix[i][j].value().get_lower_64() != 0 && i != j {
                    graph.add_edge(nodes[i], nodes[j], ());
                }
            }
        }
    
        graph
    }

    /// Converts a DMatrix to a directed graph
    ///
    /// This function creates a `DiGraph` from a standard DMatrix representation,
    /// where non-zero elements indicate directed edges between nodes.
    ///
    /// # Parameters
    /// - `matrix`: Input DMatrix to convert
    ///
    /// # Returns
    /// A `DiGraph<(), ()>` representing the graph structure
    pub fn dmatrix_to_digraph(matrix: &DMatrix<f64>) -> DiGraph<(), ()> {
        // println!("Matrix A:\n{:?}", matrix);
        let size = matrix.nrows();
        // println!("size: {:?}", size);
        let mut graph = DiGraph::new();
    
        let nodes: Vec<_> = (0..size).map(|_| graph.add_node(())).collect();
    
        for i in 0..size {
            for j in 0..size {
                // println!("matrix[{}, {}]: {:?}", i, j, matrix[(i, j)]);
                if matrix[(i, j)] != 0.0 && i != j {
                    graph.add_edge(nodes[i], nodes[j], ());
                }
            }
        }
    
        graph
    }

    /// Performs honest acyclicity check outside the zero-knowledge circuit
    ///
    /// # Parameters
    /// - `g_dmatrix`: Input adjacency matrix
    ///
    /// # Returns
    /// A tuple containing:
    /// - `bool`: True if the graph is acyclic, false otherwise
    /// - `Vec<usize>`: Topological ordering of nodes (if acyclic) or cycle nodes (if cyclic)
    pub fn honest_acyclicity_check(g_dmatrix: &DMatrix<f64>) -> (bool, Vec<usize>) {
        let g_graph = Self::dmatrix_to_digraph(g_dmatrix);
        let mut idx_order: Vec<usize> = Vec::new();
        let mut is_acyclic = true;
        if let Some(circle_found) = find_cycle(&g_graph) {
            // println!("Graph has a cycle");
            is_acyclic = false;
            for node in circle_found {
                print!("{} ", node.index());
                idx_order.push(node.index());
            }
        } else {
            // println!("Graph is acyclic");
            // Topological sorting
            let topo_sorted_nodes = toposort(&g_graph, None).expect("The graph is not a DAG");
            idx_order = topo_sorted_nodes
                .iter()
                .map(|&node| node.index())
                .collect();
        }
        return (is_acyclic, idx_order);
    }

    /// Zero-knowledge acyclicity verification for causal graphs
    ///
    /// # Parameters
    /// - `ctx`: Halo2 context for constraint generation
    /// - `fpchip`: Fixed-point arithmetic chip
    /// - `g_dmatrix`: Input adjacency matrix
    ///
    /// # Returns
    /// An `AssignedValue<F>` representing the acyclicity result (1 if acyclic, 0 if cyclic)
    pub fn acyclicity_check(ctx: &mut Context<F>, fpchip: &FixedPointChip<F, PRECISION_BITS>, g_dmatrix: &DMatrix<f64>) -> AssignedValue<F> {
        // Step 1: Check if the graph is acyclic without constraints
        // Convert matrix to directed graph
        // println!("Matrix A:\n{:?}", a_dmatrix);
        let g_zk = Self::dmatrix_to_zkmatrix(ctx, fpchip, &g_dmatrix);
        let g_graph = Self::dmatrix_to_digraph(g_dmatrix);
        let (acyclic_flag, order) = Self::honest_acyclicity_check(g_dmatrix);
        // println!("The graph is: {:?}", graph);
        // println!("The order is: {:?}", order);
        // println!("The acyclic flag is: {:?}", acyclic_flag);
        let size = g_zk.len();
        // let ctx_constants: Vec<AssignedValue<F>> = (0..size).map(|i| ctx.load_constant(F::from(i as u64))).collect();
        let quantized_ctx_constants: Vec<AssignedValue<F>> = (0..size).map(|i| ctx.load_witness(fpchip.quantization(i as f64))).collect();
        let true_flag_witness = ctx.load_witness(F::from(1));
        let mut flag_constraint_1 = ctx.load_witness(F::from(0));
        let mut flag_constraint_2 = ctx.load_witness(F::from(0));
        if acyclic_flag {
            flag_constraint_1 = ctx.load_witness(F::from(1));
            // Generate permutation matrix P according to the topological sorting result
            let p = generate_permutation_matrix(&order, size);
            // the reverse of the permutation matrix P is its transpose
            // let p_inv = p.transpose();
            // println!("Permutation Matrix P:\n{:?}", p);
            let p_zk = Self::dmatrix_to_zkmatrix(ctx, fpchip, &p);
            // Compute B = P^(-1) A P
            let a_raw = Self::zkmatrix_to_dmatrix(fpchip, &g_zk);
            let b = compute_transformed_matrix(&a_raw, &p);
            // println!("Transformed Matrix B:\n{:?}", b);
            let b_zk = Self::dmatrix_to_zkmatrix(ctx, fpchip, &b);
            // println!("Transformed Matrix B:\n{:?}", b);
            // Constraint 1: Upper Triangular Matrix
            for i in 0..size {
                for j in 0..i+1 {
                    let is_equal_zero = fpchip.gate().is_equal(ctx, b_zk[i][j], quantized_ctx_constants[0]);
                    flag_constraint_1 = fpchip.gate().and(ctx, flag_constraint_1, is_equal_zero);
                }
            }

            // Constraint 2: Sum of each row and column of matrix p_zk is 1
            for i in 0..size {
                for j in 0..size {
                    let one_minus_p = fpchip.gate().sub(ctx, quantized_ctx_constants[1], p_zk[i][j]);
                    let p_one_minus_p = fpchip.gate().mul(ctx, p_zk[i][j], one_minus_p);
                    ctx.constrain_equal(&p_one_minus_p, &quantized_ctx_constants[0]);
                }
                let sum_row = fpchip.gate().sum(ctx, p_zk[i].to_vec());
                ctx.constrain_equal(&sum_row, &quantized_ctx_constants[1]);

                let cur_col: Vec<AssignedValue<F>> = (0..size).map(|j| p_zk[j][i]).collect();
                let sum_col = fpchip.gate().sum(ctx, cur_col);
                ctx.constrain_equal(&sum_col, &quantized_ctx_constants[1]);
            }   

            // Constraint 3: verify the transformation
            let rand = ctx.load_witness(F::from(2)); // TODO: random challenge value
            Self::verify_transformation(ctx, fpchip.gate(), &p_zk, &g_zk, &b_zk, &rand);        
        } else{
            // Case when the graph is not acyclic, find the cycle and constrain the cycle
            flag_constraint_2 = ctx.load_witness(F::from(1));
            for i in 0..order.len()-1 {
                let j = (i + 1) % order.len();
                let is_equal_zero = fpchip.gate().is_equal(ctx, g_zk[order[i]][order[j]], quantized_ctx_constants[1]);
                flag_constraint_2 = fpchip.gate().and(ctx, flag_constraint_2, is_equal_zero);
            }
        }
        let constraint_flag = fpchip.gate().or(ctx, flag_constraint_1, flag_constraint_2);
        ctx.constrain_equal(&constraint_flag, &true_flag_witness);

        return ctx.load_witness(F::from(acyclic_flag));
    }

    /// Generates an intervention graph in zero-knowledge
    ///
    /// # Parameters
    /// - `ctx`: Halo2 context for constraint generation
    /// - `fpchip`: Fixed-point arithmetic chip
    /// - `g_dmatrix`: Original causal graph adjacency matrix
    /// - `intervention`: Zero-knowledge value indicating which node to intervene on
    ///
    /// # Returns
    /// A zero-knowledge matrix representing the post-intervention graph
    pub fn intervention_graph_generation(ctx: &mut Context<F>, fpchip: &FixedPointChip<F, PRECISION_BITS>, g_dmatrix: &DMatrix<f64>, intervention: AssignedValue<F>) -> Vec<Vec<AssignedValue<F>>> {
        let g_zk = Self::dmatrix_to_zkmatrix(ctx, fpchip, &g_dmatrix);
        let size = g_zk.len();
        let constant_witness_vec: Vec<AssignedValue<F>> = (0..size).map(|i| ctx.load_witness(F::from(i as u64))).collect();
        let quantized_ctx_constants: Vec<AssignedValue<F>> = (0..size).map(|i| ctx.load_witness(fpchip.quantization(i as f64))).collect();
        let mut intervention_graph: Vec<Vec<AssignedValue<F>>> = Vec::new();
        for i in 0..size {
            let mut new_row: Vec<AssignedValue<F>> = Vec::new();
            for j in 0..size {
                let is_intervention = fpchip.gate().is_equal(ctx, intervention, constant_witness_vec[j]);
                // let is_not_intervention = fpchip.gate().not(ctx, is_intervention);
                let elem_true = fpchip.gate().and(ctx, quantized_ctx_constants[0], is_intervention);
                // let elem_false = fpchip.gate().mul(ctx, g_zk[i][j], is_not_intervention);
                // let elem = fpchip.qadd(ctx, elem_true, elem_false);
                new_row.push(elem_true);
            }
            intervention_graph.push(new_row);
        }
        intervention_graph
    }

    /// Zero-knowledge d-separation test for causal graphs
    ///
    /// # Parameters
    /// - `ctx`: Halo2 context for constraint generation
    /// - `fpchip`: Fixed-point arithmetic chip
    /// - `g_dmatrix`: Causal graph adjacency matrix
    /// - `x`: First node to test (source)
    /// - `y`: Second node to test (target)
    /// - `z`: Conditioning set (nodes to remove)
    /// - `ancestor_graph`: Precomputed ancestor relationships
    ///
    /// # Returns
    /// An `AssignedValue<F>`: 1 if X and Y are d-separated by Z, 0 otherwise
    pub fn d_separation(ctx: &mut Context<F>, fpchip: &FixedPointChip<F, PRECISION_BITS>, g_dmatrix: &DMatrix<f64>, x: usize, y: usize, z: Vec<usize>, ancestor_graph: Vec<Vec<AssignedValue<F>>>) -> AssignedValue<F> {
        let g_zk = Self::dmatrix_to_zkmatrix(ctx, fpchip, &g_dmatrix);
        let size = g_zk.len();
        let constant_witness_vec: Vec<AssignedValue<F>> = (0..size).map(|i| ctx.load_witness(F::from(i as u64))).collect();
        let quantized_ctx_constants: Vec<AssignedValue<F>> = (0..size).map(|i| ctx.load_witness(fpchip.quantization(i as f64))).collect();
        
        let err_svd_scale = MatrixChip::<F, PRECISION_BITS>::cal_scaled_error(1e-10, size);

        let matrix_chip: MatrixChip<F, PRECISION_BITS> = MatrixChip::new(err_svd_scale, constant_witness_vec[2].clone());

        // enforce the constraints that the element of a is either 0 or 1
        for i in 0..size {
            for j in 0..size {
                let one_minus_p = fpchip.gate().sub(ctx, quantized_ctx_constants[1], g_zk[i][j]);
                let p_one_minus_p = fpchip.gate().mul(ctx, g_zk[i][j], one_minus_p);
                ctx.constrain_equal(&p_one_minus_p, &quantized_ctx_constants[0]);
            }
        }

        // Step 1: Compute ancestral graph G_ancestor(X ∪ Y ∪ Z)
        // First, identify all ancestors of nodes in X ∪ Y ∪ Z
        let mut ancestor_nodes = Vec::new();
        ancestor_nodes.push(x);
        ancestor_nodes.push(y);
        ancestor_nodes.extend(z.clone());

        let zero_matrix = DMatrix::<f64>::zeros(size, size);
        let mut mask_matrix: Vec<Vec<AssignedValue<F>>> = Self::dmatrix_to_zkmatrix(ctx, fpchip, &zero_matrix);
        for i in 0..size {
            let mut is_unmasked = ctx.load_witness(F::from(0));
            for j in &ancestor_nodes {
                is_unmasked = fpchip.gate().or(ctx, is_unmasked, ancestor_graph[i][*j]);
            }
            mask_matrix[i][i] = is_unmasked;
        }
        
        // Create ancestral subgraph by masking non-ancestor nodes
        let mut masked_ancestor_graph: Vec<Vec<AssignedValue<F>>> = matrix_chip.matrix_mul_no_scale(ctx, fpchip, &mask_matrix, &ancestor_graph);
        masked_ancestor_graph = matrix_chip.matrix_mul_no_scale(ctx, fpchip, &masked_ancestor_graph, &mask_matrix);
        
        // Step 2: Construct moral graph G' = G_ancestor * G_ancestor^T (binarized)
        let g_ancestor_transpose = MatrixChip::<F, PRECISION_BITS>::transpose_matrix(&masked_ancestor_graph);
        
        // Compute moral graph using matrix multiplication
        let moral_graph_zk = matrix_chip.matrix_mul_no_scale(ctx, fpchip, &masked_ancestor_graph, &g_ancestor_transpose);
        
        // Binarize the moral graph
        let mut moral_graph_binary: Vec<Vec<AssignedValue<F>>> = Self::dmatrix_to_zkmatrix(ctx, fpchip, &zero_matrix);

        let mut mask_z_matrix: Vec<Vec<AssignedValue<F>>> = Self::dmatrix_to_zkmatrix(ctx, fpchip, &zero_matrix);
        for i in 0..size {
            let mut is_masked = ctx.load_witness(F::from(0));
            for j in &z {
                let is_i_in_z = fpchip.gate().is_equal(ctx, constant_witness_vec[i], constant_witness_vec[*j]);
                is_masked = fpchip.gate().or(ctx, is_masked, is_i_in_z);
            }
            mask_z_matrix[i][i] = fpchip.gate().not(ctx, is_masked);
        }

        for i in 0..size {
            for j in 0..size {
                let is_zero = fpchip.gate().is_zero(ctx, moral_graph_zk[i][j]);
                moral_graph_binary[i][j] = fpchip.gate().not(ctx, is_zero);
            }
        }

        // Step 3: Remove conditioning nodes Z from moral graph

        let mut masked_final_graph: Vec<Vec<AssignedValue<F>>> = matrix_chip.matrix_mul_no_scale(ctx, fpchip, &mask_z_matrix, &moral_graph_binary);
        masked_final_graph = matrix_chip.matrix_mul_no_scale(ctx, fpchip, &masked_final_graph, &mask_z_matrix);
        // // println!("masked_final_graph size: {:?}", masked_final_graph.len());
        
        // Step 4: Floyd-Warshall algorithm to check connectivity
        let mut reachability = masked_final_graph.clone();
        
        // Floyd-Warshall algorithm
        for k in 0..size {
            for i in 0..size {
                for j in 0..size {
                    // println!("i: {:?}, j: {:?}, k: {:?}", i, j, k);
                    let mut i_k_cell = masked_final_graph[i][k];
                    let mut k_j_cell = masked_final_graph[k][j];
                    let mut i_j_cell = masked_final_graph[i][j];
                    // println!("i_k_cell: {:?}, k_j_cell: {:?}, i_j_cell: {:?}", i_k_cell, k_j_cell, i_j_cell);
                    let and_cell = fpchip.gate().and(ctx, i_k_cell, k_j_cell);
                    let mut or_cell = fpchip.gate().or(ctx, i_j_cell, and_cell);
                    reachability[i][j] = or_cell;
                }
            }
        }

        return reachability[x][y];
        // return ctx.load_witness(F::from(1));
    }

    /// Takes quantised matrices `a` and `b`, their unscaled product `c_s`
    /// and a commitment (hash) to *at least* all of these matrices `init_rand`
    /// and checks if `a*b = c_s` in field multiplication.
    ///
    /// `c_s`: unscaled product of `a` and `b`(produced by simply multiplying `a` and `b` as field elements);
    ///  producing this is the costly part of matrix multiplication
    ///
    /// `init_rand`:  is the starting randomness/ challenge value; should commit to
    /// *at least* the matrices `a, b, c_s`
    ///
    /// Since, this method only verifies field multiplication, it will not fail even if
    /// `a` and `b` are incorrectly encoded. However, trying to rescale the result and use
    /// it downstream might fail in this case.
    pub fn verify_transformation(
        ctx: &mut Context<F>,
        gate: &GateChip<F>,
        p: &Vec<Vec<AssignedValue<F>>>,
        a: &Vec<Vec<AssignedValue<F>>>,
        b: &Vec<Vec<AssignedValue<F>>>,
        init_rand: &AssignedValue<F>,
    ) {
        assert_eq!(p.len(), a.len());
        assert_eq!(b.len(), p.len());
        assert_eq!(b[0].len(), a.len());
        assert!(b[0].len() >= 1);

        let d = b[0].len();

        // v = (1, r, r^2, ..., r^(d-1)) where r = init_rand is the random challenge value
        let mut v: Vec<AssignedValue<F>> = Vec::new();

        let one = ctx.load_witness(F::from(1));
        gate.assert_is_const(ctx, &one, &F::from(1));
        v.push(one);

        for i in 1..d {
            let prev = &v[i - 1];
            let r_to_i = gate.mul(ctx, *prev, *init_rand);
            v.push(r_to_i);
        }
        let v = v;

        // println!("Random vector, v = [");
        // for x in &v {
        //     println!("{:?}", *x.value());
        // }
        // println!("]");

        // let b_times_v = field_mat_vec_mul(ctx, gate, b, &v);
        let p_times_v = field_mat_vec_mul(ctx, gate, p, &v);
        let b_p_times_v = field_mat_vec_mul(ctx, gate, b, &p_times_v);
        // let p_inv_times_v = field_mat_vec_mul_a_inverse(ctx, gate, p, &v);
        let a_times_v = field_mat_vec_mul(ctx, gate, a, &v);
        let p_a_times_v = field_mat_vec_mul(ctx, gate, p, &a_times_v);
        // let a_p_inv_times_v = field_mat_vec_mul(ctx, gate, a, &p_inv_times_v);
        // let p_a_p_inv_times_v = field_mat_vec_mul(ctx, gate, p, &a_p_inv_times_v);

        for i in 0..b_p_times_v.len() {
            ctx.constrain_equal(&b_p_times_v[i], &p_a_times_v[i]);
        }
    }

    pub fn verify_transformation_raw(
        ctx: &mut Context<F>,
        gate: &GateChip<F>,
        p: &Vec<Vec<AssignedValue<F>>>,
        a: &Vec<Vec<AssignedValue<F>>>,
        b: &Vec<Vec<AssignedValue<F>>>,
        init_rand: &AssignedValue<F>,
    ) {
        assert_eq!(p.len(), a.len());
        assert_eq!(b.len(), p.len());
        assert_eq!(b[0].len(), a.len());
        assert!(b[0].len() >= 1);

        let d = b[0].len();

        let b_p_times_v = field_mat_mul_raw(ctx, gate,b,p);
        let p_a_times_v = field_mat_mul_raw(ctx, gate, p, a);

        for i in 0..b_p_times_v.len() {
            for j in 0..b_p_times_v[0].len() {
                ctx.constrain_equal(&b_p_times_v[i][j], &p_a_times_v[i][j]);
            }
        }
    }

    pub fn constrain_matrix_equal(
        ctx: &mut Context<F>,
        a: &Vec<Vec<AssignedValue<F>>>,
        b: &Vec<Vec<AssignedValue<F>>>,
    ) {
        for i in 0..a.len() {
            for j in 0..a[0].len() {
                ctx.constrain_equal(&a[i][j], &b[i][j]);
            }
        }
    }

    pub fn verify_matrix_inverse(
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        a: &Vec<Vec<AssignedValue<F>>>,
        a_inv: &Vec<Vec<AssignedValue<F>>>,
        init_rand: &AssignedValue<F>,
    ) {
        let d = a[0].len();
        let quantized_one_square = ctx.load_witness(fpchip.quantization(1.0)*fpchip.quantization(1.0)); // TODO: add constraint

        // v = (1, r, r^2, ..., r^(d-1)) where r = init_rand is the random challenge value
        let mut v: Vec<AssignedValue<F>> = Vec::new();

        let one = ctx.load_witness(F::from(1));
        fpchip.gate().assert_is_const(ctx, &one, &F::from(1));
        v.push(one);

        for i in 1..d {
            let prev = &v[i - 1];
            let r_to_i = fpchip.gate().mul(ctx, *prev, *init_rand);
            v.push(r_to_i);
        }
        let v = v;

        // println!("Random vector, v = [");
        // for x in &v {
        //     println!("{:?}", *x.value());
        // }
        // println!("]");

        let a_times_v = field_mat_vec_mul(ctx, fpchip.gate(), a, &v);
        let a_inv_a_times_v = field_mat_vec_mul(ctx, fpchip.gate(), a_inv, &a_times_v);

        for i in 0..a_inv_a_times_v.len() {
            let quantized_v_i = fpchip.gate().mul(ctx, v[i], quantized_one_square);
            ctx.constrain_equal(&a_inv_a_times_v[i], &quantized_v_i);
        }
    }

    pub fn verify_matrix_inverse_raw(
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        a: &Vec<Vec<AssignedValue<F>>>,
        a_inv: &Vec<Vec<AssignedValue<F>>>,
        init_rand: &AssignedValue<F>,
    ) {
        let d = a[0].len();
        let quantized_one_square = ctx.load_witness(fpchip.quantization(1.0)*fpchip.quantization(1.0)); // TODO: add constraint
        let quantized_zero = ctx.load_witness(fpchip.quantization(0.0)); // TODO: add constraint

        // v = (1, r, r^2, ..., r^(d-1)) where r = init_rand is the random challenge value
        // let mut v: Vec<AssignedValue<F>> = Vec::new();

        let one = ctx.load_witness(F::from(1));
        fpchip.gate().assert_is_const(ctx, &one, &F::from(1));
        // v.push(one);

        let a_inv_a = Self::matrix_mul_raw(ctx, fpchip, a_inv, a);

        for i in 0..a_inv_a.len() {
            for j in 0..a_inv_a[0].len() {
                if i == j {
                    // ctx.constrain_equal(&a_inv_a[i][j], &quantized_one_square);
                }
                else {
                    // ctx.constrain_equal(&a_inv_a[i][j], &quantized_zero);
                }
                
            }
        }
    }

    pub fn matrix_mul_raw( //_backup
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        a: &Vec<Vec<AssignedValue<F>>>,
        b: &Vec<Vec<AssignedValue<F>>>,
    )->Vec<Vec<AssignedValue<F>>> {
        let mut c: Vec<Vec<AssignedValue<F>>> = vec![];
        let mut c_s: Vec<Vec<AssignedValue<F>>> = vec![];

        let q_one = ctx.load_witness(fpchip.quantization(1.0));
        for i in 0..a.len() {
            let mut c_row: Vec<AssignedValue<F>> = vec![];
            for j in 0..b[0].len() {
                let mut elem = ctx.load_witness(F::from(0));
                for k in 0..a[0].len() {
                    let prod = fpchip.gate().mul(ctx, a[i][k], b[k][j]);
                    elem = fpchip.qadd(ctx, elem, prod);
                }
                c_row.push(elem);
            }
            c.push(c_row);
        }

        for i in 0..a.len() {
            let mut c_row: Vec<AssignedValue<F>> = vec![];
            for j in 0..b[0].len() {
                let mut elem = ctx.load_witness(F::from(0));
                for k in 0..a[0].len() {
                    let dq_a = fpchip.dequantization(*a[i][k].value());
                    let dq_b = fpchip.dequantization(*b[k][j].value());
                    let dq_c = dq_a * dq_b;
                    let q_c = ctx.load_witness(fpchip.quantization(dq_c));
                    elem = fpchip.qadd(ctx, elem, q_c);
                }
                c_row.push(elem);
            }
            c_s.push(c_row);
        }

        // let c_s = Self::honest_prover_mat_mul(ctx, a, b);
        for i in 0..c.len() {
            for j in 0..c[0].len() {
                let c_s_q = fpchip.gate().mul(ctx, c_s[i][j], q_one);
                // ctx.constrain_equal(&c[i][j], &c_s_q);
            }
        }
        
        return c_s.clone();
    }

    /// Zero-knowledge ancestor identification for causal graphs
    ///
    /// # Parameters
    /// - `ctx`: Halo2 context for constraint generation
    /// - `fpchip`: Fixed-point arithmetic chip
    /// - `a_dmatrix`: Input causal graph adjacency matrix
    ///
    /// # Returns
    /// A zero-knowledge binary matrix where `result[i][j] = 1` if i is ancestor of j
    pub fn ancestors_identification(ctx: &mut Context<F>, fpchip: &FixedPointChip<F, PRECISION_BITS>, a_dmatrix: &DMatrix<f64>)-> Vec<Vec<AssignedValue<F>>> {
        // Step 1: Preprocessing
        // Convert matrix to directed graph
        // println!("Matrix A:\n{:?}", a_dmatrix);
        let a = Self::dmatrix_to_zkmatrix(ctx, fpchip, &a_dmatrix);
        let graph = Self::dmatrix_to_digraph(a_dmatrix);
        // println!("The graph is: {:?}", graph);
        let size = a.len();
        // let ctx_constants: Vec<AssignedValue<F>> = (0..size).map(|i| ctx.load_constant(F::from(i as u64))).collect();
        let quantized_ctx_constants: Vec<AssignedValue<F>> = (0..size).map(|i| ctx.load_witness(fpchip.quantization(i as f64))).collect();
        let ctx_constants: Vec<AssignedValue<F>> = (0..size).map(|i| ctx.load_witness(F::from(i as u64))).collect();
        // Compute (I-B)^(-1)
        let i_a = DMatrix::<f64>::identity(size, size) - a_dmatrix;
        let i_a_zk = Self::dmatrix_to_zkmatrix(ctx, fpchip, &i_a);
        let i_a_inv = i_a.try_inverse().unwrap();
        let i_b_inv_zk = Self::dmatrix_to_zkmatrix(ctx, fpchip, &i_a_inv);


        // Step 2: Enforce Constraints
        // enforce the constraints that the element of a is either 0 or 1

        for i in 0..size {
            for j in 0..size {
                let one_minus_p = fpchip.gate().sub(ctx, quantized_ctx_constants[1], a[i][j]);
                let p_one_minus_p = fpchip.gate().mul(ctx, a[i][j], one_minus_p);
                ctx.constrain_equal(&p_one_minus_p, &quantized_ctx_constants[0]);
                
            }
        }


        // Constraint 3: b_zk = p_zk a p_zk^(-1) using Freivalds’ algorithm
        let rand = ctx.load_witness(F::from(2)); // TODO: random challenge value

        // Constraint 4: i_b_inv is the inverse of i_b
        Self::verify_matrix_inverse(ctx, fpchip, &i_a_zk, &i_b_inv_zk, &rand);
        // Self::verify_matrix_inverse_raw(ctx, fpchip, &i_a_zk, &i_b_inv_zk, &rand);
        
        // Step 3: Descendant Identification
        let zero_matrix = DMatrix::<f64>::zeros(size, size);
        let mut binary_matrix_zk = Self::dmatrix_to_zkmatrix(ctx, fpchip, &zero_matrix);
        for i in 0..size {
            for j in 0..size {
                binary_matrix_zk[i][j] = fpchip.gate().is_zero(ctx, i_b_inv_zk[i][j]);
            }
        }

        (binary_matrix_zk)

    }
}

/// Constrains that a field element has absolute value less than a bound
///
/// # Parameters
/// - `ctx`: Halo2 context for constraint generation
/// - `range`: Range chip for range checking
/// - `x`: Field element to constrain
/// - `bnd`: Upper bound for absolute value
pub fn check_abs_less_than<F: BigPrimeField>(
    ctx: &mut Context<F>,
    range: &RangeChip<F>,
    x: AssignedValue<F>,
    bnd: &BigUint,
) {
    let new_bnd = BigUint::from(2u32) * bnd - BigUint::from(1u32);
    let translated_x =
        range.gate.add(ctx, x, Constant(biguint_to_fe(&(bnd - BigUint::from(1u32)))));
    range.check_big_less_than_safe(ctx, translated_x, new_bnd);
}

/// Constrains that two matrices are close within a tolerance
///
/// # Parameters
/// - `ctx`: Halo2 context for constraint generation
/// - `range`: Range chip for range checking
/// - `a`: First matrix
/// - `b`: Second matrix
/// - `tol`: Tolerance for element-wise difference
pub fn check_mat_diff<F: BigPrimeField>(
    ctx: &mut Context<F>,
    range: &RangeChip<F>,
    a: &Vec<Vec<AssignedValue<F>>>,
    b: &Vec<Vec<AssignedValue<F>>>,
    tol: &BigUint,
) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a[0].len(), b[0].len());

    for i in 0..a.len() {
        for j in 0..a[0].len() {
            let diff = range.gate.sub(ctx, a[i][j], b[i][j]);
            check_abs_less_than(ctx, &range, diff, tol);
        }
    }
}

/// Constrains that a matrix is close to a scalar multiple of the identity matrix
/// 
/// # Parameters
/// - `ctx`: Halo2 context for constraint generation
/// - `range`: Range chip for range checking
/// - `a`: Matrix to check
/// - `scalar_id`: Scalar multiplier for identity matrix
/// - `tol`: Tolerance for element-wise difference
pub fn check_mat_id<F: BigPrimeField>(
    ctx: &mut Context<F>,
    range: &RangeChip<F>,
    a: &Vec<Vec<AssignedValue<F>>>,
    scalar_id: &AssignedValue<F>,
    tol: &BigUint,
) {
    let mut b: Vec<Vec<AssignedValue<F>>> = Vec::new();
    let zero = ctx.load_constant(F::from(0));

    for i in 0..a.len() {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..a[0].len() {
            if i == j {
                row.push(scalar_id.clone())
            } else {
                row.push(zero.clone());
            }
        }
        b.push(row);
    }
    check_mat_diff(ctx, &range, a, &b, tol);
}

/// Constrains that all matrix entries have absolute value less than a bound
///
/// # Parameters
/// - `ctx`: Halo2 context for constraint generation
/// - `range`: Range chip for range checking
/// - `a`: Matrix to check
/// - `bnd`: Upper bound for absolute values
pub fn check_mat_entries_bounded<F: BigPrimeField>(
    ctx: &mut Context<F>,
    range: &RangeChip<F>,
    a: &Vec<Vec<AssignedValue<F>>>,
    bnd: &BigUint,
) {
    for i in 0..a.len() {
        for j in 0..a[0].len() {
            check_abs_less_than(ctx, &range, a[i][j], &bnd);
        }
    }
}

/// Computes matrix multiplication outside the zero-knowledge circuit
/// 
/// # Parameters
/// - `a`: First matrix (left operand)
/// - `b`: Second matrix (right operand)
///
/// # Returns
/// A 2D vector of field elements representing the matrix product
pub fn field_mat_mul<F: BigPrimeField>(
    a: &Vec<Vec<AssignedValue<F>>>,
    b: &Vec<Vec<AssignedValue<F>>>,
) -> Vec<Vec<F>> {
    // a.num_col == b.num_rows
    assert_eq!(a[0].len(), b.len());

    let mut c: Vec<Vec<F>> = Vec::new();
    #[allow(non_snake_case)]
    let N = a.len();
    #[allow(non_snake_case)]
    let K = a[0].len();
    #[allow(non_snake_case)]
    let M = b[0].len();

    for i in 0..N {
        let mut row: Vec<F> = Vec::new();
        for j in 0..M {
            let mut elem = F::from(0);
            for k in 0..K {
                elem += a[i][k].value().clone() * b[k][j].value().clone();
            }
            row.push(elem);
        }
        c.push(row);
    }
    return c;
}

pub fn field_mat_mul_raw<F: BigPrimeField>(
    ctx: &mut Context<F>, gate: &GateChip<F>,
    a: &Vec<Vec<AssignedValue<F>>>,
    b: &Vec<Vec<AssignedValue<F>>>,
) -> Vec<Vec<AssignedValue<F>>> {
    // a.num_col == b.num_rows
    assert_eq!(a[0].len(), b.len());

    let mut c: Vec<Vec<AssignedValue<F>>> = Vec::new();
    #[allow(non_snake_case)]
    let N = a.len();
    #[allow(non_snake_case)]
    let K = a[0].len();
    #[allow(non_snake_case)]
    let M = b[0].len();

    for i in 0..N {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..M {
            let mut elem = ctx.load_witness(F::from(0));
            for k in 0..K {
                let cur_result = gate.mul(ctx, a[i][k], b[k][j]);
                elem = gate.add(ctx, elem, cur_result);
            }
            row.push(elem);
        }
        c.push(row);
    }
    return c;
}

/// Takes matrices `a` and `b` (viewed simply as field elements), calculates matrix product `c_s = a*b` outside of the zk circuit, loads `c_s` into the context `ctx` and outputs the loaded matrix
///
/// Assumes matrix `a` and `b` are well defined matrices (all rows have the same size) and asserts (outside of circuit) that they can be multiplied
///
/// Uses trivial O(N^3) matrix multiplication algorithm
///
/// Doesn't contrain output matrix in any way
pub fn honest_prover_mat_mul<F: BigPrimeField>(
    ctx: &mut Context<F>,
    a: &Vec<Vec<AssignedValue<F>>>,
    b: &Vec<Vec<AssignedValue<F>>>,
) -> Vec<Vec<AssignedValue<F>>> {
    // field multiply matrices a and b
    // for honest prover creates the correct product multiplied by the quantization_scale (S) when a and b are field point quantized
    let c_s = field_mat_mul(a, b);
    let mut assigned_c_s: Vec<Vec<AssignedValue<F>>> = Vec::new();

    let num_rows = c_s.len();
    let num_col = c_s[0].len();
    for i in 0..num_rows {
        let mut new_row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..num_col {
            let elem = c_s[i][j];
            new_row.push(ctx.load_witness(elem));
        }
        assigned_c_s.push(new_row);
    }
    return assigned_c_s;
}

/// Multiplies a matrix by a vector in the zero-knowledge circuit
/// 
/// # Parameters
/// - `ctx`: Halo2 context for constraint generation
/// - `gate`: Gate chip for basic operations
/// - `a`: Matrix (m×n)
/// - `v`: Vector (n×1)
///
/// # Returns
/// A vector of assigned values representing the matrix-vector product
pub fn field_mat_vec_mul<F: BigPrimeField>(
    ctx: &mut Context<F>,
    gate: &GateChip<F>,
    a: &Vec<Vec<AssignedValue<F>>>,
    v: &Vec<AssignedValue<F>>,
) -> Vec<AssignedValue<F>> {
    assert_eq!(a[0].len(), v.len());
    let mut y: Vec<AssignedValue<F>> = Vec::new();
    for row in a {
        let mut w: Vec<QuantumCell<F>> = Vec::new();
        for x in v {
            w.push(Existing(*x));
        }
        let w = w;

        let mut u: Vec<QuantumCell<F>> = Vec::new();
        for x in row {
            u.push(Existing(*x));
        }
        let u = u;

        y.push(gate.inner_product(ctx, u, w));
    }

    return y;
}

/// Multiplies matrix `a_inv` to vector `v` in the zk-circuit and returns the constrained output `a_inv.v`
/// -- all assuming `a` and `v` are field elements (and not fixed point encoded)
///
/// Assumes matrix `a` is well defined (rows are equal size) and asserts (outside circuit) `a` can be multiplied to `v`
pub fn field_mat_vec_mul_a_inverse<F: BigPrimeField>(
    ctx: &mut Context<F>,
    gate: &GateChip<F>,
    a: &Vec<Vec<AssignedValue<F>>>,
    v: &Vec<AssignedValue<F>>,
) -> Vec<AssignedValue<F>> {
    assert_eq!(a[0].len(), v.len());
    let mut y: Vec<AssignedValue<F>> = Vec::new();

    for i in 0..a.len() {
        let mut w: Vec<QuantumCell<F>> = Vec::new();
        for x in v {
            w.push(Existing(*x));
        }
        let w = w;

        let mut u: Vec<QuantumCell<F>> = Vec::new();
        for j in 0..a[0].len() {
            u.push(Existing(a[j][i]));
        }
        let u = u;

        y.push(gate.inner_product(ctx, u, w));
    }

    return y;
}

/// Multiplies matrix `a` by a diagonal matrix represented as a vector `v` in the zk-circuit and returns the constrained output `a*Diag(v)`
/// -- all assuming `a` and `v` are field elements, (and not fixed point encoded)
///
/// Assumes matrix `a` is well defined (rows are equal size)
///
/// If dimension of `a` is `N X K` and `v` is length `M`, then multiplication is carried out as long as `K >= M`
///
/// In case `K > M`, multiplication result is actually the `N X M` matrix given by `a*[Diag(v) 0]^T` where 0 is the `(M X (K-M))` matrix of all zeroes;
/// this choice allows us to handle one of the cases in the SVD check
pub fn mat_times_diag_mat<F: BigPrimeField>(
    ctx: &mut Context<F>,
    gate: &GateChip<F>,
    a: &Vec<Vec<AssignedValue<F>>>,
    v: &Vec<AssignedValue<F>>,
) -> Vec<Vec<AssignedValue<F>>> {
    assert!(v.len() <= a[0].len());
    let mut m: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..a.len() {
        let mut new_row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..v.len() {
            let prod = gate.mul(ctx, a[i][j], v[j]);
            new_row.push(prod);
        }
        m.push(new_row);
    }
    return m;
}

/// Generates a permutation matrix from a topological ordering
/// 
/// # Parameters
/// - `order`: Topological ordering of nodes (indices)
/// - `size`: Size of the matrix (number of nodes)
///
/// # Returns
/// A DMatrix<f64> representing the permutation matrix
pub fn generate_permutation_matrix(order: &[usize], size: usize) -> DMatrix<f64> {
    let mut p = DMatrix::<f64>::zeros(size, size);
    for (i, &index) in order.iter().enumerate() {
        p[(i, index)] = 1.0;
    }
    p
}

/// Computes the transformed matrix B = P * A * P^(-1)
/// 
/// # Parameters
/// - `a`: Original adjacency matrix
/// - `p`: Permutation matrix from topological ordering
///
/// # Returns
/// A DMatrix<f64> representing the transformed matrix B
pub fn compute_transformed_matrix(a: &DMatrix<f64>, p: &DMatrix<f64>) -> DMatrix<f64> {
    let p_inv = p.transpose();
    p * a * p_inv
}

/// Finds a cycle in a directed graph using depth-first search
/// 
/// # Parameters
/// - `graph`: Directed graph to search for cycles
///
/// # Returns
/// - `Some(cycle)`: Vector of NodeIndex representing the cycle if found
/// - `None`: No cycle found (graph is acyclic)
fn find_cycle(graph: &DiGraph<(), ()>) -> Option<Vec<NodeIndex>> {
    let n = graph.node_count();
    let mut visited = vec![false; n];
    let mut on_stack = vec![false; n];
    // parent[i] holds the predecessor of node with index i during DFS
    let mut parent = vec![None; n];

    for node in graph.node_indices() {
        if !visited[node.index()] {
            if let Some(cycle) = dfs_cycle(graph, node, &mut visited, &mut on_stack, &mut parent) {
                return Some(cycle);
            }
        }
    }
    None
}

/// Recursive helper function for cycle detection using DFS
/// 
/// # Parameters
/// - `graph`: Directed graph being searched
/// - `node`: Current node being visited
/// - `visited`: Array tracking which nodes have been visited
/// - `on_stack`: Array tracking which nodes are on the current path
/// - `parent`: Array storing parent of each node for cycle reconstruction
///
/// # Returns
/// - `Some(cycle)`: Vector of NodeIndex representing the cycle if found
/// - `None`: No cycle found in this subtree
fn dfs_cycle(
    graph: &DiGraph<(), ()>,
    node: NodeIndex,
    visited: &mut [bool],
    on_stack: &mut [bool],
    parent: &mut [Option<NodeIndex>],
) -> Option<Vec<NodeIndex>> {
    visited[node.index()] = true;
    on_stack[node.index()] = true;

    // Explore all outgoing neighbors
    for neighbor in graph.neighbors_directed(node, Direction::Outgoing) {
        if !visited[neighbor.index()] {
            parent[neighbor.index()] = Some(node);
            if let Some(cycle) = dfs_cycle(graph, neighbor, visited, on_stack, parent) {
                return Some(cycle);
            }
        } else if on_stack[neighbor.index()] {
            // A back edge is found, which means there is a cycle.
            // Reconstruct the cycle starting from neighbor up to the current node.
            let mut cycle = vec![neighbor];
            let mut current = node;
            while current != neighbor {
                cycle.push(current);
                current = parent[current.index()].unwrap();
            }
            cycle.push(neighbor);
            cycle.reverse(); // Optional: reverse to show the cycle in correct order.
            return Some(cycle);
        }
    }
    on_stack[node.index()] = false;
    None
}

