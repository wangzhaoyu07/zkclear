// #![allow(warnings)]
#![allow(dead_code)]
use core::num;
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
use snark_verifier_sdk::halo2::OptimizedPoseidonSpec;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo::toposort;
use petgraph::visit::Topo;
use ndarray::Array2;
use nalgebra::DMatrix;
use std::marker::PhantomData;


#[derive(Clone)]
pub struct MatrixChip<F: BigPrimeField, const PRECISION_BITS: u32> {
    tol: BigUint,
    init_rand: AssignedValue<F>,
    _marker: PhantomData<F>,
}
impl<F: BigPrimeField, const PRECISION_BITS: u32> MatrixChip <F, PRECISION_BITS> {
    pub fn new(tol: BigUint, init_rand: AssignedValue<F>) -> Self {
        Self {
            tol,
            init_rand,
            _marker: PhantomData,
        }
    }

    pub fn quantize(
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        matrix: &Vec<Vec<f64>>,
    ) -> Vec<Vec<AssignedValue<F>>> {
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
        return zkmatrix;
    }

    /// 
    /// Dequantizes the matrix and returns it;
    ///
    /// Action is not constrained in anyway
    pub fn dequantize(qmatrix: &Vec<Vec<F>>, fpchip: &FixedPointChip<F, PRECISION_BITS>) -> Vec<Vec<f64>> {
        let mut dq_matrix: Vec<Vec<f64>> = Vec::new();
        let num_rows = qmatrix.len();
        let num_col = qmatrix[0].len();
        for i in 0..num_rows {
            dq_matrix.push(Vec::<f64>::new());
            for j in 0..num_col {
                dq_matrix[i].push(fpchip.dequantization(qmatrix[i][j]));
            }
        }
        return dq_matrix;
    }

    /// Takes matrices `a` and `b` (viewed simply as field elements), calculates matrix product `c_s = a*b` outside of the zk circuit, loads `c_s` into the context `ctx` and outputs the loaded matrix
    ///
    /// Assumes matrix `a` and `b` are well defined matrices (all rows have the same size) and asserts (outside of circuit) that they can be multiplied
    ///
    /// Uses trivial O(N^3) matrix multiplication algorithm
    ///
    /// Doesn't contrain output matrix in any way
    pub fn honest_prover_mat_mul(
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

    pub fn honest_prover_mat_mul_no_scale(
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

    pub fn matrix_mul( //_backup
        &self,
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        a: &Vec<Vec<AssignedValue<F>>>,
        b: &Vec<Vec<AssignedValue<F>>>,
    )->Vec<Vec<AssignedValue<F>>> {
        let c_s = Self::honest_prover_mat_mul(ctx, a, b);
        self.verify_mul(ctx, fpchip, a, b, &c_s);
        let c = Self::rescale_matrix(ctx, fpchip, &c_s);

        // for i in 0..cs_times_v.len() {
        //     gate.is_equal(ctx, cs_times_v[i], ab_times_v[i]);
        // }
        return c.clone();
    }

    pub fn matrix_mul_raw( //_backup
        &self,
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
                    // let prod = fpchip.gate().mul(ctx, a[i][k], b[k][j]);
                    let prod = fpchip.qmul(ctx, a[i][k], b[k][j]);
                    elem = fpchip.qadd(ctx, elem, prod);
                }
                c_row.push(elem);
            }
            c.push(c_row);
        }

        // for i in 0..a.len() {
        //     let mut c_row: Vec<AssignedValue<F>> = vec![];
        //     for j in 0..b[0].len() {
        //         let mut elem = ctx.load_witness(F::from(0));
        //         for k in 0..a[0].len() {
        //             let dq_a = fpchip.dequantization(*a[i][k].value());
        //             let dq_b = fpchip.dequantization(*b[k][j].value());
        //             let dq_c = dq_a * dq_b;
        //             let q_c = ctx.load_witness(fpchip.quantization(dq_c));
        //             elem = fpchip.qadd(ctx, elem, q_c);
        //         }
        //         c_row.push(elem);
        //     }
        //     c_s.push(c_row);
        // }

        // // let c_s = Self::honest_prover_mat_mul(ctx, a, b);
        // for i in 0..c.len() {
        //     for j in 0..c[0].len() {
        //         let c_s_q = fpchip.gate().mul(ctx, c_s[i][j], q_one);
        //         // ctx.constrain_equal(&c[i][j], &c_s_q);
        //     }
        // }


        // for i in 0..cs_times_v.len() {
        //     gate.is_equal(ctx, cs_times_v[i], ab_times_v[i]);
        // }
        // return c_s.clone();
        return c.clone();
    }

    pub fn matrix_mul_no_scale( //_backup
        &self,
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        a: &Vec<Vec<AssignedValue<F>>>,
        b: &Vec<Vec<AssignedValue<F>>>,
    )->Vec<Vec<AssignedValue<F>>> {
        let c_s = Self::honest_prover_mat_mul(ctx, a, b);
        self.verify_mul(ctx, fpchip, a, b, &c_s);
        // let c = Self::rescale_matrix(ctx, fpchip, &c_s);

        // for i in 0..cs_times_v.len() {
        //     gate.is_equal(ctx, cs_times_v[i], ab_times_v[i]);
        // }
        return c_s.clone();
    }

    // pub fn matrix_mul(
    //     &self,
    //     ctx: &mut Context<F>,
    //     fpchip: &FixedPointChip<F, PRECISION_BITS>,
    //     a: &Vec<Vec<AssignedValue<F>>>,
    //     b: &Vec<Vec<AssignedValue<F>>>,
    // )->Vec<Vec<AssignedValue<F>>> {
    //     // a.num_col == b.num_rows
    //     assert_eq!(a[0].len(), b.len());
    
    //     let mut c_s: Vec<Vec<AssignedValue<F>>> = Vec::new();
    //     #[allow(non_snake_case)]
    //     let N = a.len();
    //     #[allow(non_snake_case)]
    //     let K = a[0].len();
    //     #[allow(non_snake_case)]
    //     let M = b[0].len();
    
    //     // use fpchip.inner_product to multiply matrices a and b
    //     for i in 0..N {
    //         let mut row: Vec<AssignedValue<F>> = Vec::new();
    //         for j in 0..M {
    //             let mut elem = ctx.load_witness(F::from(0));
    //             for k in 0..K {
    //                 let prod = fpchip.gate().mul(ctx, a[i][k], b[k][j]);
    //                 elem = fpchip.qadd(ctx, elem, prod);
    //             }
    //             row.push(elem);
    //         }
    //         c_s.push(row);
    //     }

    //     // rescale c
    //     let c = Self::rescale_matrix(ctx, fpchip, &c_s);

    //     return c;
    // }

    pub fn matrix_vec_mul(
        &self,
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        a: &Vec<Vec<AssignedValue<F>>>,
        v: &Vec<AssignedValue<F>>,
    )->Vec<AssignedValue<F>> {
        let c_s = field_mat_vec_mul(ctx, fpchip.gate(), a, v);
        let c = Self::rescale_vec(ctx, fpchip, &c_s);
        return c.clone();
    }

    pub fn honest_prover_mat_inv(
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        a: &Vec<Vec<AssignedValue<F>>>,
    ) -> Vec<Vec<AssignedValue<F>>> {        
        let a_raw = Self::zkmatrix_to_dmatrix(fpchip, &a);
        let a_inv_raw = a_raw.clone().try_inverse().unwrap();
        // println!("a_inv_raw: {:?}", a_inv_raw);
        // check whether a_raw * a_inv_raw = I
        // let id_result = a_raw.clone() * a_inv_raw.clone();
        // println!("id_result: {:?}", id_result);

        // let id_result = a_raw.clone() * a_inv_raw.clone();
        // for i in 0..a_raw.nrows() {
        //     for j in 0..a_raw.ncols() {
        //         println!("i: {:?}, j: {:?}, a: {:?}, a_inv: {:?}, id: {:?}", i, j, a_raw[(i, j)], a_inv_raw[(i, j)], id_result[(i, j)]);
        //     }
        // }
        // println!("max a_inv: {:?}", a_inv_raw.max());
        // println!("Identity matrix: {:?}", id_result);
        let a_inv = Self::dmatrix_to_zkmatrix(ctx, fpchip, &a_inv_raw);
        return a_inv;
    }

    pub fn matrix_inv(
        &self,
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        a: &Vec<Vec<AssignedValue<F>>>,
    )->Vec<Vec<AssignedValue<F>>> {
        let a_inv = Self::honest_prover_mat_inv(ctx, fpchip, a);
        self.verify_matrix_inverse(ctx, fpchip, a, &a_inv);
        // self.verify_matrix_inverse_raw(ctx, fpchip, a, &a_inv);
        return a_inv.clone();
        
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
    pub fn verify_mul(
        &self,
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        a: &Vec<Vec<AssignedValue<F>>>,
        b: &Vec<Vec<AssignedValue<F>>>,
        c_s: &Vec<Vec<AssignedValue<F>>>,
    ) {
        assert_eq!(a[0].len(), b.len());
        assert_eq!(c_s.len(), a.len());
        assert_eq!(c_s[0].len(), b[0].len());
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
            let r_to_i = fpchip.gate().mul(ctx, *prev, self.init_rand);
            v.push(r_to_i);
        }
        let v = v;

        // println!("Random vector, v = [");
        // for x in &v {
        //     println!("{:?}", *x.value());
        // }
        // println!("]");

        let cs_times_v = field_mat_vec_mul(ctx, gate, c_s, &v);
        let b_times_v = field_mat_vec_mul(ctx, gate, b, &v);
        let ab_times_v = field_mat_vec_mul(ctx, gate, a, &b_times_v);

        check_vec_diff(ctx, &fpchip.gate, &cs_times_v, &ab_times_v, &self.tol);

        // for i in 0..cs_times_v.len() {
        //     gate.is_equal(ctx, cs_times_v[i], ab_times_v[i]);
        // }
    }

    /// Takes `c_s` and divides it by the quantization factor to scale it;
    ///
    /// Useful after matrix multiplication;
    ///
    /// Is costly- leads to ~94 (when lookup_bits =12) cells per element
    ///
    /// NOTE: Each of the entries of `c_s` need to be lesser than `2^(3*PRECISION_BITS)`
    /// for the result to be correctly encoded. For rescaling after matrix multiplication,
    /// best way to ensure this is to simply make sure that the matrices being multiplied are
    /// appropriately bounded.
    pub fn rescale_matrix(
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        c_s: &Vec<Vec<AssignedValue<F>>>,
    ) -> Vec<Vec<AssignedValue<F>>> {
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
        // let raw_matrix = Self::zkmatrix_to_dmatrix(fpchip, &c);
        return c;
    }

    pub fn rescale_vec(
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        c_s: &Vec<AssignedValue<F>>,
    ) -> Vec<AssignedValue<F>> {
        // #CONSTRAINTS = 94*N^2
        // now rescale c_s
        let mut c: Vec<AssignedValue<F>> = Vec::new();
        let num_rows = c_s.len();
        for i in 0..num_rows {
            // use fpchip to rescale c_s[i][j]
            // implemented in circuit, so we know c produced is correct
            let (elem, _) = fpchip.signed_div_scale(ctx, c_s[i]);
            c.push(elem);
        }
        // let raw_matrix = Self::zkmatrix_to_dmatrix(fpchip, &c);
        return c;
    }

    /// hash all the matrices in the given list
    pub fn hash_matrix(
        ctx: &mut Context<F>,
        gate: &GateChip<F>,
        matrix: &Vec<Vec<AssignedValue<F>>>,
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
        
        let num_rows = matrix.len();
        let num_col = matrix[0].len();
        let mut flat_mat_witnesses: Vec<AssignedValue<F>> = Vec::with_capacity(num_rows * num_col);
        for row in matrix.iter() {
            for &val in row.iter() {
                flat_mat_witnesses.push(val);
            }
        }
        // let flat_adj_mat_witnesses: [AssignedValue<F>; matrix.num_nodes * matrix.num_nodes] = flat_adj_mat_witnesses.try_into().unwrap();
        let mat_hash = poseidon.hash_fix_len_array(ctx, &gate, &flat_mat_witnesses);
            
        // dbg!(init_rand.value());
        return mat_hash;
    }

    /// Outputs the transpose matrix of a matrix `a`;
    ///
    /// Doesn't create any new constraints; just outputs the a copy of the transposed Self.matrix
    pub fn transpose_matrix(a: &Vec<Vec<AssignedValue<F>>>) -> Vec<Vec<AssignedValue<F>>> {
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

        // let raw_matrix = Self::zkmatrix_to_dmatrix(fpchip, &a_trans);
        return a_trans;
    }    
    
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
        &self,
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        a: &Vec<Vec<AssignedValue<F>>>,
        a_inv: &Vec<Vec<AssignedValue<F>>>,
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
            let r_to_i = fpchip.gate().mul(ctx, *prev, self.init_rand);
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

        let mut quantized_v: Vec<AssignedValue<F>> = Vec::new();
        for i in 0..a_inv_a_times_v.len() {
            let quantized_v_i = fpchip.gate().mul(ctx, v[i], quantized_one_square);
            quantized_v.push(quantized_v_i);
            // ctx.constrain_equal(&a_inv_a_times_v[i], &quantized_v_i);
        }

        // let mut max_diff = 0.0;
        // let mut max_idx = 0;
        for i in 0..a_inv_a_times_v.len() {
            let dq_1= fpchip.dequantization(*a_inv_a_times_v[i].value());
            let dq_2 = fpchip.dequantization(*quantized_v[i].value());
            // println!("dq_1: {:?}, dq_2: {:?}", dq_1, dq_2);
            // let diff = (dq_1 - dq_2).abs();
            // if diff > max_diff {
            //     max_diff = diff;
            //     max_idx = i;
            // }
        }
        // println!("max_diff: {:?}", max_diff);
        // println!("max_idx: {:?}", max_idx);
        // println!("tol: {:?}", self.tol);

        check_vec_diff(ctx, &fpchip.gate, &a_inv_a_times_v, &quantized_v, &self.tol);
    }

    pub fn verify_matrix_inverse_raw(
        &self,
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        a: &Vec<Vec<AssignedValue<F>>>,
        a_inv: &Vec<Vec<AssignedValue<F>>>,
    ) {
        let d = a[0].len();
        let quantized_one = ctx.load_witness(fpchip.quantization(1.0)); // TODO: add constraint
        let quantized_zero = ctx.load_witness(fpchip.quantization(0.0)); // TODO: add constraint

        let a_inv_a = self.matrix_mul_raw(ctx, fpchip, a_inv, a);

        let mut idn_matrix: Vec<Vec<AssignedValue<F>>> = Vec::new();
        for i in 0..d {
            let mut row: Vec<AssignedValue<F>> = Vec::new();
            for j in 0..d {
                if i == j {
                    row.push(quantized_one.clone());
                } else {
                    row.push(quantized_zero.clone());
                }
            }
            idn_matrix.push(row);
        }

        check_mat_diff(ctx, &fpchip.gate, &a_inv_a, &idn_matrix, &self.tol);
    }

    pub fn cal_scaled_error(eps_u: f64, max_dim: usize) -> BigUint {
        let err_u = err_calc(PRECISION_BITS, max_dim, eps_u);
        let err_u_scale = BigUint::from((err_u * (2u128.pow(2 * PRECISION_BITS) as f64)).round() as u128);
        return err_u_scale;
    }
}

/// Constrains that `x` satisfies `|x| < bnd`, i.e., `x` is in the set `{-(bnd-1), -(bnd-2), ..., 0, 1, ..., (bnd-1)}`
///
/// Does so by checking that `x+(bnd-1) < 2*bnd - 1` as a range check
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

/// Takes as two matrices `a` and `b` as input and checks that `|a[i][j] - b[i][j]| < tol` for each `i,j`
/// according to the absolute value check in `check_abs_less_than`
///
/// Assumes matrix `a` and `b` are well defined matrices (all rows have the same size) and asserts (outside of circuit) that they can be compared
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

/// Takes as two vectors `a` and `b` as input and checks that `|a[i] - b[i]| < tol` for each `i`
/// according to the absolute value check in `check_abs_less_than`
///
/// Assumes matrix `a` and `b` are well defined matrices (all rows have the same size) and asserts (outside of circuit) that they can be compared
pub fn check_vec_diff<F: BigPrimeField>(
    ctx: &mut Context<F>,
    range: &RangeChip<F>,
    a: &Vec<AssignedValue<F>>,
    b: &Vec<AssignedValue<F>>,
    tol: &BigUint,
) {
    assert_eq!(a.len(), b.len());

    for i in 0..a.len() {
        let diff = range.gate.sub(ctx, a[i], b[i]);
        check_abs_less_than(ctx, &range, diff, tol);
    }
}

/// Given a matrix of field elements `a` and a field element `scalar_id`, checks that `|a[i][j] - scalar_id*Id[i][j]| < tol` for each `i,j`, where Id is the identity matrix
/// according to the absolute value check in `check_abs_less_than`
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

/// Given a matrix `a` in the fixed point representation, checks that all of its entries are less in absolute value than some bound `bnd`
///
/// Assumes matrix `a` is well formed (all rows have the same size)
///
/// COMMENT- for our specific use case- to make sure that unitaries are in (-1,1), it might be better to use range_check based checks
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

/// Takes matrices `a` and `b` (viewed simply as field elements), calculates and outputs matrix product `c = a*b` outside of the zk circuit
///
/// Assumes matrix `a` and `b` are well defined matrices (all rows have the same size) and asserts (outside of circuit) that they can be multiplied
///
/// Uses trivial O(N^3) matrix multiplication algorithm
///
/// Doesn't contrain output in any way
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



/// Multiplies matrix `a` to vector `v` in the zk-circuit and returns the constrained output `a.v`
/// -- all assuming `a` and `v` are field elements (and not fixed point encoded)
///
/// Assumes matrix `a` is well defined (rows are equal size) and asserts (outside circuit) `a` can be multiplied to `v`
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

pub fn err_calc(p: u32, size: usize, eps_u: f64) -> f64 {
    let precision = 2.0_f64.powf(-1.0 * (p as f64 + 1.0));
    let err_u = eps_u + precision * (size as f64) * (2.0 * (1.0 + eps_u) + precision);
    return err_u;
}