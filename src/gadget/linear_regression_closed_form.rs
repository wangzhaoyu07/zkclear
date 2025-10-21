use halo2_base::{
    utils::BigPrimeField,
    QuantumCell, Context, AssignedValue
};
use log::warn;
use snark_verifier_sdk::snark_verifier::halo2_ecc::fields::fp;
use super::fixed_point::{FixedPointChip, FixedPointInstructions};
use super::matrix::MatrixChip;
use std::convert::From;
use std::marker::PhantomData;
use std::cmp::min;
use ndarray::{Array2, Array3, Array, Dim, IxDyn};
use num_bigint::BigUint;
use nalgebra::{DMatrix, DVector};

#[derive(Clone, Debug)]
pub struct LinearRegressionChip<F: BigPrimeField, const PRECISION_BITS: u32> {
    weight: Vec<AssignedValue<F>>,
    bias: AssignedValue<F>,
    in_dim: usize,
    tol: BigUint,
    init_rand: AssignedValue<F>,
    _marker: PhantomData<F>,
}

impl<F: BigPrimeField, const PRECISION_BITS: u32> LinearRegressionChip<F, PRECISION_BITS> {
    pub fn new(ctx: &mut Context<F>, fpchip: &FixedPointChip<F, PRECISION_BITS>, in_dim: usize, max_dim: usize, eps_err:f64, init_rand: &AssignedValue<F>) -> Self {
        let mut w: Vec<AssignedValue<F>> = vec![];
        for _i in 0..in_dim {
            w.push(ctx.load_witness(fpchip.quantization(0.0)));
        }

        let b: AssignedValue<F> = ctx.load_witness(fpchip.quantization(0.0));
        let err_svd_scale = MatrixChip::<F, PRECISION_BITS>::cal_scaled_error(eps_err, max_dim);
        
        Self {
            weight: w,
            bias: b,
            in_dim,
            tol: err_svd_scale,
            init_rand: init_rand.clone(),
            _marker: PhantomData,
        }
    }

    pub fn inference_one_sample<QA>(
        // &mut self,
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        w: impl IntoIterator<Item = QA>,
        x: impl IntoIterator<Item = QA>,
        b: QA
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField, QA: Into<QuantumCell<F>> + Copy
    {
        
        let wx = fpchip.inner_product(ctx, w, x);
        let y = fpchip.qadd(ctx, wx, b);

        y
    }

    pub fn inference(
        &self,
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        // w: &Vec<AssignedValue<F>>,
        x: Vec<Vec<AssignedValue<F>>>,
        // b: QA
    ) -> Vec<AssignedValue<F>>
    where 
        F: BigPrimeField
    {
        let b = self.bias;
        let mut y = Vec::new();
        for cur_x in x.into_iter() {
            let w = self.weight.clone();
            let wx = fpchip.inner_product(ctx, w, cur_x);
            let y_cur = fpchip.qadd(ctx, wx, b);
            y.push(y_cur);
        }

        y
    }

    pub fn fit(
        &self,
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        x: &Vec<Vec<AssignedValue<F>>>,
        y_truth: &Vec<AssignedValue<F>>,
    ) -> Self
    where
        F: BigPrimeField,
    {
        // add 1 to the last column of x
        let mut x_bias: Vec<Vec<AssignedValue<F>>> = x.clone();
        for xi in x_bias.iter_mut() {
            xi.push(ctx.load_witness(fpchip.quantization(1.0)));
        }
        // println!("x_bias: {:?}", x_bias[0].len());
        // println!("x: {:?}", x[0].len());
        let x = &x_bias;
        println!("x: {:?}", x.len());

        let in_dim = x[0].len();
        let matrix_chip: MatrixChip<F, PRECISION_BITS> = MatrixChip::new(self.tol.clone(), self.init_rand.clone());
        let x_t: Vec<Vec<AssignedValue<F>>> = MatrixChip::<F, PRECISION_BITS>::transpose_matrix(x);
        let mut x_t_x = matrix_chip.matrix_mul(ctx, fpchip, &x_t, x);

        let lambda = ctx.load_witness(fpchip.quantization(0.0001));
        for i in 0..in_dim {
            x_t_x[i][i] = fpchip.qadd(ctx, x_t_x[i][i], lambda);
        }

        let x_t_x_inv = matrix_chip.matrix_inv(ctx, fpchip, &x_t_x);
        // let x_t_x_inv = matrix_chip.matrix_inv_raw(ctx, fpchip, &x_t_x);

        let x_t_y = matrix_chip.matrix_vec_mul(ctx, fpchip, &x_t, y_truth);
        let w_b = matrix_chip.matrix_vec_mul(ctx, fpchip, &x_t_x_inv, &x_t_y);

        let w = w_b[..in_dim-1].to_vec();
        let b = w_b[in_dim-1];

        return Self {
            weight: w,
            bias: b,
            in_dim: self.in_dim,
            tol: self.tol.clone(),
            init_rand: self.init_rand.clone(),
            _marker: PhantomData,
        };
    }

    pub fn get_weight(&self) -> Vec<AssignedValue<F>> {
        self.weight.clone()
    }

    pub fn get_bias(&self) -> AssignedValue<F> {
        self.bias
    }
}


fn vec_to_dmatrix(vec: Vec<Vec<f64>>) -> DMatrix<f64> {
    let n_rows = vec.len();
    let n_cols = if n_rows > 0 { vec[0].len() } else { 0 };

    // Flatten the Vec<Vec<f64>> into a single Vec<f64>
    let flat_vec: Vec<f64> = vec.into_iter().flat_map(|v| v).collect();

    // Create the DMatrix from the flattened vector
    DMatrix::from_row_slice(n_rows, n_cols, &flat_vec)
}

#[derive(Clone, Debug)]
pub struct LinearRegressionNative {
    weights: DVector<f64>,
    bias: f64,
}

impl LinearRegressionNative {
    pub fn new(n_features: usize) -> Self {
        LinearRegressionNative {
            weights: DVector::zeros(n_features),
            bias: 0.0,
        }
    }

    pub fn fit_native(&mut self, X: &Vec<Vec<f64>>, y: &Vec<f64>)-> LinearRegressionNative{
        let X = vec_to_dmatrix(X.clone());
        let y = DVector::from(y.clone());

        let n_samples = X.nrows();
        let n_features = X.ncols();
        
        let mut X_bias = X.clone().insert_columns(n_features, 1, 1.0);
        // println!("the value of first row of X_bias: {:?}", X_bias[(1, n_features)]);
        // 计算 (X^T * X)^{-1} * X^T * y
        let mut XTX = X_bias.transpose() * &X_bias;
        let lambda = 0.0001;
        for i in 0..n_features {
            XTX[(i, i)] += lambda;
        }
        let XTX_inv = XTX.try_inverse().expect("Matrix is not invertible");
        let XTy = X_bias.transpose() * y;

        let coefficients = XTX_inv * XTy;

        // 分离权重和偏置
        self.bias = coefficients[n_features];
        self.weights = DVector::from(coefficients.rows(0, n_features).clone_owned());
        // println!("weights[-1]: {:?}", self.weights[self.weights.len()-1]);
        // println!("bias: {:?}", self.bias);
        self.clone()

    }

    pub fn inference_one_sampe(&self, features: &Vec<f64>) -> f64 {
        let features_vec: DVector<f64> = DVector::from(features.clone());
        self.bias + self.weights.dot(&features_vec)
    }

    pub fn inference(&self, X: &Vec<Vec<f64>>) -> Vec<f64> {
        X.iter().map(|x| self.inference_one_sampe(x)).collect()
    }

    pub fn get_weight(&self) -> Vec<f64> {
        self.weights.iter().map(|&x| x).collect()
    }
    pub fn get_bias(&self) -> f64 {
        self.bias
    }
}