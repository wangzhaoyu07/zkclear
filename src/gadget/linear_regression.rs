use halo2_base::{
    utils::BigPrimeField,
    QuantumCell, Context, AssignedValue
};
use log::warn;
use super::fixed_point::{FixedPointChip, FixedPointInstructions};
use std::convert::From;
use std::marker::PhantomData;
use std::cmp::min;
use ndarray::{Array2, Array3, Array, Dim, IxDyn};

#[derive(Clone, Debug)]
pub struct LinearRegressionChip<F: BigPrimeField, const PRECISION_BITS: u32> {
    weight: Vec<AssignedValue<F>>,
    bias: AssignedValue<F>,
    in_dim: usize,
    _marker: PhantomData<F>,
}

impl<F: BigPrimeField, const PRECISION_BITS: u32> LinearRegressionChip<F, PRECISION_BITS> {
    pub fn new(ctx: &mut Context<F>, fpchip: &FixedPointChip<F, PRECISION_BITS>, in_dim: usize) -> Self {
        let mut w: Vec<AssignedValue<F>> = vec![];
        for i in 0..in_dim {
            w.push(ctx.load_witness(fpchip.quantization(0.0)));
        }

        let mut b: AssignedValue<F> = ctx.load_witness(fpchip.quantization(0.0));
        
        Self {
            weight: w,
            bias: b,
            in_dim,
            _marker: PhantomData,
        }
    }

    pub fn inference_one_sample<QA>(
        &mut self,
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

    /// Mini-batch gradient descent for training linear regression
    pub fn train_one_batch<QA>(
        &mut self,
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        x: impl IntoIterator<Item = impl IntoIterator<Item = QA>>,
        y_truth: impl IntoIterator<Item = QA>,
        learning_rate: f64
    )
    where
        F: BigPrimeField, QA: Into<QuantumCell<F>> + From<AssignedValue<F>> + Copy
    {
        let y_truth: Vec<QA> = y_truth.into_iter().collect();
        let x: Vec<Vec<QA>> = x.into_iter().map(|xi| xi.into_iter().collect()).collect();
        let n_sample = x.len() as f64;
        // debug!("n_sample: {:?}", n_sample);

        let mut w: Vec<AssignedValue<F>> = self.weight.clone();
        let mut b = self.bias;
        let dim = x[0].len();
        assert!(dim == w.len());

        let learning_rate = ctx.load_witness(fpchip.quantization(learning_rate / n_sample));

        let y: Vec<QA> = x.iter().map(|xi| {
            let xw = fpchip.inner_product(ctx, (*xi).clone(), w.iter().map(|wi| QA::from(*wi)));
            let yi = fpchip.qadd(ctx, xw, b);

            QA::from(yi)
        }).collect();
        let mut diff_y = vec![];
        for (yi, ti) in y.iter().zip(y_truth.iter()) {
            diff_y.push(fpchip.qsub(ctx, *yi, *ti));
        }
        let mut loss = 0.;
        for i in 0..diff_y.len() {
            loss += fpchip.dequantization(*diff_y[i].value()).powi(2);
        }
        // loss = 0.5 * MSE(y, t)
        loss /= n_sample * 2.0;
        warn!("loss: {:?}", loss);

        for j in 0..w.len() {
            let mut partial_wj = vec![];
            for i in 0..diff_y.len() {
                partial_wj.push(fpchip.qmul(ctx, diff_y[i], x[i][j]));
            }
            let partial_wj_sum = fpchip.qsum(ctx, partial_wj);
            let diff_wj = fpchip.qmul(ctx, learning_rate, partial_wj_sum);
            w[j] = fpchip.qsub(ctx, w[j], diff_wj);
        }

        let partial_b = fpchip.qsum(ctx, diff_y);
        let diff_b = fpchip.qmul(ctx, learning_rate, partial_b);
        b = fpchip.qsub(ctx, b, diff_b);

        self.weight = w.clone();
        self.bias = b;

        // (w.iter().map(|wi| QA::from(*wi)).collect(), QA::from(b))
    }

    pub fn train<QA>(
        mut self,
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        x: &Vec<Vec<AssignedValue<F>>>,
        y_truth: &Vec<AssignedValue<F>>,
        learning_rate: f64,
        batch_size: usize,
        n_epoch: usize
    ) -> Self
    where
        F: BigPrimeField, QA: Into<QuantumCell<F>> + From<AssignedValue<F>> + Copy
    {
        let n_batch = (x.len() as f64 / batch_size as f64).ceil() as i64;
        for _ in 0..n_epoch {
            for idx_batch in 0..n_batch {
                let batch_x = (x[idx_batch as usize * batch_size..min(x.len(), (idx_batch as usize + 1) * batch_size)]).to_vec();
                let batch_y = (y_truth[idx_batch as usize * batch_size..min(y_truth.len(), (idx_batch as usize + 1) * batch_size)]).to_vec();
                self.train_one_batch(ctx, fpchip, batch_x, batch_y, learning_rate);
            }
        }

        Self {
            weight: self.weight,
            bias: self.bias,
            in_dim: self.in_dim,
            _marker: PhantomData,
        }
    }

    pub fn get_weight(&self) -> Vec<AssignedValue<F>> {
        self.weight.clone()
    }

    pub fn get_bias(&self) -> AssignedValue<F> {
        self.bias
    }
}