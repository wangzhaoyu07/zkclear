use halo2_base::{
    utils::BigPrimeField,
    QuantumCell, Context, AssignedValue
};
use log::warn;
use petgraph::data;
use super::fixed_point::{FixedPointChip, FixedPointInstructions};
use super::matrix::MatrixChip;
use super::linear_regression_closed_form::LinearRegressionChip;
use super::linear_regression_closed_form::LinearRegressionNative;
use std::convert::From;
use std::marker::PhantomData;
use std::cmp::{self, min};
use ndarray::{Array2, Array3, Array, Dim, IxDyn};
use num_bigint::BigUint;

#[derive(Clone, Debug)]
pub struct DMLChip<F: BigPrimeField, const PRECISION_BITS: u32> {
    model_y: LinearRegressionChip<F, PRECISION_BITS>,
    model_t: LinearRegressionChip<F, PRECISION_BITS>,
    model_final: LinearRegressionChip<F, PRECISION_BITS>,
    _marker: PhantomData<F>,
}

impl<F: BigPrimeField, const PRECISION_BITS: u32> DMLChip<F, PRECISION_BITS> {
    pub fn new(ctx: &mut Context<F>, fpchip: &FixedPointChip<F, PRECISION_BITS>, in_dim: usize, max_dim_x: usize, eps_err:f64, init_rand: &AssignedValue<F>) -> Self {
        let model_y = LinearRegressionChip::new(ctx, fpchip, in_dim, max_dim_x, eps_err, init_rand);
        let model_t = LinearRegressionChip::new(ctx, fpchip, in_dim, max_dim_x, eps_err, init_rand);
        let max_dim_final = cmp::max(in_dim+1, max_dim_x);
        let model_final = LinearRegressionChip::new(ctx, fpchip, in_dim+1, max_dim_final, eps_err, init_rand);
        
        Self {
            model_y: model_y,
            model_t: model_t,
            model_final: model_final,
            _marker: PhantomData,
        }
    }

    pub fn fit(
        &self,
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        data_x: Vec<Vec<AssignedValue<F>>>,
        data_t: Vec<AssignedValue<F>>,
        data_y: Vec<AssignedValue<F>>,
    ) -> Self
    where
        F: BigPrimeField
    {
        let model_y = self.model_y.fit(ctx, fpchip, &data_x, &data_y);
        let model_t = self.model_t.fit(ctx, fpchip, &data_x, &data_t);
        // let model_final = self.model_y.fit(ctx, fpchip, &data_x, &data_y); // dummy

        let data_y_pred: Vec<AssignedValue<F>> = self.model_y.inference(ctx, fpchip, data_x.clone());
        let data_y_residual: Vec<AssignedValue<F>> = data_y.iter().zip(&data_y_pred).map(|(yi, y_pred_i)| fpchip.qsub(ctx, *yi, *y_pred_i)).collect();

        let data_t_pred: Vec<AssignedValue<F>> = self.model_t.inference(ctx, fpchip, data_x.clone());
        let data_t_residual: Vec<AssignedValue<F>> = data_t.iter().zip(&data_t_pred).map(|(ti, t_pred_i)| fpchip.qsub(ctx, *ti, *t_pred_i)).collect();
        // add data_t_residual to data_x as new feature
        let data_x_new: Vec<Vec<AssignedValue<F>>> = data_x.iter().zip(&data_t_residual).map(|(xi, ti)| xi.iter().chain(std::iter::once(ti)).cloned().collect()).collect();
        let model_final = self.model_final.fit(ctx, fpchip, &data_x_new, &data_y_residual);

        return Self {
            model_y: model_y,
            model_t: model_t,
            model_final: model_final,
            _marker: PhantomData,
        };
    }

    pub fn ate_estimate(
        &self,
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        data_x: Vec<Vec<AssignedValue<F>>>,
        t0: AssignedValue<F>,
        t1: AssignedValue<F>,
    ) -> AssignedValue<F>
    where
        F: BigPrimeField
    {
        // let data_y_pred: Vec<AssignedValue<F>> = self.model_y.inference(ctx, fpchip, data_x.clone());
        let data_t_pred: Vec<AssignedValue<F>> = self.model_t.inference(ctx, fpchip, data_x.clone());

        // data_t0_residual = t0 - data_t_pred
        let data_t0_residual: Vec<AssignedValue<F>> = data_t_pred.iter().map(|t_pred_i| fpchip.qsub(ctx, t0, *t_pred_i)).collect();
        let data_x0_new: Vec<Vec<AssignedValue<F>>> = data_x.iter().zip(&data_t0_residual).map(|(xi, ti)| xi.iter().chain(std::iter::once(ti)).cloned().collect()).collect();
        let data_y0_pred_residual: Vec<AssignedValue<F>> = self.model_final.inference(ctx, fpchip, data_x0_new);

        // data_t1_residual = t1 - data_t_pred
        let data_t1_residual: Vec<AssignedValue<F>> = data_t_pred.iter().map(|t_pred_i| fpchip.qsub(ctx, t1, *t_pred_i)).collect();
        let data_x1_new: Vec<Vec<AssignedValue<F>>> = data_x.iter().zip(&data_t1_residual).map(|(xi, ti)| xi.iter().chain(std::iter::once(ti)).cloned().collect()).collect();
        let data_y1_pred_residual: Vec<AssignedValue<F>> = self.model_final.inference(ctx, fpchip, data_x1_new);

        // ate = the average of all rows in (data_y1_pred_residual - data_y0_pred_residual)
        let ate_raw: Vec<AssignedValue<F>> = data_y1_pred_residual.iter().zip(&data_y0_pred_residual).map(|(y1, y0)| fpchip.qsub(ctx, *y1, *y0)).collect();
        let ate_sum = fpchip.qsum(ctx, ate_raw);
        let n_sample = ctx.load_constant(fpchip.quantization(data_x.len() as f64));
        let ate = fpchip.qdiv(ctx, ate_sum, n_sample);
        return ate;

    }
}

#[derive(Clone, Debug)]
pub struct DMLChipNative {
    model_y: LinearRegressionNative,
    model_t: LinearRegressionNative,
    model_final: LinearRegressionNative,
}

impl DMLChipNative {
    pub fn new(in_dim: usize) -> Self {
        let model_y = LinearRegressionNative::new(in_dim);
        let model_t = LinearRegressionNative::new(in_dim);
        let model_final = LinearRegressionNative::new(in_dim+1);
        
        Self {
            model_y: model_y,
            model_t: model_t,
            model_final: model_final,
            // _marker: PhantomData,
        }
    }

    pub fn fit(
        &mut self,
        data_x: Vec<Vec<f64>>,
        data_t: Vec<f64>,
        data_y: Vec<f64>,
    ) -> Self
    {
        let model_y = self.model_y.fit_native(&data_x, &data_y);
        let model_t = self.model_t.fit_native( &data_x, &data_t);
        // let model_final = self.model_y.fit(ctx, fpchip, &data_x, &data_y); // dummy

        let data_y_pred: Vec<f64> = self.model_y.inference(&data_x.clone());
        let data_y_residual: Vec<f64> = data_y.iter().zip(&data_y_pred).map(|(yi, y_pred_i)| (yi-y_pred_i)).collect();

        let data_t_pred: Vec<f64> = self.model_t.inference(&data_x.clone());
        let data_t_residual: Vec<f64> = data_t.iter().zip(&data_t_pred).map(|(ti, t_pred_i)| (ti-t_pred_i)).collect();

        // add data_t_residual to data_x as new feature
        let data_x_new: Vec<Vec<f64>> = data_x.iter().zip(&data_t_residual).map(|(xi, ti)| xi.iter().chain(std::iter::once(ti)).cloned().collect()).collect();
        let model_final = self.model_final.fit_native(&data_x_new, &data_y_residual);
        return Self {
            model_y: model_y,
            model_t: model_t,
            model_final: model_final,
        };
    }

    pub fn ate_estimate(
        &self,
        data_x: Vec<Vec<f64>>,
        t0: f64,
        t1: f64,
    ) -> f64
    {
        // let data_y_pred: Vec<f64> = self.model_y.inference(ctx, fpchip, data_x.clone());
        let data_t_pred: Vec<f64> = self.model_t.inference(&data_x.clone());

        // data_t0_residual = t0 - data_t_pred
        let data_t0_residual: Vec<f64> = data_t_pred.iter().map(|t_pred_i| (t0-t_pred_i)).collect();
        let data_x0_new: Vec<Vec<f64>> = data_x.iter().zip(&data_t0_residual).map(|(xi, ti)| xi.iter().chain(std::iter::once(ti)).cloned().collect()).collect();
        let data_y0_pred_residual: Vec<f64> = self.model_final.inference(&data_x0_new);

        // data_t1_residual = t1 - data_t_pred
        let data_t1_residual: Vec<f64> = data_t_pred.iter().map(|t_pred_i| (t1-*t_pred_i)).collect();
        let data_x1_new: Vec<Vec<f64>> = data_x.iter().zip(&data_t1_residual).map(|(xi, ti)| xi.iter().chain(std::iter::once(ti)).cloned().collect()).collect();
        let data_y1_pred_residual: Vec<f64> = self.model_final.inference(&data_x1_new);

        // ate = the average of all rows in (data_y1_pred_residual - data_y0_pred_residual)
        let ate_raw: Vec<f64> = data_y1_pred_residual.iter().zip(&data_y0_pred_residual).map(|(y1, y0)| (y1-y0)).collect();
        let ate_sum = ate_raw.iter().sum::<f64>();
        let n_sample = data_x.len();
        let ate = ate_sum / n_sample as f64;
        return ate;

    }
}