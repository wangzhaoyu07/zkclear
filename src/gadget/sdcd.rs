use std::{iter, vec};
use std::marker::PhantomData;
// use ethers_core::abi::Hash;
// use log::{debug, warn};
use halo2_base::{
    utils::{BigPrimeField, fe_to_biguint},
    Context, AssignedValue, gates::{GateInstructions, RangeInstructions}, QuantumCell
};
use petgraph::adj;
use super::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::QuantumCell::Constant;
use std::collections::HashMap;
use ndarray::{Array2, Array3, Array, Dim, IxDyn};
// use itertools::{izip, Itertools};

const PRECISION_BITS: u32 = 63;
const MASK_CLS: u64 = 255;

// #[derive(Clone, Debug)]

#[derive(Debug)]
struct Stage1Kwargs {
    learning_rate: f64,
    batch_size: usize,
    n_epochs: usize,
    alpha: f64,
    beta: f64,
    gamma_increment: f64,
    n_epochs_check: usize,
    mask_threshold: f64,
}

impl Stage1Kwargs {
    fn new() -> Self {
        Self {
            learning_rate: 2e-3,
            batch_size: 256,
            n_epochs: 1,
            alpha: 1e-2,
            beta: 2e-4,
            gamma_increment: 0.0,
            n_epochs_check: 100,
            mask_threshold: 0.2,
        }
    }
    
}

#[derive(Debug)]
struct Stage2Kwargs {
    learning_rate: f64,
    batch_size: usize,
    n_epochs: usize,
    alpha: f64,
    beta: f64,
    gamma_increment: f64,
    gamma_schedule: String,
    freeze_gamma_at_dag: bool,
    freeze_gamma_threshold: f64,
    threshold: f64,
    n_epochs_check: usize,
    dag_penalty_flavor: String,
}

impl Stage2Kwargs {
    fn new() -> Self {
        Self {
            learning_rate: 1e-3,
            batch_size: 256,
            n_epochs: 2000,
            alpha: 5e-4,
            beta: 5e-3,
            gamma_increment: 0.005,
            gamma_schedule: "linear".to_string(),
            freeze_gamma_at_dag: true,
            freeze_gamma_threshold: 0.01,
            threshold: 0.1,
            n_epochs_check: 100,
            dag_penalty_flavor: "power_iteration".to_string(),
        }
    }
}


//TODO: 把模型打包成一个struct




// Mask?
struct DispatcherLayer<F: BigPrimeField, const PRECISION_BITS: u32> {
    weight: Array<AssignedValue<F>, Dim<[usize; 3]>>,
    bias: Array<AssignedValue<F>, Dim<[usize; 2]>>,
    in_dim: usize,
    out_dim: usize,
    hidden_dim: usize,
}

impl <F: BigPrimeField, const PRECISION_BITS: u32> DispatcherLayer<F, PRECISION_BITS> {
    pub fn new(ctx: &mut Context<F>, fpchip: &FixedPointChip<F, PRECISION_BITS>, in_dim: usize, out_dim: usize, hidden_dim: usize) -> Self{
        let mut w: Vec<Vec<Vec<AssignedValue<F>>>> = vec![];
        for i in 0..in_dim {
            let mut w_i = vec![];
            for j in 0..out_dim {
                let mut w_ij = vec![];
                for k in 0..hidden_dim {
                    w_ij.push(ctx.load_witness(fpchip.quantization(0.0))); //TODO: random initialization
                }
                w_i.push(w_ij);
            }
            w.push(w_i);
        }

        // initiall array_w as 0 array
        let mut array_w = Array::from_shape_vec((in_dim, out_dim, hidden_dim), w.into_iter().flatten().flatten().collect()).unwrap();

        let mut b: Vec<Vec<AssignedValue<F>>> = vec![];
        for i in 0..out_dim {
            let mut b_i = vec![];
            for j in 0..hidden_dim {
                b_i.push(ctx.load_witness(fpchip.quantization(0.0))); //TODO: random initialization
            }
            b.push(b_i);
        }
        let mut array_b = Array::from_shape_vec((out_dim, hidden_dim), b.into_iter().flatten().collect()).unwrap();

         
        DispatcherLayer{
            weight: array_w,
            bias: array_b,
            in_dim: in_dim,
            out_dim: out_dim,
            hidden_dim: hidden_dim,
        }
    
    }

    pub fn forward(&self, ctx: &mut Context<F>, fpchip: &FixedPointChip<F, PRECISION_BITS>, x: &Array<AssignedValue<F>, Dim<[usize; 2]>>,) -> Array<AssignedValue<F>, Dim<[usize; 3]>> {
        let mut result = vec![];
        for b in 0..x.shape()[0] {
            for o in 0..self.out_dim {
                for h in 0..self.hidden_dim {
                    let mut sum = ctx.load_witness(fpchip.quantization(0.0));
                    for i in 0..self.in_dim {
                        let mut mul_result: AssignedValue<F>;
                        if i == o {
                            mul_result = ctx.load_witness(fpchip.quantization(0.0));
                        }
                        else{
                            mul_result = fpchip.qmul(ctx, x[[b, i]], self.weight[[i, o, h]]);
                        }
                        
                        sum = fpchip.qadd(ctx, sum, mul_result);
                    }
                    sum = fpchip.qadd(ctx, sum, self.bias[[o, h]]);
                    result.push(sum);
                }
            }
        }

        let out_shape = [x.shape()[0], self.out_dim, self.hidden_dim];
        Array::from_shape_vec(out_shape, result).unwrap()
    }

    //TODO: check the correctness of the backward pass
    pub fn backward(
        &mut self,
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        x: &Array<AssignedValue<F>, Dim<[usize; 2]>>,       // 输入张量
        grad_output: &Array<AssignedValue<F>, Dim<[usize; 3]>>, // 输出梯度
        learning_rate: f64,                                // 学习率
    ) -> (Array<AssignedValue<F>, Dim<[usize; 3]>>, Array<AssignedValue<F>, Dim<[usize; 2]>>, Array<AssignedValue<F>, Dim<[usize; 2]>>) {
        let mut grad_weight = vec![];
        let mut grad_bias = vec![];
        let mut grad_input = vec![];

        let batch_size = x.shape()[0];
        let quantized_batch_size = ctx.load_witness(fpchip.quantization(batch_size as f64));

        // 计算权重梯度
        for i in 0..self.in_dim {
            for o in 0..self.out_dim {
                for h in 0..self.hidden_dim {
                    let mut grad_w_ioh = ctx.load_witness(fpchip.quantization(0.0));
                    for b in 0..batch_size {
                        let product = fpchip.qmul(ctx, x[[b, i]], grad_output[[b, o, h]]);
                        grad_w_ioh = fpchip.qadd(ctx, grad_w_ioh, product);
                    } 
                    let per_grad_w_ioh = fpchip.qdiv(ctx, grad_w_ioh, quantized_batch_size);
                    grad_weight.push(per_grad_w_ioh);
                }
            }
        }

        // 计算偏置梯度
        for o in 0..self.out_dim {
            for h in 0..self.hidden_dim {
                let mut grad_b_oh = ctx.load_witness(fpchip.quantization(0.0));
                for b in 0..batch_size {
                    grad_b_oh = fpchip.qadd(ctx, grad_b_oh, grad_output[[b, o, h]]);
                }
                let quantized_batch_size = ctx.load_witness(fpchip.quantization(batch_size as f64));
                let per_grad_b_oh = fpchip.qdiv(ctx, grad_b_oh, quantized_batch_size);
                grad_bias.push(per_grad_b_oh);
            }
        }

        // 计算输入梯度
        for b in 0..batch_size {
            for i in 0..self.in_dim {
                let mut grad_x_bi = ctx.load_witness(fpchip.quantization(0.0));
                for o in 0..self.out_dim {
                    for h in 0..self.hidden_dim {
                        let product = fpchip.qmul(ctx, grad_output[[b, o, h]], self.weight[[i, o, h]]);
                        grad_x_bi = fpchip.qadd(ctx, grad_x_bi, product);
                    }
                }
                grad_input.push(grad_x_bi);
            }
        }

        // // 更新权重和偏置
        // for idx in 0..self.weight.len() {
        //     let scaled_grad = fpchip.qmul_const(ctx, grad_weight[idx], -learning_rate);
        //     self.weight[idx] = fpchip.qadd(ctx, self.weight[idx], scaled_grad);
        // }

        // for idx in 0..self.bias.len() {
        //     let scaled_grad = fpchip.qmul_const(ctx, grad_bias[idx], -learning_rate);
        //     self.bias[idx] = fpchip.qadd(ctx, self.bias[idx], scaled_grad);
        // }

        // 转换梯度为数组
        let grad_weight_array = Array::from_shape_vec(self.weight.raw_dim(), grad_weight).unwrap();
        let grad_bias_array = Array::from_shape_vec(self.bias.raw_dim(), grad_bias).unwrap();
        let grad_input_array = Array::from_shape_vec(x.raw_dim(), grad_input).unwrap();

        (grad_weight_array, grad_bias_array, grad_input_array)
    }

    pub fn update(&mut self, ctx: &mut Context<F>, fpchip: &FixedPointChip<F, PRECISION_BITS>, grad_weight: &Array<AssignedValue<F>, Dim<[usize; 3]>>, grad_bias: &Array<AssignedValue<F>, Dim<[usize; 2]>>, learning_rate: f64) {
        let quantized_learning_rate = ctx.load_witness(fpchip.quantization(learning_rate));
        for i in 0..self.in_dim {
            for j in 0..self.out_dim {
                for k in 0..self.hidden_dim {
                    let scaled_grad = fpchip.qmul(ctx, grad_weight[[i,j,k]], quantized_learning_rate);
                    self.weight[[i,j,k]] = fpchip.qsub(ctx, self.weight[[i,j,k]], scaled_grad);
                }
            }
        }

        for i in 0..self.out_dim {
            for j in 0..self.hidden_dim {
                let scaled_grad = fpchip.qmul(ctx, grad_bias[[i,j]], quantized_learning_rate);
                self.bias[[i,j]] = fpchip.qsub(ctx, self.bias[[i,j]], scaled_grad);
            }
        }
    }
}

struct LinearParallel<F: BigPrimeField, const PRECISION_BITS: u32> {
    weight: Array<AssignedValue<F>, Dim<[usize; 3]>>,
    bias: Array<AssignedValue<F>, Dim<[usize; 2]>>,
    in_dim: usize,
    out_dim: usize,
    parallel_dim: usize,
}

impl <F: BigPrimeField, const PRECISION_BITS: u32> LinearParallel<F, PRECISION_BITS> {
    pub fn new(ctx: &mut Context<F>, fpchip: &FixedPointChip<F, PRECISION_BITS>, in_dim: usize, out_dim: usize, parallel_dim: usize) -> Self{
        let mut w: Vec<Vec<Vec<AssignedValue<F>>>> = vec![];
        for i in 0..parallel_dim {
            let mut w_i = vec![];
            for j in 0..in_dim {
                let mut w_ij = vec![];
                for k in 0..out_dim {
                    w_ij.push(ctx.load_witness(fpchip.quantization(0.0))); //TODO: random initialization
                }
                w_i.push(w_ij);
            }
            w.push(w_i);
        }
        let mut array_w = Array::from_shape_vec((parallel_dim, in_dim, out_dim), w.into_iter().flatten().flatten().collect()).unwrap();

        let mut b: Vec<Vec<AssignedValue<F>>> = vec![];
        for i in 0..parallel_dim {
            let mut b_i = vec![];
            for j in 0..out_dim {
                b_i.push(ctx.load_witness(fpchip.quantization(0.0))); //TODO: random initialization
            }
            b.push(b_i);
        }
        let mut array_b = Array::from_shape_vec((parallel_dim, out_dim), b.into_iter().flatten().collect()).unwrap();
    
        LinearParallel{
            weight: array_w,
            bias: array_b,
            in_dim: in_dim,
            out_dim: out_dim,
            parallel_dim: parallel_dim,
        }
    
    }

    pub fn forward(&self, ctx: &mut Context<F>, fpchip: &FixedPointChip<F, PRECISION_BITS>, x: &Array<AssignedValue<F>, Dim<[usize; 3]>>,) -> Array<AssignedValue<F>, Dim<[usize; 3]>> {
        let mut result = vec![];

        let batch_size = x.shape()[0];
        for n in 0..batch_size {
            for p in 0..self.parallel_dim {
                for o in 0..self.out_dim {
                    let mut sum = ctx.load_witness(fpchip.quantization(0.0));
                    for i in 0..self.in_dim {
                        let product = fpchip.qmul(ctx, x[[n,p,i]], self.weight[[p,i,o]]);
                        sum = fpchip.qadd(ctx, sum, product);
                    }
                    sum = fpchip.qadd(ctx, sum, self.bias[[p,o]]);
                    result.push(sum);
                }
            }
        }

        let out_shape = [x.shape()[0], self.parallel_dim, self.out_dim];
        Array::from_shape_vec(out_shape, result).unwrap()
    }


    //TODO: check the correctness of the backward pass
    pub fn backward(
        &self,
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        x: &Array<AssignedValue<F>, Dim<[usize; 3]>>,
        grad_output: &Array<AssignedValue<F>, Dim<[usize; 3]>>,
    ) -> (Array<AssignedValue<F>, Dim<[usize; 3]>>, Array<AssignedValue<F>, Dim<[usize; 2]>>, Array<AssignedValue<F>, Dim<[usize; 3]>>) {
        let mut grad_weight = vec![];
        let mut grad_bias = vec![];
        let mut grad_input = vec![];

        let batch_size = x.shape()[0];
        let quantized_batch_size = ctx.load_witness(fpchip.quantization(batch_size as f64));

        for p in 0..self.parallel_dim {
            for i in 0..self.in_dim {
                for o in 0..self.out_dim {
                    let mut grad_w_pio = ctx.load_witness(fpchip.quantization(0.0));
                    for n in 0..batch_size {
                        let product = fpchip.qmul(ctx, x[[n, p, i]], grad_output[[n, p, o]]);
                        grad_w_pio = fpchip.qadd(ctx, grad_w_pio, product);
                    }
                    let per_grad_w_pio = fpchip.qdiv(ctx, grad_w_pio, quantized_batch_size);
                    grad_weight.push(per_grad_w_pio);
                }
            }
        }


        for p in 0..self.parallel_dim {
            for o in 0..self.out_dim {
                let mut grad_b_po = ctx.load_witness(fpchip.quantization(0.0));
                for n in 0..batch_size {
                    grad_b_po = fpchip.qadd(ctx, grad_b_po, grad_output[[n, p, o]]);
                }
                let per_grad_b_po = fpchip.qdiv(ctx, grad_b_po, quantized_batch_size);
                grad_bias.push(per_grad_b_po);
            }
        }

        for n in 0..batch_size {
            for p in 0..self.parallel_dim {
                for i in 0..self.in_dim {
                    let mut grad_x_npi = ctx.load_witness(fpchip.quantization(0.0));
                    for o in 0..self.out_dim {
                        let product = fpchip.qmul(ctx, grad_output[[n, p, o]], self.weight[[p, i, o]]);
                        grad_x_npi = fpchip.qadd(ctx, grad_x_npi, product);
                    }
                    grad_input.push(grad_x_npi);
                }
            }
        }

        // // Update weights and biases
        // for i in 0..self.parallel_dim {
        //     for j in 0..self.in_dim {
        //         for k in 0..self.out_dim {
        //             let scaled_grad = fpchip.qmul(ctx, grad_weight[[i,j,k]], ctx.load_witness(fpchip.quantization(learning_rate)));
        //             self.weight[[i,j,k]] = fpchip.qsub(ctx, self.weight[[i,j,k]], scaled_grad);
        //         }
        // }

        // for i in 0..self.parallel_dim {
        //     for j in 0..self.out_dim {
        //         let scaled_grad = fpchip.qmul(ctx, grad_bias[[i,j]], ctx.load_witness(fpchip.quantization(learning_rate)));
        //         self.bias[[i,j]] = fpchip.qsub(ctx, self.bias[[i,j]], scaled_grad);
        //     }
        // }

        let grad_weight_array = Array::from_shape_vec(self.weight.raw_dim(), grad_weight).unwrap();
        let grad_bias_array = Array::from_shape_vec(self.bias.raw_dim(), grad_bias).unwrap();
        let grad_input_array = Array::from_shape_vec(x.raw_dim(), grad_input).unwrap();

        (grad_weight_array, grad_bias_array, grad_input_array)
    }

    pub fn update(&mut self, ctx: &mut Context<F>, fpchip: &FixedPointChip<F, PRECISION_BITS>, grad_weight: &Array<AssignedValue<F>, Dim<[usize; 3]>>, grad_bias: &Array<AssignedValue<F>, Dim<[usize; 2]>>, learning_rate: f64) {
        let quantized_learning_rate = ctx.load_witness(fpchip.quantization(learning_rate));
        for i in 0..self.parallel_dim {
            for j in 0..self.in_dim {
                for k in 0..self.out_dim {
                    let scaled_grad = fpchip.qmul(ctx, grad_weight[[i,j,k]], quantized_learning_rate);
                    self.weight[[i,j,k]] = fpchip.qsub(ctx, self.weight[[i,j,k]], scaled_grad);
                }
            }
        }

        for i in 0..self.parallel_dim {
            for j in 0..self.out_dim {
                let scaled_grad = fpchip.qmul(ctx, grad_bias[[i,j]], quantized_learning_rate);
                self.bias[[i,j]] = fpchip.qsub(ctx, self.bias[[i,j]], scaled_grad);
            }
        }
    }
}


#[derive(Debug)]
struct ModelKwargs {
    num_layers: usize,
    dim_hidden: usize,
    power_iteration_n_steps: usize,
}

impl ModelKwargs {
    fn new() -> Self {
        Self {
            num_layers: 1,
            dim_hidden: 10,
            power_iteration_n_steps: 15,
        }
    }
}

pub struct SDCDChip<F: BigPrimeField, const PRECISION_BITS: u32> {
    pub stage1_kwargs: Stage1Kwargs,
    pub stage2_kwargs: Stage2Kwargs,
    pub model_kwargs: ModelKwargs,
    _marker: PhantomData<F>,

}

impl<F: BigPrimeField, const PRECISION_BITS: u32> SDCDChip<F, PRECISION_BITS> {
    pub fn new() -> Self {
        let stage1_kwargs = Stage1Kwargs::new();
        let stage2_kwargs = Stage2Kwargs::new();
        let model_kwargs = ModelKwargs::new();
        Self {
            stage1_kwargs,
            stage2_kwargs,
            model_kwargs,
            _marker: PhantomData,
        }
    }

    // pub fn inference<QA>(
    //     &self,
    //     ctx: &mut Context<F>,
    //     w: impl IntoIterator<Item = QA>,
    //     x: impl IntoIterator<Item = QA>,
    //     b: QA
    // ) -> AssignedValue<F>
    // where 
    //     F: BigPrimeField, QA: Into<QuantumCell<F>> + Copy
    // {
    //     let wx = self.chip.inner_product(ctx, w, x);
    //     let logit = self.chip.qadd(ctx, wx, b);
    //     let neg_logit = self.chip.neg(ctx, logit);
    //     let exp_logit = self.chip.qexp(ctx, neg_logit);
    //     let one = Constant(self.chip.quantization(1.0));
    //     let exp_logit_p1 = self.chip.qadd(ctx, exp_logit, one);
    //     let y = self.chip.qdiv(ctx, one, exp_logit_p1);
    //     y
    // }

    // pub fn train_multi_batch<QA>(
    //     &self,
    //     ctx: &mut Context<F>,
    //     w: impl IntoIterator<Item = AssignedValue<F>>,
    //     b: AssignedValue<F>,
    //     x: impl IntoIterator<Item = impl IntoIterator<Item = impl IntoIterator<Item = QA>>>,
    //     y_truth: impl IntoIterator<Item = impl IntoIterator<Item = QA>>,
    //     learning_rate_batch: f64
    // ) -> (Vec<AssignedValue<F>>, AssignedValue<F>)
    // where
    //     F: BigPrimeField, QA: Into<QuantumCell<F>> + From<AssignedValue<F>> + Copy
    // {
    //     let x_multi_batch = x.into_iter();
    //     let y_truth_multi_batch = y_truth.into_iter();
    //     let mut w = w.into_iter().collect_vec();
    //     let mut b = b;
    //     for (cur_x, cur_y) in x_multi_batch.zip(y_truth_multi_batch) {
    //         (w, b) = self.train_one_batch(ctx, w, b, cur_x, cur_y, learning_rate_batch);
    //     }

    //     (w, b)
    // }

    /// Mini-batch gradient descent for training linear regression
    /// 所有epoch的运行在外面
    pub fn train_one_batch(
        &self,
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        // w: impl IntoIterator<Item = impl IntoIterator<Item = impl IntoIterator<Item = QA>>>,
        // b: AssignedValue<F>,
        x: Vec<Vec<AssignedValue<F>>>,
        learning_rate_batch: f64
    ) -> Vec<Vec<AssignedValue<F>>> // here should be model weight
    where
        F: BigPrimeField
    {
        let x_vec: Vec<Vec<AssignedValue<F>>> = x.into_iter().map(|xi| xi.into_iter().collect()).collect();
        let x = Array::from_shape_vec((x_vec.len(), x_vec[0].len()), x_vec.into_iter().flatten().collect()).unwrap();

        
        let n_sample = x.len() as f64;

        // Stage 1
        // define the model
        let mut graph_dims: usize = x.shape()[1];
        let mut dispatcher_layer = DispatcherLayer::new(ctx, fpchip, graph_dims, graph_dims, self.model_kwargs.dim_hidden);
        let mut output_layer = LinearParallel::new(ctx, fpchip, self.model_kwargs.dim_hidden, 1, graph_dims);
        let mut var_layer = LinearParallel::new(ctx, fpchip, self.model_kwargs.dim_hidden, 1, graph_dims);

        // TODO: forward pass and backward pass
        // Forward pass
        for epoch in 0..self.stage1_kwargs.n_epochs {

            // Backward pass
            let outp_disp: ndarray::ArrayBase<ndarray::OwnedRepr<AssignedValue<F>>, Dim<[usize; 3]>> = dispatcher_layer.forward(ctx, fpchip, &x);
            // let outp_disp_act = ActivationLayer::forward(ctx, fpchip, &outp_disp, "sigmoid");
            let x_mean = output_layer.forward(ctx, fpchip, &outp_disp); //outp_disp_act
            let outp_var = var_layer.forward(ctx, fpchip, &outp_disp); //outp_disp_act
            let x_var = ActivationLayer::forward(ctx, fpchip, &outp_var, "softplus");

            // Backward pass
            let grad_dispatch_l1 = LossFunc::l1_reg_dispatcher_backward(ctx, fpchip, &dispatcher_layer);
            let (grad_dispatch_l2, grad_output_l2, grad_var_l2) = LossFunc::l2_reg_all_weights_backward(ctx, fpchip, &dispatcher_layer, &output_layer, &var_layer);
            
            let (grad_x_mean, grad_x_var) = LossFunc::nll_backward(ctx, fpchip, &x_mean, &x_var, &x); //x_var
            let grad_dx_var_doutp_var = ActivationLayer::backward(ctx, fpchip, &outp_var, "softplus");
            let grad_outp_var = per_elem_mul(ctx, fpchip, &grad_dx_var_doutp_var, &grad_x_var);

            let (grad_weight_var_layer, grad_bias_var_layer, grad_outp_disp_act_var_layer) 
                = var_layer.backward(ctx, fpchip, &outp_disp, &grad_outp_var); //outp_disp_act, grad_outp_var
            let (grad_weight_output_layer, grad_bias_output_layer, grad_outp_disp_act_output_layer) 
                = output_layer.backward(ctx, fpchip, &outp_disp, &grad_x_mean); //outp_disp_act, grad_x_mean
            let grad_outp_disp_act = per_elem_add(ctx, fpchip, &grad_outp_disp_act_output_layer, &grad_outp_disp_act_var_layer);
            // let grad_doutp_disp_act_doutp_disp = ActivationLayer::backward(ctx, fpchip, &outp_disp_act, "sigmoid");
            // let grad_outp_disp = per_elem_mul(ctx, fpchip, &grad_doutp_disp_act_doutp_disp, &grad_outp_disp_act);
            let (grad_weight_dispatcher_layer, grad_bias_dispatcher_layer, grad_x) 
                = dispatcher_layer.backward(ctx, fpchip, &x, &grad_outp_disp_act, learning_rate_batch); //grad_outp_disp

            // Merge gradients
            //grad_weight_dispatch_l1+grad_weight_dispatch_l2+grad_weight_dispatcher_layer
            let grad_weight_dispatcher_layer_all = per_elem_add(ctx, fpchip, &grad_dispatch_l1, &grad_dispatch_l2);
            let grad_weight_dispatcher_layer_all = per_elem_add(ctx, fpchip, &grad_weight_dispatcher_layer_all, &grad_weight_dispatcher_layer);
            let grad_bias_dispatcher_layer_all = grad_bias_dispatcher_layer;

            //grad_weight_output_layer+grad_output_l2
            let grad_weight_output_layer_all = per_elem_add(ctx, fpchip, &grad_weight_output_layer, &grad_output_l2);
            let grad_bias_output_layer_all = grad_bias_output_layer;

            //grad_weight_var_layer+grad_var_l2
            let grad_weight_var_layer_all = per_elem_add(ctx, fpchip, &grad_weight_var_layer, &grad_var_l2);
            let grad_bias_var_layer_all = grad_bias_var_layer;

            // Update weights and biases
            dispatcher_layer.update(ctx, fpchip, &grad_weight_dispatcher_layer_all, &grad_bias_dispatcher_layer_all, learning_rate_batch);
            output_layer.update(ctx, fpchip, &grad_weight_output_layer_all, &grad_bias_output_layer_all, learning_rate_batch);
            var_layer.update(ctx, fpchip, &grad_weight_var_layer_all, &grad_bias_var_layer_all, learning_rate_batch);
        }

        // Stage 2




        // Generate the causal graph
        let graph_weights = dispatcher_layer.weight.clone();
        let mut adj_matrix: Vec<Vec<AssignedValue<F>>> = vec![];
        for i in 0..graph_dims {
            let mut b_i = vec![];
            for j in 0..graph_dims {
                // torch.linalg.vector_norm(self.weight, dim=2, ord=self.adjacency_p)
                let mut norm = ctx.load_witness(fpchip.quantization(0.0));
                for k in 0..self.model_kwargs.dim_hidden {
                    let weight = graph_weights[[i, j, k]];
                    let weight_square = fpchip.qmul(ctx, weight, weight);
                    norm = fpchip.qadd(ctx, norm, weight_square);
                }
                b_i.push(norm); //TODO: random initialization
            }
            adj_matrix.push(b_i);
        }

        adj_matrix
    }
}



struct ActivationLayer<F: BigPrimeField, const PRECISION_BITS: u32>{
    _marker: PhantomData<F>,
}


impl<F: BigPrimeField, const PRECISION_BITS: u32> ActivationLayer<F, PRECISION_BITS> {
    pub fn new() -> Self {
        Self{
            _marker: PhantomData,
        }
    }

    pub fn forward(ctx: &mut Context<F>, fpchip: &FixedPointChip<F, PRECISION_BITS>, x: &Array<AssignedValue<F>, Dim<[usize; 3]>>, name: &str) 
    -> Array<AssignedValue<F>, Dim<[usize; 3]>> {
        let mut result = vec![];
        for b in 0..x.shape()[0] {
            for o in 0..x.shape()[1] {
                for h in 0..x.shape()[2] {
                    let y: AssignedValue<F>;
                    if name == "softplus" {
                        y = ActivationLayer::softplus(ctx, fpchip, x[[b, o, h]]);
                    } else if name == "sigmoid" {
                        y = ActivationLayer::sigmoid(ctx, fpchip, x[[b, o, h]]);
                    } else {
                        panic!("Activation function not supported");
                    }
                    result.push(y);
                }
            }
        }

        let out_shape = [x.shape()[0], x.shape()[1], x.shape()[2]];
        Array::from_shape_vec(out_shape, result).unwrap()
    }

    pub fn backward(ctx: &mut Context<F>, fpchip: &FixedPointChip<F, PRECISION_BITS>, x: &Array<AssignedValue<F>, Dim<[usize; 3]>>, name: &str)
    -> Array<AssignedValue<F>, Dim<[usize; 3]>> {
        let mut result = vec![];
        for b in 0..x.shape()[0] {
            for o in 0..x.shape()[1] {
                for h in 0..x.shape()[2] {
                    let y: AssignedValue<F>;
                    if name == "softplus" {
                        y = ActivationLayer::softplus_backward(ctx, fpchip, x[[b, o, h]]);
                    } else if name == "sigmoid" {
                        y = ActivationLayer::sigmoid_backward(ctx, fpchip, x[[b, o, h]]);
                    } else {
                        panic!("Activation function not supported");
                    }
                    result.push(y);
                }
            }
        }

        let out_shape = [x.shape()[0], x.shape()[1], x.shape()[2]];
        Array::from_shape_vec(out_shape, result).unwrap()
    }

    pub fn softplus_backward(
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        x: AssignedValue<F>,
    ) -> AssignedValue<F> 
    where
        F: BigPrimeField
    {
        Self::sigmoid(ctx, fpchip, x)
    }

    pub fn sigmoid_backward(
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        x: AssignedValue<F>,
    ) -> AssignedValue<F> 
    where
        F: BigPrimeField
    {
        let sigmoid_x = Self::sigmoid(ctx, fpchip, x);
        let one = ctx.load_constant(fpchip.quantization(1.0));
        let one_s_sigmoid_x = fpchip.qsub(ctx, one, sigmoid_x);
        fpchip.qmul(ctx, sigmoid_x, one_s_sigmoid_x)
    }


    pub fn softplus(
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        x: AssignedValue<F>,
    ) -> AssignedValue<F> 
    where
        F: BigPrimeField
    {
        let one = ctx.load_constant(fpchip.quantization(1.0));
        // let threshold = ctx.load_constant(fpchip.quantization(threshold));
        // let x = fpchip.qmul(ctx, x, threshold);
        let exp_x = fpchip.qexp(ctx, x);
        let one_p_exp_x = fpchip.qadd(ctx, one, exp_x);
        let log_one_p_exp_x = fpchip.qlog(ctx, one_p_exp_x);
        log_one_p_exp_x
    }
    
    pub fn sigmoid(
        ctx: &mut Context<F>,
        fpchip: &FixedPointChip<F, PRECISION_BITS>,
        x: AssignedValue<F>,
    ) -> AssignedValue<F> 
    where
        F: BigPrimeField
    {
        let one = ctx.load_constant(fpchip.quantization(1.0));
        let neg_x = fpchip.neg(ctx, x); // TODO: check whether neg is correct
        let exp_neg_x = fpchip.qexp(ctx, neg_x);
        let one_p_exp_neg_x = fpchip.qadd(ctx, one, exp_neg_x);
        let one_div_one_p_exp_neg_x = fpchip.qdiv(ctx, one, one_p_exp_neg_x);
        one_div_one_p_exp_neg_x
    }
    
}


struct LossFunc<F: BigPrimeField, const PRECISION_BITS: u32>{
    _marker: PhantomData<F>,
}


impl<F: BigPrimeField, const PRECISION_BITS: u32> LossFunc<F, PRECISION_BITS> {
    pub fn new() -> Self {
        Self{
            _marker: PhantomData,
        }
    }

    pub fn nll(ctx: &mut Context<F>, fpchip: &FixedPointChip<F, PRECISION_BITS>, x_mean: &Array<AssignedValue<F>, Dim<[usize; 3]>>, 
        x_var: &Array<AssignedValue<F>, Dim<[usize; 3]>>, x: &Array<AssignedValue<F>, Dim<[usize; 2]>>) -> AssignedValue<F> {
        // 确保 x_mean, x_var 和 x 的形状一致
        assert_eq!(x_mean.len(), x_var.len());
        assert_eq!(x_var.len(), x.len());
    
        let mut nll_sum = ctx.load_witness(fpchip.quantization(0.0));
        let quantized_2 = ctx.load_constant(fpchip.quantization(2.0));
        let quantized_pi = ctx.load_constant(fpchip.quantization(std::f64::consts::PI));
        let quantized_minus_0_5 = ctx.load_constant(fpchip.quantization(-0.5));
    
        for i in 0..x.shape()[0] {
            for j in 0..x.shape()[1] {
                    let mean = x_mean[[i,j,0]];
                    let var = x_var[[i,j,0]];
                    let obs = x[[i,j]];
    
                    //obs-mean
                    let diff_obs_mean = fpchip.qsub(ctx, obs, mean);
                    // (obs-mean)^2
                    let diff_obs_mean_square = fpchip.qmul(ctx, diff_obs_mean, diff_obs_mean);
                    // (obs-mean)^2 / var
                    let diff_obs_mean_square_div_var = fpchip.qdiv(ctx, diff_obs_mean_square, var);
                    // log(2 * PI * var)
                    let pi_times_2 = fpchip.qmul(ctx, quantized_2, quantized_pi); 
                    let pi_times_2_var = fpchip.qmul(ctx, pi_times_2, var);
                    let log_2_pi_var = fpchip.qlog(ctx, pi_times_2_var);
                    // -0.5 * ((x - mean)^2 / var + log(2 * PI * var))
                    let log_prob_times_2 = fpchip.qadd(ctx, diff_obs_mean_square_div_var, log_2_pi_var);
                    let log_prob = fpchip.qmul(ctx, quantized_minus_0_5, log_prob_times_2);

                    nll_sum = fpchip.qsub(ctx, nll_sum, log_prob);
            }
        }
    
        nll_sum
    }    

    pub fn nll_backward(ctx: &mut Context<F>, fpchip: &FixedPointChip<F, PRECISION_BITS>, x_mean: &Array<AssignedValue<F>, Dim<[usize; 3]>>, 
        x_var: &Array<AssignedValue<F>, Dim<[usize; 3]>>, x: &Array<AssignedValue<F>, Dim<[usize; 2]>>) -> (Array<AssignedValue<F>, Dim<[usize; 3]>>, Array<AssignedValue<F>, Dim<[usize; 3]>>) {
        // 确保 x_mean, x_var 和 x 的形状一致
        assert_eq!(x_mean.len(), x_var.len());
        assert_eq!(x_var.len(), x.len());
    
        let mut nll_sum = ctx.load_witness(fpchip.quantization(0.0));
        let quantized_2 = ctx.load_constant(fpchip.quantization(2.0));
        let quantized_pi = ctx.load_constant(fpchip.quantization(std::f64::consts::PI));
        let quantized_minus_0_5 = ctx.load_constant(fpchip.quantization(-0.5));
        let quantized_0_5 = ctx.load_constant(fpchip.quantization(0.5));
    
        let mut grad_mean = vec![];
        let mut grad_var = vec![];
    
        for i in 0..x.shape()[0] {
            for j in 0..x.shape()[1] {
                    let mean = x_mean[[i,j,0]];
                    let var = x_var[[i,j,0]];
                    let obs = x[[i,j]];
    
                    //obs-mean
                    let diff_obs_mean = fpchip.qsub(ctx, obs, mean);
                    // (obs-mean)^2
                    let diff_obs_mean_square = fpchip.qmul(ctx, diff_obs_mean, diff_obs_mean);
                    // (obs-mean)^2 / var
                    let diff_obs_mean_square_div_var = fpchip.qdiv(ctx, diff_obs_mean_square, var);
                    // log(2 * PI * var)
                    let pi_times_2 = fpchip.qmul(ctx, quantized_2, quantized_pi);
                    let pi_times_2_var = fpchip.qmul(ctx, pi_times_2, var);
                    let log_2_pi_var = fpchip.qlog(ctx, pi_times_2_var);
                    // -0.5 * ((x - mean)^2 / var + log(2 * PI * var))
                    let log_prob_times_2 = fpchip.qadd(ctx, diff_obs_mean_square_div_var, log_2_pi_var);
                    let log_prob = fpchip.qmul(ctx, quantized_minus_0_5, log_prob_times_2);

                    nll_sum = fpchip.qsub(ctx, nll_sum, log_prob);
    
                    // grad_mean
                    // - (x - mean) / var
                    let minus_grad_mean_i = fpchip.qdiv(ctx, diff_obs_mean, var);
                    let grad_mean_i = fpchip.neg(ctx, minus_grad_mean_i);
                    grad_mean.push(grad_mean_i);

                    // grad_var
                    // 1 / (2 * var) - (x - mean)^2 / (2 * var^2)
                    // 1 / (2 * var)
                    let frac_1_2_var = fpchip.qdiv(ctx, quantized_0_5, var);
                    // (x - mean)^2 / (2 * var^2)
                    let var_square = fpchip.qmul(ctx, var, var);
                    let var_square_times_2 = fpchip.qmul(ctx, var_square, quantized_2);
                    let diff_obs_mean_square_div_2_var_square = fpchip.qdiv(ctx, diff_obs_mean_square, var_square_times_2);
                    let grad_var_i = fpchip.qsub(ctx, frac_1_2_var, diff_obs_mean_square_div_2_var_square);
                    grad_var.push(grad_var_i);
            }
        }

        let out_shape = [x.shape()[0], x.shape()[1], 1];
        let grad_mean = Array::from_shape_vec(out_shape, grad_mean).unwrap();
        let grad_var = Array::from_shape_vec(out_shape, grad_var).unwrap();

        (grad_mean, grad_var)
    }

    pub fn l1_reg_dispatcher(ctx: &mut Context<F>, fpchip: &FixedPointChip<F, PRECISION_BITS>, dispatcher_layer: &DispatcherLayer<F, PRECISION_BITS>) -> AssignedValue<F> {
        let mut l1_sum = ctx.load_witness(fpchip.quantization(0.0));
        for i in 0..dispatcher_layer.in_dim {
            for j in 0..dispatcher_layer.out_dim {
                for k in 0..dispatcher_layer.hidden_dim {
                    let abs_weight = fpchip.qabs(ctx, dispatcher_layer.weight[[i,j,k]]);
                    l1_sum = fpchip.qadd(ctx, l1_sum, abs_weight);
                }
            }
        }
        l1_sum
    }

    pub fn l1_reg_dispatcher_backward(ctx: &mut Context<F>, fpchip: &FixedPointChip<F, PRECISION_BITS>, dispatcher_layer: &DispatcherLayer<F, PRECISION_BITS>) -> Array<AssignedValue<F>, Dim<[usize; 3]>> {
        let mut grad = vec![];
        let quantized_minus_1 = ctx.load_constant(fpchip.quantization(-1.0));
        let quantized_1 = ctx.load_constant(fpchip.quantization(1.0));
        for i in 0..dispatcher_layer.in_dim {
            for j in 0..dispatcher_layer.out_dim {
                for k in 0..dispatcher_layer.hidden_dim {
                    let is_neg = fpchip.is_neg(ctx, dispatcher_layer.weight[[i,j,k]]);
                    let grad_i = fpchip.gate().select(ctx, quantized_minus_1, quantized_1, is_neg);
                    grad.push(grad_i);
                }
            }
        }
        let out_shape = [dispatcher_layer.in_dim, dispatcher_layer.out_dim, dispatcher_layer.hidden_dim];
        Array::from_shape_vec(out_shape, grad).unwrap()
    }


    pub fn l2_reg_all_weights(ctx: &mut Context<F>, fpchip: &FixedPointChip<F, PRECISION_BITS>, dispatcher_layer: &DispatcherLayer<F, PRECISION_BITS>,
    output_layer: &LinearParallel<F, PRECISION_BITS>, var_layer: &LinearParallel<F, PRECISION_BITS>
    ) -> AssignedValue<F> {
        let mut l2_sum = ctx.load_witness(fpchip.quantization(0.0));
        for i in 0..dispatcher_layer.in_dim {
            for j in 0..dispatcher_layer.out_dim {
                for k in 0..dispatcher_layer.hidden_dim {
                    let weight_square = fpchip.qmul(ctx, dispatcher_layer.weight[[i,j,k]], dispatcher_layer.weight[[i,j,k]]);
                    l2_sum = fpchip.qadd(ctx, l2_sum, weight_square);
                }
            }
        }

        for i in 0..output_layer.parallel_dim {
            for j in 0..output_layer.in_dim {
                for k in 0..output_layer.out_dim {
                    let weight_square = fpchip.qmul(ctx, output_layer.weight[[i,j,k]], output_layer.weight[[i,j,k]]);
                    l2_sum = fpchip.qadd(ctx, l2_sum, weight_square);
                }
            }
        }

        for i in 0..var_layer.parallel_dim {
            for j in 0..var_layer.in_dim {
                for k in 0..var_layer.out_dim {
                    let weight_square = fpchip.qmul(ctx, var_layer.weight[[i,j,k]], var_layer.weight[[i,j,k]]);
                    l2_sum = fpchip.qadd(ctx, l2_sum, weight_square);
                }
            }
        }
        l2_sum
    }

    pub fn l2_reg_all_weights_backward(ctx: &mut Context<F>, fpchip: &FixedPointChip<F, PRECISION_BITS>, dispatcher_layer: &DispatcherLayer<F, PRECISION_BITS>,
    output_layer: &LinearParallel<F, PRECISION_BITS>, var_layer: &LinearParallel<F, PRECISION_BITS>
    ) -> (Array<AssignedValue<F>, Dim<[usize; 3]>>, Array<AssignedValue<F>, Dim<[usize; 3]>>, Array<AssignedValue<F>, Dim<[usize; 3]>>) {
        //TODO: add bias
        let mut grad_dispatcher = vec![];
        let quantized_2 = ctx.load_constant(fpchip.quantization(2.0));
        for i in 0..dispatcher_layer.in_dim {
            for j in 0..dispatcher_layer.out_dim {
                for k in 0..dispatcher_layer.hidden_dim {
                    let grad_i = fpchip.qmul(ctx, quantized_2, dispatcher_layer.weight[[i,j,k]]);
                    grad_dispatcher.push(grad_i);
                }
            }
        }
        let out_shape_dispatcher = [dispatcher_layer.in_dim, dispatcher_layer.out_dim, dispatcher_layer.hidden_dim];
        let grad_dispatcher = Array::from_shape_vec(out_shape_dispatcher, grad_dispatcher).unwrap();

        let mut grad_output = vec![];
        for i in 0..output_layer.parallel_dim {
            for j in 0..output_layer.in_dim {
                for k in 0..output_layer.out_dim {
                    let grad_i = fpchip.qmul(ctx, quantized_2, output_layer.weight[[i,j,k]]);
                    grad_output.push(grad_i);
                }
            }
        }
        let out_shape_output = [output_layer.parallel_dim, output_layer.in_dim, output_layer.out_dim];
        let grad_output = Array::from_shape_vec(out_shape_output, grad_output).unwrap();

        let mut grad_var = vec![];
        for i in 0..var_layer.parallel_dim {
            for j in 0..var_layer.in_dim {
                for k in 0..var_layer.out_dim {
                    let grad_i = fpchip.qmul(ctx, quantized_2, var_layer.weight[[i,j,k]]);
                    grad_var.push(grad_i);
                }
            }
        }
        let out_shape_var = [var_layer.parallel_dim, var_layer.in_dim, var_layer.out_dim];
        let grad_var = Array::from_shape_vec(out_shape_var, grad_var).unwrap();

        (grad_dispatcher, grad_output, grad_var)
    }
}


pub fn per_elem_mul<F: BigPrimeField, const PRECISION_BITS: u32>(
    ctx: &mut Context<F>,
    fpchip: &FixedPointChip<F, PRECISION_BITS>,
    x: &Array<AssignedValue<F>, Dim<[usize; 3]>>,
    y: &Array<AssignedValue<F>, Dim<[usize; 3]>>,
) -> Array<AssignedValue<F>, Dim<[usize; 3]>> {
    let mut result = vec![];
    for i in 0..x.shape()[0] {
        for j in 0..x.shape()[1] {
            for k in 0..x.shape()[2] {
                let product = fpchip.qmul(ctx, x[[i,j,k]], y[[i,j,k]]);
                result.push(product);
            }
        }
    }

    let out_shape = [x.shape()[0], x.shape()[1], x.shape()[2]];
    Array::from_shape_vec(out_shape, result).unwrap()
}

pub fn per_elem_add<F: BigPrimeField, const PRECISION_BITS: u32>(
    ctx: &mut Context<F>,
    fpchip: &FixedPointChip<F, PRECISION_BITS>,
    x: &Array<AssignedValue<F>, Dim<[usize; 3]>>,
    y: &Array<AssignedValue<F>, Dim<[usize; 3]>>,
) -> Array<AssignedValue<F>, Dim<[usize; 3]>> {
    let mut result = vec![];
    for i in 0..x.shape()[0] {
        for j in 0..x.shape()[1] {
            for k in 0..x.shape()[2] {
                let product = fpchip.qadd(ctx, x[[i,j,k]], y[[i,j,k]]);
                result.push(product);
            }
        }
    }

    let out_shape = [x.shape()[0], x.shape()[1], x.shape()[2]];
    Array::from_shape_vec(out_shape, result).unwrap()
}

pub fn per_elem_scale<F: BigPrimeField, const PRECISION_BITS: u32>(
    ctx: &mut Context<F>,
    fpchip: &FixedPointChip<F, PRECISION_BITS>,
    x: &Array<AssignedValue<F>, Dim<[usize; 3]>>,
    scale_factor: AssignedValue<F>,
) -> Array<AssignedValue<F>, Dim<[usize; 3]>> {
    let mut result = vec![];
    for i in 0..x.shape()[0] {
        for j in 0..x.shape()[1] {
            for k in 0..x.shape()[2] {
                let product = fpchip.qmul(ctx, x[[i,j,k]], scale_factor);
                result.push(product);
            }
        }
    }

    let out_shape = [x.shape()[0], x.shape()[1], x.shape()[2]];
    Array::from_shape_vec(out_shape, result).unwrap()
}