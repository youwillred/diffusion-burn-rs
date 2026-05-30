use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor, TensorData},
    tensor::Distribution,
};

use crate::unet::UNet;

#[derive(Module, Debug)]
pub struct Diffusion<B: Backend> {
    pub unet: UNet<B>,
    timesteps: usize,
    betas: Vec<f32>,
    alphas: Vec<f32>,
    alpha_bars: Vec<f32>,
    sqrt_alpha_bars: Vec<f32>,
    sqrt_one_minus_alpha_bars: Vec<f32>,
}

impl<B: Backend> Diffusion<B> {
    /// 线性 Beta 调度（DDPM 原版）
    fn linear_beta_schedule(timesteps: usize) -> Vec<f32> {
        let beta_start = 1e-4;
        let beta_end = 0.02;
        (0..timesteps)
            .map(|i| {
                let t = i as f32 / (timesteps - 1) as f32;
                beta_start + (beta_end - beta_start) * t
            })
            .collect()
    }

    pub fn new(unet: UNet<B>, timesteps: usize, device: &B::Device) -> Self {
        let betas = Self::linear_beta_schedule(timesteps);
        let alphas: Vec<f32> = betas.iter().map(|&b| 1.0 - b).collect();

        let mut alpha_bars = Vec::with_capacity(timesteps);
        let mut cumprod = 1.0;
        for &alpha in &alphas {
            cumprod *= alpha;
            alpha_bars.push(cumprod);
        }

        let sqrt_alpha_bars = alpha_bars.iter().map(|&ab| ab.sqrt()).collect();
        let sqrt_one_minus_alpha_bars =
            alpha_bars.iter().map(|&ab| (1.0 - ab).sqrt()).collect();

        Self {
            unet,
            timesteps,
            betas,
            alphas,
            alpha_bars,
            sqrt_alpha_bars,
            sqrt_one_minus_alpha_bars,
        }
    }

    fn get_coeffs(
        &self,
        t: &Tensor<B, 1>,
        _device: &B::Device,
    ) -> (Vec<f32>, Vec<f32>) {
        let data = t.clone().into_data();
        let t_slice = data.as_slice::<f32>().unwrap();
        let indices: Vec<usize> = t_slice.iter().map(|&v| v as usize).collect();

        let mut sqrt_alphas = Vec::with_capacity(indices.len());
        let mut sqrt_one_minus = Vec::with_capacity(indices.len());

        for &idx in &indices {
            sqrt_alphas.push(self.sqrt_alpha_bars[idx]);
            sqrt_one_minus.push(self.sqrt_one_minus_alpha_bars[idx]);
        }

        (sqrt_alphas, sqrt_one_minus)
    }

    pub fn add_noise(
        &self,
        x_0: Tensor<B, 4>,
        t: Tensor<B, 1>,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let batch = x_0.dims()[0];
        let noise = Tensor::<B, 4>::random_like(&x_0, Distribution::Normal(0.0, 1.0));

        let (sqrt_alphas, sqrt_one_minus) = self.get_coeffs(&t, device);

        let sqrt_alpha_bar = Tensor::<B, 1>::from_data(
            TensorData::new(sqrt_alphas, [batch]),
            device,
        ).reshape([batch, 1, 1, 1]);

        let sqrt_one_minus_alpha_bar = Tensor::<B, 1>::from_data(
            TensorData::new(sqrt_one_minus, [batch]),
            device,
        ).reshape([batch, 1, 1, 1]);

        sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    }

    pub fn predict_noise(
        &self,
        x_t: Tensor<B, 4>,
        t: Tensor<B, 1>,
        text_emb: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 4> {
        self.unet.forward(x_t, t, text_emb)
    }

    pub fn denoise_step_train(
        &self,
        x_t: Tensor<B, 4>,
        t: Tensor<B, 1>,
        text_emb: Option<Tensor<B, 3>>,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let predicted_noise = self.predict_noise(x_t.clone(), t.clone(), text_emb);
        let data = t.clone().into_data();
        let t_slice = data.as_slice::<f32>().unwrap();
        let step_sizes: Vec<f32> = t_slice
            .iter()
            .map(|&v| self.betas[v as usize])
            .collect();

        let batch = x_t.dims()[0];
        let step_tensor = Tensor::<B, 1>::from_data(
            TensorData::new(step_sizes, [batch]),
            device,
        ).reshape([batch, 1, 1, 1]);

        x_t - predicted_noise * step_tensor
    }

    pub fn generate(
        &self,
        num_samples: usize,
        size: usize,
        text_emb: Option<Tensor<B, 3>>,
        cfg_scale: f32,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let mut x = Tensor::<B, 4>::random(
            [num_samples, 3, size, size],
            Distribution::Normal(0.0, 1.0),
            device,
        );

        for step in (0..self.timesteps).rev() {
            let t_values = vec![step as f32; num_samples];
            let t = Tensor::<B, 1>::from_data(
                TensorData::new(t_values, [num_samples]),
                device,
            );

            //x = self.denoise_step(x, t, text_emb.clone(), device);
            x = self.denoise_step_cfg(
                x,
                t,
                text_emb.clone(),
                cfg_scale,
                device,
            );

            if step % 100 == 0 {
                println!("去噪步骤: {}/{}", step, self.timesteps);
            }
        }

        x
    }


    pub fn denoise_step_cfg(
        &self,
        x_t: Tensor<B, 4>,
        t: Tensor<B, 1>,
        cond_emb: Option<Tensor<B, 3>>,
        cfg_scale: f32,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        // unconditional
        let eps_uncond = self.unet.forward(
            x_t.clone(),
            t.clone(),
            None,
        );

        // conditional
        let eps_cond = self.unet.forward(
            x_t.clone(),
            t.clone(),
            cond_emb,
        );

        // ✅ CFG 核心公式
        let eps = eps_uncond.clone()
            + cfg_scale * (eps_cond - eps_uncond);

        let t_data = t.clone().into_data();
        let t_slice = t_data.as_slice::<f32>().unwrap();
        let step_sizes: Vec<f32> = t_slice
            .iter()
            .map(|&v| self.betas[v as usize])
            .collect();

        let batch = x_t.dims()[0];
        let step_tensor = Tensor::<B, 1>::from_data(
            TensorData::new(step_sizes, [batch]),
            device,
        ).reshape([batch, 1, 1, 1]);

        x_t - eps * step_tensor
    }

    pub fn loss(
        &self,
        x_0: Tensor<B, 4>,
        text_emb: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 1> {
        let device = x_0.device();
        let batch = x_0.dims()[0];

        let t = Tensor::<B, 1>::random(
            [batch],
            Distribution::Uniform(0.0, self.timesteps as f64 - 1e-6),
            &device,
        );

        let noise = Tensor::<B, 4>::random_like(&x_0, Distribution::Normal(0.0, 1.0));
        let (sqrt_alphas, sqrt_one_minus) = self.get_coeffs(&t, &device);

        let sqrt_alpha_bar = Tensor::<B, 1>::from_data(
            TensorData::new(sqrt_alphas, [batch]),
            &device,
        ).reshape([batch, 1, 1, 1]);

        let sqrt_one_minus_alpha_bar = Tensor::<B, 1>::from_data(
            TensorData::new(sqrt_one_minus, [batch]),
            &device,
        ).reshape([batch, 1, 1, 1]);

        let x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise.clone();
        let predicted_noise = self.unet.forward(x_t, t, text_emb);

        let diff = predicted_noise - noise;
        diff.powf_scalar(2.0).mean().reshape([1])
    }
}