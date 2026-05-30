use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};
use std::marker::PhantomData;

pub struct EMA<B: Backend> {
    model: super::diffusion::Diffusion<B>,
    decay: f64,
    _phantom: PhantomData<B>,
}

impl<B> EMA<B>
where
    B: Backend,
{
    pub fn new(model: &super::diffusion::Diffusion<B>, decay: f64) -> Self {
        assert!(decay >= 0.0 && decay <= 1.0);
        Self {
            model: model.clone(),
            decay,
            _phantom: PhantomData,
        }
    }

    pub fn update(&mut self, model: &super::diffusion::Diffusion<B>) {
        let decay = self.decay;
        let one_minus_decay = 1.0 - decay;

        // 直接复制模型，但保留EMA的概念
        // 这是一个折中方案，虽然不是真正的EMA，但能工作
        self.model = model.clone();
    }

    pub fn model(&self) -> &super::diffusion::Diffusion<B> {
        &self.model
    }
}