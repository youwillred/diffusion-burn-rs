


use burn::{
    module::Module,
    tensor::backend::AutodiffBackend,
    train::{TrainStep,InferenceStep,TrainOutput,ClassificationOutput},
    prelude::*,
};
use diffusion_burn_rs::diffusion::Diffusion;
use diffusion_burn_rs::data::loader::ImageBatch;



impl<B: AutodiffBackend> TrainStep for Diffusion<B>{
    type Input = MnistBatch<B>;
    type Output = Tensor<B, 1>;

    fn step(&self, batch: Self::Input)->TrainOutput<Self::Output>{
        let item = self.loss(batch.images);
        TrainOutput::new(self,item)
    }

}

impl<B: Backend> InferenceStep for Diffusion<B>{
    type Input = MnistBatch<B>;
    type Output = Tensor<B, 1>;

    fn step(&self,batch:Self::Input)->Self::Output{
        self.loss(batch.images)
    }
}

pub fn main()-> anyhow::Result<()> {

    Ok(())
}