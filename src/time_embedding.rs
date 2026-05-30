use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
    tensor::activation::silu,
};

#[derive(Module, Debug)]
pub struct TimeEmbedding<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
}

impl<B: Backend> TimeEmbedding<B> {
    // ✅ 修改：第一个参数应该是 emb_dim（内部维度），第二个是输出维度
    pub fn new(emb_dim: usize, out_dim: usize, device: &B::Device) -> Self {
        // 输入是 [batch, 1]，所以 in_features = 1
        let linear1 = LinearConfig::new(1, emb_dim).init(device);
        let linear2 = LinearConfig::new(emb_dim, out_dim).init(device);
        Self { linear1, linear2 }
    }

    pub fn forward(&self, timesteps: Tensor<B, 1>) -> Tensor<B, 2> {
        // timesteps: [batch]
        let batch_size = timesteps.dims()[0];
        let t_emb = timesteps.reshape([batch_size, 1]);  // [batch, 1]
        
        let t_emb = self.linear1.forward(t_emb);   // [batch, emb_dim]
        let t_emb = silu(t_emb);
        self.linear2.forward(t_emb)                 // [batch, out_dim]
    }
}