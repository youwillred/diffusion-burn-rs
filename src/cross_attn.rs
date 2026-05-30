use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
    tensor::activation::softmax,
};

#[derive(Module, Debug)]
pub struct CrossAttention<B: Backend> {
    q_proj: Linear<B>,  // 图像 → Q
    k_proj: Linear<B>,  // 文本 → K
    v_proj: Linear<B>,  // 文本 → V
    out_proj: Linear<B>, // 输出投影
}

impl<B: Backend> CrossAttention<B> {
    pub fn new(
        img_dim: usize,    // 图像通道数
        txt_dim: usize,    // 文本通道数
        d_model: usize,    // Q/K/V 的统一维度
        device: &B::Device,
    ) -> Self {
        let q_proj = LinearConfig::new(img_dim, d_model).init(device);
        let k_proj = LinearConfig::new(txt_dim, d_model).init(device);
        let v_proj = LinearConfig::new(txt_dim, d_model).init(device);
        let out_proj = LinearConfig::new(d_model, img_dim).init(device);
        
        Self { q_proj, k_proj, v_proj, out_proj }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 4>,       // [B, C_img, H, W]
        context: Tensor<B, 3>, // [B, T, C_txt]
    ) -> Tensor<B, 4> {
        let [batch, img_dim, height, width] = x.dims();
        
        // 1. 图像 → 序列 [B, H*W, C_img]
        let x_flat = x.clone().reshape([batch, img_dim, height * width]);
        let x_seq = x_flat.swap_dims(1, 2);
        
        // 2. 生成 Q、K、V（关键步骤）
        let q = self.q_proj.forward(x_seq);      // [B, H*W, d_model]
        let k = self.k_proj.forward(context.clone());    // [B, T, d_model]
        let v = self.v_proj.forward(context);    // [B, T, d_model]
        
        // 3. 计算注意力：Q·K^T
        let scores = q.matmul(k.swap_dims(1, 2));  // [B, H*W, T]
        let attn = softmax(scores, 2);
        
        // 4. 加权求和
        let out_seq = attn.matmul(v);               // [B, H*W, d_model]
        
        // 5. 投影回图像维度
        let out_seq = self.out_proj.forward(out_seq); // [B, H*W, C_img]
        
        // 6. 恢复形状
        let out_flat = out_seq.swap_dims(1, 2);      // [B, C_img, H*W]
        let attention_out = out_flat.reshape([batch, img_dim, height, width]);
        // ✅ 残差连接：原始输入 + 注意力输出
        x + attention_out
    }
}