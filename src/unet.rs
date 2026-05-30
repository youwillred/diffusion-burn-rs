use burn::nn::{
    conv::{Conv2d, Conv2dConfig},
    BatchNorm, BatchNormConfig,
    Linear, LinearConfig,
    Relu,
};
use burn::nn::interpolate::{Interpolate2d, Interpolate2dConfig, InterpolateMode};
use burn::module::Module;
use burn::nn::PaddingConfig2d;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use crate::time_embedding::TimeEmbedding;
use crate::cross_attn::CrossAttention;


/// 定义 UNet 的基本卷积块：Conv -> BatchNorm -> ReLU
#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B>,
    activation: Relu,
    time_proj: Linear<B>,
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(
        in_channels: usize, 
        out_channels: usize, 
        time_emb_dim: usize,
        device: &B::Device
    ) -> Self {
        let conv1 = Conv2dConfig::new([in_channels, out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let bn1 = BatchNormConfig::new(out_channels).init(device);
        let conv2 = Conv2dConfig::new([out_channels, out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let bn2 = BatchNormConfig::new(out_channels).init(device);
        let activation = Relu::new();
        let time_proj = LinearConfig::new(time_emb_dim, out_channels).init(device);

        Self { conv1, bn1, conv2, bn2, activation, time_proj }
    }

    pub fn forward(&self, x: Tensor<B, 4>, t_emb: Tensor<B, 2>) -> Tensor<B, 4> {

        
        let t_proj = self.time_proj.forward(t_emb);  // [batch, out_channels]
        
        // ✅ 正确的方法：使用 reshape 而不是 unsqueeze
        let batch_size = t_proj.dims()[0];
        let channels = t_proj.dims()[1];
        let t_proj = t_proj.reshape([batch_size, channels, 1, 1]);  // [batch, channels, 1, 1] 
        
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = self.activation.forward(x);
        let x = x + t_proj.clone();

        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);
        self.activation.forward(x)
    }
}

/// 下采样块：卷积块 + MaxPool
#[derive(Module, Debug)]
pub struct DownBlock<B: Backend> {
    conv_block: ConvBlock<B>,
    pool: burn::nn::pool::MaxPool2d,
}

impl<B: Backend> DownBlock<B> {
    pub fn new(
        in_channels: usize, 
        out_channels: usize, 
        time_emb_dim: usize,
        device: &B::Device
    ) -> Self {
        let conv_block = ConvBlock::new(in_channels, out_channels, time_emb_dim, device);
        let pool = burn::nn::pool::MaxPool2dConfig::new([2, 2]).init();
        Self { conv_block, pool }
    }

    pub fn forward(&self, x: Tensor<B, 4>, t_emb: Tensor<B, 2>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let skip = self.conv_block.forward(x, t_emb.clone());
        let pooled = self.pool.forward(skip.clone());
        (pooled, skip)
    }
}

/// 上采样块：上采样（插值） -> 拼接 -> 卷积块
#[derive(Module, Debug)]
pub struct UpBlock<B: Backend> {
    conv_block: ConvBlock<B>,
    interpolate: Interpolate2d,
}

impl<B: Backend> UpBlock<B> {
    pub fn new(
        in_channels: usize, 
        out_channels: usize, 
        time_emb_dim: usize, 
        device: &B::Device
    ) -> Self {
        let conv_block = ConvBlock::new(in_channels, out_channels, time_emb_dim, device);
        
        let interpolate = Interpolate2dConfig::new()
            .with_mode(InterpolateMode::Nearest)
            .with_scale_factor(Some([2.0, 2.0]))
            .init();

        Self { conv_block, interpolate }
    }

    pub fn forward(&self, x: Tensor<B, 4>, skip: Tensor<B, 4>, t_emb: Tensor<B, 2>) -> Tensor<B, 4> {
        let upsampled = self.interpolate.forward(x);
        let cat = Tensor::cat(vec![upsampled, skip], 1);
        self.conv_block.forward(cat, t_emb)
    }
}

/// 全 UNet 模型
#[derive(Module, Debug)]
pub struct UNet<B: Backend> {
    down1: DownBlock<B>,
    down2: DownBlock<B>,
    down3: DownBlock<B>,
    down4: DownBlock<B>,
    bottleneck: ConvBlock<B>,
    up1: UpBlock<B>,
    up2: UpBlock<B>,
    up3: UpBlock<B>,
    up4: UpBlock<B>,
    final_conv: Conv2d<B>,
    time_embedding: TimeEmbedding<B>,
    cross_attn: CrossAttention<B>,
}

impl<B: Backend> UNet<B> {
    pub fn new(
        time_emb_dim: usize,
        time_proj_dim: usize,
        device: &B::Device
    ) -> Self {
        let time_embedding = TimeEmbedding::new(time_emb_dim, time_proj_dim, device);

        let down1 = DownBlock::new(3, 64, time_proj_dim, device);
        let down2 = DownBlock::new(64, 128, time_proj_dim, device);
        let down3 = DownBlock::new(128, 256, time_proj_dim, device);
        let down4 = DownBlock::new(256, 512, time_proj_dim, device);

        let bottleneck = ConvBlock::new(512, 1024, time_proj_dim, device);

        let up1 = UpBlock::new(1024 + 512, 512, time_proj_dim, device);
        let up2 = UpBlock::new(512 + 256, 256, time_proj_dim, device);
        let up3 = UpBlock::new(256 + 128, 128, time_proj_dim, device);
        let up4 = UpBlock::new(128 + 64, 64, time_proj_dim, device);

        let final_conv = Conv2dConfig::new([64, 3], [1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .init(device);

        let cross_attn = CrossAttention::new(
            1024,   // img_dim: bottleneck 的输出通道
            768,    // txt_dim: 文本嵌入维度（例如 CLIP 是 768）
            512,    // d_model: 注意力内部维度
            device,
        );
        
        Self { down1, down2, down3, down4, bottleneck, up1, up2, up3, up4, final_conv, time_embedding, cross_attn }
    }

    // ✅ 修复：参数名改为 timesteps（不要和内部变量 t_emb 冲突）
    pub fn forward(
        &self, 
        x: Tensor<B, 4>, 
        timesteps: Tensor<B, 1>,
        text_emb: Option<Tensor<B, 3>>
    ) -> Tensor<B, 4> {

        // 把时间步转换成时间嵌入
        let t_emb = self.time_embedding.forward(timesteps);

        // 编码阶段
        let (x, skip1) = self.down1.forward(x, t_emb.clone());
        let (x, skip2) = self.down2.forward(x, t_emb.clone());
        let (x, skip3) = self.down3.forward(x, t_emb.clone());
        let (x, skip4) = self.down4.forward(x, t_emb.clone());
        
        // 瓶颈
        let x = self.bottleneck.forward(x, t_emb.clone());
        
        // ⭐ 文本条件注入点
        let x = if let Some(text) = text_emb {
            self.cross_attn.forward(x, text)
        } else {
            x
        };
        
        // 解码阶段
        let x = self.up1.forward(x, skip4, t_emb.clone());
        let x = self.up2.forward(x, skip3, t_emb.clone());
        let x = self.up3.forward(x, skip2, t_emb.clone());
        let x = self.up4.forward(x, skip1, t_emb.clone());
        
        // 输出
        self.final_conv.forward(x)
    }
}