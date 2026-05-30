mod unet;
mod time_embedding;
mod diffusion;
mod cross_attn;  // ← 添加

use burn::backend::Wgpu;
use burn::tensor::{Tensor, Distribution};

fn main() {
    let device = Default::default();
    
    let unet = unet::UNet::<Wgpu>::new(1, 64, &device);
    let diffusion = diffusion::Diffusion::new(unet.clone(), 1000,&device);
    
    // 原始测试（无文本）
    let original = Tensor::<Wgpu, 4>::zeros([1, 3, 64, 64], &device);
    let t = Tensor::<Wgpu, 1>::from_data([500.0], &device);
    let noisy = diffusion.add_noise(original, t, &device);
    println!("加噪后形状: {:?}", noisy.shape());
    
    // ⭐ 测试带文本的 forward
    println!("测试文本条件...");
    let batch = 1;
    let seq_len = 16;
    let channels = 768; // 匹配 bottleneck 的输出通道
    
    let fake_text = Tensor::<Wgpu, 3>::random(
        [batch, seq_len, channels],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    
    // 直接用 UNet forward，传入 Some(fake_text)
    let test_input = Tensor::<Wgpu, 4>::random([1, 3, 64, 64], Distribution::Normal(0.0, 1.0), &device);
    let t_test = Tensor::<Wgpu, 1>::from_data([100.0], &device);
    let output = unet.forward(test_input, t_test, Some(fake_text));
    println!("带文本的输出形状: {:?}", output.shape());
}