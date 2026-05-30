#![recursion_limit = "256"]
use rand::random_bool;
use std::path::PathBuf;
use rand::Rng;
use burn::{
    backend::{Autodiff, Wgpu},
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{Tensor, Distribution},
    optim::{AdamConfig, Optimizer},
    module::Module,
    data::dataloader::DataLoaderBuilder,
};
use burn::optim::GradientsParams;
use burn::tensor::backend::Backend;
use burn::backend::ndarray::NdArrayDevice;
use burn::record::Recorder;
use diffusion_burn_rs::{
    data::loader::{DataConfig, ImageDataset, ImageBatcher},
    diffusion::Diffusion,
    unet::UNet,
    tokenizer::{Vocab, tokenize_simple},
    text_encoder::{TextEncoder, TextBatch},
};
use diffusion_burn_rs::ema::EMA;

type MyBackend = Autodiff<Wgpu<f32, i32>>;

fn main() -> anyhow::Result<()> {
    let device = burn::backend::wgpu::WgpuDevice::default();
    println!("=== 扩散模型训练（带文本条件）===\n");

    let img_size = 32;
    let batch_size = 2;
    let learning_rate = 1e-4;
    let max_seq_len = 8;
    let epochs = 3;
    let ema_decay = 0.999;

    // --------------------------------------------------
    // 1. 数据
    // --------------------------------------------------
    let data_config = DataConfig {
        csv_path: "input/trainLabels.csv".into(),
        image_root: "input/cifar10_train".into(),
        target_size: (img_size, img_size),
        batch_size,
        shuffle_seed: 42,
        num_workers: 0,
        has_header: true,
        max_samples: Some(100),
        ..Default::default()
    };

    let dataset = ImageDataset::new(&data_config)?;
    let batcher = ImageBatcher::<MyBackend>::new(device.clone(), (img_size, img_size));
    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(batch_size)
        .shuffle(42)
        .build(dataset);

    // --------------------------------------------------
    // 2. 文本组件
    // --------------------------------------------------
    let mut vocab = Vocab::new();
    vocab.add_token("airplane".to_string());
    vocab.add_token("automobile".to_string());
    vocab.add_token("bird".to_string());
    vocab.add_token("cat".to_string());
    vocab.add_token("deer".to_string());
    vocab.add_token("dog".to_string());
    vocab.add_token("frog".to_string());
    vocab.add_token("horse".to_string());
    vocab.add_token("ship".to_string());
    vocab.add_token("truck".to_string());

    let text_encoder = TextEncoder::<MyBackend>::new(
        vocab.vocab_size(),
        768,
        max_seq_len,
        &device,
    );

    // --------------------------------------------------
    // 3. 模型
    // --------------------------------------------------
    let unet = UNet::new(3, 768, &device);
    let mut model = Diffusion::new(unet, 100, &device);
    let mut ema_diffusion = EMA::new(&model,ema_decay);

    // --------------------------------------------------
    // 4. 优化器
    // --------------------------------------------------
    let mut optimizer = AdamConfig::new().init();

    // --------------------------------------------------
    // 5. 训练
    // --------------------------------------------------
    for epoch in 1..=epochs {
        let mut total_loss = 0.0;
        let mut count = 0;

        
        for (batch_idx, batch) in dataloader.iter().enumerate() {
            // ✅ 使用真实的标签，而不是硬编码
            let captions: Vec<String> = batch.targets.clone().into_data()
                .as_slice::<i32>()
                .unwrap()
                .iter()
                .map(|&label| match label {
                    0 => "airplane",
                    1 => "automobile",
                    2 => "bird",
                    3 => "cat",
                    4 => "deer",
                    5 => "dog",
                    6 => "frog",
                    7 => "horse",
                    8 => "ship",
                    9 => "truck",
                    _ => "object",
                }.to_string())
                .collect();

            // 10% 概率无条件 这就是论文里的 unconditional guidance dropout
            let use_text = random_bool(0.9);

            let text_emb = if use_text{
                let text_batch =TextBatch::encode(&vocab, &captions, max_seq_len, &device);
                Some(text_encoder.forward(text_batch.tokens))
            }else{
                None // ✅ unconditional 
            };

            let loss = model.loss(batch.images.clone(), text_emb.clone());
            let loss_val = loss.clone().into_scalar();

            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);
            let new_model = optimizer.step(learning_rate, model, grads_params);

            ema_diffusion.update(&new_model);

            model = new_model;

            total_loss += loss_val;
            count += 1;

            // 在训练循环里加一个「denoise 验证块」
            if batch_idx == 0 {
                validate_denoise_step(
                    &model,
                    batch.images.clone(),
                    text_emb.clone(),
                    &device,
                );
            }
        }

        if count > 0 {
            println!(
                "Epoch {}: Avg Loss = {:.6}",
                epoch,
                total_loss / count as f32
            );
        }


    }

    // --------------------------------------------------
    // 6. 保存模型（✅ 不爆显存）
    // --------------------------------------------------
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model_dir = PathBuf::from("checkpoints");
    std::fs::create_dir_all(&model_dir)?;

    recorder.record(
        ema_diffusion.model().clone().into_record(),
        model_dir.join("diffusion.mpk"),
    );
    println!("✅ 模型已保存到 checkpoints/diffusion.mpk");

    // --------------------------------------------------
    // 7. 生成 & 保存图片
    // --------------------------------------------------
    println!("\n生成样本...");

    let gen_captions = vec![
        "cat red".to_string(),
        "dog blue".to_string(),
    ];

    let gen_text_batch =
        TextBatch::encode(&vocab, &gen_captions, max_seq_len, &device);

    let gen_text_emb = text_encoder.forward(gen_text_batch.tokens);

    let samples = model.generate(
        2,
        img_size as usize,
        Some(gen_text_emb),
        7.5, // ✅ cfg_scale
        &device,
    );

    println!("生成形状: {:?}", samples.shape());

    save_images(&samples, img_size as u32);

    println!("\n✅ 测试完成！");
    Ok(())
}

// --------------------------------------------------
// ✅ 图片保存（CPU + 安全）
// --------------------------------------------------
fn save_images<B: Backend>(samples: &Tensor<B, 4>, size: u32) {
    use image::{ImageBuffer, Rgb};

    let cpu_device = B::Device::default(); // ✅
    let samples = samples.clone().to_device(&cpu_device);

    let data = samples.into_data();
    let slice = data.as_slice::<f32>().unwrap();

    for i in 0..2 {
        let img = ImageBuffer::from_fn(size, size, |x, y| {
            let idx = i * 3 * (size as usize) * (size as usize)
                + y as usize * (size as usize)
                + x as usize;

            let r = slice[idx];
            let g = slice[idx + (size as usize) * (size as usize)];
            let b = slice[idx + 2 * (size as usize) * (size as usize)];

            Rgb([
                ((r.clamp(-1.0, 1.0) * 127.5 + 127.5) as u8),
                ((g.clamp(-1.0, 1.0) * 127.5 + 127.5) as u8),
                ((b.clamp(-1.0, 1.0) * 127.5 + 127.5) as u8),
            ])
        });

        img.save(format!("sample_{}.png", i)).unwrap();
    }

    println!("✅ 图片已保存为 sample_0.png / sample_1.png");
}

fn validate_denoise_step<B: Backend>(
    model: &diffusion_burn_rs::diffusion::Diffusion<B>,
    images: Tensor<B, 4>,
    text_emb: Option<Tensor<B, 3>>,
    device: &B::Device,
) {
    let batch = images.dims()[0];

    // 1️⃣ 随机时间步
    let t = Tensor::<B, 1>::random(
        [batch],
        Distribution::Uniform(0.0, 99.0),
        device,
    );

    // 2️⃣ 构造噪声
    let noise = Tensor::random_like(&images, Distribution::Normal(0.0, 1.0));

    // 3️⃣ q(x_t | x_0)
    let x_t = model.add_noise(
        images.clone(),
        t.clone(),
        device,
    );

    // 4️⃣ denoise_step_train
    let x_denoised = model.denoise_step_train(
        x_t,
        t,
        text_emb,
        device,
    );

    // 5️⃣ 形状断言
    assert_eq!(
        x_denoised.shape(),
        images.shape(),
        "denoise_step_train 输出形状与输入不一致"
    );

    println!(
        "✅ denoise_step_train 验证通过，shape = {:?}",
        x_denoised.shape()
    );
}