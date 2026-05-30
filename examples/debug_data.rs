// examples/debug_data.rs - 调试数据加载
use anyhow::Result;
use diffusion_burn_rs::data::loader::{DataConfig, ImageDataset};
use burn::data::dataloader::Dataset;

type MyBackend = burn::backend::Wgpu<f32, i32>;

fn main() -> Result<()> {
    let config = DataConfig {
        csv_path: "input/trainLabels.csv".into(),
        image_root: "input/cifar10_train".into(),
        target_size: (64, 64),
        batch_size: 8,
        shuffle_seed: 42,
        num_workers: 1,
        image_col: 0,
        label_col: 1,
        has_header: true,
    };
    
    println!("📋 配置:");
    println!("   CSV: {:?}", config.csv_path);
    println!("   图片目录: {:?}", config.image_root);
    println!();
    
    // 只创建数据集，不加载图片
    let dataset = ImageDataset::new(&config)?;
    println!("✅ 数据集大小: {} 张图片", dataset.len());
    
    // 检查前5个图片路径是否存在
    println!("\n🔍 检查前5个图片路径:");
    for i in 0..5 {
        if let Some(item) = dataset.get(i) {
            println!("   {}: {:?} -> 存在: {}", i, item.path, item.path.exists());
        }
    }
    
    Ok(())
}