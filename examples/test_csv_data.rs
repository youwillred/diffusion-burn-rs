// examples/test_csv_data.rs
use anyhow::Result;
use burn::data::dataloader::DataLoaderBuilder;
use diffusion_burn_rs::data::loader::{DataConfig, ImageDataset, ImageBatcher};

// 明确指定后端类型
type MyBackend = burn::backend::Wgpu<f32, i32>;

fn main() -> Result<()> {
    let device = burn::backend::wgpu::WgpuDevice::default();
    
    println!("=== CSV 数据加载器测试 ===\n");
    
    // 创建测试数据
    create_test_data()?;
    
    // 配置
    let config = DataConfig {
        csv_path: "input/trainLabels.csv".into(),
        image_root: "input/cifar10_train".into(),
        target_size: (64, 64),
        batch_size: 8,
        shuffle_seed: 42,
        num_workers: 2,
        image_col: 0,
        label_col: 1,
        has_header: true,
    };
    
    println!("📋 配置信息:");
    println!("   CSV: {:?}", config.csv_path);
    println!("   图片目录: {:?}", config.image_root);
    println!("   目标尺寸: {}x{}", config.target_size.0, config.target_size.1);
    println!("   批次大小: {}", config.batch_size);
    println!();
    
    // 创建 dataset
    println!("🚀 创建数据集...");
    let dataset = ImageDataset::new(&config)?;
    
    // 显式指定 batcher 的类型
    let batcher: ImageBatcher<MyBackend> = ImageBatcher::new(device.clone(), config.target_size);
    
    // 构建 DataLoader
    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.shuffle_seed)
        .num_workers(config.num_workers)
        .build(dataset);
    
    // 迭代测试
    println!("🔄 迭代数据...\n");
    for (idx, batch) in dataloader.iter().enumerate() {
        println!("   Batch {}: 图片 {:?}", idx, batch.images.shape());
        if idx >= 2 {
            break;
        }
    }
    
    println!("\n✅ 测试完成！");
    Ok(())
}

/// 创建测试数据
fn create_test_data() -> Result<()> {
    use std::fs;
    use std::io::Write;
    use std::path::Path;
    
    let csv_path = Path::new("input/trainLabels.csv");
    let img_dir = Path::new("input/cifar10_train");
    
    if csv_path.exists() && img_dir.exists() {
        println!("📁 测试数据已存在\n");
        return Ok(());
    }
    
    println!("📁 创建测试数据...");
    fs::create_dir_all(img_dir)?;
    
    // 创建 CSV
    let mut csv_file = fs::File::create(csv_path)?;
    writeln!(csv_file, "image_path,label")?;
    writeln!(csv_file, "cat_001.jpg,cat")?;
    writeln!(csv_file, "cat_002.jpg,cat")?;
    writeln!(csv_file, "dog_001.jpg,dog")?;
    writeln!(csv_file, "dog_002.jpg,dog")?;
    writeln!(csv_file, "bird_001.jpg,bird")?;
    
    // 创建测试图片
    for i in 1..=2 {
        let img = image::RgbImage::from_fn(64, 64, |_, _| image::Rgb([255, 100, 100]));
        img.save(img_dir.join(format!("cat_00{}.jpg", i)))?;
        
        let img = image::RgbImage::from_fn(64, 64, |_, _| image::Rgb([100, 100, 255]));
        img.save(img_dir.join(format!("dog_00{}.jpg", i)))?;
    }
    
    let img = image::RgbImage::from_fn(64, 64, |_, _| image::Rgb([100, 255, 100]));
    img.save(img_dir.join("bird_001.jpg"))?;
    
    println!("   ✅ 创建了 5 张测试图片\n");
    Ok(())
}