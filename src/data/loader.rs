// src/data/loader.rs - 简洁版，不封装复杂类型
use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::Dataset,
    },
    tensor::{Tensor, backend::Backend, Int, TensorData},
};
use anyhow::{Result, Context};
use std::path::{Path, PathBuf};
use std::collections::HashMap;



// ============================================================
// 辅助函数：支持多种图片格式
// ============================================================
/// 查找图片文件，支持多种扩展名
fn find_image_file(root: &Path, base_name: &str) -> Option<PathBuf> {
    let extensions = ["png", "jpg", "jpeg", "bmp", "tiff", "webp"];
    for ext in extensions {
        let path = root.join(format!("{}.{}", base_name, ext));
        if path.exists() {
            return Some(path);
        }
    }
    None
}

// ============================================================
// 配置
// ============================================================
#[derive(Debug, Clone)]
pub struct DataConfig {
    pub csv_path: PathBuf,
    pub image_root: PathBuf,
    pub target_size: (u32, u32),
    pub batch_size: usize,
    pub shuffle_seed: u64,
    pub num_workers: usize,
    pub image_col: usize,
    pub label_col: usize,
    pub has_header: bool,
    pub max_samples: Option<usize>,  // 新增：最大样本数限制
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            csv_path: "data/train.csv".into(),
            image_root: "data/images".into(),
            target_size: (224, 224),
            batch_size: 32,
            shuffle_seed: 42,
            num_workers: 4,
            image_col: 0,
            label_col: 1,
            has_header: true,
            max_samples: None,  // 默认不限，保持原有行为
        }
    }
}

// ============================================================
// 数据项
// ============================================================
#[derive(Clone, Debug)]
pub struct ImageItem {
    pub path: PathBuf,
    pub label: i64,
}

// ============================================================
// 数据集实现
// ============================================================
pub struct ImageDataset {
    items: Vec<ImageItem>,
    num_classes: usize,
}

impl ImageDataset {
    pub fn new(config: &DataConfig) -> Result<Self> {
        let mut items = Vec::new();
        let mut label_to_idx = HashMap::new();
        let mut next_label = 0;
        
        let file = std::fs::File::open(&config.csv_path)?;
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(config.has_header)
            .from_reader(file);
        
        for result in reader.records() {
            let record = result?;
            if record.len() <= config.image_col.max(config.label_col) {
                continue;
            }
            
            let file_name = &record[config.image_col];
            
            // ✅ 使用 find_image_file 支持多种格式，并检查文件是否存在
            let image_path = match find_image_file(&config.image_root, file_name) {
                Some(path) => path,
                None => {
                    // 文件不存在，跳过这条记录（不计数）
                    continue;
                }
            };
            
            let label_name = record[config.label_col].to_string();
            
            let label = *label_to_idx.entry(label_name).or_insert_with(|| {
                let idx = next_label;
                next_label += 1;
                idx
            });
            
            items.push(ImageItem {
                path: image_path,
                label,
            });
            
            // 添加限制：只加载 max_samples 张存在的图片
            if let Some(max) = config.max_samples {
                if items.len() >= max {
                    break;
                }
            }
        }
        
        println!("✅ 加载 {} 张图片，{} 个类别", items.len(), next_label);
        Ok(Self { items, num_classes: next_label as usize })
    }
    
    pub fn num_classes(&self) -> usize { self.num_classes }
    
    // 新增：获取数据集大小
    pub fn len(&self) -> usize { self.items.len() }
}

impl Dataset<ImageItem> for ImageDataset {
    fn len(&self) -> usize { self.items.len() }
    fn get(&self, index: usize) -> Option<ImageItem> {
        self.items.get(index).cloned()
    }
}

// ============================================================
// Batcher
// ============================================================
#[derive(Clone)]
pub struct ImageBatcher<B: Backend> {
    device: B::Device,
    target_size: (u32, u32),
}

impl<B: Backend> ImageBatcher<B> {
    pub fn new(device: B::Device, target_size: (u32, u32)) -> Self {
        Self { device, target_size }
    }
    
    fn load_image(&self, path: &Path) -> Result<Tensor<B, 3>> {
        let img = image::open(path)?;
        let img = img.resize_exact(
            self.target_size.0,
            self.target_size.1,
            image::imageops::FilterType::Lanczos3,
        );
        let rgb = img.to_rgb8();
        let (h, w) = (self.target_size.1 as usize, self.target_size.0 as usize);
        let mut data = Vec::with_capacity(3 * h * w);
        
        for channel in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let p = rgb.get_pixel(x as u32, y as u32);
                    let v = match channel {
                        0 => p[0] as f32 / 255.0,
                        1 => p[1] as f32 / 255.0,
                        2 => p[2] as f32 / 255.0,
                        _ => unreachable!(),
                    };
                    data.push(v);
                }
            }
        }
        
        Ok(Tensor::<B, 3>::from_data(TensorData::new(data, [3, h, w]), &self.device))
    }
}

#[derive(Clone, Debug)]
pub struct ImageBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, ImageItem, ImageBatch<B>> for ImageBatcher<B> {
    fn batch(&self, items: Vec<ImageItem>, device: &B::Device) -> ImageBatch<B> {
        let mut images = Vec::new();
        let mut targets = Vec::new();
        let mut failed_count = 0;
        
        for item in items {
            match self.load_image(&item.path) {
                Ok(img) => {
                    images.push(img.unsqueeze());  // [C, H, W] -> [1, C, H, W]
                    targets.push(Tensor::<B, 1, Int>::from_data(
                        TensorData::new(vec![item.label], [1]),
                        device,
                    ));
                }
                Err(e) => {
                    failed_count += 1;
                    if failed_count <= 5 {  // 只打印前5个错误
                        eprintln!("⚠️ 跳过图片 {:?}: {}", item.path, e);
                    }
                }
            }
        }
        
        if failed_count > 5 {
            eprintln!("⚠️ 还有 {} 张图片加载失败", failed_count - 5);
        }
        
        if images.is_empty() {
            // 如果所有图片都加载失败，返回空 batch（但这种情况应该避免）
            panic!("没有成功加载任何图片，请检查图片路径和格式");
        }
        
       // println!("✅ 成功加载 {} / {} 张图片", images.len(), images.len() + failed_count);
        
        ImageBatch {
            images: Tensor::cat(images, 0),
            targets: Tensor::cat(targets, 0),
        }
    }
}



// 使用方法
// use my_app::data::loader::{DataConfig, ImageDataset, ImageBatcher};
// use burn::data::dataloader::DataLoaderBuilder;

// fn main() -> Result<()> {
//     let device = WgpuDevice::default();
//     let config = DataConfig {
//         csv_path: "your_data/train.csv".into(),
//         image_root: "your_data/images".into(),
//         target_size: (224, 224),
//         batch_size: 32,
//         ..Default::default()
//     };
    
//     let dataset = ImageDataset::new(&config)?;
//     let batcher: ImageBatcher<MyBackend> = ImageBatcher::new(device, config.target_size);
    
//     let dataloader = DataLoaderBuilder::new(batcher)
//         .batch_size(config.batch_size)
//         .shuffle(42)
//         .num_workers(4)
//         .build(dataset);
    
//     // 训练循环
//     for batch in dataloader.iter() {
//         // batch.images: [batch, 3, H, W]
//         // batch.targets: [batch]
//         let output = model.forward(batch.images);
//         // ...
//     }
// }