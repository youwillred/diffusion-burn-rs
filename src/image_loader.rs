// image_loader.rs - 保存这个文件，永久复用！
use burn::{
    data::{
        dataloader::{DataLoaderBuilder, batcher::Batcher},
        dataset::{Dataset, vision::ImageFolderDataset},
    },
    tensor::{Tensor, backend::Backend, Int, TensorData},
};
use anyhow::Result;

/// 图片批次数据结构
#[derive(Clone, Debug)]
pub struct ImageBatch<B: Backend> {
    pub images: Tensor<B, 4>,  // [batch, 3, H, W]
    pub targets: Tensor<B, 1, Int>,  // [batch]
}

/// 图片加载器（适用于 ImageFolderDataset 格式）
#[derive(Clone, Debug)]
pub struct ImageBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> ImageBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

// PixelDepth 转换（处理所有变体）
fn pixel_to_f32(pixel: &burn::data::dataset::vision::PixelDepth) -> f32 {
    match pixel {
        burn::data::dataset::vision::PixelDepth::U8(v) => *v as f32,
        burn::data::dataset::vision::PixelDepth::U16(v) => *v as f32,
        burn::data::dataset::vision::PixelDepth::F32(v) => *v,
    }
}

impl<B: Backend> Batcher<B, burn::data::dataset::vision::ImageDatasetItem, ImageBatch<B>> 
    for ImageBatcher<B> 
{
    fn batch(&self, items: Vec<burn::data::dataset::vision::ImageDatasetItem>, device: &B::Device) -> ImageBatch<B> {
        let mut images = Vec::new();
        let mut targets = Vec::new();

        for item in items {
            // 像素转换
            let data: Vec<f32> = item.image.iter().map(pixel_to_f32).collect();
            let tensor_data = TensorData::new(
                data, 
                [3, item.image_height as usize, item.image_width as usize]
            );
            let tensor = Tensor::<B, 3>::from_data(tensor_data, device) / 255.0;
            images.push(tensor.unsqueeze());

            // 标签转换
            let label = match item.annotation {
                burn::data::dataset::vision::Annotation::Label(idx) => idx as i32,
                _ => 0,
            };
            targets.push(Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![label], [1]),
                device,
            ));
        }

        ImageBatch {
            images: Tensor::cat(images, 0),
            targets: Tensor::cat(targets, 0),
        }
    }
}

/// 一行代码创建数据加载器
pub fn create_image_loader<B: Backend>(
    data_path: &str,
    batch_size: usize,
    device: B::Device,
) -> Result<DataLoader<ImageBatch<B>, ImageBatch<B>>> {
    let dataset = ImageFolderDataset::new_classification(data_path)?;
    let batcher = ImageBatcher::new(device.clone());
    
    Ok(DataLoaderBuilder::new(batcher)
        .batch_size(batch_size)
        .shuffle(42)
        .num_workers(4)
        .build(dataset))
}



// 下面是使用方法

//// main.rs 或 train.rs
// use my_burn_app::image_loader::create_image_loader;  // 导入你的模板

// fn main() -> Result<()> {
//     let device = burn::backend::wgpu::WgpuDevice::default();
    
//     // 就这一行！创建数据加载器
//     let dataloader = create_image_loader::<burn::backend::Wgpu<f32, i32>>(
//         "input/cifar10_train",  // 图片文件夹路径
//         32,                      // 批次大小
//         device,
//     )?;
    
//     // 训练循环
//     for batch in dataloader.iter() {
//         println!("Batch: {:?}", batch.images.shape());
//     }
    
//     Ok(())
// }