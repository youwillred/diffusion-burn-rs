// src/train_config.rs
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct TrainConfig {
    // 数据
    pub data_dir: PathBuf,
    pub img_size: u32,
    pub batch_size: usize,

    // 模型
    pub timesteps: usize,
    pub cond_dim: usize,       // ✅ 原来是 text_embed_dim
    pub max_seq_len: usize,

    // 训练
    pub learning_rate: f64,
    pub num_epochs: usize,
    pub save_every: usize,

    // 条件控制
    pub use_text: bool,        // ✅ 新增

    // 保存
    pub checkpoint_dir: PathBuf,
    pub vocab_path: Option<PathBuf>, // ✅ 更安全
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("data"),
            img_size: 64,
            batch_size: 8,
            timesteps: 100,
            cond_dim: 256,
            max_seq_len: 16,
            learning_rate: 1e-4,
            num_epochs: 20,
            save_every: 5,
            use_text: false,   // ✅ 当前阶段
            checkpoint_dir: PathBuf::from("checkpoints"),
            vocab_path: None,
        }
    }
}