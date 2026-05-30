#[cfg(never)]
// src/text_encoder.rs
use burn::{
    nn::{Linear, LinearConfig, Embedding, EmbeddingConfig},
    tensor::{backend::Backend, Tensor, Int},
    tensor::activation::gelu,
    module::Module,  // 合并到这里
};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

// ============================================================
// 词表 (Vocab)
// ============================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocab {
    token_to_id: HashMap<String, usize>,
    id_to_token: HashMap<usize, String>,
    vocab_size: usize,
    pad_token: String,
    unk_token: String,
}

impl Vocab {
    pub fn new() -> Self {
        let mut vocab = Self {
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            vocab_size: 0,
            pad_token: "[PAD]".to_string(),
            unk_token: "[UNK]".to_string(),
        };
        
        vocab.add_token("[PAD]".to_string());
        vocab.add_token("[UNK]".to_string());
        vocab.add_token("[CLS]".to_string());
        vocab.add_token("[SEP]".to_string());
        
        vocab
    }
    
    pub fn add_token(&mut self, token: String) -> usize {
        if let Some(&id) = self.token_to_id.get(&token) {
            return id;
        }
        let id = self.vocab_size;
        self.token_to_id.insert(token.clone(), id);
        self.id_to_token.insert(id, token);
        self.vocab_size += 1;
        id
    }
    
    pub fn token_to_id(&self, token: &str) -> usize {
        *self.token_to_id.get(token).unwrap_or(&self.token_to_id[&self.unk_token])
    }
    
    pub fn id_to_token(&self, id: usize) -> &str {
        self.id_to_token.get(&id).map(|s| s.as_str()).unwrap_or(&self.unk_token)
    }
    
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    
    pub fn pad_id(&self) -> usize {
        self.token_to_id[&self.pad_token]
    }
    
    pub fn encode(&self, text: &str, max_len: usize) -> Vec<usize> {
        let tokens: Vec<String> = tokenize_simple(text);
        let mut ids = vec![self.token_to_id("[CLS]")];
        
        for token in tokens.iter().take(max_len - 2) {
            ids.push(self.token_to_id(token));
        }
        
        ids.push(self.token_to_id("[SEP]"));
        
        while ids.len() < max_len {
            ids.push(self.pad_id());
        }
        
        ids
    }
    
    pub fn decode(&self, ids: &[usize]) -> String {
        let tokens: Vec<String> = ids.iter()
            .filter(|&&id| id != self.pad_id())
            .map(|&id| self.id_to_token(id).to_string())
            .collect();
        tokens.join(" ")
    }
    
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
    
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let vocab = serde_json::from_str(&json)?;
        Ok(vocab)
    }
    
    pub fn build_from_texts(texts: &[String], vocab_size: usize) -> Self {
        let mut vocab = Vocab::new();
        let mut word_counts = HashMap::new();
        
        for text in texts {
            for token in tokenize_simple(text) {
                *word_counts.entry(token).or_insert(0) += 1;
            }
        }
        
        let mut words: Vec<String> = word_counts.keys().cloned().collect();
        words.sort_by_key(|w| std::cmp::Reverse(word_counts[w]));
        
        for word in words.iter().take(vocab_size - 4) {
            vocab.add_token(word.clone());
        }
        
        vocab
    }
}

impl Default for Vocab {
    fn default() -> Self {
        Self::new()
    }
}

fn tokenize_simple(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

// ============================================================
// 文本嵌入模型
// ============================================================
#[derive(Module, Debug)]
pub struct TextEncoder<B: Backend> {
    embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    linear1: Linear<B>,
    linear2: Linear<B>,
    embed_dim: usize,
    max_seq_len: usize,
}

impl<B: Backend> TextEncoder<B> {
    pub fn new(vocab_size: usize, embed_dim: usize, max_seq_len: usize, device: &B::Device) -> Self {
        let embedding = EmbeddingConfig::new(vocab_size, embed_dim)
            .init(device);
        
        let position_embedding = EmbeddingConfig::new(max_seq_len, embed_dim)
            .init(device);
        
        let linear1 = LinearConfig::new(embed_dim, embed_dim * 2)
            .init(device);
        let linear2 = LinearConfig::new(embed_dim * 2, embed_dim)
            .init(device);
        
        Self {
            embedding,
            position_embedding,
            linear1,
            linear2,
            embed_dim,
            max_seq_len,
        }
    }
    
    pub fn forward(&self, tokens: Tensor<B, 2, Int>, device: &B::Device) -> Tensor<B, 3> {
        let [batch_size, seq_len] = tokens.dims();
        
        // 词嵌入
        let token_embeds = self.embedding.forward(tokens);
        
        // 位置嵌入
        let positions = Tensor::<B, 1, Int>::arange(0..seq_len as i64, device)
            .reshape([1, seq_len]);
        
        let positions = positions.repeat(&[batch_size, 1]);
        let pos_embeds = self.position_embedding.forward(positions);
        
        // 合并
        let embeds = token_embeds + pos_embeds;
        
        // MLP 处理
        let embeds = self.linear1.forward(embeds);
        let embeds = gelu(embeds);
        let embeds = self.linear2.forward(embeds);
        
        embeds
    }
}

// ============================================================
// 批处理
// ============================================================
pub struct TextBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,
    pub attention_mask: Tensor<B, 2, Int>,
    pub embeddings: Option<Tensor<B, 3>>,
}

impl<B: Backend> TextBatch<B> {
    pub fn encode(vocab: &Vocab, texts: &[String], max_seq_len: usize, device: &B::Device) -> Self {
        let batch_size = texts.len();
        let mut token_ids = vec![vec![0; max_seq_len]; batch_size];
        let mut attention_masks = vec![vec![0; max_seq_len]; batch_size];
        
        for (i, text) in texts.iter().enumerate() {
            let ids = vocab.encode(text, max_seq_len);
            for (j, &id) in ids.iter().enumerate() {
                token_ids[i][j] = id as i64;
                attention_masks[i][j] = 1;
            }
        }
        
        let flat_tokens: Vec<i64> = token_ids.into_iter().flatten().collect();
        let flat_masks: Vec<i64> = attention_masks.into_iter().flatten().collect();
        
        let tokens = Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(flat_tokens, [batch_size, max_seq_len]),
            device,
        );
        
        let attention_mask = Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(flat_masks, [batch_size, max_seq_len]),
            device,
        );
        
        Self {
            tokens,
            attention_mask,
            embeddings: None,
        }
    }
}