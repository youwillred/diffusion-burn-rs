// src/data/text_loader.rs
use crate::tokenizer::{Vocab};
use crate::text_encoder::TextBatch;
use burn::tensor::backend::Backend;
use anyhow::Result;

pub struct TextDataset {
    pub texts: Vec<String>,
    pub labels: Vec<i64>,
    pub vocab: Vocab,
    pub max_seq_len: usize,
}

impl TextDataset {
    pub fn new(texts: Vec<String>, labels: Vec<i64>, max_seq_len: usize) -> Self {
        let vocab = Vocab::build_from_texts(&texts, 10000);
        Self {
            texts,
            labels,
            vocab,
            max_seq_len,
        }
    }
    
    pub fn len(&self) -> usize {
        self.texts.len()
    }
    
    pub fn get_item(&self, idx: usize) -> (String, i64) {
        (self.texts[idx].clone(), self.labels[idx])
    }
    
    pub fn encode_batch<B: Backend>(&self, indices: &[usize], device: &B::Device) -> TextBatch<B> {
        let texts: Vec<String> = indices.iter().map(|&i| self.texts[i].clone()).collect();
        TextBatch::encode(&self.vocab, &texts, self.max_seq_len, device)
    }
}