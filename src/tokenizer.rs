use std::collections::HashMap;
use serde::{Serialize, Deserialize};

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
        self.id_to_token.get(&id).unwrap_or(&self.unk_token)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn pad_id(&self) -> usize {
        self.token_to_id[&self.pad_token]
    }

    pub fn encode(&self, text: &str, max_len: usize) -> Vec<usize> {
        let mut ids = vec![self.token_to_id("[CLS]")];

        for token in tokenize_simple(text).iter().take(max_len - 2) {
            ids.push(self.token_to_id(token));
        }

        ids.push(self.token_to_id("[SEP]"));

        while ids.len() < max_len {
            ids.push(self.pad_id());
        }

        ids
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter(|&&id| id != self.pad_id())
            .map(|&id| self.id_to_token(id).to_string())
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn build_from_texts(texts: &[String], vocab_size: usize) -> Self {
        let mut vocab = Vocab::new();
        let mut counts = HashMap::new();

        for text in texts {
            for token in tokenize_simple(text) {
                *counts.entry(token).or_insert(0) += 1;
            }
        }

        let mut words: Vec<String> = counts.keys().cloned().collect();
        words.sort_by_key(|w| std::cmp::Reverse(counts[w]));

        for w in words.iter().take(vocab_size - 4) {
            vocab.add_token(w.clone());
        }

        vocab
    }
}

impl Default for Vocab {
    fn default() -> Self {
        Self::new()
    }
}

pub fn tokenize_simple(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|s| !s.is_empty())
        .collect()
}