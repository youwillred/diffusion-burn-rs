use burn::{
    nn::{Linear, LinearConfig, Embedding, EmbeddingConfig},
    tensor::{backend::Backend, Tensor, Int},
    tensor::activation::gelu,
    module::Module,
};

use crate::tokenizer::Vocab;

pub struct TextBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,
    pub attention_mask: Tensor<B, 2, Int>,
    pub embeddings: Option<Tensor<B, 3>>,
}

impl<B: Backend> TextBatch<B> {
    pub fn encode(
        vocab: &Vocab,
        texts: &[String],
        max_seq_len: usize,
        device: &B::Device,
    ) -> Self {
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

        let tokens = Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(
                token_ids.into_iter().flatten().collect(),
                [batch_size, max_seq_len],
            ),
            device,
        );

        let attention_mask = Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(
                attention_masks.into_iter().flatten().collect(),
                [batch_size, max_seq_len],
            ),
            device,
        );

        Self {
            tokens,
            attention_mask,
            embeddings: None,
        }
    }
}

#[derive(Module, Debug)]
pub struct TextEncoder<B: Backend> {
    embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    linear1: Linear<B>,
    linear2: Linear<B>,
    max_seq_len: usize,
}

impl<B: Backend> TextEncoder<B> {
    pub fn new(
        vocab_size: usize,
        embed_dim: usize,
        max_seq_len: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            embedding: EmbeddingConfig::new(vocab_size, embed_dim).init(device),
            position_embedding: EmbeddingConfig::new(max_seq_len, embed_dim).init(device),
            linear1: LinearConfig::new(embed_dim, embed_dim * 2).init(device),
            linear2: LinearConfig::new(embed_dim * 2, embed_dim).init(device),
            max_seq_len,
        }
    }

    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, seq_len] = tokens.dims();

        let token_emb = self.embedding.forward(tokens);

        let pos = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &token_emb.device())
            .reshape([1, seq_len])
            .repeat(&[batch, 1]);

        let pos_emb = self.position_embedding.forward(pos);

        let x = token_emb + pos_emb;
        let x = self.linear1.forward(x);
        let x = gelu(x);
        self.linear2.forward(x)
    }
}