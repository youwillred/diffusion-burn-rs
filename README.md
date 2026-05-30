# Diffusion-Burn-RS

[![Rust](https://img.shields.io/badge/rust-2024+-blue.svg)](https://www.rust-lang.org/)
[![Burn](https://img.shields.io/badge/burn-0.20-orange.svg)](https://github.com/tracel-ai/burn)
[![License](https://img.shields.io/badge/license-Apache2.0-green.svg)](LICENSE)

> 基于 Rust + Burn 框架实现的文本条件扩散模型。从零训练，支持文本引导的图像生成。

## 项目定位

- 纯 Rust 实现，基于 Burn 0.20 框架
- 支持文本条件生成（Text-to-Image）
- 内置 EMA（指数移动平均）提升生成质量
- 完整的训练/推理/数据加载链路

## 快速开始

### 1. 环境准备

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default nightly

### 2. 克隆项目

git clone https://github.com/youwillred/diffusion-burn-rs.git
cd diffusion-burn-rs
unzip input/cifar10_train.zip -d input/

### 3. 运行训练

cargo run --example train_with_text --release

训练输出示例：

✅ 加载 100 张图片，10 个类别
Epoch 1: Avg Loss = 0.809446
Epoch 2: Avg Loss = 0.373010
✅ 模型已保存到 checkpoints/diffusion.mpk
✅ 图片已保存为 sample_0.png / sample_1.png

## 命令大全

cargo run --example train_with_text : 完整训练（推荐）
cargo run --example test_model : 测试已训练模型
cargo run --example test_csv_data : 验证数据加载

## 许可证

Apache 2.0

## 致谢

Burn - Rust 深度学习框架
CIFAR-10 - 数据集
