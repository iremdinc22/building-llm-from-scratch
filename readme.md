# 🧠 Building LLM from Scratch

This project is an end-to-end implementation of a GPT-style Large Language Model built entirely from scratch.  
The goal is to deeply understand how modern LLMs work by implementing every core component manually, from tokenization to text generation.

---

## 🚀 Project Overview

Instead of relying on high-level libraries, this project reconstructs the full LLM pipeline step by step, covering:

- Tokenization
- Embedding layers
- Self-attention mechanism
- Transformer architecture
- Training loop
- Inference and text generation

The project aims to bridge the gap between theoretical knowledge and practical implementation of LLM systems.

---

## 🏗️ Architecture

The model follows a simplified GPT-style pipeline:

Text → Tokenizer → Embedding → Multi-Head Attention → Transformer Blocks → Output Layer → Sampling → Generated Text


### 🔍 Component Breakdown

- **Tokenizer**  
  Converts raw text into tokens (word/subword level).

- **Embedding Layer**  
  Maps tokens into dense vector representations.

- **Multi-Head Self-Attention**  
  Captures relationships between tokens in a sequence.

- **Transformer Blocks**  
  Stacked layers including attention + feed-forward (MLP) + residual connections + normalization.

- **Output Layer**  
  Projects hidden states to vocabulary logits.

- **Sampling**  
  Generates text using strategies like temperature and top-k.

---

### 📊 Architecture Diagram (Mermaid)

```mermaid
flowchart LR
    A[Raw Text] --> B[Tokenizer]
    B --> C[Token IDs]
    C --> D[Embedding Layer]
    D --> E[Positional Encoding]
    E --> F[Transformer Blocks]
    F --> G[Multi-Head Attention]
    G --> H[Feed Forward Network]
    H --> I[Output Logits]
    I --> J[Sampling (Top-K / Temperature)]
    J --> K[Generated Text]