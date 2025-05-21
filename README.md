# nanoGPT
nanoGPT is a minimal, educational implementation of transformer-based language models, focusing on clarity and simplicity. The codebase includes two main components:

## Components

### `bigram.py`
Implements a simple Bigram language model for character-level text generation. This model learns the probability of each character given the previous character, serving as an accessible introduction to language modeling concepts.

### `gpt.py`
Implements a minimal Generative Pre-trained Transformer (GPT) model using PyTorch. This script demonstrates the core ideas behind transformer architectures, including self-attention and positional encoding, in a concise and readable format.

## Dataset
The codebase uses the `tiny-shakespeare` dataset for training and evaluation. This dataset is a small, character-level corpus of Shakespeare's works, commonly used for language modeling experiments.

- `train.bin` contains 1,003,854 tokens
- `val.bin` contains 111,540 tokens

## Model Overview

### `bigram.py`
This script implements a simple Bigram language model for character-level text generation. The Bigram model predicts the next character based solely on the previous character, making it a straightforward baseline for language modeling.

### `gpt.py`
This script implements a minimal Generative Pre-trained Transformer (GPT) model using PyTorch. The GPT model is based on the transformer decoder architecture, which leverages self-attention mechanisms to capture dependencies across sequences. GPT uses only the decoder stack, making it suitable for autoregressive text generation.

#### Text Encoder and Text Decoder
In language models, a text encoder converts input text into a sequence of embeddings (numerical representations), while a text decoder generates output text from these embeddings. In the context of GPT, the model acts as a decoder, generating text by predicting the next token based on previous tokens.

#### Self-Attention and Mathematical Trick
Self-attention allows each token to attend to previous tokens in the sequence, capturing dependencies regardless of distance. A key mathematical trick in GPT's self-attention is the use of the `torch.tril()` function to create a lower triangular mask. This mask ensures that each token can only attend to itself and previous tokens, preventing information from future tokens from being used during training or inference.

## Usage
- To train or sample from the Bigram model, run `bigram.py` with your dataset.
- To train or sample from the GPT model, run `gpt.py` and follow the script's instructions or modify the configuration at the top of the file.