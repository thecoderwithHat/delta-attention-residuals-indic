# Delta Attention Residuals NewInML



## Overview

Delta Attention Residuals replace cumulative hidden states with per-sublayer deltas as routing sources for cross-layer connectivity. The key insight: routing over *what changed* rather than *what accumulated* yields 3x sharper routing and consistently better perplexity across all tested scales (220M--8B).

Two variants:
- **Delta AttnRes**: per-sublayer deltas (2L sources), best quality
- **Delta Block**: block-level deltas (~L/B sources), practical default with minimal overhead

## Repository Structure

```
Attention-Residuals/
  modeling_qwen3_attnres.py   # Core model: Qwen3 + Delta Attention Residuals
train_scratch.py              # From-scratch training (DDP, up to ~1B)
train_scratch_fsdp.py         # From-scratch training (FSDP, 7B+)
train_finetune.py             # Fine-tuning pretrained models
eval_downstream.py            # Downstream evaluation (lm-eval-harness)
run_8b_delta_block.sh         # Launch script for 8B training
```

## Quick Start

### Requirements

```bash
pip install torch transformers datasets wandb
```

### Training from scratch (220M--1B, DDP)

```bash
# FineWeb 2 defaults to the small aai_Latn subset for smoke testing
torchrun --standalone --nproc_per_node=1 train_scratch.py \
    --mode baseline --steps 2 --eval_every 0

# Delta Block (recommended)
torchrun --standalone --nproc_per_node=8 train_scratch.py --mode delta_block --compile_model

# Delta AttnRes (per-sublayer)
torchrun --standalone --nproc_per_node=8 train_scratch.py --mode delta --compile_model
```

Choose another FineWeb 2 language by passing its language-script subset:

```bash
torchrun --standalone --nproc_per_node=8 train_scratch.py \
    --dataset HuggingFaceFW/fineweb-2 --dataset_name hrv_Latn
```

### Routing sharpness plot

Compare the mean maximum depth-routing probability at each layer using two
trained checkpoints:

```bash
python plot_routing_sharpness.py \
    --attnres-checkpoint output/attnres/final \
    --delta-checkpoint output/delta/final \
    --text-file evaluation_text.txt \
    --output routing_sharpness.png
```

The script averages over tokens, input samples, and the attention and MLP
routing decisions in each layer. It saves the plotted values to a CSV file
next to the image. Use an AttnRes `block` or `full` checkpoint and a
Delta-AttnRes `delta` checkpoint.

### Training from scratch (7B+, FSDP)

```bash
torchrun --standalone --nproc_per_node=8 train_scratch_fsdp.py \
    --mode delta_block \
    --hidden_size 4096 --num_layers 36 --num_heads 32 --num_kv_heads 8 \
    --intermediate_size 12288 \
    --batch_size 4 --grad_accum 2 \
    --compile_model --shard_grad_op \
    --steps 10000
```

### Fine-tuning pretrained models

```bash
torchrun --standalone --nproc_per_node=4 train_finetune.py \
    --base_model Qwen/Qwen3-0.6B \
    --mode delta_block \
    --lr 5e-5 --lr_attnres 5e-3 \
    --steps 20000
```

### Indic benchmarks

`eval_downstream.py` includes the official MILU task definitions for all 11
languages, NVIDIA IFEval-Hi with Hindi-aware instruction-following metrics,
GSM8K-Hi for mathematical reasoning, and BFCL-Hi for function calling.

MILU is gated. Accept the dataset terms at
<https://huggingface.co/datasets/ai4bharat/MILU> and set `HF_TOKEN` before
running the published 5-shot setup:

```bash
python eval_downstream.py \
    --model_path Qwen/Qwen3-0.6B \
    --mode baseline \
    --tasks milu \
    --num_fewshot 5 \
    --apply_chat_template
```

Use `--tasks milu_Hindi` for Hindi only. Run the linked IFEval-Hi benchmark
separately because it is zero-shot and generates up to 4,096 tokens:

```bash
python eval_downstream.py \
    --model_path Qwen/Qwen3-0.6B \
    --mode baseline \
    --tasks ifeval_hi \
    --apply_chat_template
```

Run GSM8K-Hi zero-shot with NVIDIA's Hindi chain-of-thought prompt:

```bash
python eval_downstream.py \
    --model_path Qwen/Qwen3-0.6B \
    --mode baseline \
    --tasks gsm8khi \
    --apply_chat_template
```

BFCL-Hi downloads the benchmark's raw JSONL files through the Hugging Face Hub
and uses the tokenizer's tool-aware chat template. It evaluates simple,
multiple, parallel, parallel-multiple, relevance, and irrelevance categories:

```bash
python eval_downstream.py \
    --model_path Qwen/Qwen3-0.6B \
    --mode baseline \
    --tasks bfcl_hi
```

For a short smoke test, add `--limit 10`. Select categories with
`--bfcl_categories simple,parallel`; the default is `all`.





## License

MIT
