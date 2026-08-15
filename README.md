# Delta Attention Residuals

Official code for **"Delta Attention Residuals: Per-Sublayer Sources for Cross-Layer Information Flow"**.

Cheng Luo, Zefan Cai, Junjie Hu

[[Paper]](https://github.com/wdlctc/delta-attention-residuals-arxiv)

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

The descriptive plot compares the mean maximum depth-routing probability at
each layer using two trained checkpoints:

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

This cross-checkpoint comparison is correlational. To test whether routing
sharpness itself affects quality, run the within-checkpoint logit-scale
intervention:

```bash
python test_routing_causality.py \
    --checkpoint AttnRes=output/attnres/final \
    --checkpoint Delta-AttnRes=output/delta/final \
    --text-file evaluation_text.txt \
    --logit-scales 0.5 0.75 1 1.5 2 \
    --output routing_causal_intervention.png
```

The intervention multiplies only the learned Q/K and MLP depth-routing logits.
Values above `1` sharpen their softmax distributions and values below `1`
flatten them; the learned scoring directions and all other parameters remain
fixed. The script measures the achieved mean maximum weight and next-token
NLL/perplexity on the same examples, then reports paired sample-bootstrap
confidence intervals for the NLL change relative to the untouched `1x`
checkpoint. Use a held-out text file with many independent documents for
meaningful intervals. Supported modes are `block`, `full`, `delta`, and
`delta_block`.

Flattening that reliably hurts while moderate sharpening reliably helps is
evidence for a local causal effect of sharpness in that checkpoint. If both
directions hurt, the checkpoint is instead calibrated near its learned routing
temperature; that result does not support the stronger claim that sharpness by
itself explains the quality gap between architectures.

### Routing source-selection intervention

Logit scaling changes routing confidence but usually preserves the winning
source. Test whether the learned source identity matters by comparing it with
probability-preserving source permutations and exactly uniform routing:

```bash
python test_routing_selection.py \
    --checkpoint Delta-AttnRes=output/delta/final \
    --text-file evaluation_text.txt \
    --permutation-seeds 0 1 2 3 4 \
    --output routing_source_selection.png
```

The script first captures every learned routing distribution for each document.
Each permutation seed then replays those exact probabilities while applying a
deterministic derangement at every routing site, so none remains attached to its
original source. This cache-and-replay design prevents upstream perturbations
from indirectly changing later routing probabilities. The script averages the
permutation effects across seeds and reports paired document-bootstrap
confidence intervals relative to learned routing. Uniform routing sets every
source probability to exactly `1/N`.

If permutation hurts while max-weight remains similar, performance depends on
selecting the correct historical representation rather than merely producing a
sharp distribution. If uniform routing also hurts, the learned depth-selection
policy is useful. As with the sharpness experiment, supported modes are
`block`, `full`, `delta`, and `delta_block`.

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
    --pretrained Qwen/Qwen3-0.6B \
    --mode delta_block \
    --lr 5e-5 --lr_attnres 5e-3 \
    --steps 20000 \
    --gradient_checkpointing
```

Fine-tuning checkpoints include the model, optimizer, learning-rate scheduler,
per-rank RNG state, and streaming-dataset position. Resume from the latest
checkpoint in `--out_dir` with a bare `--resume`, or select one explicitly:

```bash
torchrun --standalone --nproc_per_node=4 train_finetune.py \
    --pretrained Qwen/Qwen3-0.6B \
    --mode delta_block \
    --steps 20000 \
    --out_dir ./output/my-run \
    --resume ./output/my-run/step-10000
```

Resume with the same process count and optimization flags used to create the
checkpoint. `--steps` remains the final target step, not an additional count.
Intermediate `step-*` checkpoints are retained if training is interrupted. After
successful completion, the script saves `final/` and removes the numbered
checkpoints so only the final checkpoint remains.

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

## Results & W&B Runs

The exact W&B run for every paper experiment is listed in [`WANDB_RUNS.md`](./WANDB_RUNS.md) (training/validation curves, configs, and system metrics). Project: <https://wandb.ai/wdlctc_abr/attention-residual-h100>.

### From-Scratch Training (10K steps, FineWeb-Edu)

| Scale | Baseline | AttnRes | Delta Block | Delta AttnRes |
|-------|----------|---------|-------------|---------------|
| 220M  | 38.71    | 37.39   | **37.08**   | **36.83**     |
| 533M  | 32.00    | 31.75   | **31.16**   | **31.05**     |
| 1044M | 29.70    | 31.76   | **29.19**   | **29.13**     |

### Fine-tuning Qwen3-0.6B (downstream avg accuracy)

| Baseline FT | AttnRes | Delta Block |
|-------------|---------|-------------|
| 55.0%       | 54.1%   | **55.6%**   |

## Citation

```bibtex
@article{luo2026delta,
  title={Delta Attention Residuals},
  author={Luo, Cheng and Cai, Zefan and Hu, Junjie},
  year={2026}
}
```

## License

MIT
