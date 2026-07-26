# Local lm-eval tasks

This directory contains task definitions loaded by `eval_downstream.py` through
lm-evaluation-harness's `TaskManager(include_path=...)` API.

- `milu` is adapted from the official
  [AI4Bharat/MILU](https://github.com/AI4Bharat/MILU) lm-eval task. The dataset
  is gated and requires an accepted Hugging Face account token.
- `ifevalhi` (also exposed as the `ifeval_hi` group alias) is adapted from the
  evaluator referenced by the
  [nvidia/IFEval-Hi](https://huggingface.co/datasets/nvidia/IFEval-Hi) dataset
  card. It includes Hindi-specific normalization and instruction checkers.
- `gsm8khi` uses NVIDIA's Hindi prompt and exact-match extraction settings for
  [nvidia/GSM8K-Hi](https://huggingface.co/datasets/nvidia/GSM8K-Hi).
- `bfcl_hi` uses a tool-aware evaluator because
  [nvidia/BFCL-Hi](https://huggingface.co/datasets/nvidia/BFCL-Hi) is published
  as raw JSONL files that are incompatible with `datasets.load_dataset`.
  It reports structural function-call accuracy for all six BFCL-Hi categories.

The vendored IFEval checker code retains its upstream Apache-2.0 headers. The
MILU task configuration is sourced from the MIT-licensed benchmark repository.
The GSM8K-Hi task follows NVIDIA's published lm-eval configuration. BFCL-Hi is
downloaded at evaluation time and is not vendored in this repository.
