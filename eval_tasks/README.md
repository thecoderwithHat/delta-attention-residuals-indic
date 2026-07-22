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

The vendored IFEval checker code retains its upstream Apache-2.0 headers. The
MILU task configuration is sourced from the MIT-licensed benchmark repository.
