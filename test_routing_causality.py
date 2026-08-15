"""Measure the causal effect of routing sharpness with a logit-scale intervention.

Unlike ``plot_routing_sharpness.py``, this experiment does not compare sharpness
across independently trained models.  It evaluates each checkpoint repeatedly
while multiplying only its learned depth-routing logits by a fixed scale.  A
scale above 1 sharpens routing, a scale below 1 flattens it, and scale 1 is the
unaltered checkpoint.

Example:
    python test_routing_causality.py \
        --checkpoint AttnRes=output/attnres/final \
        --checkpoint Delta-AttnRes=output/delta/final \
        --text-file evaluation_text.txt \
        --logit-scales 0.5 0.75 1 1.5 2 \
        --output routing_causal_intervention.png
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "Attention-Residuals"))

from modeling_qwen3_attnres import Qwen3AttnResForCausalLM
from transformers import AutoTokenizer

from plot_routing_sharpness import DEFAULT_TEXTS


@dataclass
class ConditionResult:
    checkpoint: str
    scale: float
    routing_max: float
    nll: float
    perplexity: float
    sample_loss_sums: torch.Tensor
    sample_token_counts: torch.Tensor
    delta_nll: float = 0.0
    delta_nll_ci_low: float = 0.0
    delta_nll_ci_high: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Intervene on depth-routing sharpness and measure next-token "
            "performance on the same examples."
        )
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="[LABEL=]PATH",
        help="Checkpoint to test. May be supplied more than once.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer path or model ID. Defaults to the first checkpoint.",
    )
    parser.add_argument("--text", action="append", default=[])
    parser.add_argument(
        "--text-file",
        default=None,
        help="UTF-8 file containing samples separated by blank lines.",
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--logit-scales",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 1.0, 1.5, 2.0],
        help="Positive multipliers for learned routing logits; must include 1.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Paired sample-bootstrap draws for the delta-NLL 95%% CI (0 disables).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="routing_causal_intervention.png")
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Defaults to the image path with a .csv suffix.",
    )
    return parser.parse_args()


def parse_checkpoint(spec: str) -> tuple[str, str]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        if not label or not path:
            raise ValueError(f"Invalid checkpoint specification: {spec!r}")
        return label, path
    path = spec
    label = Path(path.rstrip(os.sep)).name or path
    return label, path


def load_texts(args: argparse.Namespace) -> list[str]:
    texts = list(args.text)
    if args.text_file:
        content = Path(args.text_file).read_text(encoding="utf-8")
        texts.extend(part.strip() for part in content.split("\n\n") if part.strip())
    return texts or DEFAULT_TEXTS


def load_model(checkpoint: str, device: torch.device) -> Qwen3AttnResForCausalLM:
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = Qwen3AttnResForCausalLM.from_pretrained(checkpoint, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model


def routing_projection_weights(model: Qwen3AttnResForCausalLM) -> list[torch.Tensor]:
    """Return projections represented by ``routing_max_weights``.

    The model's sharpness diagnostic contains the Q/K and MLP route at each
    decoder layer.  Deliberately omit final and V-stream routes so the measured
    manipulation and reported manipulation check refer to the same decisions.
    """
    weights = []
    for layer in model.model.layers:
        weights.extend([layer.attn_res_proj.weight, layer.mlp_res_proj.weight])
    return weights


@contextmanager
def scaled_routing_logits(
    model: Qwen3AttnResForCausalLM, scale: float
) -> Iterator[None]:
    """Temporarily multiply logits without changing learned score directions."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"Routing logit scale must be finite and positive, got {scale}")
    weights = routing_projection_weights(model)
    originals = [weight.detach().clone() for weight in weights]
    try:
        with torch.no_grad():
            for weight in weights:
                weight.mul_(scale)
        yield
    finally:
        with torch.no_grad():
            for weight, original in zip(weights, originals):
                weight.copy_(original)


@torch.inference_mode()
def evaluate_condition(
    model: Qwen3AttnResForCausalLM,
    checkpoint_label: str,
    tokenizer,
    texts: list[str],
    device: torch.device,
    max_length: int,
    scale: float,
) -> ConditionResult:
    sample_loss_sums = []
    sample_token_counts = []
    routing_weighted_sum = 0.0
    routing_token_count = 0

    with scaled_routing_logits(model, scale):
        for sample_index, text in enumerate(texts):
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            else:
                attention_mask = attention_mask.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=0,
                return_routing_max=True,
            )
            routing_values = outputs.routing_max_weights
            if routing_values is None:
                mode = getattr(model.config, "attnres_mode", "unknown")
                raise RuntimeError(
                    f"Routing statistics are not implemented for mode {mode!r}. "
                    "Use a 'block', 'full', 'delta', or 'delta_block' checkpoint."
                )

            target_mask = attention_mask[:, 1:].bool()
            target_count = int(target_mask.sum())
            if target_count == 0:
                raise ValueError(
                    f"Evaluation sample {sample_index} has fewer than two tokens"
                )
            token_losses = F.cross_entropy(
                outputs.logits[:, :-1, :].float().transpose(1, 2),
                input_ids[:, 1:],
                reduction="none",
            )
            loss_sum = token_losses.masked_select(target_mask).sum().item()
            sample_loss_sums.append(loss_sum)
            sample_token_counts.append(target_count)

            routing_tokens = int(attention_mask.sum())
            routing_mean = routing_values.float().nanmean().item()
            routing_weighted_sum += routing_mean * routing_tokens
            routing_token_count += routing_tokens

    loss_sums = torch.tensor(sample_loss_sums, dtype=torch.float64)
    token_counts = torch.tensor(sample_token_counts, dtype=torch.float64)
    nll = (loss_sums.sum() / token_counts.sum()).item()
    perplexity = math.exp(nll) if nll < math.log(torch.finfo(torch.float64).max) else math.inf
    return ConditionResult(
        checkpoint=checkpoint_label,
        scale=scale,
        routing_max=routing_weighted_sum / routing_token_count,
        nll=nll,
        perplexity=perplexity,
        sample_loss_sums=loss_sums,
        sample_token_counts=token_counts,
    )


def add_paired_effects(
    results: list[ConditionResult], bootstrap_samples: int, seed: int
) -> None:
    baseline = next(result for result in results if result.scale == 1.0)
    generator = torch.Generator().manual_seed(seed)
    sample_count = len(baseline.sample_loss_sums)

    for result in results:
        result.delta_nll = result.nll - baseline.nll
        if result.scale == 1.0 or bootstrap_samples == 0:
            result.delta_nll_ci_low = result.delta_nll
            result.delta_nll_ci_high = result.delta_nll
            continue

        if not torch.equal(result.sample_token_counts, baseline.sample_token_counts):
            raise RuntimeError("Paired conditions evaluated different target-token counts")
        indices = torch.randint(
            sample_count,
            (bootstrap_samples, sample_count),
            generator=generator,
        )
        loss_differences = result.sample_loss_sums - baseline.sample_loss_sums
        sampled_difference = loss_differences[indices].sum(dim=1)
        sampled_tokens = baseline.sample_token_counts[indices].sum(dim=1)
        bootstrap_effects = sampled_difference / sampled_tokens
        quantiles = torch.quantile(
            bootstrap_effects, torch.tensor([0.025, 0.975], dtype=torch.float64)
        )
        result.delta_nll_ci_low = quantiles[0].item()
        result.delta_nll_ci_high = quantiles[1].item()


def save_csv(path: Path, results: list[ConditionResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "checkpoint",
                "logit_scale",
                "mean_max_routing_weight",
                "next_token_nll",
                "perplexity",
                "delta_nll_vs_scale_1",
                "delta_nll_ci_95_low",
                "delta_nll_ci_95_high",
                "evaluation_samples",
                "evaluation_tokens",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.checkpoint,
                    result.scale,
                    result.routing_max,
                    result.nll,
                    result.perplexity,
                    result.delta_nll,
                    result.delta_nll_ci_low,
                    result.delta_nll_ci_high,
                    len(result.sample_loss_sums),
                    int(result.sample_token_counts.sum()),
                ]
            )


def plot_results(path: Path, grouped_results: dict[str, list[ConditionResult]]) -> None:
    import matplotlib.pyplot as plt

    fig, (sharpness_ax, performance_ax) = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for checkpoint, results in grouped_results.items():
        scales = [result.scale for result in results]
        positions = range(len(scales))
        sharpness_ax.plot(
            positions,
            [result.routing_max for result in results],
            marker="o",
            linewidth=1.6,
            label=checkpoint,
        )
        effects = [result.delta_nll for result in results]
        effect_line = performance_ax.plot(
            positions,
            effects,
            marker="o",
            linewidth=1.6,
            label=checkpoint,
        )[0]
        performance_ax.fill_between(
            positions,
            [result.delta_nll_ci_low for result in results],
            [result.delta_nll_ci_high for result in results],
            color=effect_line.get_color(),
            alpha=0.15,
        )

    scale_labels = [f"{result.scale:g}×" for result in next(iter(grouped_results.values()))]
    for axis in (sharpness_ax, performance_ax):
        axis.set_xticks(range(len(scale_labels)), scale_labels)
        axis.set_xlabel("Routing-logit scale (1× = checkpoint)")
        axis.grid(True, alpha=0.3)
    sharpness_ax.set_ylabel("Mean maximum routing weight")
    sharpness_ax.set_title("Manipulation check")
    performance_ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    performance_ax.set_ylabel("Δ next-token NLL vs 1× (lower is better)")
    performance_ax.set_title("Causal performance effect (paired 95% CI)")
    sharpness_ax.legend()
    performance_ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def validate_args(args: argparse.Namespace) -> list[float]:
    scales = sorted(set(args.logit_scales))
    if any(not math.isfinite(scale) or scale <= 0 for scale in scales):
        raise ValueError("All --logit-scales must be finite and positive")
    if 1.0 not in scales:
        raise ValueError("--logit-scales must include 1 for the unaltered baseline")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples cannot be negative")
    if args.max_length < 2:
        raise ValueError("--max-length must be at least 2")
    return scales


def main() -> None:
    args = parse_args()
    scales = validate_args(args)
    checkpoints = [parse_checkpoint(spec) for spec in args.checkpoint]
    labels = [label for label, _ in checkpoints]
    if len(labels) != len(set(labels)):
        raise ValueError("Checkpoint labels must be unique")

    device = torch.device(args.device)
    texts = load_texts(args)
    tokenizer_path = args.tokenizer or checkpoints[0][1]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    grouped_results = {}

    for checkpoint_index, (label, checkpoint_path) in enumerate(checkpoints):
        model = load_model(checkpoint_path, device)
        results = []
        for scale in scales:
            result = evaluate_condition(
                model, label, tokenizer, texts, device, args.max_length, scale
            )
            results.append(result)
            print(
                f"{label}: scale={scale:g} max_weight={result.routing_max:.4f} "
                f"NLL={result.nll:.4f} PPL={result.perplexity:.3f}"
            )
        add_paired_effects(
            results,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + checkpoint_index,
        )
        for result in results:
            print(
                f"{label}: scale={result.scale:g} "
                f"delta_NLL={result.delta_nll:+.5f} "
                f"95% CI=[{result.delta_nll_ci_low:+.5f}, "
                f"{result.delta_nll_ci_high:+.5f}]"
            )
        grouped_results[label] = results
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    all_results = [result for results in grouped_results.values() for result in results]
    output_path = Path(args.output)
    csv_path = Path(args.csv_output) if args.csv_output else output_path.with_suffix(".csv")
    plot_results(output_path, grouped_results)
    save_csv(csv_path, all_results)
    print(f"Saved causal intervention plot to {output_path}")
    print(f"Saved intervention results to {csv_path}")


if __name__ == "__main__":
    main()
