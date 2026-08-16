"""Sweep the post-softmax strength of Delta-AttnRes retrieval branches.

The learned routing logits, softmax probabilities, source ordering, and model
parameters are not modified.  For each Delta routing operation, only the
additive merge changes from ``residual + routed`` to
``residual + branch_scale * routed``.

Example:
    python test_routing_branch_strength.py \
        --checkpoint Delta-AttnRes=output/delta/final \
        --text-file hindi_eval.txt \
        --branch-scales 0 .25 .5 .75 1 1.25 1.5 2 \
        --output routing_branch_strength.png
"""

from __future__ import annotations

import argparse
import csv
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F

from test_routing_causality import (
    ConditionResult,
    add_paired_effects,
    load_model,
    load_texts,
    parse_checkpoint,
)
from transformers import AutoTokenizer


DEFAULT_BRANCH_SCALES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


@dataclass
class BranchStrengthResult(ConditionResult):
    mean_routed_l2: float = 0.0
    mean_scaled_routed_ratio: float = 0.0
    layer_routed_l2: tuple[float, ...] = ()
    layer_scaled_routed_ratio: tuple[float, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scale only the post-softmax routed contribution in Delta-AttnRes "
            "and measure paired next-token performance."
        )
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="[LABEL=]PATH",
        help="Delta or Delta-Block checkpoint to test. May be supplied repeatedly.",
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
        "--branch-scales",
        type=float,
        nargs="+",
        default=DEFAULT_BRANCH_SCALES,
        help="Nonnegative routed-branch multipliers; must include 1.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Paired sample-bootstrap draws for the delta-NLL 95%% CI (0 disables).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="routing_branch_strength.png")
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Defaults to the image path with a .csv suffix.",
    )
    parser.add_argument(
        "--layer-csv-output",
        default=None,
        help="Defaults to <csv-output stem>_layers.csv.",
    )
    return parser.parse_args()


@contextmanager
def scaled_routing_branch(model, scale: float) -> Iterator[None]:
    """Temporarily scale additive routes without touching checkpoint tensors."""
    if not math.isfinite(scale) or scale < 0:
        raise ValueError(f"Routing branch scale must be finite and nonnegative, got {scale}")

    layers = list(model.model.layers)
    old_values = [getattr(layer, "routing_branch_scale", 1.0) for layer in layers]
    try:
        for layer in layers:
            layer.routing_branch_scale = scale
        yield
    finally:
        for layer, old_value in zip(layers, old_values):
            layer.routing_branch_scale = old_value


@torch.inference_mode()
def evaluate_condition(
    model,
    checkpoint_label: str,
    tokenizer,
    texts: list[str],
    device: torch.device,
    max_length: int,
    scale: float,
) -> BranchStrengthResult:
    sample_loss_sums = []
    sample_token_counts = []
    routing_weighted_sum = 0.0
    routing_token_count = 0
    layer_routed_l2_sum = None
    layer_scaled_ratio_sum = None

    with scaled_routing_branch(model, scale):
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
                return_routing_magnitude=True,
            )
            routing_values = outputs.routing_max_weights
            routed_l2_values = outputs.routing_routed_l2
            scaled_ratio_values = outputs.routing_scaled_ratio
            if (
                routing_values is None
                or routed_l2_values is None
                or scaled_ratio_values is None
            ):
                mode = getattr(model.config, "attnres_mode", "unknown")
                raise RuntimeError(
                    f"Routing diagnostics are not implemented for mode {mode!r}."
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
            sample_loss_sums.append(
                token_losses.masked_select(target_mask).sum().item()
            )
            sample_token_counts.append(target_count)

            routing_tokens = int(attention_mask.sum())
            routing_weighted_sum += (
                routing_values.float().nanmean().item() * routing_tokens
            )
            routing_token_count += routing_tokens
            routed_l2_values = routed_l2_values.detach().float().cpu()
            scaled_ratio_values = scaled_ratio_values.detach().float().cpu()
            if layer_routed_l2_sum is None:
                layer_routed_l2_sum = torch.zeros_like(routed_l2_values)
                layer_scaled_ratio_sum = torch.zeros_like(scaled_ratio_values)
            if routed_l2_values.shape != layer_routed_l2_sum.shape:
                raise RuntimeError("Routing layer count changed between samples")
            layer_routed_l2_sum += routed_l2_values * routing_tokens
            layer_scaled_ratio_sum += scaled_ratio_values * routing_tokens

    loss_sums = torch.tensor(sample_loss_sums, dtype=torch.float64)
    token_counts = torch.tensor(sample_token_counts, dtype=torch.float64)
    nll = (loss_sums.sum() / token_counts.sum()).item()
    perplexity = (
        math.exp(nll)
        if nll < math.log(torch.finfo(torch.float64).max)
        else math.inf
    )
    layer_routed_l2 = layer_routed_l2_sum / routing_token_count
    layer_scaled_ratio = layer_scaled_ratio_sum / routing_token_count
    return BranchStrengthResult(
        checkpoint=checkpoint_label,
        scale=scale,
        routing_max=routing_weighted_sum / routing_token_count,
        nll=nll,
        perplexity=perplexity,
        sample_loss_sums=loss_sums,
        sample_token_counts=token_counts,
        mean_routed_l2=layer_routed_l2.mean().item(),
        mean_scaled_routed_ratio=layer_scaled_ratio.mean().item(),
        layer_routed_l2=tuple(layer_routed_l2.tolist()),
        layer_scaled_routed_ratio=tuple(layer_scaled_ratio.tolist()),
    )


def save_csv(path: Path, results: list[BranchStrengthResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline_by_checkpoint = {
        result.checkpoint: result
        for result in results
        if result.scale == 1.0
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "checkpoint",
                "routing_branch_scale",
                "mean_max_routing_weight",
                "max_routing_weight_change_vs_scale_1",
                "mean_routed_l2",
                "mean_scaled_routed_l2_over_residual_l2",
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
            baseline = baseline_by_checkpoint[result.checkpoint]
            writer.writerow(
                [
                    result.checkpoint,
                    result.scale,
                    result.routing_max,
                    result.routing_max - baseline.routing_max,
                    result.mean_routed_l2,
                    result.mean_scaled_routed_ratio,
                    result.nll,
                    result.perplexity,
                    result.delta_nll,
                    result.delta_nll_ci_low,
                    result.delta_nll_ci_high,
                    len(result.sample_loss_sums),
                    int(result.sample_token_counts.sum()),
                ]
            )


def save_layer_csv(path: Path, results: list[BranchStrengthResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "checkpoint",
                "routing_branch_scale",
                "layer",
                "mean_routed_l2",
                "mean_scaled_routed_l2_over_residual_l2",
            ]
        )
        for result in results:
            for layer, (routed_l2, scaled_ratio) in enumerate(
                zip(result.layer_routed_l2, result.layer_scaled_routed_ratio)
            ):
                writer.writerow(
                    [result.checkpoint, result.scale, layer, routed_l2, scaled_ratio]
                )


def plot_results(path: Path, grouped: dict[str, list[BranchStrengthResult]]) -> None:
    import matplotlib.pyplot as plt

    fig, (routing_ax, magnitude_ax, performance_ax) = plt.subplots(
        1, 3, figsize=(15.8, 4.4)
    )
    for checkpoint, results in grouped.items():
        scales = [result.scale for result in results]
        routing_ax.plot(
            scales,
            [result.routing_max for result in results],
            marker="o",
            linewidth=1.6,
            label=checkpoint,
        )
        magnitude_ax.plot(
            scales,
            [result.mean_scaled_routed_ratio for result in results],
            marker="o",
            linewidth=1.6,
            label=checkpoint,
        )
        effect_line = performance_ax.plot(
            scales,
            [result.delta_nll for result in results],
            marker="o",
            linewidth=1.6,
            label=checkpoint,
        )[0]
        performance_ax.fill_between(
            scales,
            [result.delta_nll_ci_low for result in results],
            [result.delta_nll_ci_high for result in results],
            color=effect_line.get_color(),
            alpha=0.15,
        )

    for axis in (routing_ax, magnitude_ax, performance_ax):
        axis.set_xlabel("Routed-branch scale λ (1 = checkpoint)")
        axis.grid(True, alpha=0.3)
    routing_ax.set_ylabel("Mean maximum routing weight")
    routing_ax.set_title("Router manipulation check")
    magnitude_ax.set_ylabel("Mean ||λr||₂ / ||residual||₂")
    magnitude_ax.set_title("Effective retrieval magnitude")
    performance_ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    performance_ax.set_ylabel("Δ next-token NLL vs λ=1 (lower is better)")
    performance_ax.set_title("Branch-strength effect (paired 95% CI)")
    routing_ax.legend()
    magnitude_ax.legend()
    performance_ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def validate_args(args: argparse.Namespace) -> list[float]:
    scales = sorted(set(args.branch_scales))
    if any(not math.isfinite(scale) or scale < 0 for scale in scales):
        raise ValueError("All --branch-scales must be finite and nonnegative")
    if 1.0 not in scales:
        raise ValueError("--branch-scales must include 1 for the checkpoint baseline")
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
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or checkpoints[0][1])
    grouped_results = {}

    for checkpoint_index, (label, checkpoint_path) in enumerate(checkpoints):
        model = load_model(checkpoint_path, device)
        mode = getattr(model.config, "attnres_mode", "unknown")
        if mode not in {"delta", "delta_block"}:
            raise ValueError(
                f"{label} uses mode {mode!r}; branch-strength scaling supports "
                "only additive 'delta' and 'delta_block' checkpoints"
            )

        results = [
            evaluate_condition(
                model, label, tokenizer, texts, device, args.max_length, scale
            )
            for scale in scales
        ]
        add_paired_effects(
            results,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + checkpoint_index,
        )
        baseline = next(result for result in results if result.scale == 1.0)
        for result in results:
            print(
                f"{label}: lambda={result.scale:g} "
                f"max_weight={result.routing_max:.4f} "
                f"delta_max={result.routing_max - baseline.routing_max:+.5f} "
                f"route_ratio={result.mean_scaled_routed_ratio:.4f} "
                f"NLL={result.nll:.4f} PPL={result.perplexity:.3f} "
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
    layer_csv_path = (
        Path(args.layer_csv_output)
        if args.layer_csv_output
        else csv_path.with_name(f"{csv_path.stem}_layers.csv")
    )
    plot_results(output_path, grouped_results)
    save_csv(csv_path, all_results)
    save_layer_csv(layer_csv_path, all_results)
    print(f"Saved branch-strength plot to {output_path}")
    print(f"Saved branch-strength results to {csv_path}")
    print(f"Saved per-layer routing magnitudes to {layer_csv_path}")


if __name__ == "__main__":
    main()
