"""Test whether performance depends on selecting the correct routing source.

This complements the logit-scale experiment in ``test_routing_causality.py``.
It compares learned routing with two within-checkpoint interventions:

* permuted: preserve the learned probabilities but assign them to a
  deterministic derangement of the sources;
* uniform: replace every routing distribution with exactly 1/N.

Example:
    python test_routing_selection.py \
        --checkpoint Delta-AttnRes=output/delta/final \
        --text-file evaluation_text.txt \
        --permutation-seeds 0 1 2 3 4 \
        --output routing_source_selection.png
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "Attention-Residuals"))

from test_routing_causality import (
    load_model,
    load_texts,
    parse_checkpoint,
)
from transformers import AutoTokenizer


@dataclass
class SelectionResult:
    checkpoint: str
    condition: str
    permutation_seed: int | None
    routing_max: float
    nll: float
    perplexity: float
    sample_loss_sums: torch.Tensor
    sample_token_counts: torch.Tensor
    routing_weight_caches: list[list[torch.Tensor]] | None = None
    delta_nll: float = 0.0
    delta_nll_ci_low: float = 0.0
    delta_nll_ci_high: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare learned routing with source permutation and exactly "
            "uniform routing on the same evaluation documents."
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
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--permutation-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Seeds for repeatable source derangements.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed.")
    parser.add_argument("--output", default="routing_source_selection.png")
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Defaults to the image path with a .csv suffix.",
    )
    return parser.parse_args()


def safe_perplexity(nll: float) -> float:
    limit = math.log(torch.finfo(torch.float64).max)
    return math.exp(nll) if nll < limit else math.inf


@torch.inference_mode()
def evaluate_condition(
    model,
    checkpoint_label: str,
    tokenizer,
    texts: list[str],
    device: torch.device,
    max_length: int,
    condition: str,
    permutation_seed: int | None = None,
    routing_weight_caches: list[list[torch.Tensor]] | None = None,
) -> SelectionResult:
    sample_loss_sums = []
    sample_token_counts = []
    routing_weighted_sum = 0.0
    routing_token_count = 0
    captured_caches = [] if condition == "learned" else None

    if condition == "permute" and routing_weight_caches is None:
        raise ValueError("Permuted evaluation requires learned routing caches")
    if routing_weight_caches is not None and len(routing_weight_caches) != len(texts):
        raise ValueError("Routing cache count does not match evaluation sample count")

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

        model_intervention = "capture" if condition == "learned" else condition
        model_kwargs = {
            "routing_intervention": model_intervention,
            "routing_permutation_seed": permutation_seed or 0,
        }
        if condition == "permute":
            model_kwargs["routing_weight_cache"] = routing_weight_caches[sample_index]
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=0,
            return_routing_max=True,
            **model_kwargs,
        )
        routing_values = outputs.routing_max_weights
        if routing_values is None:
            mode = getattr(model.config, "attnres_mode", "unknown")
            raise RuntimeError(
                f"Routing statistics are not implemented for mode {mode!r}. "
                "Use a 'block', 'full', 'delta', or 'delta_block' checkpoint."
            )
        if captured_caches is not None:
            if outputs.routing_weight_cache is None:
                raise RuntimeError("The model did not return captured routing weights")
            captured_caches.append(outputs.routing_weight_cache)

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
        routing_weighted_sum += routing_values.float().nanmean().item() * routing_tokens
        routing_token_count += routing_tokens

    loss_sums = torch.tensor(sample_loss_sums, dtype=torch.float64)
    token_counts = torch.tensor(sample_token_counts, dtype=torch.float64)
    nll = (loss_sums.sum() / token_counts.sum()).item()
    return SelectionResult(
        checkpoint=checkpoint_label,
        condition=condition,
        permutation_seed=permutation_seed,
        routing_max=routing_weighted_sum / routing_token_count,
        nll=nll,
        perplexity=safe_perplexity(nll),
        sample_loss_sums=loss_sums,
        sample_token_counts=token_counts,
        routing_weight_caches=captured_caches,
    )


def average_permutations(results: list[SelectionResult]) -> SelectionResult:
    if not results:
        raise ValueError("At least one permutation result is required")
    token_counts = results[0].sample_token_counts
    if any(not torch.equal(result.sample_token_counts, token_counts) for result in results):
        raise RuntimeError("Permutation runs evaluated different target-token counts")

    mean_loss_sums = torch.stack(
        [result.sample_loss_sums for result in results]
    ).mean(dim=0)
    nll = (mean_loss_sums.sum() / token_counts.sum()).item()
    return SelectionResult(
        checkpoint=results[0].checkpoint,
        condition="permuted_mean",
        permutation_seed=None,
        routing_max=sum(result.routing_max for result in results) / len(results),
        nll=nll,
        perplexity=safe_perplexity(nll),
        sample_loss_sums=mean_loss_sums,
        sample_token_counts=token_counts.clone(),
    )


def add_paired_effect(
    result: SelectionResult,
    baseline: SelectionResult,
    bootstrap_samples: int,
    seed: int,
) -> None:
    if not torch.equal(result.sample_token_counts, baseline.sample_token_counts):
        raise RuntimeError("Paired conditions evaluated different target-token counts")

    result.delta_nll = result.nll - baseline.nll
    if result is baseline or bootstrap_samples == 0:
        result.delta_nll_ci_low = result.delta_nll
        result.delta_nll_ci_high = result.delta_nll
        return

    sample_count = len(baseline.sample_loss_sums)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        sample_count,
        (bootstrap_samples, sample_count),
        generator=generator,
    )
    loss_differences = result.sample_loss_sums - baseline.sample_loss_sums
    bootstrap_effects = (
        loss_differences[indices].sum(dim=1)
        / baseline.sample_token_counts[indices].sum(dim=1)
    )
    quantiles = torch.quantile(
        bootstrap_effects,
        torch.tensor([0.025, 0.975], dtype=torch.float64),
    )
    result.delta_nll_ci_low = quantiles[0].item()
    result.delta_nll_ci_high = quantiles[1].item()


def save_csv(path: Path, results: list[SelectionResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "checkpoint",
                "condition",
                "permutation_seed",
                "mean_max_routing_weight",
                "next_token_nll",
                "perplexity",
                "delta_nll_vs_learned",
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
                    result.condition,
                    "" if result.permutation_seed is None else result.permutation_seed,
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


def plot_results(path: Path, grouped: dict[str, list[SelectionResult]]) -> None:
    import matplotlib.pyplot as plt

    display_conditions = ["learned", "permuted_mean", "uniform"]
    labels = ["Learned", "Permuted sources", "Uniform"]
    positions = range(len(display_conditions))
    fig, (routing_ax, performance_ax) = plt.subplots(1, 2, figsize=(10.8, 4.4))

    for checkpoint, results in grouped.items():
        by_condition = {result.condition: result for result in results}
        displayed = [by_condition[condition] for condition in display_conditions]
        routing_ax.plot(
            positions,
            [result.routing_max for result in displayed],
            marker="o",
            linewidth=1.6,
            label=checkpoint,
        )
        effect_line = performance_ax.plot(
            positions,
            [result.delta_nll for result in displayed],
            marker="o",
            linewidth=1.6,
            label=checkpoint,
        )[0]
        performance_ax.vlines(
            positions,
            [result.delta_nll_ci_low for result in displayed],
            [result.delta_nll_ci_high for result in displayed],
            color=effect_line.get_color(),
            linewidth=2,
        )

    for axis in (routing_ax, performance_ax):
        axis.set_xticks(positions, labels)
        axis.grid(True, alpha=0.3)
    routing_ax.set_ylabel("Mean maximum routing weight")
    routing_ax.set_title("Probability control")
    performance_ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    performance_ax.set_ylabel("Δ next-token NLL vs learned")
    performance_ax.set_title("Source-selection effect (paired 95% CI)")
    routing_ax.legend()
    performance_ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def validate_args(args: argparse.Namespace) -> list[int]:
    if args.max_length < 2:
        raise ValueError("--max-length must be at least 2")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples cannot be negative")
    seeds = list(dict.fromkeys(args.permutation_seeds))
    if not seeds:
        raise ValueError("At least one --permutation-seeds value is required")
    return seeds


def main() -> None:
    args = parse_args()
    permutation_seeds = validate_args(args)
    checkpoints = [parse_checkpoint(spec) for spec in args.checkpoint]
    labels = [label for label, _ in checkpoints]
    if len(labels) != len(set(labels)):
        raise ValueError("Checkpoint labels must be unique")

    device = torch.device(args.device)
    texts = load_texts(args)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or checkpoints[0][1])
    grouped_results = {}
    csv_results = []

    for checkpoint_index, (label, checkpoint_path) in enumerate(checkpoints):
        model = load_model(checkpoint_path, device)
        baseline = evaluate_condition(
            model, label, tokenizer, texts, device, args.max_length, "learned"
        )
        permutation_results = [
            evaluate_condition(
                model,
                label,
                tokenizer,
                texts,
                device,
                args.max_length,
                "permute",
                permutation_seed=seed,
                routing_weight_caches=baseline.routing_weight_caches,
            )
            for seed in permutation_seeds
        ]
        for result in permutation_results:
            if not math.isclose(
                result.routing_max,
                baseline.routing_max,
                rel_tol=1e-6,
                abs_tol=1e-7,
            ):
                raise RuntimeError(
                    "Permutation failed its probability-preservation check: "
                    f"learned max={baseline.routing_max}, "
                    f"permuted max={result.routing_max}"
                )
        baseline.routing_weight_caches = None
        permutation_mean = average_permutations(permutation_results)
        uniform = evaluate_condition(
            model, label, tokenizer, texts, device, args.max_length, "uniform"
        )

        all_conditions = [baseline, *permutation_results, permutation_mean, uniform]
        for condition_index, result in enumerate(all_conditions):
            add_paired_effect(
                result,
                baseline,
                args.bootstrap_samples,
                args.seed + checkpoint_index * 1000 + condition_index,
            )
            seed_text = (
                "" if result.permutation_seed is None
                else f" seed={result.permutation_seed}"
            )
            print(
                f"{label}: {result.condition}{seed_text} "
                f"max_weight={result.routing_max:.4f} NLL={result.nll:.4f} "
                f"PPL={result.perplexity:.3f} delta_NLL={result.delta_nll:+.5f} "
                f"95% CI=[{result.delta_nll_ci_low:+.5f}, "
                f"{result.delta_nll_ci_high:+.5f}]"
            )

        grouped_results[label] = [baseline, permutation_mean, uniform]
        csv_results.extend(all_conditions)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_path = Path(args.output)
    csv_path = Path(args.csv_output) if args.csv_output else output_path.with_suffix(".csv")
    plot_results(output_path, grouped_results)
    save_csv(csv_path, csv_results)
    print(f"Saved source-selection plot to {output_path}")
    print(f"Saved source-selection results to {csv_path}")


if __name__ == "__main__":
    main()
