"""Plot the mean maximum depth-routing probability at every model layer.

Example:
    python plot_routing_sharpness.py \
        --attnres-checkpoint output/attnres/final \
        --delta-checkpoint output/delta/final \
        --text-file evaluation_text.txt \
        --output routing_sharpness.png
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "Attention-Residuals"))

from modeling_qwen3_attnres import Qwen3AttnResForCausalLM
from transformers import AutoTokenizer


DEFAULT_TEXTS = [
    (
        "Attention residuals allow a transformer layer to select information "
        "from earlier layers instead of relying only on the latest hidden state."
    ),
    (
        "Delta attention residuals route over the changes produced by individual "
        "sublayers, which can make cross-layer routing more selective."
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare per-layer routing sharpness for two AttnRes checkpoints."
    )
    parser.add_argument("--attnres-checkpoint", required=True)
    parser.add_argument("--delta-checkpoint", required=True)
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer path or model ID. Defaults to the AttnRes checkpoint.",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Evaluation text. May be supplied more than once.",
    )
    parser.add_argument(
        "--text-file",
        default=None,
        help="UTF-8 file containing samples separated by blank lines.",
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="routing_sharpness.png")
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Optional CSV path. Defaults to the image path with a .csv suffix.",
    )
    return parser.parse_args()


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


@torch.inference_mode()
def measure_sharpness(
    model: Qwen3AttnResForCausalLM,
    tokenizer,
    texts: list[str],
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    """Average max routing probability over tokens, samples, and sublayers."""
    weighted_sum = None
    token_count = 0

    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            sample_tokens = int(attention_mask.sum())
        else:
            sample_tokens = input_ids.numel()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=1,
            return_routing_max=True,
        )
        values = outputs.routing_max_weights
        if values is None:
            mode = getattr(model.config, "attnres_mode", "unknown")
            raise RuntimeError(
                f"Routing statistics are not implemented for mode {mode!r}. "
                "Use an AttnRes 'block' or 'full' checkpoint and a Delta-AttnRes "
                "'delta' checkpoint."
            )

        values = values.float().cpu()
        weighted_sum = (
            values * sample_tokens
            if weighted_sum is None
            else weighted_sum + values * sample_tokens
        )
        token_count += sample_tokens

    if weighted_sum is None or token_count == 0:
        raise ValueError("No non-empty evaluation text was provided")
    return weighted_sum / token_count


def save_csv(path: Path, attnres: torch.Tensor, delta: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", "AttnRes", "Delta-AttnRes"])
        for layer, (attnres_value, delta_value) in enumerate(zip(attnres, delta)):
            writer.writerow([layer, float(attnres_value), float(delta_value)])


def plot(path: Path, attnres: torch.Tensor, delta: torch.Tensor) -> None:
    import matplotlib.pyplot as plt

    layers = range(len(attnres))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(
        layers,
        attnres.numpy(),
        color="#e74c3c",
        marker="o",
        markersize=4,
        linewidth=1.6,
        label="AttnRes",
    )
    ax.plot(
        layers,
        delta.numpy(),
        color="#27ae60",
        marker="s",
        markersize=4,
        linewidth=1.6,
        label="Delta-AttnRes",
    )
    ax.set_title("(a) Routing Sharpness (per layer)", fontweight="bold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Max Attention Weight")
    ax.set_xlim(-0.5, len(attnres) - 0.5)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    texts = load_texts(args)
    tokenizer_path = args.tokenizer or args.attnres_checkpoint
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    attnres_model = load_model(args.attnres_checkpoint, device)
    attnres_values = measure_sharpness(
        attnres_model, tokenizer, texts, device, args.max_length
    )
    del attnres_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    delta_model = load_model(args.delta_checkpoint, device)
    delta_values = measure_sharpness(
        delta_model, tokenizer, texts, device, args.max_length
    )

    if len(attnres_values) != len(delta_values):
        raise ValueError(
            "The checkpoints have different layer counts: "
            f"{len(attnres_values)} and {len(delta_values)}"
        )

    output_path = Path(args.output)
    csv_path = (
        Path(args.csv_output)
        if args.csv_output
        else output_path.with_suffix(".csv")
    )
    plot(output_path, attnres_values, delta_values)
    save_csv(csv_path, attnres_values, delta_values)
    print(f"Saved plot to {output_path}")
    print(f"Saved values to {csv_path}")


if __name__ == "__main__":
    main()
