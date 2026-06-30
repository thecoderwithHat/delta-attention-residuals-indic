"""
Fine-tune pretrained Sarvam-1 with Delta AttnRes on FineWeb 2 or Sangraha.

Sarvam-1 (sarvamai/sarvam-1) is a Llama-2-architecture model:
  hidden_size=2048, intermediate_size=11008, num_attention_heads=16,
  num_key_value_heads=8, num_hidden_layers=28, vocab_size=68096, head_dim=128.

Loads a pretrained Sarvam-1 model, injects Delta AttnRes parameters (zero-initialized
+ optional null source), and fine-tunes on FineWeb 2 (default) or
ai4bharat/sangraha (verified/hin) for Hindi continued pretraining.

Usage:
    # Baseline fine-tune (no AttnRes, for comparison)
    torchrun --nproc_per_node=4 train_finetune_sarvam1.py --pretrained sarvamai/sarvam-1 --mode baseline

    # Delta AttnRes fine-tune with null source
    torchrun --nproc_per_node=4 train_finetune_sarvam1.py --pretrained sarvamai/sarvam-1 --mode delta_block --null_source

    # Delta-V with null source
    torchrun --nproc_per_node=4 train_finetune_sarvam1.py --pretrained sarvamai/sarvam-1 --mode delta_v --null_source

    # Hindi continued pretraining on Sangraha (verified/hin), 250M tokens
    torchrun --nproc_per_node=4 train_finetune_sarvam1.py \
        --pretrained sarvamai/sarvam-1 --mode delta_block --null_source \
        --dataset ai4bharat/sangraha --dataset_name verified/hin --max_tokens 250000000
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Attention-Residuals"))

from modeling_sarvam1_attnres import Sarvam1AttnResConfig, Sarvam1AttnResForCausalLM, enable_compile as enable_attnres_compile
from transformers import AutoTokenizer, AutoConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", default="sarvamai/sarvam-1",
                   help="Pretrained model name or path (default: sarvamai/sarvam-1)")
    p.add_argument("--mode", default="delta_block",
                   choices=["baseline", "block", "block_v", "full", "full_v",
                            "delta", "delta_block", "delta_block_v", "delta_v"],
                   help="AttnRes mode (baseline = standard fine-tune)")
    p.add_argument("--num_blocks", type=int, default=4,
                   help="Number of AttnRes blocks (for block modes)")
    p.add_argument("--gate_type", default="bias",
                   choices=["bias", "sigmoid_scalar", "sigmoid_vector", "learnable_alpha"])
    p.add_argument("--null_source", action="store_true",
                   help="Add null source for zero-disruption init")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-2",
                   help="HF dataset id. For Sarvam-1 Hindi CPT use ai4bharat/sangraha.")
    p.add_argument("--dataset_name", default="aai_Latn",
                   help="Dataset config (small Arifama-Miniafia subset by default). "
                        "For Sangraha Hindi use verified/hin.")
    p.add_argument("--max_tokens", type=int, default=None,
                   help="Cap total training tokens (across all ranks). "
                        "Default: None (stream until --steps). Example: 250000000 for 250M.")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=2, help="per-GPU")
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_attnres", type=float, default=None,
                   help="Separate LR for AttnRes params (default: same as --lr)")
    p.add_argument("--lr_min", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--max_norm", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--wandb_project", default="residual")
    p.add_argument("--wandb_entity", default="wdlctc_abr")
    p.add_argument("--run_name", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compile", action="store_true",
                   help="Enable torch.compile on AttnRes kernels")
    p.add_argument("--compile_model", action="store_true",
                   help="torch.compile the entire model (fuses attention+MLP+routing)")
    p.add_argument("--freeze_base", action="store_true",
                   help="Freeze all pretrained params, only train AttnRes (like LoRA)")
    return p.parse_args()


def cosine_with_warmup(step, warmup, total, lr_min_ratio):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return lr_min_ratio + (1 - lr_min_ratio) * cos


def token_stream(dataset_name, config_name, tokenizer, seq_len, rank, world_size, seed,
                 max_tokens_per_rank=None):
    """Stream tokens. Stops early once max_tokens_per_rank is hit on this rank.

    max_tokens_per_rank=None → stream indefinitely (until outer loop breaks).
    """
    from datasets import load_dataset
    ds = load_dataset(dataset_name, name=config_name, split="train",
                      streaming=True)
    ds = ds.shuffle(seed=seed + rank, buffer_size=10_000)
    ds = ds.skip(rank)
    buf = []
    produced = 0
    for sample in ds:
        if max_tokens_per_rank is not None and produced >= max_tokens_per_rank:
            return
        text = sample.get("text") or sample.get("content") or ""
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)
        buf.extend(ids)
        while len(buf) >= seq_len + 1:
            if max_tokens_per_rank is not None and produced >= max_tokens_per_rank:
                return
            chunk = buf[:seq_len + 1]
            buf = buf[world_size * seq_len:]
            produced += seq_len
            yield torch.tensor(chunk, dtype=torch.long)


def eval_validation(model, tokenizer, seq_len, eval_steps, device):
    """Quick validation on WikiText-2."""
    from datasets import load_dataset
    model.eval()
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    nlls = []
    total_tokens = 0
    for begin in range(0, min(input_ids.size(1), eval_steps * seq_len), seq_len):
        end = min(begin + seq_len, input_ids.size(1))
        chunk = input_ids[:, begin:end]
        with torch.no_grad():
            outputs = model(input_ids=chunk, use_cache=False)
            logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
        nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                       shift_labels.view(-1))
        nlls.append(nll.item())
        total_tokens += shift_labels.numel()

    avg_nll = sum(nlls) / total_tokens
    ppl = math.exp(avg_nll)
    model.train()
    return avg_nll, ppl


def build_model(args, device):
    """Load pretrained Sarvam-1 and optionally inject AttnRes parameters."""
    if args.mode == "baseline":
        model = LlamaForCausalLM.from_pretrained(
            args.pretrained, torch_dtype=torch.bfloat16)
        model = model.to(device)
        return model

    # Load pretrained config, extend with AttnRes params
    base_config = AutoConfig.from_pretrained(args.pretrained)

    attnres_config = Sarvam1AttnResConfig(
        attnres_num_blocks=args.num_blocks,
        attnres_mode=args.mode,
        attnres_gate_type=args.gate_type,
        attnres_use_null_source=args.null_source,
        **{k: v for k, v in base_config.to_dict().items()
           if k not in ("model_type", "_name_or_path", "architectures",
                        "auto_map", "transformers_version")},
    )

    # Build AttnRes model with random init
    model = Sarvam1AttnResForCausalLM(attnres_config)

    # Load pretrained weights (strict=False to skip AttnRes-specific params)
    pretrained_state = LlamaForCausalLM.from_pretrained(
        args.pretrained, torch_dtype=torch.bfloat16).state_dict()
    missing, unexpected = model.load_state_dict(pretrained_state, strict=False)

    # Verify only AttnRes params are missing
    attnres_params = [k for k in missing if "res_" in k or "null_source" in k]
    other_missing = [k for k in missing if k not in attnres_params]
    if other_missing:
        raise RuntimeError(f"Unexpected missing keys (not AttnRes): {other_missing}")

    model = model.to(dtype=torch.bfloat16, device=device)
    return model


def main():
    args = parse_args()

    pretrained_short = args.pretrained.split("/")[-1]
    if args.run_name is None:
        args.run_name = f"ft-{pretrained_short}-{args.mode}-{args.steps//1000}k"
    if args.out_dir is None:
        args.out_dir = f"./output/ft-{pretrained_short}-{args.mode}-{args.steps//1000}k"

    # ── distributed ──
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_main = rank == 0

    torch.manual_seed(args.seed + rank)

    # ── W&B ──
    use_wandb = False
    if is_main:
        try:
            import wandb
            wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                       name=args.run_name, config=vars(args))
            use_wandb = True
        except Exception as e:
            print(f"W&B init failed ({e}), continuing without logging")

    # ── model ──
    if is_main:
        print(f"Loading pretrained {args.pretrained}, injecting {args.mode} AttnRes...")

    model = build_model(args, device)

    if args.compile and args.mode != "baseline":
        enable_attnres_compile()
        if is_main:
            print("torch.compile enabled for AttnRes kernels")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    if is_main:
        print(f"Model: {n_params:.1f}M params | mode={args.mode}")
        if args.mode != "baseline":
            n_attnres = sum(p.numel() for n, p in model.named_parameters() if "res_" in n or "null_source" in n)
            print(f"AttnRes params: {n_attnres/1e3:.1f}K ({n_attnres/sum(p.numel() for p in model.parameters())*100:.3f}%)")
            if args.null_source:
                print("Null source enabled (zero-disruption init)")

    # ── freeze base (LoRA-style) ──
    if args.freeze_base:
        if args.mode == "baseline":
            raise ValueError("--freeze_base requires an AttnRes mode, not baseline")
        n_frozen = 0
        n_trainable = 0
        for name, param in model.named_parameters():
            if "res_" in name or "null_source" in name:
                param.requires_grad = True
                n_trainable += param.numel()
            else:
                param.requires_grad = False
                n_frozen += param.numel()
        if is_main:
            print(f"Freeze base: {n_frozen/1e6:.1f}M frozen, {n_trainable/1e3:.1f}K trainable "
                  f"({n_trainable/(n_frozen+n_trainable)*100:.3f}%)")

    # torch.compile the full model before DDP wrapping.
    # Gives ~2.5-2.9x throughput improvement for all modes.
    if args.compile_model:
        model = torch.compile(model)
        if is_main:
            print("torch.compile enabled for full model")

    model = DDP(model, device_ids=[local_rank])

    # ── optimizer ──
    if args.lr_attnres is not None and args.mode != "baseline":
        # Separate param groups: base transformer vs AttnRes
        base_params = []
        attnres_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "res_" in name or "null_source" in name:
                attnres_params.append(param)
            else:
                base_params.append(param)
        param_groups = [
            {"params": base_params, "lr": args.lr},
            {"params": attnres_params, "lr": args.lr_attnres},
        ]
        optimizer = AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)
        lr_min_ratio_base = args.lr_min / args.lr
        lr_min_ratio_attnres = args.lr_min / args.lr_attnres
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=[
                lambda s: cosine_with_warmup(s, args.warmup, args.steps, lr_min_ratio_base),
                lambda s: cosine_with_warmup(s, args.warmup, args.steps, lr_min_ratio_attnres),
            ],
        )
        if is_main:
            print(f"Optimizer: base LR={args.lr}, AttnRes LR={args.lr_attnres} "
                  f"({len(base_params)} base groups, {len(attnres_params)} attnres groups)")
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=args.lr,
                          betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)
        lr_min_ratio = args.lr_min / args.lr
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda s: cosine_with_warmup(s, args.warmup, args.steps, lr_min_ratio),
        )

    # ── data ──
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)

    # Per-rank token budget: total / world_size (so the *sum* across ranks hits the cap).
    # None → no cap, stream until --steps is reached.
    if args.max_tokens is not None:
        if args.max_tokens <= 0:
            raise ValueError(f"--max_tokens must be positive, got {args.max_tokens}")
        if args.max_tokens < world_size * args.batch_size * args.grad_accum * args.seq_len:
            raise ValueError(
                f"--max_tokens ({args.max_tokens}) is too small for one optimizer step "
                f"(need at least {world_size * args.batch_size * args.grad_accum * args.seq_len} tokens)")
        max_tokens_per_rank = args.max_tokens // world_size
        if is_main:
            print(f"Token budget: {args.max_tokens/1e6:.1f}M total "
                  f"({max_tokens_per_rank/1e6:.1f}M per rank)")
    else:
        max_tokens_per_rank = None

    stream = token_stream(args.dataset, args.dataset_name, tokenizer,
                          args.seq_len, rank, world_size, args.seed,
                          max_tokens_per_rank=max_tokens_per_rank)

    # ── training ──
    os.makedirs(args.out_dir, exist_ok=True)
    model.train()
    optimizer.zero_grad()

    global_step = 0
    accum_step = 0
    accum_loss = 0.0
    t0 = time.time()
    tokens_seen = 0

    batch_buf = []
    for chunk in stream:
        if global_step >= args.steps:
            break

        batch_buf.append(chunk[:-1])
        if len(batch_buf) < args.batch_size:
            continue

        input_ids = torch.stack(batch_buf).to(device)
        labels = input_ids
        batch_buf = []

        out = model(input_ids=input_ids, labels=labels, use_cache=False)
        loss = out.loss / args.grad_accum
        loss.backward()

        accum_loss += loss.item()
        accum_step += 1
        tokens_seen += args.seq_len * args.batch_size

        if accum_step < args.grad_accum:
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        global_step += 1
        accum_step = 0

        if global_step % args.log_every == 0:
            loss_t = torch.tensor(accum_loss, device=device)
            dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)

            if is_main:
                elapsed = time.time() - t0
                tok_sec = tokens_seen * world_size / elapsed
                avg_loss = loss_t.item()
                lr_now = scheduler.get_last_lr()[0]
                print(f"step {global_step:6d} | loss {avg_loss:.4f} | "
                      f"lr {lr_now:.2e} | grad_norm {grad_norm:.3f} | "
                      f"{tok_sec/1e3:.1f}k tok/s")

                if use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr": lr_now,
                        "train/grad_norm": grad_norm,
                        "train/tok_per_s": tok_sec,
                    }, step=global_step)

                tokens_seen = 0
                t0 = time.time()
        accum_loss = 0.0

        # ── validation ──
        if is_main and args.eval_every > 0 and global_step % args.eval_every == 0:
            val_loss, val_ppl = eval_validation(
                model.module, tokenizer, args.seq_len, args.eval_steps, device)
            print(f"  [val] step {global_step} | WT2 loss {val_loss:.4f} | PPL {val_ppl:.2f}")
            if use_wandb:
                import wandb
                wandb.log({"val/wt2_loss": val_loss, "val/wt2_ppl": val_ppl}, step=global_step)

        if is_main and global_step % args.save_every == 0:
            ckpt_dir = os.path.join(args.out_dir, f"step-{global_step}")
            model.module.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"Saved checkpoint → {ckpt_dir}")

    if is_main:
        final_dir = os.path.join(args.out_dir, "final")
        model.module.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"Training done. Final model → {final_dir}")
        if use_wandb:
            import wandb
            wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
