#!/bin/bash
# Sarvam-1 Hindi continued pretraining on ai4bharat/sangraha (verified/hin)
# Runs two back-to-back jobs on the same node so results are directly comparable:
#   1) Baseline fine-tune (no AttnRes)        → ./output/sarvam1-hin-baseline-250M/
#   2) Delta AttnRes fine-tune (null source)  → ./output/sarvam1-hin-delta_block-250M/
# Both train on the same 250M Hindi tokens so eval perplexity is apples-to-apples.
#
# Adjust NPROC, SEQ_LEN, BATCH_SIZE, GRAD_ACCUM, STEPS, MAX_TOKENS for your hardware.
# Defaults assume 4x GPU (24-48GB VRAM each).

set -e

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

NPROC=${NPROC:-4}
SEQ_LEN=${SEQ_LEN:-2048}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-4}
STEPS=${STEPS:-10000}
MAX_TOKENS=${MAX_TOKENS:-250000000}
LR=${LR:-3e-4}
LR_MIN=${LR_MIN:-3e-5}
WARMUP=${WARMUP:-500}
SEED=${SEED:-42}

DATASET="ai4bharat/sangraha"
DATASET_NAME="verified/hin"
PRETRAINED="sarvamai/sarvam-1"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

run_job() {
    local mode=$1
    local tag=$2
    local out_dir="./output/sarvam1-hin-${tag}-250M"

    log "================================================================"
    log "Starting job: mode=${mode}  tag=${tag}  out=${out_dir}"
    log "================================================================"

    # Build the mode-specific flag list
    if [ "${mode}" = "baseline" ]; then
        mode_flags=""
    else
        mode_flags="--mode ${mode} --null_source"
    fi

    torchrun --standalone --nproc_per_node=${NPROC} \
        train_finetune_sarvam1.py \
        --pretrained "${PRETRAINED}" \
        ${mode_flags} \
        --dataset "${DATASET}" \
        --dataset_name "${DATASET_NAME}" \
        --max_tokens ${MAX_TOKENS} \
        --seq_len ${SEQ_LEN} \
        --batch_size ${BATCH_SIZE} \
        --grad_accum ${GRAD_ACCUM} \
        --steps ${STEPS} \
        --lr ${LR} \
        --lr_min ${LR_MIN} \
        --warmup ${WARMUP} \
        --seed ${SEED} \
        --save_every 2000 \
        --eval_every 500 \
        --eval_steps 50 \
        --log_every 10 \
        --run_name "sarvam1-hin-${tag}-250M" \
        --out_dir "${out_dir}" \
        2>&1 | tee "${out_dir}.log"

    log "Finished job: ${tag}"
}

# ── 1) Baseline: standard fine-tune, no AttnRes ──
run_job "baseline" "baseline"

# ── 2) Delta AttnRes with null source (zero-disruption init) ──
run_job "delta_block" "delta_block"

log "Both jobs complete."
log "Compare:  baseline       → ./output/sarvam1-hin-baseline-250M/final"
log "          delta_block    → ./output/sarvam1-hin-delta_block-250M/final"
