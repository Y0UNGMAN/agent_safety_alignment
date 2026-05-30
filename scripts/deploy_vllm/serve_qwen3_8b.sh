#!/usr/bin/env bash

MODE=${1:-lora}   # base or lora

MODEL_PATH="/root/autodl-tmp/models/Qwen/Qwen3-8B"
LORA_PATH="outputs/sft_model_qwen3_8b_0527_1128"

PORT=8000
GPU_ID=0

BASE_MODEL_NAME="qwen3-8b-base"
LORA_MODEL_NAME="agent-safety-lora"

echo "Starting vLLM server..."
echo "Mode: $MODE"
echo "Model Path: $MODEL_PATH"
echo "Port: $PORT"

if [ "$MODE" = "base" ]; then
  CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$BASE_MODEL_NAME" \
    --max-model-len 4096 \
    --max-num-seqs 8 \
    --tensor-parallel-size 1 \
    --port $PORT \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml

elif [ "$MODE" = "lora" ]; then
  echo "LoRA Path: $LORA_PATH"

  CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$BASE_MODEL_NAME" \
    --max-model-len 4096 \
    --max-num-seqs 8 \
    --tensor-parallel-size 1 \
    --port $PORT \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --enable-lora \
    --lora-modules "$LORA_MODEL_NAME=$LORA_PATH"

else
  echo "Usage: bash scripts/deploy_vllm/serve_qwen3_8b.sh [base|lora]"
  exit 1
fi
