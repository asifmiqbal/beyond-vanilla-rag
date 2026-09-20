#!/usr/bin/env bash
# ==============================================================================
# vLLM High-Performance Local Serving Script (Linux / GPU / CPU Fallback)
# Target Model: Qwen/Qwen2.5-3B-Instruct or Qwen/Qwen3.5-2B
# ==============================================================================

set -eo pipefail

MODEL_NAME="${1:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${2:-8000}"
HOST="0.0.0.0"

echo "======================================================================"
echo " Starting local vLLM OpenAI-Compatible Inference Server"
echo " Model: ${MODEL_NAME}"
echo " Endpoint: http://${HOST}:${PORT}/v1"
echo "======================================================================"

# Check if vLLM is installed
if ! command -v vllm &> /dev/null; then
    echo "vLLM CLI not found in current PATH. Checking Python module..."
    if ! python3 -c "import vllm" &> /dev/null; then
        echo "Error: vLLM is not installed."
        echo "Install via: pip install vllm"
        echo ""
        echo "Note: If you have Ollama running with Qwen 3 (qwen3:1.7b),"
        echo "the workbench automatically connects to http://localhost:11434"
        echo "without needing to run vLLM."
        exit 1
    fi
fi

# Detect NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "[GPU DETECTED] Launching vLLM with CUDA acceleration..."
    exec python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_NAME}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --gpu-memory-utilization 0.90 \
        --max-model-len 4096 \
        --dtype bfloat16 \
        --tensor-parallel-size 1 \
        --trust-remote-code \
        --disable-log-requests
else
    echo "[CPU MODE] No NVIDIA GPU detected. Launching in CPU fallback mode..."
    export VLLM_TARGET_DEVICE=cpu
    exec python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_NAME}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --max-model-len 2048 \
        --dtype float32 \
        --device cpu \
        --trust-remote-code
fi
