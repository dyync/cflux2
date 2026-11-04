#!/usr/bin/env bash
# Start the vLLM-Omni server for FLUX.1-dev (Linux + NVIDIA GPU, CUDA 12.9).
# Install first:
#   uv venv --python 3.12 --seed && source .venv/bin/activate
#   uv pip install "https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" --torch-backend=cu129
#   uv pip install vllm-omni==0.32.0 --torch-backend=cu129
# FLUX.1-dev is gated: accept the license on Hugging Face, then export HF_TOKEN.
set -euo pipefail

MODEL="${MODEL:-black-forest-labs/FLUX.1-dev}"
PORT="${PORT:-8000}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "warning: HF_TOKEN is not set; downloading the gated FLUX.1-dev weights will fail." >&2
fi

exec vllm serve "$MODEL" --omni --host 0.0.0.0 --port "$PORT" "$@"
