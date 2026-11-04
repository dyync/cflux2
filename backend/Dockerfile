# vLLM 0.32.0 built against CUDA 12.9 + vLLM-Omni on top.
# (The official vllm/vllm-omni image only ships the default CUDA 13 build.)
FROM vllm/vllm-openai:v0.32.0-cu129

RUN pip install --no-cache-dir vllm-omni==0.32.0 \
 && python3 -c "import torch; v = torch.version.cuda; print('torch CUDA', v); assert v.startswith('12.9'), v"

ENTRYPOINT []
