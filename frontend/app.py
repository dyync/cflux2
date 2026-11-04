"""Gradio frontend for FLUX.1-dev served by vLLM-Omni (/v1/images/generations)."""

import base64
import io
import os
import random

import gradio as gr
import requests
from PIL import Image

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
MODEL = os.environ.get("MODEL", "black-forest-labs/FLUX.1-dev")
TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "600"))
MAX_SEED = 2**31 - 1

ASPECTS = {
    "1:1 (1024x1024)": (1024, 1024),
    "4:3 (1152x864)": (1152, 864),
    "3:4 (864x1152)": (864, 1152),
    "16:9 (1344x768)": (1344, 768),
    "9:16 (768x1344)": (768, 1344),
    "Custom": None,
}


def backend_status() -> str:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if r.ok:
            return f"🟢 Backend online: `{BACKEND_URL}` · model `{MODEL}`"
        return f"🟠 Backend responded {r.status_code}: `{BACKEND_URL}`"
    except requests.RequestException:
        return f"🔴 Backend unreachable: `{BACKEND_URL}` (still loading weights?)"


def on_aspect(choice):
    dims = ASPECTS[choice]
    if dims is None:
        return gr.update(interactive=True), gr.update(interactive=True)
    w, h = dims
    return gr.update(value=w, interactive=False), gr.update(value=h, interactive=False)


def generate(prompt, width, height, steps, guidance, seed, randomize, n, progress=gr.Progress()):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")
    if randomize:
        seed = random.randint(0, MAX_SEED)
    seed = int(seed)

    # FLUX works on 16-px multiples.
    width, height = int(width) // 16 * 16, int(height) // 16 * 16

    payload = {
        "model": MODEL,
        "prompt": prompt.strip(),
        "size": f"{width}x{height}",
        "n": int(n),
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance),
        "seed": seed,
        "response_format": "b64_json",
    }

    progress(0, desc="Generating on vLLM-Omni…")
    try:
        r = requests.post(f"{BACKEND_URL}/v1/images/generations", json=payload, timeout=TIMEOUT)
    except requests.RequestException as e:
        raise gr.Error(f"Could not reach backend at {BACKEND_URL}: {e}")
    if not r.ok:
        raise gr.Error(f"Backend error {r.status_code}: {r.text[:500]}")

    images = [
        Image.open(io.BytesIO(base64.b64decode(item["b64_json"])))
        for item in r.json().get("data", [])
        if item.get("b64_json")
    ]
    if not images:
        raise gr.Error("Backend returned no images.")
    return images, seed


EXAMPLES = [
    "A cat holding a sign that says hello world",
    "A cinematic photo of a lighthouse on a cliff during a thunderstorm, dramatic lighting, 35mm",
    "Isometric cutaway illustration of a cozy underground library, warm lamps, intricate detail",
    "Macro photograph of a dewdrop on a fern leaf reflecting a sunrise",
]

with gr.Blocks(title="FLUX.1-dev · vLLM", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# FLUX.1-dev\nText-to-image served by **vLLM-Omni**.")
    status = gr.Markdown(backend_status())

    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Describe the image…")
            run = gr.Button("Generate", variant="primary")
            with gr.Accordion("Settings", open=True):
                aspect = gr.Dropdown(list(ASPECTS), value="1:1 (1024x1024)", label="Aspect ratio")
                with gr.Row():
                    width = gr.Slider(256, 2048, value=1024, step=16, label="Width", interactive=False)
                    height = gr.Slider(256, 2048, value=1024, step=16, label="Height", interactive=False)
                steps = gr.Slider(1, 100, value=28, step=1, label="Inference steps")
                guidance = gr.Slider(0.0, 10.0, value=3.5, step=0.1, label="Guidance scale")
                n = gr.Slider(1, 4, value=1, step=1, label="Images per prompt")
                with gr.Row():
                    seed = gr.Number(value=42, label="Seed", precision=0, minimum=0, maximum=MAX_SEED)
                    randomize = gr.Checkbox(value=True, label="Randomize seed")
            refresh = gr.Button("Check backend", size="sm")
        with gr.Column(scale=3):
            gallery = gr.Gallery(label="Result", columns=2, height=640, format="png")
            used_seed = gr.Number(label="Seed used", interactive=False, precision=0)

    gr.Examples(EXAMPLES, inputs=prompt)

    aspect.change(on_aspect, aspect, [width, height])
    inputs = [prompt, width, height, steps, guidance, seed, randomize, n]
    run.click(generate, inputs, [gallery, used_seed], concurrency_limit=2)
    prompt.submit(generate, inputs, [gallery, used_seed], concurrency_limit=2)
    refresh.click(backend_status, None, status)

if __name__ == "__main__":
    demo.queue(max_size=32).launch(
        server_name=os.environ.get("GRADIO_HOST", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_PORT", "7860")),
    )
