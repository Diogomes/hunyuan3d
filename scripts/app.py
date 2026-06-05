#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app.py — Interface web (Gradio) para converter foto em objeto 3D.

O usuário envia uma imagem, ajusta a qualidade e recebe o objeto 3D num
visualizador interativo, com botão de download (.glb).

Os pipelines são carregados UMA vez na inicialização (a 1ª vez baixa os pesos).
Sobe em http://localhost:7860
"""

import os
import time

import gradio as gr

# Rede de segurança para um bug do gradio_client ("argument of type 'bool' is
# not iterable") ao gerar o schema da API com certos componentes. Inofensivo
# se o bug não ocorrer.
import gradio_client.utils as _gcu
_orig_js2pt = _gcu._json_schema_to_python_type
def _safe_js2pt(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_js2pt(schema, defs)
_gcu._json_schema_to_python_type = _safe_js2pt

from core import Hunyuan3DConverter

# Diretórios (montados como volumes no container).
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspace/output")
MODEL = os.environ.get("HY3D_MODEL", "tencent/Hunyuan3D-2mini")
SUBFOLDER = os.environ.get("HY3D_SUBFOLDER", "hunyuan3d-dit-v2-mini")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(msg: str) -> None:
    print(f"[app] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Carrega o conversor uma única vez (custoso). Em CPU isso pode demorar.
# ---------------------------------------------------------------------------
log("Inicializando conversor Hunyuan3D (pode demorar na 1ª vez)...")
CONVERTER = Hunyuan3DConverter(
    model=MODEL,
    subfolder=SUBFOLDER,
    device="auto",
    enable_texture=True,  # só ativa de fato se houver GPU CUDA
    log=log,
)
DEVICE = CONVERTER.device
TEXTURE_OK = CONVERTER.texture_available
log(f"Pronto. Dispositivo={DEVICE} | textura disponível={TEXTURE_OK}")


def generate(image, steps, octree, max_faces, remove_bg, with_texture, seed,
             progress=gr.Progress()):
    """Callback do botão Gerar. Retorna (caminho_glb_para_viewer, caminho_para_download, status)."""
    if image is None:
        raise gr.Error("Envie uma imagem primeiro.")

    progress(0.1, desc="Preparando imagem...")
    stem = time.strftime("modelo_%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"{stem}.glb")

    t0 = time.time()
    progress(0.3, desc=f"Gerando malha em {DEVICE} (pode levar minutos)...")
    CONVERTER.convert(
        image,
        out_path,
        steps=int(steps),
        octree_resolution=int(octree),
        max_faces=int(max_faces),
        seed=int(seed),
        remove_bg=bool(remove_bg),
        with_texture=bool(with_texture),
    )
    dt = time.time() - t0
    progress(1.0, desc="Concluído")

    tex_note = "" if (with_texture and TEXTURE_OK) else \
        "  (sem textura — requer GPU NVIDIA/CUDA)"
    status = f"✅ Gerado em {dt:.0f}s no dispositivo **{DEVICE}**{tex_note}\n\nArquivo: `{out_path}`"
    return out_path, out_path, status


DEVICE_BANNER = (
    "🟢 **GPU CUDA detectada** — forma + textura PBR disponíveis."
    if TEXTURE_OK else
    "🟡 **Rodando em CPU** (sem GPU NVIDIA). Gera a *forma* 3D — porém **lento** "
    "(minutos por objeto) e **sem textura PBR** (a textura exige GPU). "
    "Use steps/octree menores para testes rápidos."
)

with gr.Blocks(title="Foto → 3D (Hunyuan3D)") as demo:
    gr.Markdown("# 📷 → 🧊 Foto para Objeto 3D (Hunyuan3D)")
    gr.Markdown(DEVICE_BANNER)

    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(type="pil", label="Imagem de entrada", height=320)
            gr.Markdown(
                "_Dica: objeto único, centralizado, bem iluminado e fundo simples._"
            )
            with gr.Accordion("Qualidade / parâmetros", open=False):
                steps = gr.Slider(10, 50, value=30, step=5, label="Passos de difusão (mais = melhor/lento)")
                octree = gr.Slider(128, 512, value=256, step=64, label="Resolução do octree (granularidade)")
                max_faces = gr.Slider(5000, 100000, value=40000, step=5000, label="Faces máximas")
                seed = gr.Number(value=42, precision=0, label="Seed")
                remove_bg = gr.Checkbox(value=True, label="Remover fundo automaticamente")
                with_texture = gr.Checkbox(
                    value=TEXTURE_OK, label="Gerar textura PBR (só GPU)", interactive=TEXTURE_OK
                )
            btn = gr.Button("🚀 Gerar objeto 3D", variant="primary")

        with gr.Column(scale=1):
            model_out = gr.Model3D(label="Objeto 3D gerado", height=420)
            file_out = gr.File(label="Baixar .glb")
            status = gr.Markdown()

    btn.click(
        fn=generate,
        inputs=[image_in, steps, octree, max_faces, remove_bg, with_texture, seed],
        outputs=[model_out, file_out, status],
    )

if __name__ == "__main__":
    # Fila evita execuções simultâneas (inferência pesada) e mostra progresso.
    demo.queue(max_size=8).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7861")),
        show_error=True,
    )
