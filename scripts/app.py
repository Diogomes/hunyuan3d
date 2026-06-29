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

from core import Hunyuan3DConverter, best_model, quality_preset, level_preset

# Diretórios (montados como volumes no container).
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspace/output")
# Modelo escolhido pelo dispositivo: GPU -> Hunyuan3D-2 completo (mais realista
# + textura PBR); CPU -> mini (viável sem GPU). Pode ser sobrescrito por env.
_DEF_MODEL, _DEF_SUBFOLDER = best_model("auto")
MODEL = os.environ.get("HY3D_MODEL", _DEF_MODEL)
SUBFOLDER = os.environ.get("HY3D_SUBFOLDER", _DEF_SUBFOLDER)
TEXTURE_MODEL = os.environ.get("HY3D_TEXTURE_MODEL", "tencent/Hunyuan3D-2")
# Padrões de qualidade dos controles (GPU = máximo; CPU = moderado).
QP = quality_preset("auto")
ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
LOGO_PATH = os.path.join(ASSET_DIR, "gigaverse-logo.png")

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
    texture_model=TEXTURE_MODEL,
    log=log,
)
DEVICE = CONVERTER.device
TEXTURE_OK = CONVERTER.texture_available
log(f"Pronto. Dispositivo={DEVICE} | textura disponível={TEXTURE_OK}")


def _common_kwargs(steps, octree, max_faces, remove_bg, with_texture, enhance,
                   smooth, target_mm, want_stl, seed):
    """Monta os kwargs compartilhados entre 1-foto e multi-view."""
    return dict(
        steps=int(steps),
        octree_resolution=int(octree),
        max_faces=int(max_faces),
        seed=int(seed),
        remove_bg=bool(remove_bg),
        with_texture=bool(with_texture),
        enhance=bool(enhance),
        smooth=int(smooth),
        target_size_mm=float(target_mm),
        extra_formats=(".stl",) if want_stl else (),
        make_solid=bool(want_stl),
    )


def _status(out_path, want_stl, with_texture, dt):
    stl_path = os.path.splitext(out_path)[0] + ".stl" if want_stl else None
    tex_note = "" if (with_texture and TEXTURE_OK) else \
        "  (sem textura — requer GPU NVIDIA/CUDA)"
    stl_note = f"\nSTL p/ impressão: `{stl_path}`" if stl_path else ""

    info = getattr(CONVERTER, "last_info", None)
    info_note = ""
    if info:
        sx, sy, sz = info["size_mm"]
        vol = f" · volume {info['volume_cm3']} cm³" if info.get("volume_cm3") is not None else ""
        wt = "sólido ✅" if info.get("watertight") else "aberto ⚠️"
        info_note = (f"\n\n**Peça:** {info['faces']:,} faces · "
                     f"{sx}×{sy}×{sz} mm · {wt}{vol}")

    status = (f"✅ Gerado em {dt:.0f}s no dispositivo **{DEVICE}**{tex_note}"
              f"\n\nArquivo: `{out_path}`{stl_note}{info_note}")
    return out_path, out_path, stl_path, status


def generate(image, steps, octree, max_faces, remove_bg, with_texture, recenter,
             enhance, smooth, target_mm, want_stl, seed, progress=gr.Progress()):
    """Callback (1 foto). Retorna (glb_viewer, download_glb, download_stl, status)."""
    if image is None:
        raise gr.Error("Envie uma imagem primeiro.")

    progress(0.1, desc="Preparando imagem...")
    out_path = os.path.join(OUTPUT_DIR, time.strftime("modelo_%Y%m%d_%H%M%S") + ".glb")

    t0 = time.time()
    progress(0.3, desc=f"Gerando malha em {DEVICE} (pode levar minutos)...")
    CONVERTER.convert(
        image, out_path, recenter=bool(recenter),
        **_common_kwargs(steps, octree, max_faces, remove_bg, with_texture,
                         enhance, smooth, target_mm, want_stl, seed),
    )
    progress(1.0, desc="Concluído")
    return _status(out_path, want_stl, with_texture, time.time() - t0)


def generate_mv(front, back, left, right, steps, octree, max_faces, remove_bg,
                with_texture, enhance, smooth, target_mm, want_stl, seed,
                progress=gr.Progress()):
    """Callback (multi-view). Usa várias vistas do mesmo objeto."""
    if front is None:
        raise gr.Error("A vista frontal ('Frente') é obrigatória.")

    progress(0.1, desc="Preparando vistas...")
    images = {"front": front, "back": back, "left": left, "right": right}
    out_path = os.path.join(OUTPUT_DIR, time.strftime("modelo_mv_%Y%m%d_%H%M%S") + ".glb")

    t0 = time.time()
    progress(0.3, desc=f"Gerando malha multi-view em {DEVICE} (pode levar minutos)...")
    CONVERTER.convert_multiview(
        images, out_path,
        **_common_kwargs(steps, octree, max_faces, remove_bg, with_texture,
                         enhance, smooth, target_mm, want_stl, seed),
    )
    progress(1.0, desc="Concluído")
    return _status(out_path, want_stl, with_texture, time.time() - t0)


DEVICE_BANNER = (
    f"🟢 **GPU CUDA detectada** — modelo completo `{MODEL}`, forma de alta "
    "resolução + textura PBR. Padrões já no máximo de qualidade."
    if TEXTURE_OK else
    "🟡 **Rodando em CPU** (sem GPU NVIDIA). Gera a *forma* 3D — porém **lento** "
    "(minutos por objeto) e **sem textura PBR** (a textura exige GPU). "
    "Use steps/octree menores para testes rápidos."
)

CUSTOM_CSS = """
:root {
  --gv-bg: #05070d;
  --gv-panel: rgba(11, 16, 28, 0.88);
  --gv-panel-2: rgba(16, 23, 38, 0.92);
  --gv-border: rgba(81, 173, 255, 0.34);
  --gv-blue: #0084ff;
  --gv-cyan: #41d8ff;
  --gv-text: #e8f2ff;
  --gv-muted: #99abc4;
  --gv-silver: #d8e0ea;
}

body,
.gradio-container {
  background:
    radial-gradient(circle at 50% 0%, rgba(0, 132, 255, 0.28), transparent 34rem),
    radial-gradient(circle at 15% 20%, rgba(65, 216, 255, 0.12), transparent 22rem),
    linear-gradient(145deg, #02040a 0%, #07101d 42%, #02040a 100%) !important;
  color: var(--gv-text) !important;
}

.gradio-container {
  max-width: 1240px !important;
}

.gv-hero {
  position: relative;
  display: grid;
  grid-template-columns: minmax(128px, 180px) 1fr;
  gap: 26px;
  align-items: center;
  margin: 8px 0 22px;
  padding: 22px;
  border: 1px solid var(--gv-border);
  border-radius: 18px;
  background:
    linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.015)),
    linear-gradient(145deg, rgba(6, 10, 19, 0.96), rgba(9, 18, 34, 0.86));
  box-shadow: 0 0 38px rgba(0, 132, 255, 0.22), inset 0 0 0 1px rgba(255, 255, 255, 0.06);
  overflow: hidden;
}

.gv-hero::after {
  content: "";
  position: absolute;
  inset: auto 24px 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--gv-cyan), var(--gv-blue), transparent);
  box-shadow: 0 0 22px var(--gv-blue);
}

.gv-logo {
  width: min(180px, 100%);
  aspect-ratio: 1;
  object-fit: cover;
  border-radius: 14px;
  box-shadow: 0 0 34px rgba(0, 132, 255, 0.42);
}

.gv-kicker {
  margin: 0 0 8px;
  color: var(--gv-cyan);
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
}

.gv-title {
  margin: 0;
  color: var(--gv-silver);
  font-size: clamp(2rem, 4vw, 4rem);
  line-height: 1;
  font-weight: 900;
  text-shadow: 0 0 18px rgba(65, 216, 255, 0.28);
}

.gv-title span {
  color: var(--gv-blue);
  text-shadow: 0 0 20px rgba(0, 132, 255, 0.72);
}

.gv-subtitle {
  max-width: 720px;
  margin: 12px 0 0;
  color: var(--gv-muted);
  font-size: 1rem;
}

.gv-status {
  border: 1px solid rgba(65, 216, 255, 0.22);
  border-radius: 12px;
  padding: 12px 14px;
  background: rgba(0, 132, 255, 0.08);
  color: var(--gv-text);
}

.gv-workbench {
  gap: 18px;
}

.gv-panel,
.gv-panel > div {
  border-color: rgba(81, 173, 255, 0.24) !important;
}

.gv-panel {
  padding: 16px;
  border: 1px solid rgba(81, 173, 255, 0.22);
  border-radius: 16px;
  background: var(--gv-panel);
  box-shadow: 0 18px 60px rgba(0, 0, 0, 0.38);
}

.gv-panel label,
.gv-panel span,
.gv-panel p {
  color: var(--gv-text) !important;
}

.gv-note {
  color: var(--gv-muted);
}

#generate-btn {
  border: 1px solid rgba(65, 216, 255, 0.58) !important;
  background: linear-gradient(135deg, #0074ff, #2ed7ff) !important;
  color: #f7fbff !important;
  box-shadow: 0 0 24px rgba(0, 132, 255, 0.48);
}

button.primary:hover,
#generate-btn:hover {
  filter: brightness(1.08);
}

.gv-panel .wrap,
.gv-panel .block,
.gv-panel .form,
.gv-panel .container {
  background: var(--gv-panel-2) !important;
}

@media (max-width: 760px) {
  .gv-hero {
    grid-template-columns: 1fr;
    text-align: center;
  }

  .gv-logo {
    margin: 0 auto;
    width: 150px;
  }
}
"""

HERO_HTML = f"""
<section class="gv-hero">
  <img class="gv-logo" src="/file={LOGO_PATH}" alt="Gigaverse3D logo">
  <div>
    <p class="gv-kicker">Impressão 3D • Tecnologia • Universo</p>
    <h1 class="gv-title">Gigaverse<span>3D</span></h1>
    <p class="gv-subtitle">Imagem para Objetos 3D com geração local, visualização interativa e exportação em GLB.</p>
  </div>
</section>
"""

with gr.Blocks(title="Gigaverse3D imagem para Objetos 3D", css=CUSTOM_CSS) as demo:
    gr.HTML(HERO_HTML)
    gr.Markdown(DEVICE_BANNER, elem_classes="gv-status")

    with gr.Row(elem_classes="gv-workbench"):
        with gr.Column(scale=1, elem_classes="gv-panel"):
            with gr.Tabs():
                with gr.Tab("Uma foto"):
                    image_in = gr.Image(type="pil", label="Imagem de entrada", height=300)
                    gr.Markdown(
                        "_Dica: objeto único, centralizado, bem iluminado e fundo simples._",
                        elem_classes="gv-note",
                    )
                    recenter = gr.Checkbox(
                        value=True,
                        label="Centralizar e recortar o objeto (recomendado — mais fiel)",
                    )
                    btn = gr.Button("Gerar objeto 3D", variant="primary", elem_id="generate-btn")

                with gr.Tab("Multi-view (várias fotos)"):
                    gr.Markdown(
                        "_O **mesmo objeto** em vistas diferentes → geometria muito mais "
                        "fiel. A vista **Frente** é obrigatória; as outras ajudam._",
                        elem_classes="gv-note",
                    )
                    with gr.Row():
                        img_front = gr.Image(type="pil", label="Frente (obrigatória)", height=180)
                        img_back = gr.Image(type="pil", label="Trás", height=180)
                    with gr.Row():
                        img_left = gr.Image(type="pil", label="Esquerda", height=180)
                        img_right = gr.Image(type="pil", label="Direita", height=180)
                    btn_mv = gr.Button("Gerar (multi-view)", variant="primary", elem_id="generate-btn")

            quality_level = gr.Radio(
                ["Rascunho", "Equilibrado", "Máximo"],
                value=("Máximo" if DEVICE == "cuda" else "Equilibrado"),
                label="Preset de qualidade (ajusta os parâmetros abaixo)",
            )
            with gr.Accordion("Qualidade / parâmetros", open=DEVICE == "cuda"):
                steps = gr.Slider(10, 50, value=QP["steps"], step=5, label="Passos de difusão (mais = melhor/lento)")
                octree = gr.Slider(128, 512, value=QP["octree_resolution"], step=64, label="Resolução do octree (granularidade)")
                max_faces = gr.Slider(5000, 200000, value=QP["max_faces"], step=5000, label="Faces máximas (mais = mais detalhe)")
                smooth = gr.Slider(0, 20, value=0, step=1, label="Suavização da malha (Taubin; 0 = desligado)")
                target_mm = gr.Number(value=0, label="Tamanho p/ impressão em mm (0 = tamanho original)")
                seed = gr.Number(value=42, precision=0, label="Seed")
                remove_bg = gr.Checkbox(value=True, label="Remover fundo automaticamente")
                enhance = gr.Checkbox(
                    value=True,
                    label="Melhorar resolução da foto pequena (Real-ESRGAN / Lanczos)",
                )
                with_texture = gr.Checkbox(
                    value=TEXTURE_OK, label="Gerar textura PBR (só GPU)", interactive=TEXTURE_OK
                )
                want_stl = gr.Checkbox(
                    value=False,
                    label="Gerar .STL para impressão 3D (sólido/watertight)",
                )

        with gr.Column(scale=1, elem_classes="gv-panel"):
            model_out = gr.Model3D(label="Objeto 3D gerado", height=420)
            file_out = gr.File(label="Baixar .glb")
            file_out_stl = gr.File(label="Baixar .stl (impressão 3D)")
            status = gr.Markdown()

    # Seletor de qualidade -> atualiza os 3 sliders.
    _LEVELS = {"Rascunho": "rascunho", "Equilibrado": "equilibrado", "Máximo": "maximo"}

    def apply_level(level):
        p = level_preset(_LEVELS.get(level, ""))
        if not p:
            return gr.update(), gr.update(), gr.update()
        return p["steps"], p["octree_resolution"], p["max_faces"]

    quality_level.change(apply_level, inputs=[quality_level],
                         outputs=[steps, octree, max_faces])

    _outputs = [model_out, file_out, file_out_stl, status]
    btn.click(
        fn=generate,
        inputs=[image_in, steps, octree, max_faces, remove_bg, with_texture, recenter,
                enhance, smooth, target_mm, want_stl, seed],
        outputs=_outputs,
    )
    btn_mv.click(
        fn=generate_mv,
        inputs=[img_front, img_back, img_left, img_right, steps, octree, max_faces,
                remove_bg, with_texture, enhance, smooth, target_mm, want_stl, seed],
        outputs=_outputs,
    )

if __name__ == "__main__":
    # Fila evita execuções simultâneas (inferência pesada) e mostra progresso.
    demo.queue(max_size=8).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7861")),
        show_error=True,
        allowed_paths=[ASSET_DIR],
    )
