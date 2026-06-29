#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
img2mesh.py — CLI: converte foto(s) em objeto(s) 3D usando o Hunyuan3D.

A lógica de conversão vive em core.py (compartilhada com a interface web app.py).

Uso típico (dentro do container):
    python img2mesh.py --image /workspace/input/foto.png
    python img2mesh.py --input-dir /workspace/input --output-dir /workspace/output
"""

import argparse
import sys
from pathlib import Path

from core import IMG_EXTS, Hunyuan3DConverter, best_model, quality_preset, level_preset


def log(msg: str) -> None:
    print(f"[img2mesh] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Converte foto em objeto 3D (Hunyuan3D).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--image", type=str, help="Caminho de uma única imagem.")
    src.add_argument(
        "--input-dir",
        type=str,
        default="/workspace/input",
        help="Diretório com imagens a processar (usado se --image não for dado).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/output",
        help="Diretório de saída para os arquivos 3D.",
    )

    # model/subfolder/steps/octree/max-faces ficam None por padrão e são
    # resolvidos pelo preset do dispositivo (GPU = modelo completo + qualidade
    # máxima; CPU = mini + moderado). Passe o flag para sobrescrever.
    p.add_argument("--model", type=str, default=None,
                   help="Default: Hunyuan3D-2 (GPU) ou Hunyuan3D-2mini (CPU).")
    p.add_argument("--subfolder", type=str, default=None)
    p.add_argument("--variant", type=str, default="fp16")

    p.add_argument("--preset", type=str, default=None,
                   choices=["rascunho", "equilibrado", "maximo"],
                   help="Nível de qualidade (define steps/octree/max-faces; flags explícitos têm prioridade).")
    p.add_argument("--steps", type=int, default=None,
                   help="Passos de difusão. Default por dispositivo (GPU 50 / CPU 30).")
    p.add_argument("--octree-resolution", type=int, default=None,
                   help="Resolução do octree. Default por dispositivo (GPU 512 / CPU 256).")
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--max-faces", type=int, default=None,
                   help="Faces máximas. Default por dispositivo (GPU 120000 / CPU 40000).")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--smooth", type=int, default=0,
                   help="Iterações de suavização Taubin da malha (0 = desligado).")
    p.add_argument("--size-mm", type=float, default=0.0,
                   help="Escala a peça p/ que a maior aresta meça N mm (0 = original).")

    # Multi-view: várias vistas do MESMO objeto -> geometria mais fiel.
    # Se --front for dado, roda em modo multi-view (ignora --image/--input-dir).
    p.add_argument("--front", type=str, help="Vista frontal (ativa multi-view).")
    p.add_argument("--back", type=str, help="Vista traseira (multi-view).")
    p.add_argument("--left", type=str, help="Vista esquerda (multi-view).")
    p.add_argument("--right", type=str, help="Vista direita (multi-view).")

    p.add_argument("--no-rembg", action="store_true",
                   help="Não remover o fundo (use se a imagem já é PNG transparente).")
    p.add_argument("--no-recenter", action="store_true",
                   help="Não recortar/centralizar o objeto (por padrão reenquadra p/ mais fidelidade).")
    p.add_argument("--no-enhance", action="store_true",
                   help="Não melhorar/upscale fotos pequenas (por padrão usa Real-ESRGAN/Lanczos).")
    p.add_argument("--texture", action="store_true",
                   help="Tentar textura PBR. Só funciona com GPU CUDA; ignorado em CPU.")
    p.add_argument("--also-obj", action="store_true",
                   help="Exportar também .obj além do .glb.")
    p.add_argument("--stl", action="store_true",
                   help="Exportar .stl sólido/watertight para impressão 3D.")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def collect_images(args) -> list:
    if args.image:
        path = Path(args.image)
        if not path.is_file():
            log(f"ERRO: imagem não encontrada: {path}")
            sys.exit(1)
        return [path]
    in_dir = Path(args.input_dir)
    imgs = sorted(q for q in in_dir.iterdir() if q.suffix.lower() in IMG_EXTS) \
        if in_dir.is_dir() else []
    if not imgs:
        log(f"Nenhuma imagem encontrada em {in_dir}. Coloque um .png/.jpg lá.")
        sys.exit(1)
    return imgs


def main() -> None:
    args = parse_args()

    # Resolve os padrões dependentes do dispositivo (modelo + qualidade).
    qp = quality_preset(args.device)
    def_model, def_subfolder = best_model(args.device)
    model = args.model or def_model
    subfolder = args.subfolder or def_subfolder
    # Preset de nível define a base; flags explícitos têm prioridade; senão, o
    # preset do dispositivo.
    lp = level_preset(args.preset) if args.preset else {}
    steps = args.steps if args.steps is not None else lp.get("steps", qp["steps"])
    octree = args.octree_resolution if args.octree_resolution is not None \
        else lp.get("octree_resolution", qp["octree_resolution"])
    max_faces = args.max_faces if args.max_faces is not None \
        else lp.get("max_faces", qp["max_faces"])
    # Na GPU, liga textura PBR por padrão (foco em realismo); --texture força.
    with_texture = args.texture or qp["with_texture"]

    converter = Hunyuan3DConverter(
        model=model,
        subfolder=subfolder,
        variant=args.variant,
        device=args.device,
        enable_texture=with_texture,
        log=log,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Formatos extras exportados junto do .glb (na mesma passada de inferência).
    extra_formats = []
    if args.also_obj:
        extra_formats.append(".obj")
    if args.stl:
        extra_formats.append(".stl")

    # Parâmetros de geração comuns aos dois modos.
    common = dict(
        steps=steps,
        octree_resolution=octree,
        guidance_scale=args.guidance_scale,
        max_faces=max_faces,
        seed=args.seed,
        remove_bg=not args.no_rembg,
        with_texture=with_texture,
        enhance=not args.no_enhance,
        smooth=args.smooth,
        target_size_mm=args.size_mm,
        extra_formats=tuple(extra_formats),
        make_solid=args.stl,
    )

    # --- Modo multi-view: várias vistas do MESMO objeto -> uma malha ----------
    if args.front:
        images = {"front": args.front, "back": args.back,
                  "left": args.left, "right": args.right}
        for k, v in images.items():
            if v and not Path(v).is_file():
                log(f"ERRO: vista '{k}' não encontrada: {v}")
                sys.exit(1)
        glb_path = out_dir / "modelo_mv.glb"
        log("=== Multi-view: " + ", ".join(k for k, v in images.items() if v) + " ===")
        converter.convert_multiview(images, str(glb_path), **common)
        log("Tudo pronto. Arquivos em: " + str(out_dir))
        return

    # --- Modo padrão: 1 imagem -> 1 malha (lote do diretório) ----------------
    images = collect_images(args)
    log(f"{len(images)} imagem(ns) para processar.")
    for idx, img_path in enumerate(images, 1):
        log(f"=== [{idx}/{len(images)}] {img_path.name} ===")
        glb_path = out_dir / f"{img_path.stem}.glb"
        converter.convert(
            str(img_path),
            str(glb_path),
            recenter=not args.no_recenter,
            **common,
        )

    log("Tudo pronto. Arquivos em: " + str(out_dir))


if __name__ == "__main__":
    main()
