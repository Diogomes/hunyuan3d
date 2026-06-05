#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
img2mesh.py — Converte foto(s) em objeto(s) 3D usando o Hunyuan3D.

Projetado para rodar em CPU (esta máquina não tem GPU NVIDIA), mas detecta
CUDA automaticamente: se houver GPU, usa fp16 e habilita textura; em CPU usa
fp32, marching cubes puro ('mc') e gera apenas a FORMA (mesh).

Uso típico (dentro do container):
    python img2mesh.py --image /workspace/input/foto.png
    python img2mesh.py --input-dir /workspace/input --output-dir /workspace/output

Os pesos do modelo são baixados do HuggingFace na primeira execução e
ficam em /workspace/models (volume montado).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

# Extensões de imagem aceitas ao varrer um diretório.
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


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

    # Modelo
    p.add_argument(
        "--model",
        type=str,
        default="tencent/Hunyuan3D-2mini",
        help="Repositório HuggingFace do modelo de forma.",
    )
    p.add_argument(
        "--subfolder",
        type=str,
        default="hunyuan3d-dit-v2-mini",
        help="Subpasta dos pesos dentro do repositório.",
    )
    p.add_argument(
        "--variant",
        type=str,
        default="fp16",
        help="Variante dos pesos no HF (geralmente só existe 'fp16').",
    )

    # Qualidade / granularidade
    p.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Passos de difusão. Mais = melhor e mais lento (CPU: 20-50).",
    )
    p.add_argument(
        "--octree-resolution",
        type=int,
        default=256,
        help="Resolução do octree. Mais = mais detalhe/granularidade e mais lento (256/384/512).",
    )
    p.add_argument(
        "--guidance-scale", type=float, default=7.5, help="Aderência à imagem."
    )
    p.add_argument(
        "--max-faces",
        type=int,
        default=40000,
        help="Limite de faces após simplificação (qualidade x tamanho do arquivo).",
    )
    p.add_argument("--seed", type=int, default=42, help="Semente para reprodutibilidade.")

    # Comportamento
    p.add_argument(
        "--no-rembg",
        action="store_true",
        help="Não remover o fundo (use se a imagem já tiver fundo transparente).",
    )
    p.add_argument(
        "--texture",
        action="store_true",
        help="Tentar gerar textura PBR. Só funciona com GPU CUDA; ignorado em CPU.",
    )
    p.add_argument(
        "--also-obj",
        action="store_true",
        help="Exportar também .obj além do .glb.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Dispositivo de execução.",
    )
    return p.parse_args()


def resolve_device(choice: str):
    """Resolve dispositivo e dtype. fp16 só na GPU; CPU sempre fp32."""
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        log("AVISO: CUDA pedido mas indisponível — caindo para CPU.")
        choice = "cpu"
    dtype = torch.float16 if choice == "cuda" else torch.float32
    return choice, dtype


def collect_images(args) -> list:
    if args.image:
        path = Path(args.image)
        if not path.is_file():
            log(f"ERRO: imagem não encontrada: {path}")
            sys.exit(1)
        return [path]
    in_dir = Path(args.input_dir)
    imgs = sorted(
        q for q in in_dir.iterdir() if q.suffix.lower() in IMG_EXTS
    ) if in_dir.is_dir() else []
    if not imgs:
        log(f"Nenhuma imagem encontrada em {in_dir}. Coloque um .png/.jpg lá.")
        sys.exit(1)
    return imgs


def main() -> None:
    args = parse_args()
    device, dtype = resolve_device(args.device)

    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    log(f"Dispositivo: {device} | dtype: {dtype}")
    log(f"Modelo: {args.model} / {args.subfolder} (variant={args.variant})")

    # Imports do hy3dgen (disponível via PYTHONPATH=/opt/Hunyuan3D).
    from hy3dgen.shapegen import (
        Hunyuan3DDiTFlowMatchingPipeline,
        FaceReducer,
        FloaterRemover,
        DegenerateFaceRemover,
    )
    from hy3dgen.rembg import BackgroundRemover

    # Em CPU, o marching cubes precisa ser 'mc' (puro Python/skimage);
    # 'dmc' exige a lib 'diso' compilada com CUDA.
    mc_algo = "mc" if device == "cpu" else "dmc"

    log("Carregando pipeline de forma (baixa os pesos na 1ª vez)...")
    t0 = time.time()
    shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model,
        subfolder=args.subfolder,
        variant=args.variant,
        device=device,
        dtype=dtype,
    )
    log(f"Pipeline pronto em {time.time() - t0:.1f}s")

    # Textura: só se houver CUDA e o usuário pedir.
    tex_pipe = None
    if args.texture:
        if device == "cuda":
            try:
                from hy3dgen.texgen import Hunyuan3DPaintPipeline

                log("Carregando pipeline de textura (GPU)...")
                tex_pipe = Hunyuan3DPaintPipeline.from_pretrained(args.model)
            except Exception as e:  # noqa: BLE001
                log(f"AVISO: falha ao carregar textura ({e}). Seguindo sem textura.")
        else:
            log(
                "AVISO: --texture ignorado. A geração de textura exige GPU NVIDIA/CUDA "
                "(rasterizador customizado). Em CPU só a forma é gerada."
            )

    rembg = None if args.no_rembg else BackgroundRemover()
    floater = FloaterRemover()
    degenerate = DegenerateFaceRemover()
    reducer = FaceReducer()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(args)
    log(f"{len(images)} imagem(ns) para processar.")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    for idx, img_path in enumerate(images, 1):
        log(f"=== [{idx}/{len(images)}] {img_path.name} ===")
        t_img = time.time()

        image = Image.open(img_path).convert("RGBA")
        # Remove fundo se a imagem não tiver canal alfa útil.
        if rembg is not None:
            has_alpha = image.mode == "RGBA" and image.getextrema()[3][0] < 255
            if not has_alpha:
                log("Removendo fundo...")
                image = rembg(image.convert("RGB"))

        log(f"Gerando forma (steps={args.steps}, octree={args.octree_resolution})...")
        mesh = shape_pipe(
            image=image,
            num_inference_steps=args.steps,
            octree_resolution=args.octree_resolution,
            guidance_scale=args.guidance_scale,
            generator=generator,
            mc_algo=mc_algo,
            output_type="trimesh",
        )[0]

        log("Limpando malha (floaters / faces degeneradas / redução)...")
        mesh = floater(mesh)
        mesh = degenerate(mesh)
        mesh = reducer(mesh, max_facenum=args.max_faces)

        if tex_pipe is not None:
            log("Aplicando textura (GPU)...")
            mesh = tex_pipe(mesh, image=image)

        stem = img_path.stem
        glb_path = out_dir / f"{stem}.glb"
        mesh.export(str(glb_path))
        log(f"Salvo: {glb_path}")

        if args.also_obj:
            obj_path = out_dir / f"{stem}.obj"
            mesh.export(str(obj_path))
            log(f"Salvo: {obj_path}")

        log(f"Concluído em {time.time() - t_img:.1f}s")

    log("Tudo pronto. Arquivos em: " + str(out_dir))


if __name__ == "__main__":
    main()
