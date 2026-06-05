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

from core import IMG_EXTS, Hunyuan3DConverter


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

    p.add_argument("--model", type=str, default="tencent/Hunyuan3D-2mini")
    p.add_argument("--subfolder", type=str, default="hunyuan3d-dit-v2-mini")
    p.add_argument("--variant", type=str, default="fp16")

    p.add_argument("--steps", type=int, default=30,
                   help="Passos de difusão. Mais = melhor e mais lento (CPU: 20-50).")
    p.add_argument("--octree-resolution", type=int, default=256,
                   help="Resolução do octree. Mais = mais granularidade e mais lento.")
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--max-faces", type=int, default=40000)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--no-rembg", action="store_true",
                   help="Não remover o fundo (use se a imagem já é PNG transparente).")
    p.add_argument("--texture", action="store_true",
                   help="Tentar textura PBR. Só funciona com GPU CUDA; ignorado em CPU.")
    p.add_argument("--also-obj", action="store_true",
                   help="Exportar também .obj além do .glb.")
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

    converter = Hunyuan3DConverter(
        model=args.model,
        subfolder=args.subfolder,
        variant=args.variant,
        device=args.device,
        enable_texture=args.texture,
        log=log,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(args)
    log(f"{len(images)} imagem(ns) para processar.")

    for idx, img_path in enumerate(images, 1):
        log(f"=== [{idx}/{len(images)}] {img_path.name} ===")
        glb_path = out_dir / f"{img_path.stem}.glb"
        converter.convert(
            str(img_path),
            str(glb_path),
            steps=args.steps,
            octree_resolution=args.octree_resolution,
            guidance_scale=args.guidance_scale,
            max_faces=args.max_faces,
            seed=args.seed,
            remove_bg=not args.no_rembg,
            with_texture=args.texture,
        )
        if args.also_obj:
            # Reexporta a partir do .glb já gerado (evita rodar a inferência 2x).
            import trimesh
            obj_path = out_dir / f"{img_path.stem}.obj"
            trimesh.load(str(glb_path)).export(str(obj_path))
            log(f"Salvo: {obj_path}")

    log("Tudo pronto. Arquivos em: " + str(out_dir))


if __name__ == "__main__":
    main()
