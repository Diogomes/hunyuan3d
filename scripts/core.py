#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
core.py — Lógica compartilhada de conversão foto -> objeto 3D (Hunyuan3D).

Usado tanto pela CLI (img2mesh.py) quanto pela interface web (app.py).
Carrega os pipelines uma única vez e expõe um método convert() reutilizável.
"""

import os
import time
import torch
from PIL import Image

# Extensões de imagem aceitas ao varrer um diretório.
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def resolve_device(choice: str = "auto"):
    """Resolve dispositivo e dtype. fp16 só na GPU; CPU sempre fp32."""
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        choice = "cpu"
    dtype = torch.float16 if choice == "cuda" else torch.float32
    return choice, dtype


class Hunyuan3DConverter:
    """Encapsula os pipelines de forma (e textura, se houver GPU)."""

    def __init__(
        self,
        model: str = "tencent/Hunyuan3D-2mini",
        subfolder: str = "hunyuan3d-dit-v2-mini",
        variant: str = "fp16",
        device: str = "auto",
        enable_texture: bool = False,
        texture_model: str = "tencent/Hunyuan3D-2",
        log=print,
    ):
        self.log = log
        self.device, self.dtype = resolve_device(device)
        # 'dmc' (Differentiable Marching Cubes) gera malhas mais suaves, mas exige
        # a lib 'diso' (CUDA). Em CPU — ou sem diso — usa 'mc' (skimage), que
        # funciona em qualquer lugar.
        self.mc_algo = "mc"
        if self.device == "cuda":
            try:
                import diso  # noqa: F401
                self.mc_algo = "dmc"
            except Exception:
                self.log("diso indisponível — usando marching cubes 'mc'.")

        if self.device == "cpu":
            torch.set_num_threads(os.cpu_count() or 4)

        self.log(f"Dispositivo: {self.device} | dtype: {self.dtype}")
        self.log(f"Modelo: {model} / {subfolder} (variant={variant})")

        from hy3dgen.shapegen import (
            Hunyuan3DDiTFlowMatchingPipeline,
            FaceReducer,
            FloaterRemover,
            DegenerateFaceRemover,
        )
        from hy3dgen.rembg import BackgroundRemover

        self.log("Carregando pipeline de forma (baixa os pesos na 1ª vez)...")
        t0 = time.time()
        self.shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model,
            subfolder=subfolder,
            variant=variant,
            device=self.device,
            dtype=self.dtype,
        )
        self.log(f"Pipeline pronto em {time.time() - t0:.1f}s")

        self.rembg = BackgroundRemover()
        self.floater = FloaterRemover()
        self.degenerate = DegenerateFaceRemover()
        self.reducer = FaceReducer()

        # Textura PBR: só com GPU CUDA (rasterizador customizado).
        self.tex_pipe = None
        self.texture_available = self.device == "cuda"
        if enable_texture:
            if self.texture_available:
                try:
                    from hy3dgen.texgen import Hunyuan3DPaintPipeline

                    # Os pesos de textura (paint) ficam no repo completo; o modelo
                    # mini de forma não os inclui.
                    self.log(f"Carregando pipeline de textura (GPU) de {texture_model}...")
                    self.tex_pipe = Hunyuan3DPaintPipeline.from_pretrained(texture_model)
                except Exception as e:  # noqa: BLE001
                    self.log(f"AVISO: falha ao carregar textura ({e}). Seguindo sem.")
                    self.tex_pipe = None
            else:
                self.log("AVISO: textura exige GPU NVIDIA/CUDA — desativada em CPU.")

    def _prepare_image(self, image, remove_bg: bool) -> Image.Image:
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGBA")
        if remove_bg:
            has_alpha = image.mode == "RGBA" and image.getextrema()[3][0] < 255
            if not has_alpha:
                self.log("Removendo fundo...")
                image = self.rembg(image.convert("RGB"))
        return image

    def convert(
        self,
        image,
        output_path: str,
        *,
        steps: int = 30,
        octree_resolution: int = 256,
        guidance_scale: float = 7.5,
        max_faces: int = 40000,
        seed: int = 42,
        remove_bg: bool = True,
        with_texture: bool = False,
    ) -> str:
        """Converte uma imagem (PIL ou caminho) em arquivo 3D. Retorna o caminho."""
        t_img = time.time()
        image = self._prepare_image(image, remove_bg)

        self.log(f"Gerando forma (steps={steps}, octree={octree_resolution})...")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        mesh = self.shape_pipe(
            image=image,
            num_inference_steps=steps,
            octree_resolution=octree_resolution,
            guidance_scale=guidance_scale,
            generator=generator,
            mc_algo=self.mc_algo,
            output_type="trimesh",
        )[0]

        self.log("Limpando malha (floaters / faces degeneradas / redução)...")
        mesh = self.floater(mesh)
        mesh = self.degenerate(mesh)
        mesh = self.reducer(mesh, max_facenum=max_faces)

        if with_texture and self.tex_pipe is not None:
            self.log("Aplicando textura (GPU)...")
            mesh = self.tex_pipe(mesh, image=image)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        mesh.export(output_path)
        self.log(f"Salvo: {output_path} (em {time.time() - t_img:.1f}s)")
        return output_path
