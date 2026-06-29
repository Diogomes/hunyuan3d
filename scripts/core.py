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


# ---------------------------------------------------------------------------
# Presets por dispositivo — o foco é QUALIDADE/REALISMO na GPU.
#
#   GPU (cuda): modelo COMPLETO (Hunyuan3D-2) — forma mais fiel e detalhada,
#               e habilita a TEXTURA PBR. Parâmetros pesados (octree/steps altos,
#               mais faces preservadas) porque a GPU dá conta.
#   CPU:        modelo "mini" — única opção viável sem GPU (mais leve), só forma.
#               Parâmetros moderados para não levar horas.
# ---------------------------------------------------------------------------
MODEL_PRESETS = {
    "cuda": ("tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0"),
    "cpu": ("tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini"),
}

QUALITY_PRESETS = {
    # max_faces alto preserva detalhe da geometria de octree 512 (bom p/ realismo
    # e impressão 3D); a textura PBR só liga de fato se houver CUDA.
    "cuda": dict(steps=50, octree_resolution=512, guidance_scale=7.5,
                 max_faces=120000, with_texture=True),
    "cpu": dict(steps=30, octree_resolution=256, guidance_scale=7.5,
                max_faces=40000, with_texture=False),
}


def best_model(device: str = "auto"):
    """(model, subfolder) recomendado para o dispositivo resolvido."""
    dev, _ = resolve_device(device)
    return MODEL_PRESETS.get(dev, MODEL_PRESETS["cpu"])


def quality_preset(device: str = "auto") -> dict:
    """Parâmetros de geração recomendados (cópia mutável) para o dispositivo."""
    dev, _ = resolve_device(device)
    return dict(QUALITY_PRESETS.get(dev, QUALITY_PRESETS["cpu"]))


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

        # Upscaler da foto de entrada (carregado sob demanda na 1ª melhoria).
        self._upsampler = None
        self._esrgan_warned = False

        # Pipeline multi-view (Hunyuan3D-2mv), carregado sob demanda.
        self._mv_pipe = None
        self._mv_model = "tencent/Hunyuan3D-2mv"
        self._mv_subfolder = "hunyuan3d-dit-v2-mv"

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

    def _realesrgan(self, rgb: Image.Image, scale: int):
        """Super-resolução aprendida (Real-ESRGAN). Retorna PIL RGB ou None.

        Usa o backend ai-forever/Real-ESRGAN (API em PIL, sem `basicsr` — este
        é incompatível com torchvision >=0.17). Totalmente guardado: qualquer
        falha (lib ausente, sem pesos, OOM) retorna None p/ cair no Lanczos.
        """
        try:
            if self._upsampler is None:
                from RealESRGAN import RealESRGAN

                models_dir = os.environ.get("HY3DGEN_MODELS", "/workspace/models")
                weights = os.path.join(models_dir, "realesrgan", "RealESRGAN_x4.pth")
                os.makedirs(os.path.dirname(weights), exist_ok=True)
                m = RealESRGAN(self.device, scale=4)
                m.load_weights(weights, download=True)
                self._upsampler = m
            out = self._upsampler.predict(rgb)  # x4, retorna PIL
            return out
        except Exception as e:  # noqa: BLE001
            if not self._esrgan_warned:
                self.log(f"Real-ESRGAN indisponível ({e}); usando Lanczos.")
                self._esrgan_warned = True
            return None

    def _enhance_image(self, image: Image.Image, target_min: int = 1024,
                       max_side: int = 2048) -> Image.Image:
        """Aumenta a resolução de fotos pequenas (lado menor < target_min).

        Tenta Real-ESRGAN; cai para Lanczos. Preserva o canal alfa. Fotos já
        grandes passam intactas. `max_side` limita o tamanho final (memória).
        """
        w, h = image.size
        if min(w, h) >= target_min:
            return image

        alpha = image.split()[-1] if image.mode in ("RGBA", "LA") else None
        rgb = image.convert("RGB")

        up = self._realesrgan(rgb, 4)
        if up is not None:
            self.log(f"Upscale Real-ESRGAN: {(w, h)} -> {up.size}")
        else:
            import math
            scale = min(4, max(2, math.ceil(target_min / min(w, h))))
            up = rgb.resize((w * scale, h * scale), Image.LANCZOS)
            self.log(f"Upscale Lanczos x{scale}: {(w, h)} -> {up.size}")

        # Não estourar memória com imagens enormes.
        if max(up.size) > max_side:
            r = max_side / max(up.size)
            up = up.resize((max(1, int(up.size[0] * r)), max(1, int(up.size[1] * r))),
                           Image.LANCZOS)

        if alpha is not None:
            up = up.convert("RGBA")
            up.putalpha(alpha.resize(up.size, Image.LANCZOS))
        return up

    @staticmethod
    def _recenter(image: Image.Image, frame_ratio: float = 0.9) -> Image.Image:
        """Recorta no contorno (alfa) e centraliza num quadro quadrado com margem.

        O Hunyuan3D gera formas mais fiéis quando o objeto está centralizado e
        preenchendo o quadro. `frame_ratio` = fração do lado ocupada pelo objeto
        (0.9 => 10% de margem). Imagem sem alfa/totalmente transparente passa
        intacta.
        """
        if image.mode != "RGBA":
            return image
        bbox = image.split()[-1].getbbox()  # caixa do conteúdo não-transparente
        if bbox is None:
            return image  # nada opaco para enquadrar
        obj = image.crop(bbox)
        w, h = obj.size
        side = max(1, int(round(max(w, h) / max(frame_ratio, 0.1))))
        canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        canvas.paste(obj, ((side - w) // 2, (side - h) // 2))
        return canvas

    def _prepare_image(self, image, remove_bg: bool, recenter: bool = True,
                       frame_ratio: float = 0.9, enhance: bool = True) -> Image.Image:
        if isinstance(image, str):
            image = Image.open(image)
        # Melhoria/upscale da foto crua antes de tudo (mais detalhe p/ o rembg
        # e p/ o encoder do modelo). Só age em imagens pequenas.
        if enhance:
            image = self._enhance_image(image)
        image = image.convert("RGBA")
        if remove_bg:
            has_alpha = image.mode == "RGBA" and image.getextrema()[3][0] < 255
            if not has_alpha:
                self.log("Removendo fundo...")
                image = self.rembg(image.convert("RGB"))
        if recenter:
            before = image.size
            image = self._recenter(image, frame_ratio)
            if image.size != before:
                self.log(f"Reenquadrado: {before} -> {image.size} (objeto centralizado)")
        return image

    def _make_watertight(self, mesh):
        """Repara a malha para um sólido fechado (bom p/ impressão 3D).

        Funde vértices, fecha furos e corrige normais/orientação. Não-fatal:
        qualquer erro mantém a malha original. Cenas texturizadas viram uma
        malha única (a cor não importa para STL de impressão).
        """
        try:
            import trimesh

            m = mesh
            if isinstance(m, trimesh.Scene):
                m = trimesh.util.concatenate(tuple(m.geometry.values()))
            m = m.copy()
            m.merge_vertices()
            trimesh.repair.fill_holes(m)
            trimesh.repair.fix_winding(m)
            trimesh.repair.fix_normals(m)
            m.remove_unreferenced_vertices()
            self.log(f"Reparo p/ impressão: watertight={m.is_watertight}")
            return m
        except Exception as e:  # noqa: BLE001
            self.log(f"AVISO: reparo watertight falhou ({e}). Mantendo malha original.")
            return mesh

    def _smooth_mesh(self, mesh, iterations: int):
        """Suaviza a superfície (Taubin — preserva volume). Não-fatal."""
        if not iterations or iterations <= 0:
            return mesh
        try:
            import trimesh

            if isinstance(mesh, trimesh.Scene):
                return mesh  # não suaviza cena texturizada (preserva UVs)
            trimesh.smoothing.filter_taubin(mesh, iterations=int(iterations))
            self.log(f"Suavização Taubin: {iterations} iterações")
        except Exception as e:  # noqa: BLE001
            self.log(f"AVISO: suavização falhou ({e}). Mantendo malha.")
        return mesh

    def _scale_mesh(self, mesh, target_size_mm: float):
        """Escala a malha p/ que a maior aresta do bounding-box meça target_size_mm.

        Útil p/ impressão 3D (GLB/STL em mm). Não-fatal. Em cena, escala cada
        geometria pelo mesmo fator.
        """
        if not target_size_mm or target_size_mm <= 0:
            return mesh
        try:
            import trimesh

            geoms = (mesh.geometry.values() if isinstance(mesh, trimesh.Scene) else [mesh])
            longest = 0.0
            for g in geoms:
                longest = max(longest, float(max(g.bounding_box.extents)))
            if longest > 0:
                factor = target_size_mm / longest
                mesh.apply_scale(factor)
                self.log(f"Escala p/ impressão: maior aresta -> {target_size_mm:.1f} mm "
                         f"(fator {factor:.4f})")
        except Exception as e:  # noqa: BLE001
            self.log(f"AVISO: escala falhou ({e}). Mantendo tamanho.")
        return mesh

    def _lay_on_floor(self, mesh):
        """Centraliza em XY e apoia a base em Z=0 (pronto p/ a mesa de impressão).

        Não-fatal. Em cena, opera no conjunto (mesma translação p/ todas as geos).
        """
        try:
            b = mesh.bounds  # [[minx,miny,minz],[maxx,maxy,maxz]]
            tx = -(b[0][0] + b[1][0]) / 2.0
            ty = -(b[0][1] + b[1][1]) / 2.0
            tz = -b[0][2]
            mesh.apply_translation([tx, ty, tz])
            self.log("Posicionado na mesa de impressão (centro XY, base em Z=0).")
        except Exception as e:  # noqa: BLE001
            self.log(f"AVISO: posicionamento na mesa falhou ({e}). Mantendo posição.")
        return mesh

    def _postprocess_export(self, mesh, image, output_path, *, max_faces,
                            with_texture, smooth, target_size_mm,
                            extra_formats, make_solid, t_img):
        """Limpeza -> suavização -> textura -> escala -> export (principal + extras).

        Compartilhado entre `convert` (1 foto) e `convert_multiview`.
        """
        self.log("Limpando malha (floaters / faces degeneradas / redução)...")
        mesh = self.floater(mesh)
        mesh = self.degenerate(mesh)
        mesh = self.reducer(mesh, max_facenum=max_faces)

        # Suaviza antes da textura (mudar vértices depois quebraria as UVs).
        mesh = self._smooth_mesh(mesh, smooth)

        if with_texture and self.tex_pipe is not None:
            self.log("Aplicando textura (GPU)...")
            mesh = self.tex_pipe(mesh, image=image)

        # Escala física (mm) por último, p/ valer no GLB e nos formatos extras.
        mesh = self._scale_mesh(mesh, target_size_mm)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        mesh.export(output_path)
        self.log(f"Salvo: {output_path} (em {time.time() - t_img:.1f}s)")

        # Formatos extras (ex.: .stl para impressão). Opcionalmente solidifica
        # (watertight); o GLB principal acima preserva a textura para o viewer.
        if extra_formats:
            export_mesh = mesh
            if make_solid:
                export_mesh = self._make_watertight(mesh)
                export_mesh = self._lay_on_floor(export_mesh)
            stem = os.path.splitext(output_path)[0]
            for ext in extra_formats:
                ext = ext if ext.startswith(".") else "." + ext
                ep = stem + ext
                export_mesh.export(ep)
                self.log(f"Salvo: {ep}")
        return output_path

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
        recenter: bool = True,
        frame_ratio: float = 0.9,
        enhance: bool = True,
        smooth: int = 0,
        target_size_mm: float = 0.0,
        extra_formats=(),
        make_solid: bool = False,
    ) -> str:
        """Converte uma imagem (PIL ou caminho) em arquivo 3D. Retorna o caminho."""
        t_img = time.time()
        image = self._prepare_image(image, remove_bg, recenter, frame_ratio, enhance)

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

        return self._postprocess_export(
            mesh, image, output_path,
            max_faces=max_faces, with_texture=with_texture, smooth=smooth,
            target_size_mm=target_size_mm, extra_formats=extra_formats,
            make_solid=make_solid, t_img=t_img,
        )

    def _load_mv(self):
        """Carrega (1x) o pipeline multi-view Hunyuan3D-2mv."""
        if self._mv_pipe is None:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

            self.log(f"Carregando pipeline multi-view {self._mv_model} "
                     f"(baixa os pesos na 1ª vez)...")
            t0 = time.time()
            self._mv_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                self._mv_model,
                subfolder=self._mv_subfolder,
                variant="fp16",
                device=self.device,
                dtype=self.dtype,
            )
            self.log(f"Pipeline multi-view pronto em {time.time() - t0:.1f}s")
        return self._mv_pipe

    def convert_multiview(
        self,
        images: dict,
        output_path: str,
        *,
        steps: int = 50,
        octree_resolution: int = 512,
        guidance_scale: float = 7.5,
        max_faces: int = 120000,
        seed: int = 42,
        remove_bg: bool = True,
        with_texture: bool = False,
        enhance: bool = True,
        smooth: int = 0,
        target_size_mm: float = 0.0,
        extra_formats=(),
        make_solid: bool = False,
    ) -> str:
        """Gera 3D a partir de VÁRIAS vistas do mesmo objeto (geometria mais fiel).

        `images` é um dict {vista: imagem} com chaves entre 'front'/'back'/
        'left'/'right' (PIL ou caminho). 'front' é obrigatória. Não recentraliza
        (manteria as vistas em escalas inconsistentes).
        """
        t_img = time.time()
        valid = ("front", "back", "left", "right")
        views = {}
        for key in valid:
            im = images.get(key)
            if im is None:
                continue
            # Sem recenter: as vistas precisam de enquadramento consistente.
            views[key] = self._prepare_image(im, remove_bg, recenter=False,
                                             enhance=enhance).convert("RGBA")
        if "front" not in views:
            raise ValueError("Multi-view exige ao menos a vista 'front'.")
        self.log(f"Multi-view com vistas: {', '.join(views.keys())}")

        pipe = self._load_mv()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        mesh = pipe(
            image=views,
            num_inference_steps=steps,
            octree_resolution=octree_resolution,
            guidance_scale=guidance_scale,
            generator=generator,
            mc_algo=self.mc_algo,
            output_type="trimesh",
        )[0]

        # Textura usa a vista frontal como referência.
        return self._postprocess_export(
            mesh, views["front"], output_path,
            max_faces=max_faces, with_texture=with_texture, smooth=smooth,
            target_size_mm=target_size_mm, extra_formats=extra_formats,
            make_solid=make_solid, t_img=t_img,
        )
