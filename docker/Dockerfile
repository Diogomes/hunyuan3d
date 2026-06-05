# ============================================================================
#  Hunyuan3D — imagem CPU-only para geração de FORMA (mesh) a partir de fotos.
#
#  Esta máquina não tem GPU NVIDIA/CUDA, então:
#   - PyTorch é instalado na variante CPU.
#   - O pipeline de FORMA roda na CPU (lento, porém funcional).
#   - O pipeline de TEXTURA (custom_rasterizer/differentiable_renderer) exige
#     CUDA e NÃO é compilado aqui. O driver detecta CUDA em runtime e só ativa
#     textura se houver GPU — útil se um dia rodar esta mesma imagem na nuvem.
# ============================================================================
FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Caches de modelo apontam para /workspace/models (volume montado), para
    # que os downloads (~2-3 GB) persistam entre execuções.
    HF_HOME=/workspace/models/hf \
    HY3DGEN_MODELS=/workspace/models/hy3dgen \
    PYTHONPATH=/opt/Hunyuan3D

# Bibliotecas de sistema exigidas por pymeshlab / opencv-headless / onnxruntime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libgl1 \
        libopengl0 \
        libegl1 \
        libglu1-mesa \
        libglib2.0-0 \
        libxrender1 \
        libxext6 \
        libsm6 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU (versões testadas pelo projeto, índice oficial de wheels CPU).
RUN pip install --upgrade pip && \
    pip install \
        torch==2.5.1 \
        torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cpu

# Demais dependências do hy3dgen (variante CPU, sem módulos de textura CUDA).
COPY docker/requirements-cpu.txt /tmp/requirements-cpu.txt
RUN pip install -r /tmp/requirements-cpu.txt

# Código-fonte do Hunyuan3D-2 (fica disponível via PYTHONPATH=/opt/Hunyuan3D).
# Fixar um commit garante builds reproduzíveis; atualize quando quiser.
ARG HY3D_REPO=https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git
ARG HY3D_REF=main
RUN git clone --depth 1 --branch "${HY3D_REF}" "${HY3D_REPO}" /opt/Hunyuan3D

# Nosso driver e ponto de entrada.
WORKDIR /workspace
COPY scripts/img2mesh.py /workspace/scripts/img2mesh.py
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--help"]
