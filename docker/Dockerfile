# ============================================================================
#  Hunyuan3D — imagem GPU/CUDA: geração de FORMA + TEXTURA PBR.
# ============================================================================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/models/hf \
    HY3DGEN_MODELS=/workspace/models/hy3dgen \
    PYTHONPATH="/opt/Hunyuan3D:/workspace/scripts" \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip \
        git build-essential ninja-build \
        libgl1 libopengl0 libegl1 libglu1-mesa libglib2.0-0 \
        libxrender1 libxext6 libsm6 libgomp1 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip && \
    pip install torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu124

COPY docker/requirements-cpu.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && \
    (pip install diso || echo "WARN: diso falhou — usará marching cubes 'mc'")

ARG HY3D_REPO=https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git
ARG HY3D_REF=main
RUN git clone --depth 1 --branch "${HY3D_REF}" "${HY3D_REPO}" /opt/Hunyuan3D

RUN cd /opt/Hunyuan3D/hy3dgen/texgen/custom_rasterizer && \
    (pip install --no-build-isolation -e . || echo "WARN: custom_rasterizer falhou")

RUN cd /opt/Hunyuan3D/hy3dgen/texgen/differentiable_renderer && \
    (pip install --no-build-isolation -e . || python setup.py install || echo "WARN: differentiable_renderer falhou")

WORKDIR /workspace
COPY scripts/ /workspace/scripts/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--help"]