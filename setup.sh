#!/usr/bin/env bash
# ============================================================================
#  setup.sh — Prepara e sobe o projeto Foto -> 3D (Hunyuan3D) passo a passo.
#
#  Detecta automaticamente o sistema e a GPU:
#    - Com GPU NVIDIA acessível ao Docker  -> imagem CUDA (forma + textura PBR)
#    - Sem GPU                              -> imagem CPU  (só forma, mais lento)
#
#  Roda em Linux e no Windows via WSL2 / Git Bash (com Docker Desktop).
#
#  Uso:
#    ./setup.sh                 # detecta GPU e sobe a interface web
#    FORCE_MODE=cpu ./setup.sh  # força CPU
#    FORCE_MODE=gpu ./setup.sh  # força GPU
#    PORT=7870 ./setup.sh       # muda a porta da interface
#    NO_WEB=1 ./setup.sh        # só builda, não sobe o servidor
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PORT="${PORT:-7861}"
CONTAINER="hunyuan3d-web"

# --- helpers de log -------------------------------------------------------
c_step()  { printf "\n\033[1;36m▶ %s\033[0m\n" "$*"; }
c_ok()    { printf "\033[1;32m✓ %s\033[0m\n" "$*"; }
c_warn()  { printf "\033[1;33m! %s\033[0m\n" "$*"; }
c_err()   { printf "\033[1;31m✗ %s\033[0m\n" "$*" >&2; }

# --- 1) pré-requisitos ----------------------------------------------------
c_step "1/6  Verificando pré-requisitos"
if ! command -v docker >/dev/null 2>&1; then
    c_err "Docker não encontrado. Instale o Docker (ou Docker Desktop no Windows)."
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    c_err "O daemon do Docker não está rodando. Inicie o Docker e tente de novo."
    exit 1
fi
c_ok "Docker disponível"

# --- 2) detectar SO e GPU -------------------------------------------------
c_step "2/6  Detectando sistema e GPU"
OS="$(uname -s)"
c_ok "Sistema: ${OS}"

GPU=0
if [ "${FORCE_MODE:-}" = "cpu" ]; then
    c_warn "FORCE_MODE=cpu — ignorando GPU."
elif [ "${FORCE_MODE:-}" = "gpu" ]; then
    GPU=1
    c_warn "FORCE_MODE=gpu — assumindo GPU disponível."
else
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        if docker info 2>/dev/null | grep -qiE 'nvidia|Runtimes:.*nvidia'; then
            GPU=1
        else
            c_warn "GPU NVIDIA detectada, mas o Docker não a expõe."
            c_warn "Instale o nvidia-container-toolkit (Linux) ou habilite GPU no"
            c_warn "Docker Desktop (Windows/WSL2). Seguindo em modo CPU por ora."
        fi
    fi
fi

if [ "$GPU" = 1 ]; then
    MODE="gpu";  DOCKERFILE="docker/Dockerfile.gpu"; IMAGE="hunyuan3d-gpu:local"
    c_ok "Modo: GPU (forma + textura PBR)"
else
    MODE="cpu";  DOCKERFILE="docker/Dockerfile";     IMAGE="hunyuan3d-cpu:local"
    c_ok "Modo: CPU (só forma — lento; textura exige GPU)"
fi

# --- 3) pastas de trabalho ------------------------------------------------
c_step "3/6  Preparando pastas (input / output / models)"
mkdir -p input output models
c_ok "Pastas prontas em $ROOT"

# --- 4) build da imagem ---------------------------------------------------
c_step "4/6  Construindo imagem Docker: ${IMAGE}  (pode demorar na 1ª vez)"
# No Linux usamos a rede do host no build (workaround de egress em alguns
# ambientes; inofensivo em redes normais). Em Windows/macOS isso não se aplica.
BUILD_NET=()
[ "$OS" = "Linux" ] && BUILD_NET=(--network=host)
docker build "${BUILD_NET[@]}" -f "$DOCKERFILE" -t "$IMAGE" .
c_ok "Imagem construída: ${IMAGE}"

# --- 5) subir a interface web ---------------------------------------------
if [ "${NO_WEB:-0}" = "1" ]; then
    c_step "5/6  NO_WEB=1 — pulando o start do servidor"
    c_ok "Build concluído. Para subir depois: docker run ... ${IMAGE} python /workspace/scripts/app.py"
    exit 0
fi

c_step "5/6  Subindo a interface web (porta ${PORT})"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

RUN_NET=(); GPU_FLAG=()
if [ "$OS" = "Linux" ]; then
    # Rede do host: o Gradio fica direto em localhost:${PORT}.
    RUN_NET=(--network=host)
else
    # Windows/macOS (Docker Desktop): mapeamento de porta.
    RUN_NET=(-p "${PORT}:${PORT}")
fi
[ "$MODE" = "gpu" ] && GPU_FLAG=(--gpus all)

docker run -d --name "$CONTAINER" \
    "${GPU_FLAG[@]}" "${RUN_NET[@]}" \
    -e PORT="$PORT" \
    -v "$ROOT/input:/workspace/input" \
    -v "$ROOT/output:/workspace/output" \
    -v "$ROOT/models:/workspace/models" \
    "$IMAGE" python /workspace/scripts/app.py
c_ok "Container '${CONTAINER}' iniciado"

# --- 6) instruções finais -------------------------------------------------
c_step "6/6  Pronto!"
cat <<EOF

  🌐  Interface:  http://localhost:${PORT}
      (Na 1ª vez o servidor demora a responder: está baixando os pesos do
       modelo para ./models — depois fica em cache.)

  Comandos úteis:
    docker logs -f ${CONTAINER}     # acompanhar o carregamento / logs
    docker rm -f ${CONTAINER}       # parar e remover o servidor

  Modo atual: ${MODE}$( [ "$MODE" = cpu ] && echo "  (sem GPU: gera a forma, sem textura, e é lento)" )

EOF
