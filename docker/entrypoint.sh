#!/usr/bin/env bash
# Ponto de entrada do container. Repassa todos os argumentos ao driver.
# Sem argumentos -> processa todas as imagens em /workspace/input.
set -euo pipefail

# Usa todos os núcleos disponíveis para acelerar a inferência em CPU.
CORES="$(nproc)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CORES}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CORES}"

if [ "$#" -eq 0 ]; then
    exec python /workspace/scripts/img2mesh.py --input-dir /workspace/input --output-dir /workspace/output
else
    exec python /workspace/scripts/img2mesh.py "$@"
fi
