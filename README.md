# Foto → Objeto 3D (Hunyuan3D)

Pipeline para transformar **fotos em modelos 3D** usando o
[Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) da Tencent,
rodando **localmente via Docker**.

> ℹ️ O repositório oficial é `Tencent-Hunyuan/Hunyuan3D-2`
> (o link `github.com/Tencent/Hunyuan3D` retorna 404).

---

## ⚠️ Leia primeiro: hardware desta máquina

Esta máquina **não possui GPU NVIDIA/CUDA** (só uma Intel UHD integrada).
Isso tem consequências diretas:

| Etapa | CPU (esta máquina) | GPU NVIDIA |
|---|---|---|
| **Geração de forma** (a malha 3D) | ✅ Funciona, porém **lento** (minutos por objeto) | ✅ Rápido |
| **Textura PBR de alta qualidade** | ❌ **Indisponível** — exige rasterizador CUDA | ✅ Funciona |

Por isso o projeto está configurado para o modo **CPU-only**, usando o modelo
leve **Hunyuan3D-2mini**, e gera a **forma 3D** com boa granularidade. A textura
fica desativada automaticamente (o código já detecta CUDA e religa a textura
sozinho caso você rode esta mesma imagem Docker numa GPU/nuvem no futuro).

**Quer textura/PBR de verdade?** A mesma imagem Docker roda numa GPU alugada
(RunPod, Vast.ai, etc.) — veja [Rodar em GPU](#rodar-em-gpu-opcional).

---

## Pré-requisitos

- Docker instalado (já presente nesta máquina).
- Acesso à internet na **primeira** execução (baixa ~2–3 GB de pesos).

---

## Estrutura

```
hunyuan3d/
├── input/                 # coloque aqui suas fotos (.png/.jpg/.webp)
├── output/                # arquivos 3D gerados (.glb / .obj) aparecem aqui
├── models/                # cache dos pesos do modelo (persistente)
├── scripts/img2mesh.py    # driver: foto -> malha 3D
├── docker/
│   ├── Dockerfile         # imagem Python 3.10 + PyTorch CPU + hy3dgen
│   ├── requirements-cpu.txt
│   └── entrypoint.sh
├── docker-compose.yml
└── Makefile               # atalhos
```

---

## Como usar

### 1. Construir a imagem (uma vez)

```bash
make build
# ou: docker compose build
```

### 2. Colocar uma foto

Copie uma imagem para `input/`. Dicas para melhor resultado:
- Objeto **único e centralizado**, bem iluminado.
- Fundo simples (o pipeline remove o fundo automaticamente).
- Vista frontal nítida.

### 3. Gerar o 3D

```bash
# Processa TODAS as imagens em ./input
make run

# Apenas uma imagem
make run-one IMG=minha_foto.png

# Alta qualidade (mais detalhe, bem mais lento em CPU)
make hq IMG=minha_foto.png
```

O resultado (`.glb`) aparece em `output/`. Abra em qualquer visualizador glTF
(ex.: <https://gltf-viewer.donmccurdy.com/>, Blender, Windows 3D Viewer).

---

## Parâmetros de qualidade

Passe argumentos extras via `docker compose run`:

```bash
docker compose run --rm hunyuan3d \
  --image /workspace/input/foto.png \
  --steps 50 \              # passos de difusão (20–50): mais = melhor/lento
  --octree-resolution 384 \ # 256/384/512: mais = mais granularidade/lento
  --max-faces 80000 \       # teto de faces após simplificação
  --also-obj                # exporta .obj além do .glb
```

| Flag | Padrão | Efeito |
|---|---|---|
| `--steps` | 30 | Qualidade da difusão. Em CPU, 50 já é bem lento. |
| `--octree-resolution` | 256 | Detalhe da superfície. 512 = máximo detalhe, máximo tempo. |
| `--max-faces` | 40000 | Densidade da malha final. |
| `--no-rembg` | — | Pula remoção de fundo (use se a foto já é PNG transparente). |
| `--texture` | — | Tenta textura PBR — **só com GPU**; ignorado em CPU. |
| `--seed` | 42 | Reprodutibilidade. |

---

## Rodar em GPU (opcional)

A mesma imagem funciona em GPU NVIDIA — aí a textura PBR é gerada de verdade.
Numa máquina/nuvem com GPU e o `nvidia-container-toolkit` instalado:

```bash
docker compose run --rm --gpus all hunyuan3d \
  --image /workspace/input/foto.png --texture
```

O driver detecta a GPU, usa `fp16`, troca o marching cubes para `dmc` e ativa o
pipeline de textura automaticamente. Para qualidade ainda maior, troque o modelo
para o completo: `--model tencent/Hunyuan3D-2 --subfolder hunyuan3d-dit-v2-0`.

---

## Solução de problemas

- **Primeira execução demora / "baixando do huggingface"**: normal, está
  baixando os pesos para `models/`. As próximas execuções reaproveitam o cache.
- **Muito lento**: é esperado em CPU. Reduza `--steps` (ex.: 20) e
  `--octree-resolution` (ex.: 256) para testar mais rápido.
- **Erro de memória**: reduza `--octree-resolution` e `--max-faces`. A máquina
  tem 31 GB de RAM; resoluções altas (512) podem extrapolar.
- **Resultado sem cor (cinza)**: esperado em CPU — a textura precisa de GPU.

---

## Licença

O Hunyuan3D usa a *Tencent Hunyuan Non-Commercial License*. Revise os termos
no repositório oficial antes de uso comercial.
