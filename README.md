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

O projeto **escolhe a configuração pelo dispositivo, priorizando realismo**:

- **Em GPU NVIDIA** → modelo **completo `Hunyuan3D-2`** (forma mais fiel e
  detalhada) + **textura PBR**, com parâmetros no máximo (octree 512, 50 steps,
  até 120 mil faces). Tudo automático — é só rodar na máquina com GPU.
- **Nesta máquina (CPU)** → modelo leve **Hunyuan3D-2mini**, só a **forma 3D**
  (sem textura PBR), com parâmetros moderados para não levar horas.

A troca é automática: o código detecta CUDA e religa modelo completo + textura
sozinho ao rodar a mesma imagem Docker numa GPU.

**Quer textura/PBR de verdade?** A mesma imagem Docker roda numa GPU alugada
(RunPod, Vast.ai, etc.) — veja [Rodar em GPU](#rodar-em-gpu-opcional).

---

## Pré-requisitos

- Docker instalado (Docker Desktop no Windows/macOS).
- Acesso à internet na **primeira** execução (baixa ~2–3 GB de pesos).
- (Opcional) GPU NVIDIA + `nvidia-container-toolkit` (Linux) ou GPU habilitada
  no Docker Desktop + WSL2 (Windows) para o modo de alta qualidade com textura.

---

## 🚀 Início rápido (recomendado)

Um único script prepara tudo e sobe a interface, **detectando GPU automaticamente**:

```bash
./setup.sh
```

- **Com GPU NVIDIA** → builda a imagem CUDA (forma **+ textura PBR**) e usa `--gpus all`.
- **Sem GPU** → builda a imagem CPU (só forma, mais lento).
- Roda em **Linux** e no **Windows via WSL2 / Git Bash**.

Quando terminar, abra **http://localhost:7861**.

Variáveis úteis:
```bash
FORCE_MODE=gpu ./setup.sh    # força o modo GPU
FORCE_MODE=cpu ./setup.sh    # força o modo CPU
PORT=7870 ./setup.sh         # muda a porta
NO_WEB=1 ./setup.sh          # só builda, não sobe o servidor
```

> O resto deste README detalha os passos manuais (úteis se você não usar o `setup.sh`).

---

## Estrutura

```
hunyuan3d/
├── input/                 # coloque aqui suas fotos (.png/.jpg/.webp)
├── output/                # arquivos 3D gerados (.glb / .obj) aparecem aqui
├── models/                # cache dos pesos do modelo (persistente)
├── setup.sh               # bootstrap automático (detecta GPU, builda, sobe a web)
├── scripts/
│   ├── core.py            # lógica compartilhada de conversão
│   ├── img2mesh.py        # CLI: foto(s) -> malha 3D
│   └── app.py             # interface web (Gradio)
├── docker/
│   ├── Dockerfile         # imagem CPU (PyTorch CPU + hy3dgen)
│   ├── Dockerfile.gpu     # imagem GPU/CUDA (forma + textura PBR)
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

### 2A. Interface web (recomendado) 🌐

```bash
make web
```

Abra **http://localhost:7861**, arraste uma imagem, ajuste a qualidade e clique
em **Gerar objeto 3D**. O modelo aparece num visualizador 3D interativo e pode
ser baixado em `.glb`. (Na primeira vez o servidor demora a subir porque baixa
os pesos do modelo; depois fica em cache.)

Dicas de imagem para melhor resultado:
- Objeto **único e centralizado**, bem iluminado.
- Fundo simples (o pipeline remove o fundo automaticamente).
- Vista frontal nítida.

### 2B. Linha de comando (lote) 💻

```bash
# Coloque imagens em ./input e processe todas
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
| `--steps` | por dispositivo (GPU 50 / CPU 30) | Qualidade da difusão. Em CPU, 50 já é bem lento. |
| `--octree-resolution` | por dispositivo (GPU 512 / CPU 256) | Detalhe da superfície. 512 = máximo detalhe, máximo tempo. |
| `--max-faces` | por dispositivo (GPU 120000 / CPU 40000) | Densidade da malha final. |
| `--no-rembg` | — | Pula remoção de fundo (use se a foto já é PNG transparente). |
| `--no-recenter` | — | Pula o recorte/centralização. Por padrão a imagem é recortada no contorno e centralizada num quadro quadrado — **melhora a fidelidade da forma**. |
| `--no-enhance` | — | Pula o upscale da foto. Por padrão, fotos pequenas (lado < 1024px) são ampliadas com **Real-ESRGAN** (GPU; cai p/ Lanczos) — mais detalhe para a malha. |
| `--texture` | — | Tenta textura PBR — **só com GPU**; ligada por padrão em GPU. |
| `--also-obj` | — | Exporta também `.obj` além do `.glb`. |
| `--stl` | — | Exporta `.stl` **sólido/watertight** para **impressão 3D** (fecha furos e corrige normais). O `.glb` continua texturizado para o visualizador. |
| `--smooth` | 0 | Iterações de suavização Taubin (preserva volume). 0 = desligado. |
| `--size-mm` | 0 | Escala a peça p/ que a **maior aresta** meça N **mm** (impressão). 0 = tamanho original. |
| `--front`/`--back`/`--left`/`--right` | — | **Multi-view**: vistas do mesmo objeto. Passar `--front` ativa o modo. |
| `--seed` | 42 | Reprodutibilidade. |

### Multi-view (várias fotos → geometria mais fiel)

Fotografando o **mesmo objeto** de ângulos diferentes, o modelo `Hunyuan3D-2mv`
gera uma malha bem mais fiel que a partir de uma foto só. A vista **frontal** é
obrigatória; as demais (trás/esquerda/direita) ajudam.

```bash
# CLI (dentro do container):
python img2mesh.py \
  --front /workspace/input/frente.png \
  --back  /workspace/input/tras.png \
  --left  /workspace/input/esq.png \
  --right /workspace/input/dir.png \
  --stl --size-mm 80
```

Na interface web há uma aba **"Multi-view (várias fotos)"** com os quatro campos.
Os pesos do `Hunyuan3D-2mv` são baixados sob demanda na 1ª vez.

---

## Rodar em GPU (alta qualidade + textura)

Numa máquina com GPU NVIDIA (ex.: Windows com Docker Desktop + WSL2 +
GPU habilitada), basta rodar o `setup.sh`: ele detecta a GPU e usa o
`docker/Dockerfile.gpu` (PyTorch CUDA + compila os módulos de textura).

```bash
./setup.sh            # detecta GPU automaticamente
# ou force:
FORCE_MODE=gpu ./setup.sh
```

No modo GPU o app já usa, **sem configuração extra**: o modelo **completo
`tencent/Hunyuan3D-2`** (forma de maior realismo), `fp16`, marching cubes `dmc`
(se a lib `diso` compilar), **octree 512 / 50 steps / até 120k faces** e o
**pipeline de textura PBR** ligado. Os pesos de textura vêm do mesmo repo
completo (configurável por `HY3D_TEXTURE_MODEL`).

Se quiser **forçar outro modelo** (ex.: voltar ao mini numa GPU com pouca VRAM),
use variáveis de ambiente ao subir o container:

```bash
# Forçar o mini (menos VRAM):
-e HY3D_MODEL=tencent/Hunyuan3D-2mini -e HY3D_SUBFOLDER=hunyuan3d-dit-v2-mini
```

> ⚠️ VRAM: o modelo mini cabe em ~6 GB; o completo + textura (padrão na GPU)
> pede ~16 GB. Se faltar memória, reduza `octree-resolution`/`max-faces` nos
> controles, ou force o mini com as variáveis acima.

### CLI em GPU

```bash
docker run --rm --gpus all \
  -v "$PWD/input:/workspace/input" -v "$PWD/output:/workspace/output" \
  -v "$PWD/models:/workspace/models" \
  hunyuan3d-gpu:local --image /workspace/input/foto.png --texture
```

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
