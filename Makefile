# Atalhos para o pipeline foto -> 3D (Hunyuan3D, CPU-only).
# Uso: make build | make run | make run-one IMG=foto.png | make shell | make clean

COMPOSE := docker compose

.PHONY: help build run run-one shell clean

help:
	@echo "Comandos disponíveis:"
	@echo "  make build              - constrói a imagem Docker"
	@echo "  make run                - processa TODAS as imagens em ./input"
	@echo "  make run-one IMG=x.png  - processa apenas ./input/x.png"
	@echo "  make hq IMG=x.png       - alta qualidade (octree 384, 50 steps)"
	@echo "  make shell              - abre um shell no container"
	@echo "  make clean              - remove a imagem Docker local"

build:
	$(COMPOSE) build

run:
	$(COMPOSE) run --rm hunyuan3d

run-one:
	@test -n "$(IMG)" || (echo "Use: make run-one IMG=nome.png"; exit 1)
	$(COMPOSE) run --rm hunyuan3d --image /workspace/input/$(IMG)

hq:
	@test -n "$(IMG)" || (echo "Use: make hq IMG=nome.png"; exit 1)
	$(COMPOSE) run --rm hunyuan3d --image /workspace/input/$(IMG) \
		--steps 50 --octree-resolution 384 --max-faces 80000 --also-obj

shell:
	$(COMPOSE) run --rm --entrypoint bash hunyuan3d

clean:
	-docker image rm hunyuan3d-cpu:local
