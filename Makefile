# Aulas práticas de Inteligência Artificial — FACOM/UFMS
#
#   make            sobe o JupyterLab e mostra o endereço
#   make help       lista todos os comandos disponíveis
#
# Para usar outra porta:  make up PORTA=8899

SHELL := /bin/bash

PORTA   ?= 8888
SERVICO := jupyter
URL     := http://localhost:$(PORTA)/lab

# O UID/GID do host vira argumento de build da imagem, para que os arquivos
# gravados pelo container no repositório fiquem com o seu dono, e não com root.
# Os nomes têm o prefixo HOST_ de propósito: `UID` é variável especial do bash e
# não chega ao docker se passada com esse nome.
COMPOSE := HOST_UID=$(shell id -u) HOST_GID=$(shell id -g) PORTA=$(PORTA) docker compose

.DEFAULT_GOAL := up
.PHONY: up down stop restart build rebuild logs shell abrir status check clean \
        colab colab-check guarda help

up: guarda ## Sobe o JupyterLab (constrói a imagem na primeira vez)
	@$(COMPOSE) up -d --build
	@printf 'Aguardando o JupyterLab responder'
	@ok=0; for _ in $$(seq 1 40); do \
	   if curl -fsS -o /dev/null http://127.0.0.1:$(PORTA)/api 2>/dev/null; then ok=1; break; fi; \
	   printf '.'; sleep 1; \
	 done; \
	 echo; \
	 if [ $$ok -eq 1 ]; then \
	   echo ''; \
	   echo '  JupyterLab  →  $(URL)'; \
	   echo '  Notebooks   →  Aulas Práticas/'; \
	   echo '  Encerrar    →  make down'; \
	   echo ''; \
	 else \
	   echo 'O servidor não respondeu. Veja o que aconteceu com: make logs'; \
	   exit 1; \
	 fi

down: ## Encerra e remove o container (os notebooks ficam no disco)
	@$(COMPOSE) down

stop: ## Pausa o container sem removê-lo
	@$(COMPOSE) stop

restart: ## Reinicia o servidor (útil se o Jupyter travar)
	@$(COMPOSE) restart

build: guarda ## Constrói a imagem sem subir o container
	@$(COMPOSE) build

rebuild: guarda ## Reconstrói a imagem do zero (use ao mudar requirements.txt)
	@$(COMPOSE) build --no-cache
	@$(MAKE) --no-print-directory up

logs: ## Acompanha o log do servidor (Ctrl+C para sair)
	@$(COMPOSE) logs -f $(SERVICO)

shell: ## Abre um terminal bash dentro do container
	@$(COMPOSE) exec $(SERVICO) bash

abrir: ## Abre o JupyterLab no navegador
	@(command -v open >/dev/null && open '$(URL)') \
	 || (command -v xdg-open >/dev/null && xdg-open '$(URL)') \
	 || echo 'Abra manualmente: $(URL)'

status: ## Mostra o estado do container
	@$(COMPOSE) ps

check: guarda ## Confere as versões das bibliotecas dentro do container
	@$(COMPOSE) run --rm --no-deps $(SERVICO) python -c "\
import sys, numpy, pandas, sklearn, matplotlib; \
print('Python      ', sys.version.split()[0]); \
print('numpy       ', numpy.__version__); \
print('pandas      ', pandas.__version__); \
print('scikit-learn', sklearn.__version__); \
print('matplotlib  ', matplotlib.__version__)"

clean: ## Remove container, imagem e preferências do JupyterLab
	@$(COMPOSE) down --volumes --rmi local

colab: ## Atualiza os links "Abrir no Colab" (rode ao acrescentar uma aula)
	@python3 ferramentas/colab.py

colab-check: ## Verifica se os links do Colab estão atualizados
	@python3 ferramentas/colab.py --check

# Falha com uma mensagem legível quando o Docker não está instalado ou o daemon
# não está rodando — em vez do erro cru do cliente.
guarda:
	@command -v docker >/dev/null 2>&1 || { \
	   echo 'Docker não encontrado. Instale o Docker Desktop: https://www.docker.com/products/docker-desktop'; \
	   exit 1; }
	@docker info >/dev/null 2>&1 || { \
	   echo 'O Docker está instalado, mas o daemon não responde. Abra o Docker Desktop e tente de novo.'; \
	   exit 1; }

help: ## Lista os comandos disponíveis
	@echo 'Aulas práticas de IA — comandos disponíveis:'
	@echo
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":[^#]*## "} {printf "  %-9s %s\n", $$1, $$2}'
	@echo
	@echo "Porta padrão: $(PORTA)   (troque com: make up PORTA=8899)"
