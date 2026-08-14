# syntax=docker/dockerfile:1
#
# Ambiente das aulas práticas de Inteligência Artificial - FACOM/UFMS
#
# A imagem traz Python + as bibliotecas de requirements.txt + JupyterLab.
# Os notebooks e as bases de dados NÃO são copiados para dentro da imagem:
# eles entram por bind mount (ver compose.yaml). Assim, o que você edita no
# navegador é gravado direto no repositório e vai para o git normalmente.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    LANG=C.UTF-8 \
    TZ=America/Campo_Grande

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl tzdata \
 && rm -rf /var/lib/apt/lists/*

# requirements.txt vem antes de tudo para aproveitar o cache de camadas:
# mexer nos notebooks não invalida a instalação das bibliotecas.
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# UID/GID do usuário do host. Sem isso, no Linux, todo arquivo que o container
# grava no repositório (notebook salvo, checkpoint) sai com dono root.
# O Makefile preenche esses argumentos automaticamente.
#
# Estes ARG ficam DEPOIS do pip install de propósito: um ARG declarado antes de
# um RUN entra na chave de cache daquele RUN, então declará-los no topo faria
# cada máquina com UID diferente reinstalar as bibliotecas do zero (~900 MB de
# cache por UID). Aqui só a camada do useradd, que é de 65 kB, varia por máquina.
ARG UID=1000
ARG GID=1000
ARG USUARIO=aluno

# `getent` evita colisão quando o GID do host já existe na imagem (no macOS, por
# exemplo, o GID costuma ser 20 - o `dialout` do Debian).
RUN if ! getent group "${GID}" >/dev/null; then groupadd --gid "${GID}" "${USUARIO}"; fi \
 && useradd --uid "${UID}" --gid "${GID}" --create-home --shell /bin/bash "${USUARIO}" \
 && mkdir -p "/home/${USUARIO}/aulas" \
             "/home/${USUARIO}/.jupyter" \
             "/home/${USUARIO}/.local/share/jupyter" \
 && chown -R "${UID}:${GID}" "/home/${USUARIO}"

ENV HOME=/home/${USUARIO}
WORKDIR ${HOME}/aulas
USER ${UID}:${GID}

EXPOSE 8888

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=6 \
  CMD curl -fsS http://localhost:8888/api || exit 1

# Sem token e sem senha, de propósito: a porta é publicada apenas em 127.0.0.1
# (ver compose.yaml), então o servidor só é alcançável da própria máquina. Isso
# elimina a fricção de copiar a URL com token do log - o que importa em sala.
# Para exigir token, defina JUPYTER_TOKEN no compose e remova a linha do token.
#
# O root_dir do JupyterLab é o WORKDIR acima, ou seja, a raiz do repositório.
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--IdentityProvider.token=", \
     "--ServerApp.password="]
