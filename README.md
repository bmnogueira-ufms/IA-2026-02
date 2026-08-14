# Inteligência Artificial — 2026/2

[![Abrir no Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bmnogueira-ufms/IA-2026-02)

Materiais da disciplina de **Inteligência Artificial** da graduação — FACOM/UFMS.
Prof. Bruno Nogueira.

As aulas práticas são em **Python**, com **Jupyter Notebook**, e usam bases de dados
reais e publicamente disponíveis.

## Estrutura do repositório

```
IA-2026-2/
├── Aulas Práticas/          notebooks das aulas de laboratório
│   └── aula01-introducao-am/
│       ├── aula01-introducao-am.ipynb   aula guiada
│       └── aula01-gabarito.ipynb        gabarito comentado dos exercícios
├── dados/                   bases de dados usadas nas aulas
├── ferramentas/
│   └── colab.py             gera os links "Abrir no Colab" (make colab)
├── requirements.txt         dependências Python
├── Dockerfile               imagem com Python + JupyterLab
├── compose.yaml             serviço do JupyterLab
├── Makefile                 atalhos: make up, make down, make help
└── README.md
```

## Aulas práticas

| # | Aula | Conteúdo | Bases usadas |
|---|------|----------|--------------|
| 01 | [Introdução ao Aprendizado de Máquina](Aulas%20Práticas/aula01-introducao-am/) | definição de Mitchell (T, P, E); aprendizado indutivo e Navalha de Ockham; *overfitting*; paradigmas supervisionado / não supervisionado / semissupervisionado; separabilidade linear e fronteiras de decisão; árvores de decisão; matriz de confusão e métricas; *holdout* e *k-fold cross validation*; micro e macro averaging | Iris, Heart Disease (Cleveland) |

Cada aula prática corresponde ao conteúdo teórico já visto em sala. A aula 01 cobre os
slides 2 (*Introdução*), 3 (*Introdução ao Aprendizado de Máquina*) e 7 (*Introdução à
Classificação e Regressão*).

Os slides das aulas teóricas não fazem parte deste repositório — eles são distribuídos
pelo Moodle.

### Abrir os notebooks no Google Colab

Um clique, sem instalar nada. O badge do topo abre a lista completa; os links abaixo
abrem cada notebook direto:

<!-- COLAB:INICIO — gerado por ferramentas/colab.py; não edite à mão -->

| Aula | Notebook | Abrir no Colab |
|---|---|---|
| 01 | Aula guiada | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bmnogueira-ufms/IA-2026-02/blob/main/Aulas%20Pr%C3%A1ticas/aula01-introducao-am/aula01-introducao-am.ipynb) |
| 01 | Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bmnogueira-ufms/IA-2026-02/blob/main/Aulas%20Pr%C3%A1ticas/aula01-introducao-am/aula01-gabarito.ipynb) |

<!-- COLAB:FIM -->

As bases de dados são baixadas automaticamente quando o notebook roda no Colab, então
não é preciso clonar o repositório. A tabela acima é gerada por
`ferramentas/colab.py` — ao acrescentar uma aula, rode `make colab`.

## Como executar

Há quatro caminhos. O **Docker** é o recomendado para trabalhar na própria máquina:
dispensa instalar Python e garante que todos na turma usem exatamente as mesmas versões
das bibliotecas. Para só abrir e rodar, o **Colab** é o mais rápido.

### Opção 1 — Docker (recomendado)

Requer apenas o [Docker Desktop](https://www.docker.com/products/docker-desktop)
instalado e aberto.

```bash
make          # constrói a imagem (só na primeira vez) e sobe o JupyterLab
```

Ao final, o próprio comando imprime o endereço: <http://localhost:8888/lab>.
Não há senha nem token — o servidor é publicado apenas em `127.0.0.1`, ou seja, só a sua
própria máquina o alcança.

O repositório é montado dentro do container, então **os notebooks que você salva no
navegador são gravados no seu disco** e entram no git normalmente. Nada se perde ao
encerrar o container.

| Comando | O que faz |
|---|---|
| `make` ou `make up` | sobe o JupyterLab e mostra o endereço |
| `make abrir` | abre o JupyterLab no navegador |
| `make down` | encerra o container |
| `make logs` | mostra o log do servidor |
| `make shell` | abre um terminal dentro do container |
| `make check` | lista as versões das bibliotecas instaladas |
| `make rebuild` | reconstrói a imagem (após mudar o `requirements.txt`) |
| `make clean` | remove container, imagem e volumes |
| `make help` | lista todos os comandos |

Se a porta 8888 já estiver em uso: `make up PORTA=8899`.

Sem `make` (Windows sem WSL, por exemplo), o equivalente é:

```bash
docker compose up -d --build     # sobe
docker compose down              # encerra
```

Nesse caso, no Linux, defina `HOST_UID` e `HOST_GID` para que os arquivos salvos não
fiquem com dono `root`:

```bash
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose up -d --build
```

### Opção 2 — ambiente virtual local

Requer **Python 3.10 ou superior**.

```bash
# 1. crie um ambiente virtual
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. instale as dependências
pip install -r requirements.txt

# 3. abra o Jupyter
jupyter lab                      # ou: jupyter notebook
```

### Opção 3 — Anaconda

A distribuição [Anaconda](https://www.anaconda.com/download) já traz todas as
bibliotecas necessárias; basta abrir o Jupyter e navegar até os notebooks.

### Opção 4 — Google Colab

Sem instalar nada: use os links da seção
[Abrir os notebooks no Google Colab](#abrir-os-notebooks-no-google-colab).

---

Em qualquer uma das opções, os notebooks funcionam **sem conexão com a internet**: as
bases de dados estão em `dados/`. Caso o arquivo local não seja encontrado — o que é o
caso no Colab —, o notebook baixa a base automaticamente.

## Bibliografia

- MITCHELL, T. *Machine Learning*. McGraw-Hill, 1997.
- RUSSELL, S.; NORVIG, P. *Artificial Intelligence: A Modern Approach*. 3. ed. Prentice Hall, 2009.
- BISHOP, C. *Pattern Recognition and Machine Learning*. Springer, 2006.
- JAMES, G. et al. *An Introduction to Statistical Learning*. Springer, 2013.

## Contato

Prof. Bruno Nogueira — bruno.nogueira@ufms.br
Atendimento: quartas, 17h30–18h30, FACOM sala 10 (agendamento por e-mail).
Comunicação e material de apoio também no Moodle: [ava.ufms.br](https://ava.ufms.br).
