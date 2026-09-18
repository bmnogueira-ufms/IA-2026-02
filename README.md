# Inteligência Artificial - 2026/2

[![Abrir no Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bmnogueira-ufms/IA-2026-02)

Materiais da disciplina de **Inteligência Artificial** da graduação - FACOM/UFMS.
Prof. Bruno Nogueira.

As aulas práticas são em **Python**, com **Jupyter Notebook**, e usam bases de dados
reais e publicamente disponíveis.

## Estrutura do repositório

```
IA-2026-2/
├── Aulas Práticas/          notebooks das aulas de laboratório
│   ├── aula01-introducao-am/
│   │   ├── aula01-introducao-am.ipynb   aula guiada
│   │   └── aula01-gabarito.ipynb        gabarito comentado dos exercícios
│   ├── aula02-regressao-linear/
│   │   ├── aula02-regressao-linear.ipynb  aula guiada (com lacunas para implementar)
│   │   ├── aula02-gabarito.ipynb          a mesma aula, já implementada e executada
│   │   └── aula02-sklearn.ipynb           a mesma regressão, agora com o scikit-learn
│   ├── aula03-regressao-logistica/
│   │   ├── aula03.py                      FONTE única (jupytext) dos dois notebooks
│   │   ├── aula03-regressao-logistica.ipynb  aula guiada
│   │   └── aula03-gabarito.ipynb          a mesma aula, resolvida e executada
│   ├── aula04-redes-neurais/
│   │   ├── aula04.py                      FONTE única (jupytext) dos dois notebooks
│   │   ├── aula04-redes-neurais.ipynb     aula guiada
│   │   └── aula04-gabarito.ipynb          a mesma aula, resolvida e executada
│   └── aula05-otimizacao/
│       ├── aula05.py                      FONTE única (jupytext) dos dois notebooks
│       ├── aula05-otimizacao.ipynb        aula guiada
│       └── aula05-gabarito.ipynb          a mesma aula, resolvida e executada
├── dados/                   bases de dados usadas nas aulas
├── ferramentas/
│   ├── colab.py             gera os links "Abrir no Colab" (make colab)
│   └── aulas.py             gera aula + gabarito a partir da fonte .py (make aula)
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
| 02 | [Regressão Linear](Aulas%20Práticas/aula02-regressao-linear/) | tarefa de regressão; hipótese $h(x) = \theta_0 + \theta_1 x$; função de custo dos mínimos quadrados; curva e superfície de $J$; derivadas parciais e gradiente descendente com atualização simultânea; taxa de aprendizado; normalização (*min-max* e escore $z$); solução fechada e comparação com o scikit-learn; análise de resíduos. Notebook extra: a mesma regressão com o scikit-learn (`LinearRegression`, `StandardScaler`, `Pipeline`, validação cruzada, `DummyRegressor`, `SGDRegressor`) | Auto MPG, Ames Housing |
| 03 | [Regressão Logística](Aulas%20Práticas/aula03-regressao-logistica/) | por que a regressão linear com limiar falha em classificação; função logística $g(z) = 1/(1+e^{-z})$; hipótese $h(x) = g(\theta^T x)$ lida como probabilidade; **fronteira de decisão** $\theta^T x = 0$ em 1 e 2 dimensões; fronteiras não lineares com termos polinomiais; *overfitting*, *bias-variância* e Navalha de Ockham; escolha do limiar (precisão × revocação); regressão logística multinomial | Palmer Penguins, Heart Disease (Cleveland), dados sintéticos |
| 04 | [Redes Neurais](Aulas%20Práticas/aula04-redes-neurais/) | regressão logística implementada do zero (entropia cruzada e gradiente); por que o erro quadrático não serve; o neurônio artificial e as funções de ativação; **perceptron** e sua regra de aprendizado, conferida contra o traço à mão do operador AND; não convergência no XOR; regra delta e unidade sigmoide; custo dos termos polinomiais × combinação de unidades; arquitetura, matrizes $\Theta$ e **forward propagation**; a rede XNOR dos slides; a camada escondida como mudança de coordenadas; **backpropagation** conferido por diferenças centrais; classificação multiclasse com um neurônio por classe | Palmer Penguins, Digits (8×8), funções lógicas, dados sintéticos |
| 05 | [Otimização de Redes Neurais](Aulas%20Práticas/aula05-otimizacao/) | conjuntos de treino, validação e teste, e o viés de escolher olhando o teste; dados da mesma distribuição; validação cruzada **k-fold**, **repetida** e **aninhada**; diagnóstico de *bias* e variância a partir dos dois erros; curvas de complexidade e de aprendizado; **regularização L2** implementada do zero e a relação $C = 1/\lambda$; L1 × L2; *weight decay* em redes e o retorno ao regime linear; **dropout**; **data augmentation**; **early stopping** | Digits (8×8), Heart Disease (Cleveland), Palmer Penguins, dados sintéticos |

Cada aula prática corresponde ao conteúdo teórico já visto em sala. A aula 01 cobre os
slides 2 (*Introdução*), 3 (*Introdução ao Aprendizado de Máquina*) e 7 (*Introdução à
Classificação e Regressão*); a aula 02 cobre os slides 8 (*Regressão Linear*), na parte
de uma única variável; a aula 03 cobre os slides 9 (*Regressão Logística*), com exceção
da função de custo e do gradiente descendente, que ficam para a aula seguinte; a aula 04
cobre os slides 11 (*Redes Neurais*) e fecha a pendência da aula 03; a aula 05 cobre os
slides 12 (*Otimização de Redes Neurais*).

### As aulas são para ser preenchidas

A partir da aula 02 os algoritmos deixam de ser caixas-pretas: o notebook da aula guiada
tem **lacunas de propósito**, marcadas com 🔨 IMPLEMENTE, para serem escritas durante o
laboratório - por isso ele é distribuído **sem saídas**. Cada lacuna vem com uma célula
de verificação que imprime `OK` quando a implementação está correta. O
`aula02-gabarito.ipynb` é a mesma aula com tudo implementado, comentado e executado.

### Uma fonte, dois notebooks

A partir da aula 03, aula e gabarito são recortados de um **único arquivo**
(`aulaNN.py`, no formato jupytext `py:percent`), para que os dois nunca divirjam:

```bash
make aula FONTE="Aulas Práticas/aula03-regressao-logistica/aula03.py" \
          TITULO="Aula Prática 03 - Regressão Logística"
```

O script recorta a versão do aluno e a do professor, e executa o gabarito.

O terceiro notebook da aula 02, `aula02-sklearn.ipynb`, refaz a **mesma** regressão na **mesma**
base usando o scikit-learn, e mostra onde cada função escrita à mão foi parar dentro da
biblioteca (`LinearRegression`, `StandardScaler`, `Pipeline`, `SGDRegressor`). Ele vem
completo e executado, e serve de referência para o resto do semestre: daqui em diante
todo modelo da disciplina segue a mesma interface `fit` / `predict` / `score`.

Os slides das aulas teóricas não fazem parte deste repositório - eles são distribuídos
pelo Moodle.

### Abrir os notebooks no Google Colab

Um clique, sem instalar nada. O badge do topo abre a lista completa; os links abaixo
abrem cada notebook direto:

<!-- COLAB:INICIO - gerado por ferramentas/colab.py; não edite à mão -->

| Aula | Notebook | Abrir no Colab |
|---|---|---|
| 01 | Aula guiada | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bmnogueira-ufms/IA-2026-02/blob/main/Aulas%20Pr%C3%A1ticas/aula01-introducao-am/aula01-introducao-am.ipynb) |
| 01 | Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bmnogueira-ufms/IA-2026-02/blob/main/Aulas%20Pr%C3%A1ticas/aula01-introducao-am/aula01-gabarito.ipynb) |
| 02 | Aula guiada | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bmnogueira-ufms/IA-2026-02/blob/main/Aulas%20Pr%C3%A1ticas/aula02-regressao-linear/aula02-regressao-linear.ipynb) |
| 02 | Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bmnogueira-ufms/IA-2026-02/blob/main/Aulas%20Pr%C3%A1ticas/aula02-regressao-linear/aula02-gabarito.ipynb) |
| 02 | Com o scikit-learn | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bmnogueira-ufms/IA-2026-02/blob/main/Aulas%20Pr%C3%A1ticas/aula02-regressao-linear/aula02-sklearn.ipynb) |
| 03 | Aula guiada | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bmnogueira-ufms/IA-2026-02/blob/main/Aulas%20Pr%C3%A1ticas/aula03-regressao-logistica/aula03-regressao-logistica.ipynb) |
| 03 | Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bmnogueira-ufms/IA-2026-02/blob/main/Aulas%20Pr%C3%A1ticas/aula03-regressao-logistica/aula03-gabarito.ipynb) |
| 04 | Aula guiada | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bmnogueira-ufms/IA-2026-02/blob/main/Aulas%20Pr%C3%A1ticas/aula04-redes-neurais/aula04-redes-neurais.ipynb) |
| 04 | Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bmnogueira-ufms/IA-2026-02/blob/main/Aulas%20Pr%C3%A1ticas/aula04-redes-neurais/aula04-gabarito.ipynb) |
| 05 | Aula guiada | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bmnogueira-ufms/IA-2026-02/blob/main/Aulas%20Pr%C3%A1ticas/aula05-otimizacao/aula05-otimizacao.ipynb) |
| 05 | Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bmnogueira-ufms/IA-2026-02/blob/main/Aulas%20Pr%C3%A1ticas/aula05-otimizacao/aula05-gabarito.ipynb) |

<!-- COLAB:FIM -->

As bases de dados são baixadas automaticamente quando o notebook roda no Colab, então
não é preciso clonar o repositório. A tabela acima é gerada por
`ferramentas/colab.py` - ao acrescentar uma aula, rode `make colab`.

## Como executar

Há quatro caminhos. O **Docker** é o recomendado para trabalhar na própria máquina:
dispensa instalar Python e garante que todos na turma usem exatamente as mesmas versões
das bibliotecas. Para só abrir e rodar, o **Colab** é o mais rápido.

### Opção 1 - Docker (recomendado)

Requer apenas o [Docker Desktop](https://www.docker.com/products/docker-desktop)
instalado e aberto.

```bash
make          # constrói a imagem (só na primeira vez) e sobe o JupyterLab
```

Ao final, o próprio comando imprime o endereço: <http://localhost:8888/lab>.
Não há senha nem token - o servidor é publicado apenas em `127.0.0.1`, ou seja, só a sua
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

### Opção 2 - ambiente virtual local

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

### Opção 3 - Anaconda

A distribuição [Anaconda](https://www.anaconda.com/download) já traz todas as
bibliotecas necessárias; basta abrir o Jupyter e navegar até os notebooks.

### Opção 4 - Google Colab

Sem instalar nada: use os links da seção
[Abrir os notebooks no Google Colab](#abrir-os-notebooks-no-google-colab).

---

Em qualquer uma das opções, os notebooks funcionam **sem conexão com a internet**: as
bases de dados estão em `dados/`. Caso o arquivo local não seja encontrado - o que é o
caso no Colab -, o notebook baixa a base automaticamente.

## Bibliografia

- MITCHELL, T. *Machine Learning*. McGraw-Hill, 1997.
- RUSSELL, S.; NORVIG, P. *Artificial Intelligence: A Modern Approach*. 3. ed. Prentice Hall, 2009.
- BISHOP, C. *Pattern Recognition and Machine Learning*. Springer, 2006.
- JAMES, G. et al. *An Introduction to Statistical Learning*. Springer, 2013.
- HASTIE, T.; TIBSHIRANI, R.; FRIEDMAN, J. *The Elements of Statistical Learning*. 2. ed. Springer, 2009.
- NG, A. *Machine Learning* - notas de aula (Stanford CS229).

## Contato

Prof. Bruno Nogueira - bruno.nogueira@ufms.br
Atendimento: quartas, 17h30-18h30, FACOM sala 10 (agendamento por e-mail).
Comunicação e material de apoio também no Moodle: [ava.ufms.br](https://ava.ufms.br).
