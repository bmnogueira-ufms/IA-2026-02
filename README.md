# Inteligência Artificial — 2026/2

Materiais da disciplina de **Inteligência Artificial** da graduação — FACOM/UFMS.
Prof. Bruno Nogueira.

As aulas práticas são em **Python**, com **Jupyter Notebook**, e usam bases de dados
reais e publicamente disponíveis.

## Estrutura do repositório

```
IA-2026-2/
├── Aulas Teóricas/          slides das aulas expositivas (PDF)
├── Aulas Práticas/          notebooks das aulas de laboratório
│   └── aula01-introducao-am/
│       ├── aula01-introducao-am.ipynb   aula guiada
│       └── aula01-gabarito.ipynb        gabarito comentado dos exercícios
├── dados/                   bases de dados usadas nas aulas
├── requirements.txt         dependências Python
└── README.md
```

## Aulas práticas

| # | Aula | Conteúdo | Bases usadas |
|---|------|----------|--------------|
| 01 | [Introdução ao Aprendizado de Máquina](Aulas%20Práticas/aula01-introducao-am/) | definição de Mitchell (T, P, E); aprendizado indutivo e Navalha de Ockham; *overfitting*; paradigmas supervisionado / não supervisionado / semissupervisionado; separabilidade linear e fronteiras de decisão; árvores de decisão; matriz de confusão e métricas; *holdout* e *k-fold cross validation*; micro e macro averaging | Iris, Heart Disease (Cleveland) |

Cada aula prática corresponde ao conteúdo teórico já visto em sala. A aula 01 cobre os
slides 2 (*Introdução*), 3 (*Introdução ao Aprendizado de Máquina*) e 7 (*Introdução à
Classificação e Regressão*).

## Como executar

Requer **Python 3.10 ou superior**.

```bash
# 1. (recomendado) crie um ambiente virtual
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. instale as dependências
pip install -r requirements.txt

# 3. abra o Jupyter
jupyter lab                      # ou: jupyter notebook
```

Alternativamente, use a distribuição [Anaconda](https://www.anaconda.com/download), que
já traz todas as bibliotecas necessárias.

Os notebooks funcionam **sem conexão com a internet**: as bases de dados estão em
`dados/`. Caso o arquivo local não seja encontrado, o notebook tenta baixá-lo
automaticamente.

### Executando no Google Colab

Os notebooks também rodam no [Google Colab](https://colab.research.google.com) sem
instalação. Nesse caso o arquivo local não existe e o notebook usa o download
automático — basta executar as células normalmente.

## Bibliografia

- MITCHELL, T. *Machine Learning*. McGraw-Hill, 1997.
- RUSSELL, S.; NORVIG, P. *Artificial Intelligence: A Modern Approach*. 3. ed. Prentice Hall, 2009.
- BISHOP, C. *Pattern Recognition and Machine Learning*. Springer, 2006.
- JAMES, G. et al. *An Introduction to Statistical Learning*. Springer, 2013.

## Contato

Prof. Bruno Nogueira — bruno.nogueira@ufms.br
Atendimento: quartas, 17h30–18h30, FACOM sala 10 (agendamento por e-mail).
Comunicação e material de apoio também no Moodle: [ava.ufms.br](https://ava.ufms.br).
