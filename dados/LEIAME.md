# Bases de dados

Bases usadas nas aulas práticas. Todas são públicas e amplamente utilizadas em ensino e
pesquisa.

---

## `heart-cleveland.csv` - Heart Disease (Cleveland Clinic)

**Usada em:** Aulas 01 e 03.

Dados de **303 pacientes** submetidos a angiografia coronariana na Cleveland Clinic
Foundation, descritos por 13 atributos clínicos. A tarefa é prever a presença de doença
arterial coronariana.

- **Fonte original:** UCI Machine Learning Repository -
  [Heart Disease Data Set](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Referência:** DETRANO, R. et al. International application of a new probability
  algorithm for the diagnosis of coronary artery disease. *American Journal of
  Cardiology*, v. 64, p. 304-310, 1989.
- **Doadores dos dados:** Andras Janosi (Hungarian Institute of Cardiology), William
  Steinbrunn (University Hospital, Zurique), Matthias Pfisterer (University Hospital,
  Basileia) e Robert Detrano (V.A. Medical Center, Long Beach / Cleveland Clinic
  Foundation).
- **Versão utilizada:** a distribuída com o livro *An Introduction to Statistical
  Learning* (James, Witten, Hastie & Tibshirani), que mantém os atributos categóricos
  com rótulos legíveis e preserva os valores ausentes originais.
  Espelho: <https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Heart.csv>

### Colunas (nomes originais → nomes usados na aula)

| Original | Aula | Descrição |
|---|---|---|
| `Age` | `idade` | idade em anos |
| `Sex` | `sexo` | 1 = masculino, 0 = feminino |
| `ChestPain` | `dor_peito` | tipo de dor torácica: `typical`, `nontypical`, `nonanginal`, `asymptomatic` |
| `RestBP` | `pressao_repouso` | pressão arterial sistólica em repouso (mm Hg) |
| `Chol` | `colesterol` | colesterol sérico (mg/dl) |
| `Fbs` | `glicemia_alta` | glicemia em jejum > 120 mg/dl (1 = sim) |
| `RestECG` | `ecg_repouso` | resultado do eletrocardiograma em repouso (0, 1, 2) |
| `MaxHR` | `freq_card_max` | frequência cardíaca máxima atingida |
| `ExAng` | `angina_exercicio` | angina induzida por exercício (1 = sim) |
| `Oldpeak` | `depressao_st` | depressão do segmento ST induzida por exercício |
| `Slope` | `inclinacao_st` | inclinação do segmento ST no pico do exercício (1, 2, 3) |
| `Ca` | `n_vasos` | nº de vasos principais coloridos por fluoroscopia (0-3) |
| `Thal` | `talassemia` | exame de tálio: `normal`, `fixed`, `reversable` |
| `AHD` | `doenca` | **classe**: `Yes` / `No` (doença arterial coronariana) |

### Observações importantes para a aula

- Há **valores ausentes**: 4 em `Ca` e 2 em `Thal`. Isso é intencionalmente preservado -
  serve para discutir imputação.
- As classes são razoavelmente balanceadas: 164 `No` (54,1%) e 139 `Yes` (45,9%).
- ⚠️ Circula na internet uma versão numérica desta base (comum no Kaggle, também com 303
  linhas) em que a coluna `target` está **invertida** em relação à intuição: nela,
  `target = 1` corresponde a pacientes **sem** doença. Se for usar aquela versão, confira
  a semântica do rótulo antes - por exemplo, verificando se o grupo `target = 1` é o mais
  jovem e com maior frequência cardíaca máxima.

---

## `penguins.csv` - Palmer Penguins (três espécies de pinguim)

**Usada em:** Aula 03.

Medidas de **344 pinguins** de três espécies (*Pygoscelis adeliae*, *P. antarcticus* e
*P. papua*), coletadas entre 2007 e 2009 em três ilhas do arquipélago Palmer, na
Antártida, pela Dra. Kristen Gorman no âmbito do programa Palmer Station LTER. É a base
que a comunidade de ensino vem usando no lugar da Iris: mesmo formato, dados recentes e
procedência documentada.

- **Referência:** GORMAN, K. B.; WILLIAMS, T. D.; FRASER, W. R. Ecological sexual
  dimorphism and environmental variability within a community of Antarctic penguins
  (genus *Pygoscelis*). *PLoS ONE*, v. 9, n. 3, e90081, 2014.
  <https://doi.org/10.1371/journal.pone.0090081>
- **Pacote de origem:** `palmerpenguins` (HORST, A. M.; HILL, A. P.; GORMAN, K. B., 2020).
  Dados originais do Environmental Data Initiative, sob licença CC0.
- **Versão utilizada:** a distribuída com o *seaborn*, com nomes de coluna limpos.
  Espelho: <https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv>

### Colunas (nomes originais → nomes usados na aula)

| Original | Aula | Descrição |
|---|---|---|
| `species` | `especie` | **classe**: `Adelie`, `Chinstrap` ou `Gentoo` |
| `island` | `ilha` | `Biscoe`, `Dream` ou `Torgersen` |
| `bill_length_mm` | `bico_comprimento` | comprimento do bico (mm) |
| `bill_depth_mm` | `bico_profundidade` | profundidade (altura) do bico (mm) |
| `flipper_length_mm` | `nadadeira` | comprimento da nadadeira (mm) |
| `body_mass_g` | `massa_g` | massa corporal (g) |
| `sex` | `sexo` | `MALE` / `FEMALE` |

### Observações importantes para a aula

- **Valores ausentes:** 2 pinguins não têm nenhuma medida (só espécie e ilha) e outros
  11 estão sem o sexo anotado. A aula 03 descarta as 2 linhas sem medidas e diz na tela
  quantas foram.
- Classes desbalanceadas: 152 Adelie, 124 Gentoo, 68 Chinstrap.
- O par **Adelie × Chinstrap** é o problema binário da aula: as duas espécies têm bico
  de profundidade quase idêntica (18,3 × 18,4 mm) e massa parecida, mas comprimento de
  bico bem diferente (38,8 × 48,8 mm). Com esse único atributo a regressão logística
  chega a 95,9%; com os dois atributos do bico, 97,3%.
- ⚠️ Cuidado ao inverter os eixos: *bill depth* é a **altura** do bico, não a
  profundidade da narina. Gentoo é a espécie de bico raso (15,0 mm), o que a separa
  quase perfeitamente das outras duas.
- Combinações sem o comprimento do bico (profundidade + massa, por exemplo) chegam a
  68,9% de acurácia - exatamente o mesmo do classificador majoritário. É o exercício 1
  da aula.

---

## `auto-mpg.csv` - Auto MPG (consumo de combustível)

**Usada em:** Aula 02.

Dados de **398 automóveis** vendidos nos Estados Unidos entre 1970 e 1982. A tarefa é
prever o consumo de combustível a partir de características mecânicas do veículo - um
problema de **regressão**.

- **Fonte original:** UCI Machine Learning Repository -
  [Auto MPG Data Set](https://archive.ics.uci.edu/dataset/9/auto+mpg). A base vem do
  StatLib da Carnegie Mellon University e foi usada na Exposição da American Statistical
  Association de 1983.
- **Referência:** QUINLAN, R. Combining instance-based and model-based learning. In:
  *Proceedings of the Tenth International Conference on Machine Learning*, p. 236-243,
  1993.
- **Versão utilizada:** a distribuída com o pacote *seaborn*, que já tem os nomes de
  coluna limpos e mantém os valores ausentes originais.
  Espelho: <https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv>

### Colunas (nomes originais → nomes usados na aula)

| Original | Aula | Descrição |
|---|---|---|
| `mpg` | `consumo_mpg` | **alvo**: consumo em *miles per gallon* |
| `cylinders` | `cilindros` | número de cilindros (3, 4, 5, 6, 8) |
| `displacement` | `cilindrada` | cilindrada em polegadas cúbicas |
| `horsepower` | `potencia` | potência em cavalos |
| `weight` | `peso_lb` | peso em libras |
| `acceleration` | `aceleracao` | tempo de 0 a 60 mph, em segundos |
| `model_year` | `ano` | ano do modelo (70-82) |
| `origin` | `origem` | `usa`, `japan`, `europe` |
| `name` | `modelo` | nome do modelo |

A aula acrescenta duas colunas convertidas para unidades usadas no Brasil:
`peso_kg` = `peso_lb` × 0,45359237 e `consumo_kml` = `consumo_mpg` × 0,4251437.

### Observações importantes para a aula

- Há **6 valores ausentes** em `horsepower`. Preservados de propósito: aparecem no
  exercício 3, em que cada atributo é usado isoladamente e o `dropna()` muda o `n`.
- A relação entre peso e `consumo_kml` é **negativa mas curva** ($R^2 \approx 0{,}69$
  para uma reta). Trocando o alvo para litros por 100 km - que é proporcional ao gasto,
  e não ao seu inverso - a mesma reta chega a $R^2 \approx 0{,}78$. A aula usa isso para
  discutir análise de resíduos e escolha do espaço de hipóteses.
- `aceleracao` é o atributo isolado mais fraco ($R^2 \approx 0{,}18$); `ano` tem
  coeficiente **positivo**, refletindo o ganho de eficiência dos motores na década.

---

## `ames-housing.csv` - Ames Housing (preço de imóveis)

**Usada em:** Aula 02.

**2.930 casas** vendidas em Ames, Iowa, entre 2006 e 2010. É a base que substituiu a
antiga *Boston Housing* no ensino de regressão (a de Boston foi retirada do
scikit-learn por conter um atributo de conteúdo racista). Reproduz o exemplo dos slides:
**tamanho da casa × preço**.

- **Referência:** DE COCK, D. Ames, Iowa: alternative to the Boston housing data as an
  end of semester regression project. *Journal of Statistics Education*, v. 19, n. 3,
  2011. <https://doi.org/10.1080/10691898.2011.11889627>
- **Versão utilizada:** o arquivo original tem 82 colunas; aqui estão guardadas **18**,
  escolhidas por serem as úteis para regressão nas próximas aulas.
  Espelho da versão completa:
  <https://raw.githubusercontent.com/rasbt/machine-learning-book/main/ch09/AmesHousing.txt>

### Colunas guardadas

| Coluna | Descrição |
|---|---|
| `Gr Liv Area` | área construída acima do solo, em pés² (na aula: `area_m2`, × 0,09290304) |
| `Lot Area` | área do terreno, em pés² |
| `Total Bsmt SF` | área do porão, em pés² |
| `Overall Qual` | qualidade geral do acabamento (1 a 10) |
| `Overall Cond` | estado de conservação (1 a 10) |
| `Year Built` | ano de construção |
| `Year Remod/Add` | ano da última reforma |
| `Bedroom AbvGr` | quartos acima do solo |
| `Full Bath` / `Half Bath` | banheiros completos / lavabos |
| `TotRms AbvGrd` | total de peças acima do solo |
| `Fireplaces` | lareiras |
| `Garage Cars` / `Garage Area` | vagas de garagem / área da garagem |
| `Neighborhood` | bairro (28 categorias) |
| `Central Air` | ar-condicionado central (`Y`/`N`) |
| `Sale Condition` | condição da venda (`Normal`, `Partial`, `Abnorml`, ...) |
| `SalePrice` | **alvo**: preço de venda em dólares (na aula: `preco_mil`, ÷ 1000) |

### Observações importantes para a aula

- Área construída explica apenas **metade** da variação do preço
  ($R^2 \approx 0{,}50$) - é justamente o gancho para a regressão multivariada.
- Há **valores ausentes** em `Garage Cars`, `Garage Area` e `Total Bsmt SF` (1 ou 2
  casos cada). As duas colunas usadas na aula 02 (`Gr Liv Area` e `SalePrice`) estão
  completas.
- ⚠️ `Sale Condition` merece atenção: as vendas `Partial` (casas ainda em construção) e
  `Abnorml` (execuções, vendas entre familiares) são as responsáveis pelos pontos
  esquisitos do gráfico - casas grandes vendidas barato. Filtrar por
  `Sale Condition == "Normal"` eleva a correlação de 0,71 para 0,74. A aula 02 **não**
  filtra, de propósito, para que os pontos apareçam e possam ser discutidos.

---

## Iris

**Usada em:** Aula 01.

Não está neste diretório: vem embutida no scikit-learn
(`sklearn.datasets.load_iris`), o que dispensa download.

- **Referência:** FISHER, R. A. The use of multiple measurements in taxonomic problems.
  *Annals of Eugenics*, v. 7, n. 2, p. 179-188, 1936.
- 150 exemplos, 4 atributos numéricos, 3 classes perfeitamente balanceadas.
