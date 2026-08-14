# Bases de dados

Bases usadas nas aulas práticas. Todas são públicas e amplamente utilizadas em ensino e
pesquisa.

---

## `heart-cleveland.csv` - Heart Disease (Cleveland Clinic)

**Usada em:** Aula 01.

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

## Iris

**Usada em:** Aula 01.

Não está neste diretório: vem embutida no scikit-learn
(`sklearn.datasets.load_iris`), o que dispensa download.

- **Referência:** FISHER, R. A. The use of multiple measurements in taxonomic problems.
  *Annals of Eugenics*, v. 7, n. 2, p. 179-188, 1936.
- 150 exemplos, 4 atributos numéricos, 3 classes perfeitamente balanceadas.
