#!/usr/bin/env python3
"""Gera os dois notebooks de uma aula a partir de uma única fonte jupytext.

    python3 ferramentas/aulas.py "Aulas Práticas/aula03-regressao-logistica/aula03.py"

Cada aula prática tem duas versões: a **aula guiada**, com lacunas para o aluno
preencher, e o **gabarito**, com as lacunas resolvidas e todas as figuras já
executadas. Manter os dois notebooks à mão é convite para eles divergirem, então
a aula é escrita **uma vez** num arquivo `.py` no formato jupytext `py:percent`
e este script recorta as duas versões.

O formato
---------

Onde a aula tem uma lacuna, a fonte traz um bloco com três marcadores em coluna
zero. As linhas da versão do aluno vêm comentadas com o prefixo `#>>> `, para
que o arquivo continue sendo Python válido (e o editor não reclame):

    # --- TODO ---
    #>>> def sigmoide(z):
    #>>>     # TODO: devolva 1 / (1 + e^-z)
    #>>>     pass
    # --- SOLUCAO ---
    def sigmoide(z):
        return 1.0 / (1.0 + np.exp(-z))
    # --- FIM ---

Fora dos blocos, tudo é copiado igual nas duas versões. Dois marcadores de texto
mudam de acordo com a versão:

    @@TITULO@@   título da aula (o gabarito ganha o sufixo "- Gabarito")
    @@AVISO@@    aviso de abertura, diferente em cada versão

Saída
-----

`aulaNN-slug.ipynb`  (aula guiada, **sem** saídas - as saídas estragariam as
respostas) e `aulaNN-gabarito.ipynb` (executado, com as figuras). O nome sai do
nome da pasta, então basta seguir a convenção `aulaNN-slug/`.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

INICIO_TODO = "# --- TODO ---"
INICIO_SOL = "# --- SOLUCAO ---"
FIM = "# --- FIM ---"
PREFIXO = "#>>>"

TITULOS = {
    "aula": "@@TITULO@@",
    "gabarito": "@@TITULO@@ - Gabarito",
}

AVISOS = {
    "aula": (
        "> **Como usar este notebook.** Rode as células na ordem. Onde aparecer "
        "**🔨 IMPLEMENTE**, há uma função para você completar; a célula seguinte "
        "confere o resultado com `assert` e imprime `OK` quando estiver certo. "
        "Onde aparecer **🔎 PARE E PENSE**, responda antes de rodar a célula de "
        "baixo - a graça é ver se sua previsão bate com o que acontece.\n"
        ">\n"
        "> As células vêm **sem saída** de propósito: as figuras e os números são "
        "o que você vai produzir. O gabarito, já executado, está no notebook "
        "`@@GABARITO@@`."
    ),
    "gabarito": (
        "> **Gabarito.** Este é o mesmo notebook da aula, com as lacunas "
        "preenchidas e todas as células executadas. As respostas das perguntas "
        "de interpretação estão em células de markdown marcadas com "
        "**🎯 Conclusão**."
    ),
}


def recorta(linhas: list[str], versao: str) -> list[str]:
    """Resolve os blocos TODO/SOLUCAO, mantendo só o lado pedido."""
    saida: list[str] = []
    estado = "fora"  # fora | todo | solucao
    for n, linha in enumerate(linhas, 1):
        marca = linha.rstrip()
        if marca == INICIO_TODO:
            if estado != "fora":
                sys.exit(f"linha {n}: '{INICIO_TODO}' dentro de outro bloco")
            estado = "todo"
            continue
        if marca == INICIO_SOL:
            if estado != "todo":
                sys.exit(f"linha {n}: '{INICIO_SOL}' sem '{INICIO_TODO}' antes")
            estado = "solucao"
            continue
        if marca == FIM:
            if estado == "fora":
                sys.exit(f"linha {n}: '{FIM}' sem bloco aberto")
            estado = "fora"
            continue

        if estado == "fora":
            saida.append(linha)
        elif estado == "todo" and versao == "aula":
            # Tira o prefixo que mantinha o bloco comentado na fonte.
            if not linha.lstrip().startswith(PREFIXO):
                sys.exit(f"linha {n}: linha de bloco TODO sem o prefixo '{PREFIXO}'")
            saida.append(re.sub(rf"^\s*{re.escape(PREFIXO)} ?", "", linha, count=1))
        elif estado == "solucao" and versao == "gabarito":
            saida.append(linha)
    if estado != "fora":
        sys.exit(f"o arquivo terminou com um bloco '{estado}' aberto")
    return saida


def substitui(texto: str, versao: str, titulo: str, gabarito: str) -> str:
    # O aviso tem várias linhas e cai dentro de uma célula de markdown do
    # jupytext, onde toda linha precisa começar com "# " - por isso a emenda.
    aviso = AVISOS[versao].replace("\n", "\n# ")
    return (
        texto.replace("@@TITULO@@", TITULOS[versao].replace("@@TITULO@@", titulo))
        .replace("@@AVISO@@", aviso)
        .replace("@@GABARITO@@", gabarito)
    )


def roda(*args: str) -> None:
    print("  $", " ".join(args))
    subprocess.run(args, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("fonte", type=Path, help="arquivo .py em formato jupytext py:percent")
    ap.add_argument("--titulo", default=None, help="título da aula (padrão: 1ª linha '# %%%% [markdown]')")
    ap.add_argument(
        "--sem-executar",
        action="store_true",
        help="não executa o gabarito (útil para conferir a estrutura rapidamente)",
    )
    args = ap.parse_args()

    fonte: Path = args.fonte.resolve()
    pasta = fonte.parent
    slug = pasta.name  # aula03-regressao-logistica
    m = re.match(r"(aula\d+)-", slug)
    if not m:
        sys.exit(f"a pasta '{slug}' não segue a convenção 'aulaNN-slug'")
    prefixo = m.group(1)

    titulo = args.titulo or "Aula"
    destinos = {
        "aula": pasta / f"{slug}.ipynb",
        "gabarito": pasta / f"{prefixo}-gabarito.ipynb",
    }

    linhas = fonte.read_text(encoding="utf-8").splitlines(keepends=True)
    for versao, destino in destinos.items():
        texto = substitui(
            "".join(recorta(linhas, versao)), versao, titulo, destinos["gabarito"].name
        )
        temporario = pasta / f".{versao}.tmp.py"
        temporario.write_text(texto, encoding="utf-8")
        print(f"\n{versao} → {destino.name}")
        roda("jupytext", "--to", "notebook", "--output", str(destino), str(temporario))
        temporario.unlink()

        if versao == "gabarito" and not args.sem_executar:
            roda(
                "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                "--ExecutePreprocessor.timeout=600", str(destino),
            )
    print("\nPronto. Rode `make colab` para atualizar os badges e o README.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
