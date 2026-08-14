#!/usr/bin/env python3
"""Gera os links de "Abrir no Google Colab" das aulas práticas.

    python3 ferramentas/colab.py           aplica no README e nos notebooks
    python3 ferramentas/colab.py --check   só verifica; sai com erro se faltar algo

Ou, mais curto: `make colab` e `make colab-check`.

O script descobre os notebooks pelo próprio git, então uma aula nova entra na
lista sozinha: basta `git add` no notebook e rodar `make colab`. É idempotente —
rodar duas vezes não muda nada.

Duas coisas acontecem:

1. Cada notebook recebe, como primeira célula, um badge que o abre no Colab.
   Quem estiver lendo o notebook no GitHub clica e cai direto no Colab.
2. O bloco entre os marcadores COLAB:INICIO e COLAB:FIM do README é regerado
   com a tabela de todos os notebooks.

Usa só a biblioteca padrão, de propósito: roda em qualquer Python 3.10+, sem
depender do ambiente das aulas nem do container.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

RAIZ = Path(__file__).resolve().parent.parent
README = RAIZ / "README.md"

BADGE_SVG = "https://colab.research.google.com/assets/colab-badge.svg"
COLAB_GITHUB = "https://colab.research.google.com/github"

# Marca a célula gerada, para que rodar de novo substitua em vez de duplicar.
MARCA = "<!-- colab-badge -->"
INICIO, FIM = "<!-- COLAB:INICIO", "<!-- COLAB:FIM -->"


def git(*args: str) -> str:
    """Roda um comando git na raiz do repositório e devolve a saída."""
    return subprocess.run(
        ["git", *args], cwd=RAIZ, check=True, capture_output=True, text=True
    ).stdout


def repositorio() -> str:
    """Descobre 'usuario/repo' a partir do remote origin."""
    url = git("remote", "get-url", "origin").strip()
    m = re.search(r"github\.com[:/](?P<slug>[^/]+/[^/]+?)(?:\.git)?$", url)
    if not m:
        sys.exit(f"Não reconheci um repositório do GitHub em: {url}")
    return m.group("slug")


def notebooks() -> list[str]:
    """Caminhos dos notebooks rastreados pelo git, em ordem de aula.

    Vem do git (e não de um glob no disco) por dois motivos: notebook ignorado
    ou não adicionado ainda não entra na lista, e o caminho sai exatamente com
    os bytes que o GitHub vai servir — o que importa em 'Aulas Práticas', com
    espaço e acento no nome.
    """
    saida = git("ls-files", "-z", "--", "*.ipynb")
    # Ordem didática: aula 01 antes da 02 e, dentro da aula, a aula guiada antes
    # do gabarito (a ordem alfabética faria o contrário, porque 'g' vem antes).
    def chave(caminho: str) -> tuple[str, bool, str]:
        numero, tipo = rotulo(caminho)
        return (numero, tipo == "Gabarito", caminho)

    return sorted((p for p in saida.split("\0") if p), key=chave)


def url_colab(repo: str, ramo: str, caminho: str) -> str:
    return f"{COLAB_GITHUB}/{repo}/blob/{ramo}/{quote(caminho)}"


def badge(repo: str, ramo: str, caminho: str, texto: str = "Abrir no Colab") -> str:
    return f"[![{texto}]({BADGE_SVG})]({url_colab(repo, ramo, caminho)})"


def rotulo(caminho: str) -> tuple[str, str]:
    """Devolve (número da aula, tipo do notebook) a partir do caminho.

    'Aulas Práticas/aula01-introducao-am/aula01-gabarito.ipynb' → ('01', 'Gabarito')
    """
    nome = Path(caminho).name
    m = re.search(r"aula(\d+)", caminho)
    numero = m.group(1) if m else "—"
    tipo = "Gabarito" if "gabarito" in nome.lower() else "Aula guiada"
    return numero, tipo


# --------------------------------------------------------------------------
# Notebooks
# --------------------------------------------------------------------------


def celula_badge(repo: str, ramo: str, caminho: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"{MARCA}\n",
            badge(repo, ramo, caminho, "Abrir no Google Colab"),
        ],
    }


def atualiza_notebook(caminho: str, repo: str, ramo: str, escrever: bool) -> bool:
    """Insere ou atualiza a célula do badge. Devolve True se algo mudou."""
    arquivo = RAIZ / caminho
    original = arquivo.read_text(encoding="utf-8")
    nb = json.loads(original)

    nova = celula_badge(repo, ramo, caminho)
    celulas = nb.get("cells", [])
    primeira = "".join(celulas[0]["source"]) if celulas else ""

    if primeira.startswith(MARCA):
        if celulas[0].get("source") == nova["source"]:
            return False
        celulas[0] = nova
    else:
        celulas.insert(0, nova)
    nb["cells"] = celulas

    if escrever:
        # Mesma convenção do nbformat (indent=1, sort_keys, ensure_ascii=False),
        # para que o diff mostre só a célula acrescentada e não o arquivo inteiro.
        texto = json.dumps(nb, indent=1, sort_keys=True, ensure_ascii=False)
        arquivo.write_text(texto.rstrip() + "\n", encoding="utf-8")
    return True


# --------------------------------------------------------------------------
# README
# --------------------------------------------------------------------------


def tabela(repo: str, ramo: str, caminhos: list[str]) -> str:
    linhas = [
        f"{INICIO} — gerado por ferramentas/colab.py; não edite à mão -->",
        "",
        "| Aula | Notebook | Abrir no Colab |",
        "|---|---|---|",
    ]
    for caminho in caminhos:
        numero, tipo = rotulo(caminho)
        linhas.append(f"| {numero} | {tipo} | {badge(repo, ramo, caminho)} |")
    linhas += ["", FIM]
    return "\n".join(linhas)


def atualiza_readme(repo: str, ramo: str, caminhos: list[str], escrever: bool) -> bool:
    original = README.read_text(encoding="utf-8")
    novo_bloco = tabela(repo, ramo, caminhos)

    padrao = re.compile(re.escape(INICIO) + r".*?" + re.escape(FIM), re.S)
    if not padrao.search(original):
        sys.exit(
            f"Não achei os marcadores no README. Acrescente onde a tabela deve ficar:\n"
            f"  {INICIO} -->\n  {FIM}"
        )

    atualizado = padrao.sub(lambda _: novo_bloco, original)
    if atualizado == original:
        return False
    if escrever:
        README.write_text(atualizado, encoding="utf-8")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--check",
        action="store_true",
        help="não escreve nada; sai com código 1 se houver algo desatualizado",
    )
    ap.add_argument("--ramo", default="main", help="ramo apontado pelos links")
    args = ap.parse_args()

    escrever = not args.check
    repo, ramo = repositorio(), args.ramo
    caminhos = notebooks()
    if not caminhos:
        sys.exit("Nenhum notebook rastreado pelo git.")

    mudou = []
    for caminho in caminhos:
        if atualiza_notebook(caminho, repo, ramo, escrever):
            mudou.append(caminho)
    if atualiza_readme(repo, ramo, caminhos, escrever):
        mudou.append("README.md")

    print(f"Repositório: {repo} (ramo {ramo})")
    print(f"Notebooks encontrados: {len(caminhos)}")
    for caminho in caminhos:
        print(f"  {caminho}")

    if not mudou:
        print("\nTudo em ordem: badges e README já estão atualizados.")
        return 0
    if args.check:
        print("\nDesatualizado:")
        for item in mudou:
            print(f"  {item}")
        print("\nRode `make colab` para corrigir.")
        return 1
    print("\nAtualizado:")
    for item in mudou:
        print(f"  {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
