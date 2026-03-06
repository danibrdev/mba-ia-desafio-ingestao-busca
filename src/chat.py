import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from search import search_prompt

BANNER = """
╔══════════════════════════════════════════════════╗
║         Semantic PDF RAG CLI  🔍                 ║
║   Pergunte sobre o documento carregado.          ║
║   Digite 'sair', 'exit' ou 'quit' para encerrar. ║
╚══════════════════════════════════════════════════╝
"""


def main():
    print(BANNER)

    while True:
        try:
            question = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Encerrando o chat. Até logo!")
            break

        if not question:
            continue

        if question.lower() in {"sair", "exit", "quit"}:
            print("[INFO] Encerrando o chat. Até logo!")
            break

        print("\n[Assistente]: ", end="", flush=True)
        answer = search_prompt(question)

        if answer is None:
            print("Não foi possível processar sua pergunta. Verifique os logs.")
        else:
            print(answer)

        print()


if __name__ == "__main__":
    main()