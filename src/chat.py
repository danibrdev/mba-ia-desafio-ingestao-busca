import sys
import os
import logging
sys.path.insert(0, os.path.dirname(__file__))

from search import search_prompt, get_vector_store, get_llm, AppConfig

logging.basicConfig(
    level=logging.WARNING, # INFO logs PGVector e Langchain interno demais no cli
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

BANNER = """
╔══════════════════════════════════════════════════╗
║         Semantic PDF RAG CLI  🔍                 ║
║   Pergunte sobre o documento carregado.          ║
║   Digite 'sair', 'exit' ou 'quit' para encerrar. ║
╚══════════════════════════════════════════════════╝
"""


def main():
    print("[INFO] Inicializando componentes (Banco de Dados e LLM)...")
    try:
        config = AppConfig()
        vector_store = get_vector_store(config)
        llm = get_llm(config)
    except Exception as exc:
        print(f"[ERRO CRÍTICO] Falha ao carregar configurações: {exc}")
        sys.exit(1)

    print(BANNER)
    
    chat_history = []

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
        answer = search_prompt(question, llm=llm, vector_store=vector_store, chat_history=chat_history)

        if answer is None:
            print("Não foi possível processar sua pergunta. Verifique os logs.")
        else:
            print(answer)
            # Guardamos o contexto para a próxima rodada (limitando aos últimos 5 pares)
            chat_history.append((question, answer))
            if len(chat_history) > 5:
                chat_history.pop(0)

        print()


if __name__ == "__main__":
    main()