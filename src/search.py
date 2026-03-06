import os
import sys
from typing import Optional
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "rag_documents")
PROVIDER = os.getenv("PROVIDER", "gemini").lower()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def _get_embeddings():
    """Factory: retorna o modelo de embeddings conforme o provider configurado."""
    if PROVIDER == "gemini":
        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
        return GoogleGenerativeAIEmbeddings(model=model)

    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model)


def _get_llm():
    """Factory: retorna o LLM conforme o provider configurado."""
    if PROVIDER == "gemini":
        model = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash-lite")
        return ChatGoogleGenerativeAI(model=model, temperature=0)

    model = os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano")
    return ChatOpenAI(model=model, temperature=0)


def search_prompt(question: str) -> Optional[str]:
    """
    Fluxo de busca e geração:
      1. Vetoriza a pergunta
      2. Busca por similaridade com score (k=10) no PGVector
      3. Concatena os chunks como contexto
      4. Monta o prompt anti-alucinação
      5. Invoca o LLM e retorna a resposta
    """
    if not DATABASE_URL:
        print("[ERRO] DATABASE_URL não configurada no .env", file=sys.stderr)
        return None

    if not question or not question.strip():
        return None

    try:
        embeddings = _get_embeddings()
        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            connection=DATABASE_URL,
        )

        results = vector_store.similarity_search_with_score(question, k=10)
        if not results:
            return "Não tenho informações necessárias para responder sua pergunta."

        contexto = "\n\n".join(doc.page_content for doc, _score in results)
        prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=question)

        llm = _get_llm()
        response = llm.invoke(prompt)
        return response.content

    except Exception as exc:
        print(f"[ERRO] Falha na busca/geração: {exc}", file=sys.stderr)
        return None