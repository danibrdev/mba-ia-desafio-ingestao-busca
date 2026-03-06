import sys
import logging
from typing import Optional, List, Tuple
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

class AppConfig(BaseSettings):
    """
    Centraliza e valida as configurações de ambiente.
    Funciona como esquema 'Fail-Fast': se faltar algo crítico, a app nem sobe.
    """
    database_url: str = Field(..., alias="DATABASE_URL")
    collection_name: str = Field("rag_documents", alias="PG_VECTOR_COLLECTION_NAME")
    provider: str = Field("gemini", alias="PROVIDER")
    
    # OpenAI config
    openai_api_key: Optional[str] = Field(None, alias="OPENAI_API_KEY")
    openai_embedding_model: str = Field("text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")
    openai_llm_model: str = Field("gpt-4o-mini", alias="OPENAI_LLM_MODEL")
    
    # Gemini config
    google_api_key: Optional[str] = Field(None, alias="GOOGLE_API_KEY")
    google_embedding_model: str = Field("models/gemini-embedding-001", alias="GOOGLE_EMBEDDING_MODEL")
    gemini_llm_model: str = Field("gemini-2.5-flash-lite", alias="GEMINI_LLM_MODEL")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

PROMPT_TEMPLATE = """
CONTEXTO DO DOCUMENTO:
{contexto}

HISTÓRICO DA CONVERSA:
{historico}

REGRAS:
- Responda somente com base no CONTEXTO DO DOCUMENTO e, se necessário para continuidade, no HISTÓRICO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def get_embeddings(config: AppConfig):
    """Factory: retorna o modelo de embeddings conforme o provider."""
    provider = config.provider.lower()
    if provider == "gemini":
        if not config.google_api_key:
            raise ValueError("GOOGLE_API_KEY deve ser informada para o provider 'gemini'.")
        return GoogleGenerativeAIEmbeddings(
            model=config.google_embedding_model,
            google_api_key=config.google_api_key
        )

    if not config.openai_api_key:
        raise ValueError("OPENAI_API_KEY deve ser informada para o provider 'openai'.")
    return OpenAIEmbeddings(
        model=config.openai_embedding_model,
        api_key=config.openai_api_key
    )


def get_llm(config: AppConfig):
    """Factory: retorna o LLM conforme o provider configurado."""
    provider = config.provider.lower()
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=config.gemini_llm_model, 
            temperature=0.0,
            google_api_key=config.google_api_key
        )

    return ChatOpenAI(
        model=config.openai_llm_model, 
        temperature=0.0,
        api_key=config.openai_api_key
    )


def get_vector_store(config: AppConfig) -> PGVector:
    """Retorna instância única configurada do PGVector"""
    embeddings = get_embeddings(config)
    return PGVector(
        embeddings=embeddings,
        collection_name=config.collection_name,
        connection=config.database_url,
    )


def build_chat_history_str(chat_history: List[Tuple[str, str]]) -> str:
    """Formata histórico conversacional em texto para o Prompt."""
    if not chat_history:
        return "Nenhum histórico anterior."
    lines = []
    for user_msg, ai_msg in chat_history:
        lines.append(f"Usuário: {user_msg}")
        lines.append(f"Assistente: {ai_msg}")
    return "\n".join(lines)


def search_prompt(question: str, llm, vector_store: PGVector, chat_history: List[Tuple[str, str]]) -> Optional[str]:
    """
    Fluxo de busca otimizado com Injeção de Dependências.
    Usa o LLM e VectorStore já injetados.
    """
    if not question or not question.strip():
        logger.warning("Pergunta vazia informada.")
        return None

    try:
        logger.info(f"Executando busca de similaridade para a pergunta '{question[:30]}...'")
        results = vector_store.similarity_search_with_score(question, k=10)
        
        if not results:
            logger.info("Nenhum resultado encontrado no PGVector.")
            return "Não tenho informações necessárias para responder sua pergunta."

        contexto = "\n\n".join(doc.page_content for doc, _score in results)
        historico_formatado = build_chat_history_str(chat_history)
        
        prompt = PROMPT_TEMPLATE.format(
            contexto=contexto, 
            historico=historico_formatado, 
            pergunta=question
        )

        logger.debug("Invocando LLM com o contexto.")
        response = llm.invoke(prompt)
        return response.content

    except Exception as exc:
        logger.error(f"Falha na busca/geração: {exc}", exc_info=True)
        return None