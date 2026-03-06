import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "rag_documents")
PROVIDER = os.getenv("PROVIDER", "gemini").lower()


def _get_embeddings():
    """Factory: retorna o modelo de embeddings conforme o provider configurado."""
    if PROVIDER == "gemini":
        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
        return GoogleGenerativeAIEmbeddings(model=model)

    # Default: OpenAI
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model)


def ingest_pdf():
    """
    Pipeline de ingestão:
      1. Carrega o PDF
      2. Divide em chunks (size=1000, overlap=150)
      3. Gera embeddings
      4. Persiste vetores no PostgreSQL via PGVector
    """
    if not DATABASE_URL:
        print("[ERRO] DATABASE_URL não configurada no .env", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(PDF_PATH):
        print(f"[ERRO] PDF não encontrado: {PDF_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Carregando PDF: {PDF_PATH}")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"[INFO] {len(documents)} página(s) carregada(s)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)
    print(f"[INFO] {len(chunks)} chunk(s) gerado(s)")

    print(f"[INFO] Gerando embeddings com provider '{PROVIDER}'...")
    embeddings = _get_embeddings()

    print("[INFO] Persistindo vetores no PostgreSQL...")
    PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        pre_delete_collection=True,   # recria a coleção a cada ingestão
    )

    print(f"[OK] Ingestão concluída. {len(chunks)} chunks armazenados na coleção '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    ingest_pdf()