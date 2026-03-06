import os
import sys
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector

# Importamos as configurações centralizadas e providers do search.py
sys.path.insert(0, os.path.dirname(__file__))
from search import AppConfig, get_embeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "100"))

def ingest_pdf():
    """
    Pipeline de ingestão otimizado:
      1. Carrega o PDF
      2. Divide em chunks
      3. Gera embeddings via factory do AppConfig
      4. Persiste no PGVector distribuído em lotes (Batching)
    """
    logger.info("Iniciando pipeline de ingestão...")

    try:
        config = AppConfig()
    except Exception as exc:
        logger.critical(f"Falha de configuração (Verifique o .env): {exc}")
        sys.exit(1)

    if not os.path.exists(PDF_PATH):
        logger.error(f"Arquivo PDF não encontrado: {PDF_PATH}")
        sys.exit(1)

    logger.info(f"Carregando PDF: {PDF_PATH}")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    logger.info(f"{len(documents)} página(s) carregada(s).")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    total_chunks = len(chunks)
    logger.info(f"{total_chunks} chunk(s) gerado(s).")

    try:
        logger.info(f"Carregando modelo de embeddings para Provider '{config.provider}'...")
        embeddings = get_embeddings(config)
    except Exception as exc:
        logger.error(f"Erro ao carregar provider: {exc}")
        sys.exit(1)

    logger.info("Iniciando persistência de vetores via PGVector (com Batching)...")
    
    # Pre-delete a coleção apenas na primeira inserção se a lista The chunks exist
    if not chunks:
        logger.warning("Nenhum chunk foi gerado a partir do documento.")
        sys.exit(0)

    try:
        # Purgamos a coleção existente na primeira chamada de PGVector.from_documents
        is_first_batch = True
        
        for i in range(0, total_chunks, BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            logger.info(f"Processando lote {i} - {i+len(batch)} de {total_chunks}...")
            
            PGVector.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=config.collection_name,
                connection=config.database_url,
                pre_delete_collection=is_first_batch,
            )
            is_first_batch = False

        logger.info(f"[OK] Ingestão concluída com sucesso. {total_chunks} chunks armazenados.")

    except Exception as exc:
        logger.error("Falha ao persistir vetores no PostgreSQL.", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    ingest_pdf()