"""
Tests for src/ingest.py

All external dependencies are injected via sys.modules before src imports,
so no real DB, API keys, or third-party packages are needed.
"""
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Pre-mock every third-party module that src/ingest.py imports at module level
# ---------------------------------------------------------------------------
def _make_mock_module(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod

for _mod_name in [
    "dotenv",
    "langchain_community",
    "langchain_community.document_loaders",
    "langchain_text_splitters",
    "langchain_postgres",
    "langchain_openai",
    "langchain_google_genai",
]:
    if _mod_name not in sys.modules:
        _make_mock_module(_mod_name)

# Provide the specific symbols that ingest.py uses at module level
sys.modules["dotenv"].load_dotenv = MagicMock()
sys.modules["langchain_community.document_loaders"].PyPDFLoader = MagicMock()
sys.modules["langchain_text_splitters"].RecursiveCharacterTextSplitter = MagicMock()
sys.modules["langchain_postgres"].PGVector = MagicMock()
sys.modules["langchain_openai"].OpenAIEmbeddings = MagicMock()
sys.modules["langchain_google_genai"].GoogleGenerativeAIEmbeddings = MagicMock()

# Add src/ to path and import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import ingest  # noqa: E402  (must come after sys.modules patching)


class TestGetEmbeddings(unittest.TestCase):
    """Unit tests for the _get_embeddings() factory."""

    def test_returns_openai_embeddings_when_provider_openai(self):
        """Should call OpenAIEmbeddings when PROVIDER=openai."""
        mock_cls = MagicMock()
        with patch.object(ingest, "PROVIDER", "openai"), \
             patch.object(ingest, "OpenAIEmbeddings", mock_cls):
            ingest._get_embeddings()
        mock_cls.assert_called_once()

    def test_returns_gemini_embeddings_when_provider_gemini(self):
        """Should call GoogleGenerativeAIEmbeddings when PROVIDER=gemini."""
        mock_cls = MagicMock()
        with patch.object(ingest, "PROVIDER", "gemini"), \
             patch.object(ingest, "GoogleGenerativeAIEmbeddings", mock_cls):
            ingest._get_embeddings()
        mock_cls.assert_called_once()

    def test_openai_uses_correct_default_model(self):
        """OpenAIEmbeddings should be called with model=text-embedding-3-small by default."""
        mock_cls = MagicMock()
        with patch.object(ingest, "PROVIDER", "openai"), \
             patch.object(ingest, "OpenAIEmbeddings", mock_cls), \
             patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_EMBEDDING_MODEL", None)
            ingest._get_embeddings()
        mock_cls.assert_called_once_with(model="text-embedding-3-small")

    def test_gemini_uses_correct_default_model(self):
        """GoogleGenerativeAIEmbeddings should use models/embedding-001 by default."""
        mock_cls = MagicMock()
        with patch.object(ingest, "PROVIDER", "gemini"), \
             patch.object(ingest, "GoogleGenerativeAIEmbeddings", mock_cls), \
             patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GOOGLE_EMBEDDING_MODEL", None)
            ingest._get_embeddings()
        mock_cls.assert_called_once_with(model="models/embedding-001")


class TestIngestPdf(unittest.TestCase):
    """Unit tests for the ingest_pdf() pipeline."""

    def setUp(self):
        """Reset PGVector mock between tests."""
        ingest.PGVector.reset_mock()

    # ------------------------------------------------------------------
    # Requisito: Chunks de 1000 caracteres com overlap de 150
    # ------------------------------------------------------------------

    def test_splitter_uses_correct_chunk_size(self):
        """RecursiveCharacterTextSplitter must use chunk_size=1000."""
        mock_loader_cls = MagicMock()
        mock_loader_cls.return_value.load.return_value = [MagicMock()]

        mock_splitter_cls = MagicMock()
        mock_splitter_cls.return_value.split_documents.return_value = [MagicMock()]

        with patch.object(ingest, "DATABASE_URL", "postgresql+psycopg://test"), \
             patch.object(ingest, "PDF_PATH", "doc.pdf"), \
             patch("os.path.exists", return_value=True), \
             patch.object(ingest, "PyPDFLoader", mock_loader_cls), \
             patch.object(ingest, "RecursiveCharacterTextSplitter", mock_splitter_cls), \
             patch.object(ingest, "_get_embeddings", return_value=MagicMock()), \
             patch.object(ingest, "PGVector") as mock_pv:
            mock_pv.from_documents = MagicMock()
            ingest.ingest_pdf()

        _, kwargs = mock_splitter_cls.call_args
        self.assertEqual(kwargs["chunk_size"], 1000)

    def test_splitter_uses_correct_chunk_overlap(self):
        """RecursiveCharacterTextSplitter must use chunk_overlap=150."""
        mock_loader_cls = MagicMock()
        mock_loader_cls.return_value.load.return_value = [MagicMock()]

        mock_splitter_cls = MagicMock()
        mock_splitter_cls.return_value.split_documents.return_value = [MagicMock()]

        with patch.object(ingest, "DATABASE_URL", "postgresql+psycopg://test"), \
             patch.object(ingest, "PDF_PATH", "doc.pdf"), \
             patch("os.path.exists", return_value=True), \
             patch.object(ingest, "PyPDFLoader", mock_loader_cls), \
             patch.object(ingest, "RecursiveCharacterTextSplitter", mock_splitter_cls), \
             patch.object(ingest, "_get_embeddings", return_value=MagicMock()), \
             patch.object(ingest, "PGVector") as mock_pv:
            mock_pv.from_documents = MagicMock()
            ingest.ingest_pdf()

        _, kwargs = mock_splitter_cls.call_args
        self.assertEqual(kwargs["chunk_overlap"], 150)

    # ------------------------------------------------------------------
    # Requisito: Ingestão no PostgreSQL via PGVector
    # ------------------------------------------------------------------

    def test_loader_called_with_configured_pdf_path(self):
        """PyPDFLoader should be instantiated with the configured PDF_PATH."""
        mock_loader_cls = MagicMock()
        mock_loader_cls.return_value.load.return_value = [MagicMock()]

        mock_splitter_cls = MagicMock()
        mock_splitter_cls.return_value.split_documents.return_value = [MagicMock()]

        with patch.object(ingest, "DATABASE_URL", "postgresql+psycopg://test"), \
             patch.object(ingest, "PDF_PATH", "my_doc.pdf"), \
             patch("os.path.exists", return_value=True), \
             patch.object(ingest, "PyPDFLoader", mock_loader_cls), \
             patch.object(ingest, "RecursiveCharacterTextSplitter", mock_splitter_cls), \
             patch.object(ingest, "_get_embeddings", return_value=MagicMock()), \
             patch.object(ingest, "PGVector") as mock_pv:
            mock_pv.from_documents = MagicMock()
            ingest.ingest_pdf()

        mock_loader_cls.assert_called_with("my_doc.pdf")

    def test_pgvector_called_with_correct_collection_name(self):
        """PGVector.from_documents must be called with the configured collection name."""
        mock_loader_cls = MagicMock()
        mock_loader_cls.return_value.load.return_value = [MagicMock()]

        chunks = [MagicMock(), MagicMock()]
        mock_splitter_cls = MagicMock()
        mock_splitter_cls.return_value.split_documents.return_value = chunks

        with patch.object(ingest, "DATABASE_URL", "postgresql+psycopg://test"), \
             patch.object(ingest, "PDF_PATH", "doc.pdf"), \
             patch.object(ingest, "COLLECTION_NAME", "my_col"), \
             patch("os.path.exists", return_value=True), \
             patch.object(ingest, "PyPDFLoader", mock_loader_cls), \
             patch.object(ingest, "RecursiveCharacterTextSplitter", mock_splitter_cls), \
             patch.object(ingest, "_get_embeddings", return_value=MagicMock()), \
             patch.object(ingest, "PGVector") as mock_pv:
            mock_pv.from_documents = MagicMock()
            ingest.ingest_pdf()

        _, kwargs = mock_pv.from_documents.call_args
        self.assertEqual(kwargs["collection_name"], "my_col")

    def test_pgvector_called_with_pre_delete_collection_true(self):
        """PGVector.from_documents must use pre_delete_collection=True to recreate collection."""
        mock_loader_cls = MagicMock()
        mock_loader_cls.return_value.load.return_value = [MagicMock()]

        mock_splitter_cls = MagicMock()
        mock_splitter_cls.return_value.split_documents.return_value = [MagicMock()]

        with patch.object(ingest, "DATABASE_URL", "postgresql+psycopg://test"), \
             patch.object(ingest, "PDF_PATH", "doc.pdf"), \
             patch("os.path.exists", return_value=True), \
             patch.object(ingest, "PyPDFLoader", mock_loader_cls), \
             patch.object(ingest, "RecursiveCharacterTextSplitter", mock_splitter_cls), \
             patch.object(ingest, "_get_embeddings", return_value=MagicMock()), \
             patch.object(ingest, "PGVector") as mock_pv:
            mock_pv.from_documents = MagicMock()
            ingest.ingest_pdf()

        _, kwargs = mock_pv.from_documents.call_args
        self.assertTrue(kwargs.get("pre_delete_collection"))

    def test_pgvector_receives_database_url(self):
        """PGVector.from_documents must receive the configured DATABASE_URL as connection."""
        mock_loader_cls = MagicMock()
        mock_loader_cls.return_value.load.return_value = [MagicMock()]

        mock_splitter_cls = MagicMock()
        mock_splitter_cls.return_value.split_documents.return_value = [MagicMock()]

        with patch.object(ingest, "DATABASE_URL", "postgresql+psycopg://mydb:5432/rag"), \
             patch.object(ingest, "PDF_PATH", "doc.pdf"), \
             patch("os.path.exists", return_value=True), \
             patch.object(ingest, "PyPDFLoader", mock_loader_cls), \
             patch.object(ingest, "RecursiveCharacterTextSplitter", mock_splitter_cls), \
             patch.object(ingest, "_get_embeddings", return_value=MagicMock()), \
             patch.object(ingest, "PGVector") as mock_pv:
            mock_pv.from_documents = MagicMock()
            ingest.ingest_pdf()

        _, kwargs = mock_pv.from_documents.call_args
        self.assertEqual(kwargs["connection"], "postgresql+psycopg://mydb:5432/rag")

    # ------------------------------------------------------------------
    # Guard clauses — should call sys.exit(1)
    # ------------------------------------------------------------------

    def test_exits_when_database_url_missing(self):
        """ingest_pdf() must sys.exit(1) when DATABASE_URL is not set."""
        with patch.object(ingest, "DATABASE_URL", None):
            with self.assertRaises(SystemExit) as ctx:
                ingest.ingest_pdf()
        self.assertEqual(ctx.exception.code, 1)

    def test_exits_when_pdf_not_found(self):
        """ingest_pdf() must sys.exit(1) when the PDF file does not exist."""
        with patch.object(ingest, "DATABASE_URL", "postgresql+psycopg://test"), \
             patch.object(ingest, "PDF_PATH", "missing.pdf"), \
             patch("os.path.exists", return_value=False):
            with self.assertRaises(SystemExit) as ctx:
                ingest.ingest_pdf()
        self.assertEqual(ctx.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
