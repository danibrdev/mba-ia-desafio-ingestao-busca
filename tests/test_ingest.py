"""
Tests for src/ingest.py
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import search
import ingest

class TestIngestPdf(unittest.TestCase):
    @patch.dict(os.environ, {
        "DATABASE_URL": "postgresql+psycopg://test",
        "PROVIDER": "openai",
        "OPENAI_API_KEY": "fake_key"
    }, clear=True)
    def test_ingest_fails_when_pdf_missing(self):
        with patch("os.path.exists", return_value=False):
            with self.assertRaises(SystemExit) as ctx:
                ingest.ingest_pdf()
            self.assertEqual(ctx.exception.code, 1)

    @patch.dict(os.environ, {
        "DATABASE_URL": "postgresql+psycopg://test",
        "PROVIDER": "openai",
        "OPENAI_API_KEY": "fake_key"
    }, clear=True)
    @patch("ingest.PyPDFLoader")
    @patch("ingest.RecursiveCharacterTextSplitter")
    @patch("ingest.PGVector")
    @patch("search.OpenAIEmbeddings")
    def test_ingest_success_flow(self, mock_emb, mock_pgv, mock_splitter, mock_loader):
        mock_loader.return_value.load.return_value = [MagicMock()]
        mock_splitter.return_value.split_documents.return_value = [MagicMock(), MagicMock()]
        
        with patch("os.path.exists", return_value=True):
            ingest.ingest_pdf()
            
        _, kwargs = mock_splitter.call_args
        self.assertEqual(kwargs["chunk_size"], 1000)
        self.assertEqual(kwargs["chunk_overlap"], 150)
        
        # PGVector.from_documents called Exactly once (because array size 2 <= BATCH SIZE 100)
        mock_pgv.from_documents.assert_called_once()
        _, kwargs_pg = mock_pgv.from_documents.call_args
        self.assertTrue(kwargs_pg["pre_delete_collection"])


if __name__ == "__main__":
    unittest.main()
