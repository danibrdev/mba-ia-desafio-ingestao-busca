"""
Tests for src/search.py
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from io import StringIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from search import AppConfig, get_embeddings, get_llm, get_vector_store, search_prompt, build_chat_history_str

class TestSearchFactories(unittest.TestCase):
    @patch.dict(os.environ, {
        "DATABASE_URL": "postgresql+psycopg://test",
        "PROVIDER": "openai",
        "OPENAI_API_KEY": "fake-openai-key"
    }, clear=True)
    def test_openai_provider_uses_openai_embeddings(self):
        config = AppConfig()
        with patch("search.OpenAIEmbeddings") as MockClass:
            get_embeddings(config)
            MockClass.assert_called_once()

    @patch.dict(os.environ, {
        "DATABASE_URL": "postgresql+psycopg://test",
        "PROVIDER": "gemini",
        "GOOGLE_API_KEY": "fake-google-key"
    }, clear=True)
    def test_gemini_provider_uses_google_embeddings(self):
        config = AppConfig()
        with patch("search.GoogleGenerativeAIEmbeddings") as MockClass:
            get_embeddings(config)
            MockClass.assert_called_once()

    @patch.dict(os.environ, {
        "DATABASE_URL": "postgresql+psycopg://test",
        "PROVIDER": "openai",
        "OPENAI_API_KEY": "fake-key"
    }, clear=True)
    def test_openai_llm_uses_zero_temperature(self):
        config = AppConfig()
        with patch("search.ChatOpenAI") as MockClass:
            get_llm(config)
            MockClass.assert_called_once_with(model=config.openai_llm_model, temperature=0.0)


class TestSearchPrompt(unittest.TestCase):
    def _make_llm(self, content: str):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content=content)
        return llm

    def _make_store(self, docs_content: list):
        mock_store = MagicMock()
        results = [(MagicMock(page_content=c), 0.9) for c in docs_content]
        mock_store.similarity_search_with_score.return_value = results
        return mock_store

    def test_returns_none_for_empty_string(self):
        self.assertIsNone(search_prompt("", self._make_llm("ok"), self._make_store(["abc"]), []))

    def test_similarity_search_called_with_k10(self):
        mock_store = self._make_store(["Texto."])
        search_prompt("Pergunta?", self._make_llm("ok"), mock_store, [])
        mock_store.similarity_search_with_score.assert_called_once_with("Pergunta?", k=10)

    def test_prompt_contains_context_from_retrieved_chunks(self):
        mock_store = self._make_store(["Trecho 0", "Trecho 1"])
        fake_llm = self._make_llm("ok")
        search_prompt("Qual conteúdo?", fake_llm, mock_store, [])
        
        prompt_sent = fake_llm.invoke.call_args[0][0]
        self.assertIn("Trecho 0", prompt_sent)
        self.assertIn("Trecho 1", prompt_sent)

    def test_prompt_contains_anti_hallucination_rules(self):
        fake_llm = self._make_llm("ok")
        search_prompt("Pergunta?", fake_llm, self._make_store(["Conteúdo."]), [])
        
        prompt_sent = fake_llm.invoke.call_args[0][0]
        self.assertIn("Responda somente com base no CONTEXTO", prompt_sent)
        self.assertIn("Nunca invente", prompt_sent)

    def test_fallback_when_no_docs_found(self):
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = []
        result = search_prompt("Quantos clientes temos?", self._make_llm("ok"), mock_store, [])
        self.assertIn("Não tenho informações", result)

    def test_handles_llm_exception_gracefully(self):
        broken_llm = MagicMock()
        broken_llm.invoke.side_effect = RuntimeError("LLM broken")
        result = search_prompt("Pergunta?", broken_llm, self._make_store(["Conteúdo."]), [])
        self.assertIsNone(result)

    def test_build_chat_history_str_empty(self):
        self.assertEqual(build_chat_history_str([]), "Nenhum histórico anterior.")

    def test_build_chat_history_str_with_data(self):
        hist = [("Q1", "A1"), ("Q2", "A2")]
        res = build_chat_history_str(hist)
        self.assertIn("Usuário: Q1", res)
        self.assertIn("Assistente: A1", res)

if __name__ == "__main__":
    unittest.main()
