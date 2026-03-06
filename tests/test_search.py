"""
Tests for src/search.py

All external dependencies are injected via sys.modules before src imports,
so no real DB, API keys, or third-party packages are needed.
"""
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch
from io import StringIO

# ---------------------------------------------------------------------------
# Pre-mock every third-party module that src/search.py imports at module level
# ---------------------------------------------------------------------------
def _make_mock_module(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod

for _mod_name in [
    "dotenv",
    "langchain_postgres",
    "langchain_openai",
    "langchain_google_genai",
]:
    if _mod_name not in sys.modules:
        _make_mock_module(_mod_name)

sys.modules["dotenv"].load_dotenv = MagicMock()
sys.modules["langchain_postgres"].PGVector = MagicMock()
sys.modules["langchain_openai"].OpenAIEmbeddings = MagicMock()
sys.modules["langchain_openai"].ChatOpenAI = MagicMock()
sys.modules["langchain_google_genai"].GoogleGenerativeAIEmbeddings = MagicMock()
sys.modules["langchain_google_genai"].ChatGoogleGenerativeAI = MagicMock()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import search  # noqa: E402


class TestGetEmbeddings(unittest.TestCase):
    """Unit tests for the _get_embeddings() factory in search."""

    def test_openai_provider_uses_openai_embeddings(self):
        mock_cls = MagicMock()
        with patch.object(search, "PROVIDER", "openai"), \
             patch.object(search, "OpenAIEmbeddings", mock_cls):
            search._get_embeddings()
        mock_cls.assert_called_once()

    def test_gemini_provider_uses_google_embeddings(self):
        mock_cls = MagicMock()
        with patch.object(search, "PROVIDER", "gemini"), \
             patch.object(search, "GoogleGenerativeAIEmbeddings", mock_cls):
            search._get_embeddings()
        mock_cls.assert_called_once()


class TestGetLlm(unittest.TestCase):
    """Unit tests for the _get_llm() factory."""

    def test_openai_provider_uses_chatopenai(self):
        mock_cls = MagicMock()
        with patch.object(search, "PROVIDER", "openai"), \
             patch.object(search, "ChatOpenAI", mock_cls):
            search._get_llm()
        mock_cls.assert_called_once()

    def test_gemini_provider_uses_chatgooglegenerativeai(self):
        mock_cls = MagicMock()
        with patch.object(search, "PROVIDER", "gemini"), \
             patch.object(search, "ChatGoogleGenerativeAI", mock_cls):
            search._get_llm()
        mock_cls.assert_called_once()

    def test_openai_llm_uses_zero_temperature(self):
        """LLM must be instantiated with temperature=0 to avoid creative answers."""
        mock_cls = MagicMock()
        with patch.object(search, "PROVIDER", "openai"), \
             patch.object(search, "ChatOpenAI", mock_cls):
            search._get_llm()
        _, kwargs = mock_cls.call_args
        self.assertEqual(kwargs.get("temperature"), 0)

    def test_gemini_llm_uses_zero_temperature(self):
        """Gemini LLM must also use temperature=0."""
        mock_cls = MagicMock()
        with patch.object(search, "PROVIDER", "gemini"), \
             patch.object(search, "ChatGoogleGenerativeAI", mock_cls):
            search._get_llm()
        _, kwargs = mock_cls.call_args
        self.assertEqual(kwargs.get("temperature"), 0)


class TestSearchPrompt(unittest.TestCase):
    """Unit tests for search_prompt()."""

    def setUp(self):
        search.PGVector.reset_mock()

    def _make_llm(self, content: str):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content=content)
        return llm

    def _make_store(self, docs_content: list):
        """Helper: return a mock PGVector store with similarity_search_with_score."""
        mock_store = MagicMock()
        results = [(MagicMock(page_content=c), 0.9) for c in docs_content]
        mock_store.similarity_search_with_score.return_value = results
        return mock_store

    # ------------------------------------------------------------------
    # Guard clauses
    # ------------------------------------------------------------------

    def test_returns_none_when_no_database_url(self):
        with patch.object(search, "DATABASE_URL", None):
            self.assertIsNone(search.search_prompt("Pergunta?"))

    def test_returns_none_for_empty_string(self):
        with patch.object(search, "DATABASE_URL", "postgresql+psycopg://test"):
            self.assertIsNone(search.search_prompt(""))

    def test_returns_none_for_whitespace_only(self):
        with patch.object(search, "DATABASE_URL", "postgresql+psycopg://test"):
            self.assertIsNone(search.search_prompt("   "))

    # ------------------------------------------------------------------
    # Requisito: similarity_search_with_score com k=10
    # ------------------------------------------------------------------

    def test_uses_similarity_search_with_score(self):
        """Must call similarity_search_with_score (not similarity_search) as required."""
        mock_store = self._make_store(["Conteúdo."])
        with patch.object(search, "DATABASE_URL", "postgresql+psycopg://test"), \
             patch.object(search, "_get_embeddings", return_value=MagicMock()), \
             patch.object(search, "_get_llm", return_value=self._make_llm("ok")), \
             patch.object(search, "PGVector", return_value=mock_store):
            search.search_prompt("Qual o tema?")

        mock_store.similarity_search_with_score.assert_called_once()
        mock_store.similarity_search.assert_not_called()

    def test_similarity_search_called_with_k10(self):
        """similarity_search_with_score must use k=10 as required by the challenge."""
        mock_store = self._make_store(["Texto."])
        with patch.object(search, "DATABASE_URL", "postgresql+psycopg://test"), \
             patch.object(search, "_get_embeddings", return_value=MagicMock()), \
             patch.object(search, "_get_llm", return_value=self._make_llm("ok")), \
             patch.object(search, "PGVector", return_value=mock_store):
            search.search_prompt("Pergunta?")

        mock_store.similarity_search_with_score.assert_called_once_with("Pergunta?", k=10)

    # ------------------------------------------------------------------
    # Requisito: Prompt anti-alucinação
    # ------------------------------------------------------------------

    def test_prompt_contains_context_from_retrieved_chunks(self):
        """The context built from retrieved chunks must appear in the prompt sent to LLM."""
        mock_store = self._make_store(["Trecho 0", "Trecho 1", "Trecho 2"])
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = MagicMock(content="ok")

        with patch.object(search, "DATABASE_URL", "postgresql+psycopg://test"), \
             patch.object(search, "_get_embeddings", return_value=MagicMock()), \
             patch.object(search, "_get_llm", return_value=fake_llm), \
             patch.object(search, "PGVector", return_value=mock_store):
            search.search_prompt("Qual conteúdo?")

        prompt_sent = fake_llm.invoke.call_args[0][0]
        for i in range(3):
            self.assertIn(f"Trecho {i}", prompt_sent)

    def test_prompt_contains_user_question(self):
        """The user's question must be embedded in the prompt sent to the LLM."""
        question = "Qual é o faturamento da empresa?"
        mock_store = self._make_store(["Algum conteúdo."])
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = MagicMock(content="R$ 10 milhões.")

        with patch.object(search, "DATABASE_URL", "postgresql+psycopg://test"), \
             patch.object(search, "_get_embeddings", return_value=MagicMock()), \
             patch.object(search, "_get_llm", return_value=fake_llm), \
             patch.object(search, "PGVector", return_value=mock_store):
            search.search_prompt(question)

        prompt_sent = fake_llm.invoke.call_args[0][0]
        self.assertIn(question, prompt_sent)

    def test_prompt_contains_anti_hallucination_rules(self):
        """The prompt must include the anti-hallucination rules from the challenge spec."""
        mock_store = self._make_store(["Conteúdo."])
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = MagicMock(content="ok")

        with patch.object(search, "DATABASE_URL", "postgresql+psycopg://test"), \
             patch.object(search, "_get_embeddings", return_value=MagicMock()), \
             patch.object(search, "_get_llm", return_value=fake_llm), \
             patch.object(search, "PGVector", return_value=mock_store):
            search.search_prompt("Pergunta?")

        prompt_sent = fake_llm.invoke.call_args[0][0]
        self.assertIn("Responda somente com base no CONTEXTO", prompt_sent)
        self.assertIn("Nunca invente ou use conhecimento externo", prompt_sent)

    # ------------------------------------------------------------------
    # Cenários do desafio: perguntas no contexto vs. fora do contexto
    # ------------------------------------------------------------------

    def test_returns_llm_answer_when_context_found(self):
        """Simulates: 'Qual o faturamento?' → found in PDF → returns LLM answer."""
        mock_store = self._make_store(["O faturamento foi de 10 milhões de reais."])

        with patch.object(search, "DATABASE_URL", "postgresql+psycopg://test"), \
             patch.object(search, "_get_embeddings", return_value=MagicMock()), \
             patch.object(search, "_get_llm", return_value=self._make_llm("O faturamento foi de 10 milhões de reais.")), \
             patch.object(search, "PGVector", return_value=mock_store):
            result = search.search_prompt("Qual o faturamento da Empresa SuperTechIABrazil?")

        self.assertEqual(result, "O faturamento foi de 10 milhões de reais.")

    def test_fallback_when_no_docs_found(self):
        """Simulates: question completely unrelated to PDF → no results → fallback message."""
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = []

        with patch.object(search, "DATABASE_URL", "postgresql+psycopg://test"), \
             patch.object(search, "_get_embeddings", return_value=MagicMock()), \
             patch.object(search, "_get_llm", return_value=MagicMock()), \
             patch.object(search, "PGVector", return_value=mock_store):
            result = search.search_prompt("Quantos clientes temos em 2024?")

        self.assertIn("Não tenho informações", result)

    def test_fallback_message_exact_wording(self):
        """Fallback message must match the exact wording defined in the challenge spec."""
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = []

        with patch.object(search, "DATABASE_URL", "postgresql+psycopg://test"), \
             patch.object(search, "_get_embeddings", return_value=MagicMock()), \
             patch.object(search, "_get_llm", return_value=MagicMock()), \
             patch.object(search, "PGVector", return_value=mock_store):
            result = search.search_prompt("Qual é a capital da França?")

        self.assertIn("Não tenho informações necessárias para responder sua pergunta", result)

    # ------------------------------------------------------------------
    # Exception handling
    # ------------------------------------------------------------------

    def test_returns_none_on_pgvector_exception(self):
        with patch.object(search, "DATABASE_URL", "postgresql+psycopg://test"), \
             patch.object(search, "_get_embeddings", return_value=MagicMock()), \
             patch.object(search, "PGVector", side_effect=RuntimeError("connection refused")):
            result = search.search_prompt("Pergunta?")

        self.assertIsNone(result)

    def test_returns_none_on_llm_exception(self):
        mock_store = self._make_store(["Conteúdo."])
        broken_llm = MagicMock()
        broken_llm.invoke.side_effect = RuntimeError("LLM timeout")

        with patch.object(search, "DATABASE_URL", "postgresql+psycopg://test"), \
             patch.object(search, "_get_embeddings", return_value=MagicMock()), \
             patch.object(search, "_get_llm", return_value=broken_llm), \
             patch.object(search, "PGVector", return_value=mock_store):
            result = search.search_prompt("Pergunta?")

        self.assertIsNone(result)

    def test_error_message_printed_on_exception(self):
        """Should print an error to stderr when an exception occurs."""
        with patch.object(search, "DATABASE_URL", "postgresql+psycopg://test"), \
             patch.object(search, "_get_embeddings", return_value=MagicMock()), \
             patch.object(search, "PGVector", side_effect=RuntimeError("boom")), \
             patch("sys.stderr", new_callable=StringIO) as mock_err:
            search.search_prompt("Pergunta?")
        self.assertIn("ERRO", mock_err.getvalue())


if __name__ == "__main__":
    unittest.main()
