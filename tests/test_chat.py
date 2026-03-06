"""
Tests for src/chat.py

input() and search_prompt() are mocked so no real I/O or API calls occur.
All third-party modules are pre-mocked via sys.modules.
"""
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch
from io import StringIO

# ---------------------------------------------------------------------------
# Pre-mock third-party modules (chat.py imports search.py which needs dotenv)
# ---------------------------------------------------------------------------
def _make_mock_module(name):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    return sys.modules[name]

for _mod_name in [
    "dotenv",
    "langchain_postgres",
    "langchain_openai",
    "langchain_google_genai",
]:
    _make_mock_module(_mod_name)

sys.modules["dotenv"].load_dotenv = MagicMock()
sys.modules["langchain_postgres"].PGVector = MagicMock()
sys.modules["langchain_openai"].ChatOpenAI = MagicMock()
sys.modules["langchain_openai"].OpenAIEmbeddings = MagicMock()
sys.modules["langchain_google_genai"].ChatGoogleGenerativeAI = MagicMock()
sys.modules["langchain_google_genai"].GoogleGenerativeAIEmbeddings = MagicMock()

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chat  # noqa: E402


class TestChatMain(unittest.TestCase):
    """Unit tests for chat.main()."""

    def _run(self, inputs: list, search_return="Resposta simulada."):
        """Helper: run chat.main() with mocked input() and search_prompt()."""
        input_iter = iter(inputs)
        captured = StringIO()

        with patch("chat.search_prompt", return_value=search_return) as mock_search, \
             patch("builtins.input", side_effect=input_iter), \
             patch("sys.stdout", captured):
            chat.main()

        return captured.getvalue(), mock_search

    # ------------------------------------------------------------------
    # Exit commands
    # ------------------------------------------------------------------

    def test_exit_command_sair(self):
        output, _ = self._run(["sair"])
        self.assertIn("Encerrando", output)

    def test_exit_command_exit(self):
        output, _ = self._run(["exit"])
        self.assertIn("Encerrando", output)

    def test_exit_command_quit(self):
        output, _ = self._run(["quit"])
        self.assertIn("Encerrando", output)

    def test_exit_commands_are_case_insensitive(self):
        """'SAIR', 'EXIT', 'QUIT' should also close the chat."""
        for cmd in ["SAIR", "EXIT", "QUIT"]:
            with self.subTest(cmd=cmd):
                output, _ = self._run([cmd])
                self.assertIn("Encerrando", output)

    # ------------------------------------------------------------------
    # Requisito: Pergunta via CLI e resposta baseada no PDF
    # ------------------------------------------------------------------

    def test_question_is_forwarded_to_search_prompt(self):
        """Every user question must be passed to search_prompt."""
        _, mock_search = self._run(["Qual é o tema?", "sair"])
        mock_search.assert_any_call("Qual é o tema?")

    def test_answer_is_printed_to_stdout(self):
        """The answer from search_prompt must be displayed to the user."""
        output, _ = self._run(["Pergunta?", "sair"], search_return="Resposta correta.")
        self.assertIn("Resposta correta.", output)

    def test_challenge_scenario_question_in_context(self):
        """Simulates the challenge example: faturamento question gets a real answer."""
        resposta = "O faturamento foi de 10 milhões de reais."
        output, _ = self._run(
            ["Qual o faturamento da Empresa SuperTechIABrazil?", "sair"],
            search_return=resposta,
        )
        self.assertIn(resposta, output)

    def test_challenge_scenario_question_out_of_context(self):
        """Simulates the challenge example: out-of-context question gets the fallback message."""
        fallback = "Não tenho informações necessárias para responder sua pergunta."
        output, _ = self._run(
            ["Quantos clientes temos em 2024?", "sair"],
            search_return=fallback,
        )
        self.assertIn(fallback, output)

    # ------------------------------------------------------------------
    # Input edge cases
    # ------------------------------------------------------------------

    def test_empty_inputs_are_ignored(self):
        """Empty strings and whitespace-only inputs must not call search_prompt."""
        _, mock_search = self._run(["", "   ", "sair"])
        mock_search.assert_not_called()

    def test_multiple_questions_each_call_search(self):
        """Each non-empty, non-exit input must call search_prompt exactly once."""
        _, mock_search = self._run(["Pergunta 1?", "Pergunta 2?", "sair"])
        self.assertEqual(mock_search.call_count, 2)

    def test_questions_trimmed_before_forwarding(self):
        """Leading/trailing whitespace should be stripped from questions."""
        _, mock_search = self._run(["  Qual o tema?  ", "sair"])
        mock_search.assert_any_call("Qual o tema?")

    # ------------------------------------------------------------------
    # Error / edge cases
    # ------------------------------------------------------------------

    def test_none_response_prints_error_message(self):
        """When search_prompt returns None, a user-friendly error must be shown."""
        output, _ = self._run(["Pergunta?", "sair"], search_return=None)
        self.assertIn("possível processar", output)

    def test_keyboard_interrupt_exits_gracefully(self):
        """Ctrl+C should exit cleanly without raising an exception."""
        captured = StringIO()
        with patch("chat.search_prompt"), \
             patch("builtins.input", side_effect=KeyboardInterrupt), \
             patch("sys.stdout", captured):
            chat.main()  # must not raise
        self.assertIn("Encerrando", captured.getvalue())

    def test_eof_error_exits_gracefully(self):
        """EOF (e.g. piped input ending) should exit cleanly without raising."""
        captured = StringIO()
        with patch("chat.search_prompt"), \
             patch("builtins.input", side_effect=EOFError), \
             patch("sys.stdout", captured):
            chat.main()  # must not raise
        self.assertIn("Encerrando", captured.getvalue())

    # ------------------------------------------------------------------
    # Banner / UI
    # ------------------------------------------------------------------

    def test_banner_is_displayed_on_startup(self):
        """The CLI banner must be displayed when the chat starts."""
        output, _ = self._run(["sair"])
        self.assertIn("RAG", output)

    def test_chat_continues_after_each_answer(self):
        """After each answer, the CLI must keep running until an exit command."""
        _, mock_search = self._run(["P1?", "P2?", "P3?", "sair"])
        self.assertEqual(mock_search.call_count, 3)


if __name__ == "__main__":
    unittest.main()
