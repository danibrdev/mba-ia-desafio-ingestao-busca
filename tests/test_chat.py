"""
Tests for src/chat.py
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from io import StringIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import chat


class TestChatMain(unittest.TestCase):
    def _run(self, inputs: list, search_return="Resposta simulada."):
        input_iter = iter(inputs)
        captured = StringIO()

        with patch.dict(os.environ, {
                 "DATABASE_URL": "postgresql+psycopg://test",
                 "PROVIDER": "openai",
                 "OPENAI_API_KEY": "fake_key"
             }, clear=True), \
             patch("chat.get_vector_store", return_value=MagicMock()), \
             patch("chat.get_llm", return_value=MagicMock()), \
             patch("chat.search_prompt", return_value=search_return) as mock_search, \
             patch("builtins.input", side_effect=input_iter), \
             patch("sys.stdout", captured):
             
             try:
                 chat.main()
             except StopIteration:
                 # when input mock is fully consumed
                 pass

        return captured.getvalue(), mock_search

    def test_exit_command_sair(self):
        output, _ = self._run(["sair"])
        self.assertIn("Encerrando", output)

    def test_question_is_forwarded_to_search_prompt(self):
        _, mock_search = self._run(["Qual é o tema?", "sair"])
        # Arg 1 is the question text
        args, _ = mock_search.call_args
        self.assertEqual(args[0], "Qual é o tema?")

    def test_empty_inputs_are_ignored(self):
        _, mock_search = self._run(["", "   ", "sair"])
        mock_search.assert_not_called()

    def test_multiple_questions_append_to_history(self):
        _, mock_search = self._run(["Perg1?", "Perg2?", "sair"])
        self.assertEqual(mock_search.call_count, 2)
        
        # chat_history appending logic occurs AFTER search_prompt.
        # By the end of the run, two pairs should have been added.
        # However, mock call args holds a reference to the same list.
        _, kwargs_last_call = mock_search.call_args
        self.assertEqual(len(kwargs_last_call["chat_history"]), 2)
        self.assertEqual(kwargs_last_call["chat_history"][0][0], "Perg1?")


if __name__ == "__main__":
    unittest.main()
