import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hipporag.llm import _get_llm_class
from hipporag.llm.orcarouter_llm import OrcaRouterLLM
from hipporag.utils.config_utils import BaseConfig


class OrcaRouterLLMTest(unittest.TestCase):
    def make_config(self, save_dir):
        return BaseConfig(
            llm_name="orcarouter/anthropic/claude-opus-4.8",
            save_dir=save_dir,
        )

    def test_provider_selection(self):
        with tempfile.TemporaryDirectory() as save_dir, patch.dict(os.environ, {"ORCAROUTER_API_KEY": "sk-orca-test"}):
            with patch("hipporag.llm.orcarouter_llm.OpenAI"):
                self.assertIsInstance(_get_llm_class(self.make_config(save_dir)), OrcaRouterLLM)

    def test_chat_completions_inference(self):
        response = SimpleNamespace(
            id="chatcmpl-orca-test",
            model="orcarouter/anthropic/claude-opus-4.8",
            choices=[SimpleNamespace(message=SimpleNamespace(content="HippoRAG OrcaRouter test passed"), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=5, total_tokens=9),
        )
        with tempfile.TemporaryDirectory() as save_dir, patch.dict(os.environ, {"ORCAROUTER_API_KEY": "sk-orca-test"}):
            with patch("hipporag.llm.orcarouter_llm.OpenAI") as openai:
                openai.return_value.chat.completions.create = MagicMock(return_value=response)
                llm = OrcaRouterLLM(self.make_config(save_dir))
                message, metadata, cached = llm.infer([{"role": "user", "content": "Test"}])

        self.assertEqual(message, "HippoRAG OrcaRouter test passed")
        self.assertEqual(metadata["prompt_tokens"], 4)
        self.assertEqual(metadata["total_tokens"], 9)
        self.assertFalse(cached)
        openai.return_value.chat.completions.create.assert_called_once_with(
            model="anthropic/claude-opus-4.8",
            max_completion_tokens=2048,
            temperature=0,
            messages=[{"role": "user", "content": "Test"}],
        )

    def test_missing_api_key_is_an_error(self):
        with tempfile.TemporaryDirectory() as save_dir, patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "ORCAROUTER_API_KEY"):
                OrcaRouterLLM(self.make_config(save_dir))

    def test_bare_prefix_is_rejected(self):
        with tempfile.TemporaryDirectory() as save_dir, patch.dict(os.environ, {"ORCAROUTER_API_KEY": "sk-orca-test"}):
            config = self.make_config(save_dir)
            config.llm_name = "orcarouter/"
            with self.assertRaisesRegex(ValueError, "vendor/model"):
                OrcaRouterLLM(config)

    def test_default_base_url_is_used_when_unset(self):
        with tempfile.TemporaryDirectory() as save_dir, patch.dict(os.environ, {"ORCAROUTER_API_KEY": "sk-orca-test"}):
            with patch("hipporag.llm.orcarouter_llm.OpenAI") as openai:
                llm = OrcaRouterLLM(self.make_config(save_dir))

        self.assertEqual(llm.llm_base_url, "https://api.orcarouter.ai/v1")
        self.assertEqual(openai.call_args.kwargs["base_url"], "https://api.orcarouter.ai/v1")


if __name__ == "__main__":
    unittest.main()
