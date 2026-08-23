import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

from hipporag.llm.bedrock_llm import BedrockLLM
from hipporag.utils.config_utils import BaseConfig


def bedrock_response(content="Bedrock response", include_usage=True):
    usage = None
    if include_usage:
        usage = SimpleNamespace(
            prompt_tokens=4,
            completion_tokens=5,
            total_tokens=9,
            prompt_tokens_details=SimpleNamespace(cached_tokens=1),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
        )
    return SimpleNamespace(
        id="bedrock-response-id",
        model="bedrock/test-model",
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
        usage=usage,
    )


class BedrockLLMTest(unittest.TestCase):
    @staticmethod
    def make_model(save_dir):
        return BedrockLLM(
            BaseConfig(
                save_dir=save_dir,
                llm_name="bedrock/test-model",
                max_retry_attempts=0,
            )
        )

    def test_valid_response_is_accounted_and_cached(self):
        with tempfile.TemporaryDirectory() as save_dir:
            model = self.make_model(save_dir)
            with patch("hipporag.llm.bedrock_llm.litellm.completion", return_value=bedrock_response()) as completion:
                first_message, first_metadata, first_cached = model.infer([{"role": "user", "content": "hello"}])
                second_message, second_metadata, second_cached = model.infer([{"role": "user", "content": "hello"}])

        self.assertEqual(first_message, "Bedrock response")
        self.assertEqual(second_message, first_message)
        self.assertFalse(first_cached)
        self.assertTrue(second_cached)
        self.assertEqual(first_metadata, second_metadata)
        self.assertEqual(first_metadata["total_tokens"], 9)
        self.assertEqual(first_metadata["cached_tokens"], 1)
        self.assertEqual(first_metadata["reasoning_tokens"], 2)
        completion.assert_called_once()

    def test_invalid_responses_are_not_cached(self):
        cases = (
            ("missing choices", SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1)), "exactly one"),
            ("empty content", bedrock_response(content=None), "non-empty text"),
            ("missing usage", bedrock_response(include_usage=False), "omitted usage"),
            (
                "invalid usage",
                SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="response"), finish_reason="stop")],
                    usage=SimpleNamespace(prompt_tokens=None, completion_tokens=1),
                ),
                "prompt_tokens",
            ),
        )
        for label, response, error_pattern in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as save_dir:
                model = self.make_model(save_dir)
                with patch("hipporag.llm.bedrock_llm.litellm.completion", return_value=response) as completion:
                    for _ in range(2):
                        with self.assertRaisesRegex(ValueError, error_pattern):
                            model.infer([{"role": "user", "content": "invalid"}])
                self.assertEqual(completion.call_count, 2)

    def test_unsupported_multi_choice_and_streaming_fail_before_request(self):
        with tempfile.TemporaryDirectory() as save_dir:
            model = self.make_model(save_dir)
            with patch("hipporag.llm.bedrock_llm.litellm.completion") as completion:
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    model.infer([{"role": "user", "content": "hello"}], n=2)
                with self.assertRaisesRegex(ValueError, "streaming"):
                    model.infer([{"role": "user", "content": "hello"}], stream=True)
                completion.assert_not_called()

    def test_explicit_region_is_forwarded_to_litellm(self):
        with tempfile.TemporaryDirectory() as save_dir:
            model = BedrockLLM(
                BaseConfig(
                    save_dir=save_dir,
                    llm_name="bedrock/test-model",
                    bedrock_region="us-west-2",
                    max_retry_attempts=0,
                )
            )
            with patch("hipporag.llm.bedrock_llm.litellm.completion", return_value=bedrock_response()) as completion:
                model.infer([{"role": "user", "content": "hello"}])

        self.assertEqual(completion.call_args.kwargs["aws_region_name"], "us-west-2")

    def test_same_key_requests_use_single_flight(self):
        with tempfile.TemporaryDirectory() as save_dir:
            model = self.make_model(save_dir)
            start_barrier = threading.Barrier(3)
            first_request_started = threading.Event()
            duplicate_request_started = threading.Event()
            release_request = threading.Event()
            call_count = 0
            call_count_lock = threading.Lock()

            def completion_side_effect(**kwargs):
                nonlocal call_count
                with call_count_lock:
                    call_count += 1
                    current_call = call_count
                if current_call == 1:
                    first_request_started.set()
                else:
                    duplicate_request_started.set()
                if not release_request.wait(timeout=2):
                    raise TimeoutError("Test did not release the mocked Bedrock request.")
                return bedrock_response()

            def infer_once():
                start_barrier.wait(timeout=2)
                return model.infer([{"role": "user", "content": "same key"}])

            with patch("hipporag.llm.bedrock_llm.litellm.completion", side_effect=completion_side_effect) as completion:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    first_future = executor.submit(infer_once)
                    second_future = executor.submit(infer_once)
                    start_barrier.wait(timeout=2)
                    try:
                        self.assertTrue(first_request_started.wait(timeout=2))
                        self.assertFalse(duplicate_request_started.wait(timeout=0.1))
                    finally:
                        release_request.set()
                    results = [first_future.result(timeout=2), second_future.result(timeout=2)]

        self.assertEqual(completion.call_count, 1)
        self.assertEqual(sorted(result[2] for result in results), [False, True])


if __name__ == "__main__":
    unittest.main()
