import importlib
import json
import os
import re
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qs

import numpy as np
import openai
from openai import AzureOpenAI, DefaultHttpxClient, OpenAI

from hipporag.embedding_model.OpenAI import OpenAIEmbeddingModel
from hipporag.llm.bedrock_mantle import BedrockMantleLLM, BedrockMantleSigV4Auth
from hipporag.llm.openai_gpt import CacheOpenAI
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.openai_utils import resolve_azure_openai_settings, validate_openai_base_url


SDK_HTTPX = importlib.import_module("httpx2" if int(openai.__version__.split(".", 1)[0]) >= 3 else "httpx")


def chat_payload(content="offline chat"):
    return {
        "id": "chatcmpl-offline",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-offline",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    }


def response_payload(content="offline response", status="completed", include_usage=True):
    payload = {
        "id": "resp-offline",
        "object": "response",
        "created_at": 1,
        "status": status,
        "model": "openai.gpt-offline",
        "output": [],
    }
    if content is not None:
        payload["output"] = [{
            "id": "msg-offline",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": content, "annotations": []}],
        }]
    if include_usage:
        payload["usage"] = {
            "input_tokens": 2,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 3,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 5,
        }
    return payload


def embeddings_payload(include_usage=True, count=2):
    payload = {
        "object": "list",
        "model": "text-embedding-offline",
        "data": [
            {"object": "embedding", "index": index, "embedding": [float(index == 0), float(index != 0)]}
            for index in reversed(range(count))
        ],
    }
    if include_usage:
        payload["usage"] = {"prompt_tokens": count, "total_tokens": count}
    return payload


class OpenAISDKCompatibilityTest(unittest.TestCase):
    def test_installed_openai_sdk_is_in_supported_range(self):
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)", openai.__version__)
        self.assertIsNotNone(match, f"Cannot parse OpenAI SDK version {openai.__version__!r}.")
        numeric_version = tuple(int(part) for part in match.groups())
        self.assertGreaterEqual(numeric_version, (3, 3, 1))
        self.assertLess(numeric_version, (4, 0, 0))

    def make_openai_client(self, handler, base_url="https://offline.invalid/v1"):
        http_client = SDK_HTTPX.Client(transport=SDK_HTTPX.MockTransport(handler))
        return OpenAI(api_key="offline-key", base_url=base_url, http_client=http_client, max_retries=0)

    def test_chat_wrapper_uses_real_typed_sdk_response(self):
        requests = []

        def handler(request):
            requests.append(request)
            return SDK_HTTPX.Response(200, json=chat_payload(), headers={"x-request-id": "request-offline"}, request=request)

        with tempfile.TemporaryDirectory() as save_dir, patch.dict(os.environ, {"OPENAI_API_KEY": "offline-key"}):
            config = BaseConfig(
                save_dir=save_dir,
                llm_name="gpt-offline",
                llm_base_url="https://offline.invalid/v1",
                max_new_tokens=17,
                max_retry_attempts=0,
                temperature=None,
                response_format={"type": "json_object"},
                llm_supports_max_completion_tokens=True,
            )
            llm = CacheOpenAI(save_dir, config, high_throughput=False, max_retries=0)
            llm.openai_client.close()
            llm.openai_client = self.make_openai_client(handler)
            message, metadata, cache_hit = llm.infer(
                [{"role": "user", "content": "hello"}],
                response_format={"type": "json_object"},
            )
            llm.close()

        self.assertEqual(message, "offline chat")
        self.assertEqual(metadata["total_tokens"], 5)
        self.assertFalse(cache_hit)
        self.assertEqual(requests[0].url.path, "/v1/chat/completions")
        body = json.loads(requests[0].content)
        self.assertEqual(body["max_completion_tokens"], 17)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertNotIn("seed", body)
        self.assertNotIn("temperature", body)
        self.assertNotIn("n", body)

    def test_chat_wrapper_rejects_unaccounted_response_without_caching(self):
        request_count = 0

        def handler(request):
            nonlocal request_count
            request_count += 1
            payload = chat_payload()
            payload.pop("usage")
            return SDK_HTTPX.Response(200, json=payload, request=request)

        with tempfile.TemporaryDirectory() as save_dir, patch.dict(os.environ, {"OPENAI_API_KEY": "offline-key"}):
            config = BaseConfig(save_dir=save_dir, llm_name="gpt-offline", max_retry_attempts=0)
            llm = CacheOpenAI(save_dir, config, high_throughput=False, max_retries=0)
            llm.openai_client.close()
            llm.openai_client = self.make_openai_client(handler)
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, "omitted usage"):
                    llm.infer([{"role": "user", "content": "hello"}])
            llm.close()

        self.assertEqual(request_count, 2)

    def test_responses_wrapper_uses_real_typed_sdk_response(self):
        requests = []

        def handler(request):
            requests.append(request)
            return SDK_HTTPX.Response(200, json=response_payload(), request=request)

        with tempfile.TemporaryDirectory() as save_dir, patch.dict(os.environ, {"AWS_BEARER_TOKEN_BEDROCK": "offline-key"}):
            config = BaseConfig(
                save_dir=save_dir,
                llm_name="bedrock-mantle/openai.gpt-offline",
                llm_base_url="https://bedrock-offline.invalid/openai/v1",
                max_new_tokens=17,
                max_retry_attempts=0,
            )
            llm = BedrockMantleLLM(config)
            llm.openai_client.close()
            llm.openai_client = self.make_openai_client(handler, config.llm_base_url)
            message, metadata, cache_hit = llm.infer([{"role": "user", "content": "hello"}])
            llm.close()

        self.assertEqual(message, "offline response")
        self.assertEqual(metadata["total_tokens"], 5)
        self.assertFalse(cache_hit)
        self.assertEqual(requests[0].url.path, "/openai/v1/responses")
        body = json.loads(requests[0].content)
        self.assertEqual(body["model"], "openai.gpt-offline")
        self.assertEqual(body["max_output_tokens"], 17)
        self.assertFalse(body["store"])

    def test_responses_wrapper_rejects_empty_output(self):
        def handler(request):
            return SDK_HTTPX.Response(200, json=response_payload(content=None), request=request)

        with tempfile.TemporaryDirectory() as save_dir, patch.dict(os.environ, {"AWS_BEARER_TOKEN_BEDROCK": "offline-key"}):
            config = BaseConfig(
                save_dir=save_dir,
                llm_name="bedrock-mantle/openai.gpt-offline",
                llm_base_url="https://bedrock-offline.invalid/openai/v1",
                max_retry_attempts=0,
            )
            llm = BedrockMantleLLM(config)
            llm.openai_client.close()
            llm.openai_client = self.make_openai_client(handler, config.llm_base_url)
            with self.assertRaisesRegex(ValueError, "non-empty output text"):
                llm.infer([{"role": "user", "content": "hello"}])
            llm.close()

    def test_responses_wrapper_rejects_chat_completions_response_format(self):
        requests = []

        def handler(request):
            requests.append(request)
            return SDK_HTTPX.Response(200, json=response_payload(), request=request)

        with tempfile.TemporaryDirectory() as save_dir, patch.dict(os.environ, {"AWS_BEARER_TOKEN_BEDROCK": "offline-key"}):
            config = BaseConfig(
                save_dir=save_dir,
                llm_name="bedrock-mantle/openai.gpt-offline",
                llm_base_url="https://bedrock-offline.invalid/openai/v1",
                max_retry_attempts=0,
            )
            llm = BedrockMantleLLM(config)
            llm.openai_client.close()
            llm.openai_client = self.make_openai_client(handler, config.llm_base_url)
            with self.assertRaisesRegex(ValueError, "does not accept Chat Completions response_format"):
                llm.infer([{"role": "user", "content": "hello"}], response_format={"type": "json_object"})
            llm.close()

        self.assertEqual(requests, [])

    def test_embedding_wrapper_sends_float_and_restores_input_order(self):
        requests = []

        def handler(request):
            requests.append(request)
            return SDK_HTTPX.Response(200, json=embeddings_payload(), request=request)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-key"}):
            config = BaseConfig(
                embedding_model_name="text-embedding-offline",
                embedding_base_url="https://offline.invalid/v1",
                embedding_return_as_normalized=False,
                max_retry_attempts=0,
            )
            model = OpenAIEmbeddingModel(config)
            model.client.close()
            model.client = self.make_openai_client(handler)
            embeddings = model.encode(["first", "second"])
            model.close()

        np.testing.assert_array_equal(embeddings, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        self.assertEqual(model.last_usage, {"prompt_tokens": 2, "total_tokens": 2})
        self.assertEqual(requests[0].url.path, "/v1/embeddings")
        self.assertEqual(json.loads(requests[0].content)["encoding_format"], "float")

    def test_embedding_wrapper_rejects_unaccounted_response(self):
        def handler(request):
            return SDK_HTTPX.Response(200, json=embeddings_payload(include_usage=False), request=request)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-key"}):
            config = BaseConfig(
                embedding_model_name="text-embedding-offline",
                embedding_base_url="https://offline.invalid/v1",
                embedding_return_as_normalized=False,
                max_retry_attempts=0,
            )
            model = OpenAIEmbeddingModel(config)
            model.client.close()
            model.client = self.make_openai_client(handler)
            with self.assertRaisesRegex(ValueError, "omitted usage"):
                model.encode(["first", "second"])
            model.close()

    def test_embedding_wrapper_preserves_known_usage_when_payload_is_invalid(self):
        def handler(request):
            payload = embeddings_payload()
            payload["data"][0]["index"] = 0
            payload["data"][1]["index"] = 0
            return SDK_HTTPX.Response(200, json=payload, request=request)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-key"}):
            config = BaseConfig(
                embedding_model_name="text-embedding-offline",
                embedding_base_url="https://offline.invalid/v1",
                embedding_return_as_normalized=False,
                max_retry_attempts=0,
            )
            model = OpenAIEmbeddingModel(config)
            model.client.close()
            model.client = self.make_openai_client(handler)
            with self.assertRaisesRegex(ValueError, "indices"):
                model.encode(["first", "second"])
            model.close()

        self.assertEqual(model.last_usage, {"prompt_tokens": 2, "total_tokens": 2, "complete": False})

    def test_embedding_wrapper_marks_transport_failure_usage_unknown(self):
        def handler(request):
            raise SDK_HTTPX.ReadTimeout("offline timeout", request=request)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-key"}):
            config = BaseConfig(
                embedding_model_name="text-embedding-offline",
                embedding_base_url="https://offline.invalid/v1",
                embedding_return_as_normalized=False,
                max_retry_attempts=0,
            )
            model = OpenAIEmbeddingModel(config)
            model.client.close()
            model.client = self.make_openai_client(handler)
            with self.assertRaises(openai.APITimeoutError):
                model.encode(["first"])
            model.close()

        self.assertEqual(model.last_usage, {"usage_unknown": True, "complete": False})

    def test_embedding_wrapper_aggregates_usage_across_batches(self):
        requests = []

        def handler(request):
            requests.append(request)
            count = len(json.loads(request.content)["input"])
            return SDK_HTTPX.Response(200, json=embeddings_payload(count=count), request=request)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-key"}):
            config = BaseConfig(
                embedding_model_name="text-embedding-offline",
                embedding_base_url="https://offline.invalid/v1",
                embedding_return_as_normalized=False,
                max_retry_attempts=0,
            )
            model = OpenAIEmbeddingModel(config)
            model.client.close()
            model.client = self.make_openai_client(handler)
            embeddings = model.batch_encode(["first", "second"], batch_size=1)
            model.close()

        self.assertEqual(embeddings.shape, (2, 2))
        self.assertEqual(model.last_usage, {"prompt_tokens": 2, "total_tokens": 2})
        self.assertEqual(len(requests), 2)

    def test_embedding_wrapper_marks_partial_batch_usage_as_incomplete(self):
        request_count = 0

        def handler(request):
            nonlocal request_count
            request_count += 1
            return SDK_HTTPX.Response(200, json=embeddings_payload(include_usage=request_count == 1, count=1), request=request)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-key"}):
            config = BaseConfig(
                embedding_model_name="text-embedding-offline",
                embedding_base_url="https://offline.invalid/v1",
                embedding_return_as_normalized=False,
                max_retry_attempts=0,
            )
            model = OpenAIEmbeddingModel(config)
            model.client.close()
            model.client = self.make_openai_client(handler)
            with self.assertRaisesRegex(ValueError, "omitted usage"):
                model.batch_encode(["first", "second"], batch_size=1)
            model.close()

        self.assertEqual(model.last_usage, {"prompt_tokens": 1, "total_tokens": 1, "usage_unknown": True, "complete": False})

    def test_azure_resource_and_legacy_urls_are_resolved_explicitly(self):
        root = resolve_azure_openai_settings(
            "https://resource.openai.azure.com",
            api_version="2025-01-01-preview",
            deployment="chat-deployment",
            operation="chat.completions",
        )
        self.assertEqual(root.endpoint, "https://resource.openai.azure.com")
        self.assertEqual(root.deployment, "chat-deployment")

        legacy_url = "https://resource.openai.azure.com/openai/deployments/embed-deployment/embeddings?api-version=2025-01-01-preview"
        with self.assertWarns(FutureWarning):
            legacy = resolve_azure_openai_settings(
                legacy_url,
                api_version=None,
                deployment=None,
                operation="embeddings",
            )
        self.assertEqual(legacy.endpoint, "https://resource.openai.azure.com")
        self.assertEqual(legacy.api_version, "2025-01-01-preview")
        self.assertEqual(legacy.deployment, "embed-deployment")

    def test_azure_sdk_builds_expected_chat_request(self):
        requests = []

        def handler(request):
            requests.append(request)
            return SDK_HTTPX.Response(200, json=chat_payload(), request=request)

        http_client = SDK_HTTPX.Client(transport=SDK_HTTPX.MockTransport(handler))
        client = AzureOpenAI(
            api_key="offline-key",
            api_version="2025-01-01-preview",
            azure_endpoint="https://resource.openai.azure.com",
            azure_deployment="chat-deployment",
            http_client=http_client,
            max_retries=0,
        )
        response = client.chat.completions.create(
            model="gpt-offline",
            messages=[{"role": "user", "content": "hello"}],
            max_completion_tokens=17,
        )
        client.close()

        self.assertEqual(response.choices[0].message.content, "offline chat")
        self.assertEqual(requests[0].url.path, "/openai/deployments/chat-deployment/chat/completions")
        self.assertEqual(parse_qs(requests[0].url.query.decode())["api-version"], ["2025-01-01-preview"])

    def test_wrappers_accept_standard_azure_resource_endpoints(self):
        requests = []

        def handler(request):
            requests.append(request)
            payload = embeddings_payload() if request.url.path.endswith("/embeddings") else chat_payload()
            return SDK_HTTPX.Response(200, json=payload, request=request)

        with tempfile.TemporaryDirectory() as save_dir, patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "offline-key"}):
            config = BaseConfig(
                save_dir=save_dir,
                llm_name="gpt-offline",
                azure_endpoint="https://resource.openai.azure.com",
                azure_api_version="2025-01-01-preview",
                azure_chat_deployment="chat-deployment",
                embedding_model_name="text-embedding-offline",
                azure_embedding_endpoint="https://resource.openai.azure.com",
                azure_embedding_api_version="2025-01-01-preview",
                azure_embedding_deployment="embedding-deployment",
                max_retry_attempts=0,
            )
            llm = CacheOpenAI(save_dir, config, high_throughput=False, max_retries=0)
            embedding_model = OpenAIEmbeddingModel(config)
            llm_base_url = str(llm.openai_client.base_url)
            embedding_base_url = str(embedding_model.client.base_url)
            embedding_timeout = embedding_model.client.timeout.read
            llm.openai_client.close()
            embedding_model.client.close()
            llm.openai_client = AzureOpenAI(
                api_key="offline-key",
                api_version=config.azure_api_version,
                azure_endpoint=config.azure_endpoint,
                azure_deployment=config.azure_chat_deployment,
                http_client=SDK_HTTPX.Client(transport=SDK_HTTPX.MockTransport(handler)),
                max_retries=0,
            )
            embedding_model.client = AzureOpenAI(
                api_key="offline-key",
                api_version=config.azure_embedding_api_version,
                azure_endpoint=config.azure_embedding_endpoint,
                azure_deployment=config.azure_embedding_deployment,
                http_client=SDK_HTTPX.Client(transport=SDK_HTTPX.MockTransport(handler)),
                max_retries=0,
            )
            llm.infer([{"role": "user", "content": "hello"}])
            embedding_model.encode(["first", "second"])
            with self.assertRaisesRegex(ValueError, "must match the configured deployment"):
                llm.infer([{"role": "user", "content": "hello"}], model="other-deployment")
            with self.assertRaisesRegex(ValueError, "extra_body"):
                llm.infer([{"role": "user", "content": "hello"}], extra_body={"model": "other-deployment"})
            llm.batch_upsert_llm_config({"generate_params": {"model": "other-deployment"}})
            with self.assertRaisesRegex(ValueError, "must match the configured deployment"):
                llm.infer([{"role": "user", "content": "hello"}])
            llm.close()
            embedding_model.close()

        self.assertTrue(llm_base_url.endswith("/openai/deployments/chat-deployment/"))
        self.assertTrue(embedding_base_url.endswith("/openai/deployments/embedding-deployment/"))
        self.assertEqual(embedding_timeout, config.embedding_request_timeout)
        bodies = [json.loads(request.content) for request in requests]
        self.assertEqual(bodies[0]["model"], "chat-deployment")
        self.assertEqual(bodies[1]["model"], "embedding-deployment")

    def test_loopback_placeholder_does_not_mutate_process_environment(self):
        with tempfile.TemporaryDirectory() as save_dir, patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            config = BaseConfig(
                save_dir=save_dir,
                llm_name="local-model",
                llm_base_url="http://127.0.0.1:8000/v1",
                embedding_model_name="local-embedding",
                embedding_base_url="http://localhost:8001/v1",
                max_retry_attempts=0,
            )
            llm = CacheOpenAI(save_dir, config, high_throughput=False, max_retries=0)
            embedding_model = OpenAIEmbeddingModel(config)
            self.assertNotIn("OPENAI_API_KEY", os.environ)
            self.assertEqual(llm.openai_client.api_key, "local-placeholder")
            self.assertEqual(embedding_model.client.api_key, "local-placeholder")
            llm.close()
            embedding_model.close()

    def test_sigv4_event_hook_works_with_sdk_native_http_client(self):
        requests = []

        def handler(request):
            requests.append(request)
            return SDK_HTTPX.Response(200, json=response_payload(), request=request)

        credentials = SimpleNamespace(get_frozen_credentials=lambda: SimpleNamespace(access_key="key", secret_key="secret", token=None))
        auth = BedrockMantleSigV4Auth.__new__(BedrockMantleSigV4Auth)
        auth.session = SimpleNamespace(get_credentials=lambda: credentials)
        auth.region_name = "us-east-2"
        http_client = DefaultHttpxClient(
            transport=SDK_HTTPX.MockTransport(handler),
            event_hooks={"request": [auth.sign_request]},
        )
        client = OpenAI(
            api_key="bedrock-sigv4",
            base_url="https://bedrock-mantle.us-east-2.api.aws/openai/v1",
            http_client=http_client,
            max_retries=0,
        )
        response = client.responses.create(model="openai.gpt-offline", input="hello", max_output_tokens=17, store=False)
        client.close()

        self.assertEqual(response.output_text, "offline response")
        self.assertTrue(requests[0].headers["Authorization"].startswith("AWS4-HMAC-SHA256"))

    def test_full_operation_base_url_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "API base URL"):
            validate_openai_base_url("http://localhost:8001/v1/embeddings", "embeddings", "embedding_base_url")


if __name__ == "__main__":
    unittest.main()
