import json
import importlib
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import igraph as ig
import numpy as np
import torch

from hipporag.HippoRAG import HippoRAG
from hipporag.StandardRAG import StandardRAG
from hipporag.embedding_store import EmbeddingStore
from hipporag.embedding_model import _get_embedding_model_class
from hipporag.information_extraction.openie_openai import OpenIE, _extract_json_list_field
from hipporag.llm.base import LLMConfig, normalize_generation_token_params
from hipporag.llm.openai_gpt import CacheOpenAI
from hipporag.llm.transformers_llm import TransformersLLM
from hipporag.preprocessing import TextPreprocessor
from hipporag.prompts.prompt_template_manager import PromptTemplateManager
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.logging_utils import redact_config
from hipporag.utils.misc_utils import NerRawOutput, TripleRawOutput, compute_mdhash_id, text_processing
from hipporag.utils.state_utils import StateConsistencyError, embedding_index_identity, validate_or_create_index_manifest
from hipporag.vector_stores.naming import build_collection_name


class ConfigurableEmbeddingModel:
    query_instruction_mode = "ignored"

    def __init__(self, dimension=2, **kwargs):
        self.dimension = dimension

    def batch_encode(self, texts, **kwargs):
        return np.ones((len(texts), self.dimension), dtype=np.float32)


class CloseTracker:
    def __init__(self):
        self.close_count = 0

    def close(self):
        self.close_count += 1

    def infer(self, messages, **kwargs):
        return "mocked", {}, False


class TrackingQueryEmbeddingModel:
    query_instruction_mode = "ignored"

    def __init__(self):
        self.calls = []

    def batch_encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return np.ones((len(texts), 2), dtype=np.float32)


class RegressionTest(unittest.TestCase):
    def test_standard_rag_qa_and_incremental_index_refresh_retrieval_state(self):
        standard_module = importlib.import_module("hipporag.StandardRAG")
        with tempfile.TemporaryDirectory() as temp_dir:
            llm = CloseTracker()
            config = BaseConfig(
                save_dir=temp_dir,
                llm_name="fake",
                embedding_model_name="fake",
                retrieval_top_k=2,
            )
            with patch.object(standard_module, "_get_llm_class", return_value=llm), patch.object(
                standard_module, "_get_embedding_model_class", return_value=ConfigurableEmbeddingModel
            ):
                rag = StandardRAG(global_config=config)

            self.assertIsInstance(rag.prompt_template_manager, PromptTemplateManager)
            rag.index(["old document"])
            first_result = rag.retrieve(["question"], num_to_retrieve=2)[0]
            self.assertEqual(first_result.docs, ["old document"])
            self.assertTrue(rag.ready_to_retrieve)

            rag.index(["new document"])
            self.assertFalse(rag.ready_to_retrieve)
            self.assertEqual(rag.passage_node_keys, [])
            self.assertEqual(rag.query_to_embedding, {"triple": {}, "passage": {}})
            rag.dense_passage_retrieval("question")
            self.assertEqual(set(rag.passage_node_keys), set(rag.chunk_embedding_store.get_all_ids()))

            second_result = rag.retrieve(["question"], num_to_retrieve=2)[0]
            self.assertEqual(set(second_result.docs), {"old document", "new document"})
            with self.assertLogs(standard_module.logger, level="WARNING") as warning_logs:
                solutions, _, _ = rag.rag_qa(["question"])
            self.assertEqual(set(solutions[0].docs), {"old document", "new document"})
            self.assertEqual(solutions[0].answer, "mocked")
            self.assertTrue(any("Using MUSIQUE's prompt template" in message for message in warning_logs.output))
            rag.close()

    def test_vector_collection_names_isolate_indexes_and_manifest_endpoints(self):
        first_config = BaseConfig(
            save_dir="/tmp/hipporag-index-a",
            llm_name="llm",
            embedding_model_name="embedding",
            vector_store_type="chroma",
            chroma_host="chroma-a.internal",
        )
        second_config = BaseConfig(
            save_dir="/tmp/hipporag-index-b",
            llm_name="llm",
            embedding_model_name="embedding",
            vector_store_type="chroma",
            chroma_host="chroma-b.internal",
        )
        first_name = build_collection_name("/tmp/hipporag-index-a/chunk_embeddings", "chunk", first_config)
        second_name = build_collection_name("/tmp/hipporag-index-b/chunk_embeddings", "chunk", second_config)
        self.assertNotEqual(first_name, second_name)
        self.assertNotEqual(embedding_index_identity(first_config), embedding_index_identity(second_config))

        first_config.vector_store_namespace = "portable-index"
        second_config.vector_store_namespace = "portable-index"
        self.assertEqual(
            build_collection_name("/tmp/hipporag-index-a/chunk_embeddings", "chunk", first_config),
            build_collection_name("/tmp/hipporag-index-b/chunk_embeddings", "chunk", second_config),
        )

    def test_explicit_embedding_provider_supports_custom_model_names(self):
        model_class = _get_embedding_model_class("custom-model-name", provider="openai")
        self.assertEqual(model_class.__name__, "OpenAIEmbeddingModel")

        vllm_class = _get_embedding_model_class("custom-model-name", provider="vllm")
        vllm_model = vllm_class(
            BaseConfig(embedding_model_name="custom-model-name", embedding_base_url="http://localhost:8001/v1/embeddings"),
            "custom-model-name",
        )
        self.assertEqual(vllm_model.model_id, "custom-model-name")

    def test_instruction_agnostic_queries_use_one_embedding_call(self):
        rag = HippoRAG.__new__(HippoRAG)
        rag.global_config = BaseConfig(linking_top_k=5)
        rag.embedding_model = TrackingQueryEmbeddingModel()
        rag.fact_node_keys = ["fact"]
        rag.passage_node_keys = ["passage"]
        rag.query_to_embedding = {"triple": {}, "passage": {}}

        rag.get_query_embeddings(["query"])

        self.assertEqual(len(rag.embedding_model.calls), 1)
        self.assertIn("query", rag.query_to_embedding["triple"])
        self.assertIn("query", rag.query_to_embedding["passage"])

        rag.global_config.linking_top_k = 0
        rag.embedding_model.calls.clear()
        rag.query_to_embedding = {"triple": {}, "passage": {}}
        rag.get_query_embeddings(["dense-only"])
        self.assertEqual(len(rag.embedding_model.calls), 1)
        self.assertEqual(rag.query_to_embedding["triple"], {})
        rag.fact_embeddings = np.ones((1, 2), dtype=np.float32)
        self.assertEqual(rag.get_fact_scores("dense-only").size, 0)
        self.assertEqual(len(rag.embedding_model.calls), 1)

    def test_base_config_preserves_original_positional_field_order(self):
        original_prefix = [
            "llm_name",
            "llm_base_url",
            "embedding_base_url",
            "azure_endpoint",
            "azure_embedding_endpoint",
        ]
        self.assertEqual([item.name for item in fields(BaseConfig)[:5]], original_prefix)
        config = BaseConfig("llm", "https://llm.invalid/v1", "https://embed.invalid/v1", "https://azure.invalid", "https://azure-embed.invalid")
        self.assertEqual(config.embedding_base_url, "https://embed.invalid/v1")
        self.assertEqual(config.azure_embedding_endpoint, "https://azure-embed.invalid")

    def test_constructor_overrides_are_revalidated_before_provider_setup(self):
        with self.assertRaisesRegex(ValueError, "azure_endpoint is required"):
            HippoRAG(azure_api_version="2025-01-01-preview")
        with self.assertRaisesRegex(ValueError, "azure_endpoint is required"):
            StandardRAG(azure_chat_deployment="chat-deployment")

    def test_constructor_failure_closes_owned_llm(self):
        hippo_module = importlib.import_module("hipporag.HippoRAG")
        standard_module = importlib.import_module("hipporag.StandardRAG")
        with tempfile.TemporaryDirectory() as temp_dir:
            hippo_llm = CloseTracker()
            config = BaseConfig(save_dir=temp_dir, llm_name="fake", embedding_model_name="fake")
            with patch.object(hippo_module, "_get_llm_class", return_value=hippo_llm), patch.object(
                hippo_module, "_get_embedding_model_class", side_effect=RuntimeError("embedding failed")
            ):
                with self.assertRaisesRegex(RuntimeError, "embedding failed"):
                    HippoRAG(global_config=config)
            self.assertEqual(hippo_llm.close_count, 1)

        with tempfile.TemporaryDirectory() as temp_dir:
            standard_llm = CloseTracker()
            config = BaseConfig(save_dir=temp_dir, llm_name="fake", embedding_model_name="fake")
            with patch.object(standard_module, "_get_llm_class", return_value=standard_llm), patch.object(
                standard_module, "_get_embedding_model_class", side_effect=RuntimeError("embedding failed")
            ):
                with self.assertRaisesRegex(RuntimeError, "embedding failed"):
                    StandardRAG(global_config=config)
            self.assertEqual(standard_llm.close_count, 1)

    def test_hipporag_close_preserves_injected_models(self):
        rag = HippoRAG.__new__(HippoRAG)
        stores = [CloseTracker(), CloseTracker(), CloseTracker()]
        external_llm = CloseTracker()
        external_embedding = CloseTracker()
        rag.chunk_embedding_store, rag.entity_embedding_store, rag.fact_embedding_store = stores
        rag.llm_model = rag.extraction_llm = rag.qa_llm = external_llm
        rag.embedding_model = external_embedding
        rag._owns_llm_model = False
        rag._owns_embedding_model = False

        rag.close()

        self.assertEqual([store.close_count for store in stores], [1, 1, 1])
        self.assertEqual(external_llm.close_count, 0)
        self.assertEqual(external_embedding.close_count, 0)

    def test_hipporag_close_closes_owned_openie_once(self):
        rag = HippoRAG.__new__(HippoRAG)
        owned_openie = CloseTracker()
        rag.openie = owned_openie
        rag._owns_openie = True
        rag._owns_llm_model = False
        rag._owns_embedding_model = False

        rag.close()
        rag.close()

        self.assertEqual(owned_openie.close_count, 1)

    def test_azure_embedding_deployment_is_part_of_index_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = str(Path(temp_dir) / "index_manifest.json")
            store = SimpleNamespace(get_all_ids=lambda: [])
            first = BaseConfig(
                embedding_model_name="text-embedding-offline",
                azure_embedding_endpoint="https://resource.openai.azure.com",
                azure_embedding_api_version="2025-01-01-preview",
                azure_embedding_deployment="deployment-a",
            )
            second = BaseConfig(
                embedding_model_name="text-embedding-offline",
                azure_embedding_endpoint="https://resource.openai.azure.com",
                azure_embedding_api_version="2025-01-01-preview",
                azure_embedding_deployment="deployment-b",
            )
            validate_or_create_index_manifest(manifest_path, {"embedding": embedding_index_identity(first)}, (store,))
            with self.assertRaisesRegex(StateConsistencyError, "schema/config mismatch"):
                validate_or_create_index_manifest(manifest_path, {"embedding": embedding_index_identity(second)}, (store,))

    def test_explicit_injected_component_identity_prevents_state_reuse(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BaseConfig(save_dir=temp_dir, embedding_model_name="fake", llm_name="fake")
            first = HippoRAG(
                global_config=config,
                extraction_llm=CloseTracker(),
                qa_llm=CloseTracker(),
                embedding_model=ConfigurableEmbeddingModel(),
                index_identity="components-v1",
            )
            first.close()
            with self.assertRaisesRegex(StateConsistencyError, "schema/config mismatch"):
                HippoRAG(
                    global_config=config,
                    extraction_llm=CloseTracker(),
                    qa_llm=CloseTracker(),
                    embedding_model=ConfigurableEmbeddingModel(),
                    index_identity="components-v2",
                )

    def test_openie_offline_to_online_phase_keeps_compatible_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = str(Path(temp_dir) / "openie_state.json")
            export_path = str(Path(temp_dir) / "openie_export.json")
            offline = HippoRAG.__new__(HippoRAG)
            offline.global_config = BaseConfig(llm_name="same-model", embedding_model_name="fake", openie_mode="offline")
            offline.openie = CloseTracker()
            offline.text_preprocessor = CloseTracker()
            offline.index_identity = "phase-compatible-v1"
            offline._openie_provenance = None
            offline.openie_state_path = state_path
            offline.openie_results_path = export_path
            offline.save_openie_results(
                [{"idx": "chunk-id", "passage": "text", "extracted_entities": [], "extracted_triples": []}],
                state_path,
            )

            online = HippoRAG.__new__(HippoRAG)
            online.global_config = BaseConfig(
                llm_name="same-model",
                llm_base_url="http://localhost:8000/v1",
                embedding_model_name="fake",
                openie_mode="online",
            )
            online.openie = CloseTracker()
            online.extraction_llm = CloseTracker()
            online.text_preprocessor = CloseTracker()
            online.index_identity = "phase-compatible-v1"
            online.openie_state_path = state_path
            online.openie_results_path = export_path

            provenance = online._openie_provenance_for_manifest()
            self.assertEqual(provenance["producer"]["mode"], "offline")
            self.assertIsNone(provenance["producer"]["endpoint"])

    def test_forced_online_reextraction_rewrites_only_empty_offline_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            working_dir = Path(temp_dir) / "same-model_fake"
            working_dir.mkdir()
            offline = HippoRAG.__new__(HippoRAG)
            offline.global_config = BaseConfig(save_dir=temp_dir, llm_name="same-model", embedding_model_name="fake", openie_mode="offline")
            offline.openie = CloseTracker()
            offline.text_preprocessor = TextPreprocessor()
            offline.index_identity = "phase-compatible-v1"
            offline._openie_provenance = None
            offline.openie_state_path = str(working_dir / "openie_state.json")
            offline.openie_results_path = str(Path(temp_dir) / "openie_results_ner_same-model.json")
            offline.save_openie_results(
                [{"idx": compute_mdhash_id("doc", "chunk-"), "passage": "doc", "extracted_entities": [], "extracted_triples": []}],
                offline.openie_state_path,
            )

            base_kwargs = {
                "save_dir": temp_dir,
                "llm_name": "same-model",
                "embedding_model_name": "fake",
                "openie_mode": "online",
            }
            first = HippoRAG(
                global_config=BaseConfig(**base_kwargs),
                extraction_llm=CloseTracker(),
                qa_llm=CloseTracker(),
                embedding_model=ConfigurableEmbeddingModel(),
                index_identity="phase-compatible-v1",
            )
            manifest_path = Path(first.index_manifest_path)
            self.assertEqual(json.loads(manifest_path.read_text())["openie"]["producer"]["mode"], "offline")
            first.close()

            forced = HippoRAG(
                global_config=BaseConfig(**base_kwargs, force_openie_from_scratch=True),
                extraction_llm=CloseTracker(),
                qa_llm=CloseTracker(),
                embedding_model=ConfigurableEmbeddingModel(),
                index_identity="phase-compatible-v1",
            )
            forced.openie = SimpleNamespace(batch_openie=lambda rows: (_ for _ in ()).throw(RuntimeError("offline extraction failure")))
            with self.assertRaisesRegex(RuntimeError, "offline extraction failure"):
                forced.index(["doc"])
            self.assertTrue(forced.chunk_embedding_store.get_all_ids())
            self.assertFalse(forced.entity_embedding_store.get_all_ids())
            self.assertFalse(forced.fact_embedding_store.get_all_ids())
            forced.close()

            forced_retry = HippoRAG(
                global_config=BaseConfig(**base_kwargs, force_openie_from_scratch=True),
                extraction_llm=CloseTracker(),
                qa_llm=CloseTracker(),
                embedding_model=ConfigurableEmbeddingModel(),
                index_identity="phase-compatible-v1",
            )
            forced_retry.openie = SimpleNamespace(
                batch_openie=lambda rows: (
                    {key: NerRawOutput(key, "", [], {}) for key in rows},
                    {key: TripleRawOutput(key, "", [], {}) for key in rows},
                )
            )
            forced_retry.add_synonymy_edges = lambda: None
            forced_retry.index(["doc"])
            forced_retry.close()
            self.assertEqual(json.loads(manifest_path.read_text())["openie"]["producer"]["mode"], "online")

            reloaded = HippoRAG(
                global_config=BaseConfig(**base_kwargs),
                extraction_llm=CloseTracker(),
                qa_llm=CloseTracker(),
                embedding_model=ConfigurableEmbeddingModel(),
                index_identity="phase-compatible-v1",
            )
            reloaded.load_existing_openie([compute_mdhash_id("doc", "chunk-")])
            reloaded.close()

    def test_synonym_graph_configuration_is_part_of_index_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = str(Path(temp_dir) / "index_manifest.json")
            store = SimpleNamespace(get_all_ids=lambda: [])

            def make_rag(threshold):
                rag = HippoRAG.__new__(HippoRAG)
                rag.global_config = BaseConfig(embedding_model_name="fake", llm_name="fake", synonymy_edge_sim_threshold=threshold)
                rag.embedding_model = ConfigurableEmbeddingModel()
                rag.extraction_llm = CloseTracker()
                rag.text_preprocessor = CloseTracker()
                rag.index_identity = "graph-config-v1"
                rag.index_manifest_path = manifest_path
                rag.chunk_embedding_store = rag.entity_embedding_store = rag.fact_embedding_store = store
                return rag

            make_rag(0.8)._validate_or_create_index_manifest()
            with self.assertRaisesRegex(StateConsistencyError, "schema/config mismatch"):
                make_rag(0.7)._validate_or_create_index_manifest()

    def test_response_format_is_scoped_to_openie_requests(self):
        config = BaseConfig(response_format={"type": "json_object"})
        llm = CacheOpenAI.__new__(CacheOpenAI)
        llm.global_config = config
        llm.request_model_name = config.llm_name
        llm._init_llm_config()
        self.assertNotIn("response_format", llm.llm_config.generate_params)

        calls = []
        recording_llm = SimpleNamespace(
            global_config=config,
            infer=lambda messages, **kwargs: (calls.append(kwargs) or ('{"named_entities": []}', {}, False)),
        )
        result = OpenIE(recording_llm).ner("chunk", "passage")
        self.assertEqual(result.unique_entities, [])
        self.assertEqual(calls[0]["response_format"], {"type": "json_object"})

    def test_public_index_rejects_a_bare_string(self):
        with self.assertRaises(TypeError):
            HippoRAG.__new__(HippoRAG).index("one document")
        with self.assertRaises(TypeError):
            StandardRAG.__new__(StandardRAG).index("one document")

    def test_unicode_text_normalization(self):
        self.assertEqual(text_processing("  北京--大学！ "), "北京 大学")
        self.assertEqual(text_processing("Straße"), "strasse")

    def test_configuration_secrets_are_redacted(self):
        redacted = redact_config({"qdrant_api_key": "secret", "milvus_token": "token", "max_new_tokens": 10})
        self.assertEqual(redacted["qdrant_api_key"], "***REDACTED***")
        self.assertEqual(redacted["milvus_token"], "***REDACTED***")
        self.assertEqual(redacted["max_new_tokens"], 10)

    def test_safe_json_extraction_skips_unrelated_objects(self):
        response = 'prefix {"example": [1]} suffix {"named_entities": ["北京", "OSU"]}'
        self.assertEqual(_extract_json_list_field(response, "named_entities"), ["北京", "OSU"])
        with self.assertRaises(ValueError):
            _extract_json_list_field("__import__('os').system('echo unsafe')", "named_entities")

    def test_merge_openie_rejects_partial_batches_without_mutation(self):
        rag = HippoRAG.__new__(HippoRAG)
        existing = [{"idx": "old", "passage": "old", "extracted_entities": [], "extracted_triples": []}]
        chunks = {"new": {"content": "new passage"}}
        with self.assertRaises(StateConsistencyError):
            rag.merge_openie_results(existing, chunks, {"new": NerRawOutput("new", "", [], {})}, {})
        self.assertEqual([row["idx"] for row in existing], ["old"])

    def test_max_token_aliases_do_not_silently_override(self):
        params = normalize_generation_token_params({"max_completion_tokens": 99}, {"max_tokens": 10}, "max_completion_tokens")
        self.assertEqual(params, {"max_completion_tokens": 10})
        with self.assertRaises(ValueError):
            normalize_generation_token_params({}, {"max_tokens": 1, "max_new_tokens": 2}, "max_tokens")

    def test_openai_equivalent_token_aliases_share_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BaseConfig(save_dir=temp_dir, llm_name="gpt-test", embedding_model_name="text-embedding-test", llm_supports_max_completion_tokens=True)
            llm = CacheOpenAI.__new__(CacheOpenAI)
            llm.global_config = config
            llm.llm_config = LLMConfig.from_dict({"generate_params": {"model": "gpt-test", "max_completion_tokens": 99, "temperature": 0}})
            llm.cache_file_name = str(Path(temp_dir) / "cache.sqlite")
            response = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )
            create = Mock(return_value=response)
            llm.openai_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            messages = [{"role": "user", "content": "hello"}]

            _, _, first_cache_hit = llm.infer(messages, max_tokens=10)
            _, _, second_cache_hit = llm.infer(messages, max_new_tokens=10)

            self.assertFalse(first_cache_hit)
            self.assertTrue(second_cache_hit)
            self.assertEqual(create.call_count, 1)
            sent_params = create.call_args.kwargs
            self.assertEqual(sent_params["max_completion_tokens"], 10)
            self.assertNotIn("max_tokens", sent_params)

    def test_transformers_llm_decodes_only_generated_tokens(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = "prompt"
        tokenizer.encode.return_value = torch.tensor([[11, 12]])
        tokenizer.decode.return_value = "answer"
        model = Mock()
        model.device = torch.device("cpu")
        model.generate.return_value = torch.tensor([[11, 12, 99]])
        cache = Mock()
        cache.read.return_value = None
        llm = TransformersLLM.__new__(TransformersLLM)
        llm.llm_config = LLMConfig.from_dict({"generate_params": {"max_new_tokens": 8, "temperature": 0}})
        llm.model_id = "mock-model"
        llm.model = model
        llm.tokenizer = tokenizer
        llm.cache = cache

        message, metadata, cache_hit = llm.infer([{"role": "user", "content": "question"}])

        self.assertEqual(message, "answer")
        self.assertEqual(metadata, {"prompt_tokens": 2, "completion_tokens": 1})
        self.assertFalse(cache_hit)
        decoded_tokens = tokenizer.decode.call_args.args[0]
        self.assertEqual(decoded_tokens.tolist(), [99])

    def test_parquet_store_merges_stale_writers_and_rejects_dimension_drift(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = ConfigurableEmbeddingModel()
            first = EmbeddingStore(model, temp_dir, 8, "chunk")
            second = EmbeddingStore(model, temp_dir, 8, "chunk")
            first.insert_strings(["one"])
            second.insert_strings(["two"])
            reloaded = EmbeddingStore(model, temp_dir, 8, "chunk")
            self.assertEqual(reloaded.get_all_texts(), {"one", "two"})

            model.dimension = 3
            with self.assertRaises(ValueError):
                reloaded.insert_strings(["three"])
            self.assertEqual(EmbeddingStore(model, temp_dir, 8, "chunk").get_all_texts(), {"one", "two"})

    def test_source_aware_fact_deletion_preserves_synonym_component(self):
        rag = HippoRAG.__new__(HippoRAG)
        rag.graph = ig.Graph(directed=True)
        rag.graph.add_vertices(2, attributes={"name": ["entity-a", "entity-b"]})
        rag.graph.add_edge(
            0,
            1,
            weight=1.0,
            edge_kind="fact+synonym",
            fact_source_counts={"chunk-a": 1},
            synonym_score=0.8,
            passage_source=None,
            source_key="entity-a",
            target_key="entity-b",
        )
        rag._remove_fact_sources_from_graph({"chunk-a"})
        self.assertEqual(rag.graph.ecount(), 1)
        self.assertEqual(rag.graph.es[0]["edge_kind"], "synonym")
        self.assertAlmostEqual(rag.graph.es[0]["weight"], 0.8)
        self.assertEqual(rag.graph.es[0]["fact_source_counts"], {})

    def test_incremental_edges_merge_typed_contributions_without_parallel_copies(self):
        rag = HippoRAG.__new__(HippoRAG)
        rag.graph = ig.Graph(directed=True)
        rag.graph.add_vertices(2, attributes={"name": ["entity-a", "entity-b"]})
        rag.graph.add_edge(
            0,
            1,
            weight=0.9,
            edge_kind="synonym",
            fact_source_counts={},
            synonym_score=0.9,
            passage_source=None,
            source_key="entity-a",
            target_key="entity-b",
        )
        rag.node_to_node_stats = {("entity-a", "entity-b"): 1.0}
        rag._fact_edge_source_counts = {("entity-a", "entity-b"): {"chunk-a": 1}}
        rag._synonym_edge_scores = {}
        rag._passage_edge_sources = {}

        rag.add_new_edges()

        self.assertEqual(rag.graph.ecount(), 1)
        self.assertEqual(rag.graph.es[0]["edge_kind"], "fact+synonym")
        self.assertEqual(rag.graph.es[0]["fact_source_counts"], {"chunk-a": 1})
        self.assertAlmostEqual(rag.graph.es[0]["weight"], 1.0)

    def test_index_rejects_legacy_graph_before_model_or_store_work(self):
        rag = HippoRAG.__new__(HippoRAG)
        rag.graph = ig.Graph(directed=False)
        rag._graph_edge_schema_available = False

        with self.assertRaisesRegex(StateConsistencyError, "predates source-aware edges"):
            rag.index(["doc"])

    def test_undirected_edges_collapse_reverse_parallel_contributions(self):
        rag = HippoRAG.__new__(HippoRAG)
        rag.graph = ig.Graph(directed=False)
        rag.graph.add_vertices(2, attributes={"name": ["entity-a", "entity-b"]})
        attributes = {
            "weight": [1.0, 0.9],
            "edge_kind": ["fact", "synonym"],
            "fact_source_counts": [{"chunk-a": 1}, {}],
            "synonym_score": [0.0, 0.9],
            "passage_source": [None, None],
            "source_key": ["entity-a", "entity-b"],
            "target_key": ["entity-b", "entity-a"],
        }
        rag.graph.add_edges([("entity-a", "entity-b"), ("entity-b", "entity-a")], attributes=attributes)
        rag.node_to_node_stats = {("entity-b", "entity-a"): 1.0}
        rag._fact_edge_source_counts = {("entity-b", "entity-a"): {"chunk-b": 1}}
        rag._synonym_edge_scores = {}
        rag._passage_edge_sources = {}

        rag.add_new_edges()

        self.assertEqual(rag.graph.ecount(), 1)
        self.assertEqual(rag.graph.es[0]["source_key"], "entity-a")
        self.assertEqual(rag.graph.es[0]["target_key"], "entity-b")
        self.assertEqual(rag.graph.es[0]["fact_source_counts"], {"chunk-a": 1, "chunk-b": 1})
        self.assertEqual(rag.graph.es[0]["edge_kind"], "fact+synonym")
        self.assertAlmostEqual(rag.graph.es[0]["weight"], 2.0)

    def test_graph_edge_metadata_must_match_physical_endpoints(self):
        rag = HippoRAG.__new__(HippoRAG)
        rag.graph = ig.Graph(directed=True)
        rag.graph.add_vertices(3, attributes={"name": ["entity-a", "entity-b", "entity-c"]})
        rag.graph.add_edge(
            "entity-a",
            "entity-b",
            weight=1.0,
            edge_kind="fact",
            fact_source_counts={"chunk-a": 1},
            synonym_score=0.0,
            passage_source=None,
            source_key="entity-a",
            target_key="entity-c",
        )
        rag.node_to_node_stats = {}

        with self.assertRaisesRegex(StateConsistencyError, "physical endpoints"):
            rag.add_new_edges()

    def test_ppr_reaches_passages_in_a_directed_storage_graph(self):
        rag = HippoRAG.__new__(HippoRAG)
        rag.global_config = BaseConfig(is_directed_graph=True)
        rag.graph = ig.Graph(directed=True)
        rag.graph.add_vertices(2)
        rag.graph.add_edge(0, 1, weight=1.0)
        rag.node_name_to_vertex_idx = {"passage": 0, "entity": 1}
        rag.passage_node_idxs = [0]

        _, passage_scores = rag.run_ppr(np.asarray([0.0, 1.0]), damping=0.5)

        self.assertGreater(passage_scores[0], 0.0)

    def test_zero_linking_top_k_skips_reranking(self):
        rag = HippoRAG.__new__(HippoRAG)
        rag.global_config = BaseConfig(linking_top_k=0)
        self.assertEqual(rag.rerank_facts("query", np.asarray([0.9]))[:2], ([], []))

    def test_save_openie_false_still_keeps_required_canonical_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rag = HippoRAG.__new__(HippoRAG)
            rag.global_config = BaseConfig(save_dir=temp_dir, save_openie=False)
            rag.openie_state_path = str(Path(temp_dir) / "state.json")
            rag.openie_results_path = str(Path(temp_dir) / "export.json")
            rag._save_openie_state([])
            self.assertTrue(Path(rag.openie_state_path).exists())
            self.assertFalse(Path(rag.openie_results_path).exists())
            state = json.loads(Path(rag.openie_state_path).read_text())
            self.assertEqual(state["docs"], [])
            self.assertEqual(state["avg_ent_chars"], 0)
            self.assertEqual(state["avg_ent_words"], 0)
            self.assertIn("provenance", state)


if __name__ == "__main__":
    unittest.main()
