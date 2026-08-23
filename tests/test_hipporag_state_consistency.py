import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hipporag.HippoRAG import HippoRAG
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.misc_utils import NerRawOutput, TripleRawOutput


class FakeLLM:
    def infer(self, messages, **kwargs):
        if messages and "fact_before_filter" in messages[-1].get("content", ""):
            return '[[ ## fact_after_filter ## ]]\n{"fact": [["alice", "knows", "bob"]]}\n\n[[ ## completed ## ]]', {}, False
        return "Answer: mocked answer", {}, False


class FakeEmbeddingModel:
    def batch_encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            lowered = text.lower()
            vector = np.asarray([lowered.count("alice") + 1, lowered.count("bob") + 1, len(lowered) % 7 + 1], dtype=np.float32)
            embeddings.append(vector / np.linalg.norm(vector))
        return np.asarray(embeddings)


class FakeOpenIE:
    def batch_openie(self, rows):
        ner_results = {}
        triple_results = {}
        for chunk_id in rows:
            ner_results[chunk_id] = NerRawOutput(chunk_id, "", ["Alice", "Bob"], {})
            triple_results[chunk_id] = TripleRawOutput(chunk_id, "", [["Alice", "knows", "Bob"]], {})
        return ner_results, triple_results


class NormalizationCollisionOpenIE:
    def batch_openie(self, rows):
        ner_results = {}
        triple_results = {}
        for chunk_id in rows:
            ner_results[chunk_id] = NerRawOutput(chunk_id, "", ["Alice", "Bob"], {})
            triple_results[chunk_id] = TripleRawOutput(
                chunk_id, "", [["Alice", "knows", "Bob"], ["Alice!", "knows", "Bob"]], {}
            )
        return ner_results, triple_results


def make_rag(save_dir):
    config = BaseConfig(save_dir=str(save_dir), embedding_model_name="fake", llm_name="fake", save_openie=True)
    rag = HippoRAG(
        global_config=config,
        extraction_llm=FakeLLM(),
        qa_llm=FakeLLM(),
        embedding_model=FakeEmbeddingModel(),
        index_identity="fake-components-v1",
    )
    rag.openie = FakeOpenIE()
    rag.add_synonymy_edges = lambda: None
    return rag


class HippoRAGStateConsistencyTest(unittest.TestCase):
    def test_index_retrieve_and_rag_qa_pipeline(self):
        with tempfile.TemporaryDirectory() as save_dir:
            docs = ["Alice knows Bob.", "Bob works with Carol."]
            with make_rag(Path(save_dir)) as rag:
                rag.index(docs)

                retrieval_results = rag.retrieve(["Who does Alice know?"], num_to_retrieve=2)
                self.assertEqual(set(retrieval_results[0].docs), set(docs))
                self.assertTrue(retrieval_results[0].graph_seeds)

                solutions, response_messages, metadata = rag.rag_qa(["Who does Alice know?"])
                self.assertEqual(set(solutions[0].docs), set(docs))
                self.assertEqual(solutions[0].answer, "mocked answer")
                self.assertEqual(response_messages, ["Answer: mocked answer"])
                self.assertEqual(metadata, [{}])

    def test_incremental_index_counts_only_new_fact_edges_and_rebuilds_entity_sources(self):
        with tempfile.TemporaryDirectory() as save_dir:
            rag = make_rag(Path(save_dir))
            rag.index(["doc A"])

            reloaded = make_rag(Path(save_dir))
            reloaded.index(["doc B"])

            alice_id = reloaded.entity_embedding_store.text_to_hash_id["alice"]
            bob_id = reloaded.entity_embedding_store.text_to_hash_id["bob"]
            chunk_ids = set(reloaded.chunk_embedding_store.get_all_ids())
            entity_pair_stats = [weight for edge, weight in reloaded.node_to_node_stats.items() if set(edge) == {alice_id, bob_id}]
            self.assertEqual(entity_pair_stats, [1.0])
            self.assertEqual(reloaded.ent_node_to_chunk_ids[alice_id], chunk_ids)
            self.assertEqual(reloaded.ent_node_to_chunk_ids[bob_id], chunk_ids)
            vertex_names = reloaded.graph.vs["name"]
            entity_ids = {alice_id, bob_id}
            fact_edge_weights = [
                edge["weight"] for edge in reloaded.graph.es
                if vertex_names[edge.source] in entity_ids and vertex_names[edge.target] in entity_ids
            ]
            self.assertEqual(fact_edge_weights, [2.0])
            self.assertTrue(all(set(edge["fact_source_counts"]) == chunk_ids for edge in reloaded.graph.es if edge["edge_kind"] == "fact"))

    def test_sequential_delete_of_shared_triple_removes_entity_sources(self):
        with tempfile.TemporaryDirectory() as save_dir:
            rag = make_rag(Path(save_dir))
            rag.index(["doc A", "doc B"])
            rag.prepare_retrieval_objects()

            doc_b_id = rag.chunk_embedding_store.text_to_hash_id["doc B"]
            rag.delete(["doc A"])
            self.assertEqual(set(rag.entity_embedding_store.get_all_texts()), {"alice", "bob"})
            self.assertTrue(all(sources == {doc_b_id} for sources in rag.ent_node_to_chunk_ids.values()))

            rag.delete(["doc B"])
            self.assertEqual(rag.chunk_embedding_store.get_all_texts(), set())
            self.assertEqual(rag.fact_embedding_store.get_all_texts(), set())
            self.assertEqual(rag.entity_embedding_store.get_all_texts(), set())
            self.assertEqual(rag.proc_triples_to_docs, {})
            self.assertEqual(rag.ent_node_to_chunk_ids, {})
            self.assertEqual(rag.graph.vcount(), 0)

    def test_batch_delete_of_shared_triple_removes_all_state(self):
        with tempfile.TemporaryDirectory() as save_dir:
            rag = make_rag(Path(save_dir))
            rag.index(["doc A", "doc B"])
            rag.delete(["doc A", "doc B"])

            self.assertEqual(rag.chunk_embedding_store.get_all_texts(), set())
            self.assertEqual(rag.fact_embedding_store.get_all_texts(), set())
            self.assertEqual(rag.entity_embedding_store.get_all_texts(), set())
            self.assertEqual(rag.proc_triples_to_docs, {})
            self.assertEqual(rag.ent_node_to_chunk_ids, {})
            self.assertEqual(rag.graph.vcount(), 0)

    def test_delete_deduplicates_triples_after_normalization(self):
        with tempfile.TemporaryDirectory() as save_dir:
            rag = make_rag(Path(save_dir))
            rag.openie = NormalizationCollisionOpenIE()
            rag.index(["doc"])
            self.assertEqual(len(rag.fact_embedding_store.get_all_ids()), 1)

            rag.delete(["doc"])
            self.assertEqual(rag.fact_embedding_store.get_all_texts(), set())
            self.assertEqual(rag.entity_embedding_store.get_all_texts(), set())
            self.assertEqual(rag.proc_triples_to_docs, {})
            self.assertEqual(rag.ent_node_to_chunk_ids, {})

    def test_delete_last_document_persists_empty_openie_state(self):
        with tempfile.TemporaryDirectory() as save_dir:
            rag = make_rag(Path(save_dir))
            rag.index(["only doc"])
            rag.delete(["only doc"])

            with open(rag.openie_results_path, encoding="utf-8") as openie_file:
                persisted_openie = json.load(openie_file)
            self.assertEqual(persisted_openie["docs"], [])
            self.assertEqual(persisted_openie["avg_ent_chars"], 0)
            self.assertEqual(persisted_openie["avg_ent_words"], 0)
            self.assertEqual(persisted_openie["provenance"]["identity"]["explicit_identity"], "fake-components-v1")

            reloaded = make_rag(Path(save_dir))
            reloaded.prepare_retrieval_objects()
            self.assertEqual(reloaded.proc_triples_to_docs, {})
            self.assertEqual(reloaded.ent_node_to_chunk_ids, {})
            self.assertEqual(reloaded.graph.vcount(), 0)


if __name__ == "__main__":
    unittest.main()
