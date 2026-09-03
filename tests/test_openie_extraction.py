import unittest
from hipporag.information_extraction.openie_openai import _extract_json_list_field, _extract_ner_from_response
from hipporag.embedding_model import _get_embedding_model_class, OpenAIEmbeddingModel


class TestOpenIEExtraction(unittest.TestCase):
    def test_standard_json_object(self):
        resp = '{"named_entities": ["Apple", "California"]}'
        self.assertEqual(_extract_ner_from_response(resp), ["Apple", "California"])

    def test_markdown_code_fence(self):
        resp = "```json\n{\n  \"named_entities\": [\"Tesla\", \"Austin\"]\n}\n```"
        self.assertEqual(_extract_ner_from_response(resp), ["Tesla", "Austin"])

    def test_embedding_model_fallback(self):
        cls = _get_embedding_model_class("Nemotron-3-Embed-1B-NVFP4")
        self.assertEqual(cls, OpenAIEmbeddingModel)
        cls2 = _get_embedding_model_class("jina-embeddings-v3")
        self.assertEqual(cls2, OpenAIEmbeddingModel)


if __name__ == "__main__":
    unittest.main()
