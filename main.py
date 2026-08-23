import argparse
import json
import logging
import os
from pathlib import Path
from typing import List

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.StandardRAG import StandardRAG
from src.hipporag.utils.config_utils import BaseConfig
from src.hipporag.utils.misc_utils import string_to_bool

PROJECT_ROOT = Path(__file__).resolve().parent


def get_gold_docs(samples: List, dataset_name: str = None) -> List:
    gold_docs = []
    for sample in samples:
        if "supporting_facts" in sample:
            gold_titles = {item[0] for item in sample["supporting_facts"]}
            supporting_contexts = [item for item in sample["context"] if item[0] in gold_titles]
            separator = "" if (dataset_name or "").startswith("hotpotqa") else " "
            gold_doc = [item[0] + "\n" + separator.join(item[1]) for item in supporting_contexts]
        elif "contexts" in sample:
            gold_doc = [item["title"] + "\n" + item["text"] for item in sample["contexts"] if item["is_supporting"]]
        else:
            if "paragraphs" not in sample:
                raise ValueError("Each sample needs supporting_facts, contexts, or paragraphs for retrieval evaluation")
            paragraphs = [item for item in sample["paragraphs"] if item.get("is_supporting", True)]
            gold_doc = [item["title"] + "\n" + (item["text"] if "text" in item else item["paragraph_text"]) for item in paragraphs]
        gold_docs.append(list(dict.fromkeys(gold_doc)))
    return gold_docs


def get_gold_answers(samples):
    gold_answers = []
    for sample in samples:
        if "answer" in sample or "gold_ans" in sample:
            gold_answer = sample.get("answer", sample.get("gold_ans"))
        elif "reference" in sample:
            gold_answer = sample["reference"]
        elif "obj" in sample:
            gold_answer = [sample["obj"], sample["o_wiki_title"]]
            for field in ("possible_answers", "o_aliases"):
                value = sample.get(field, [])
                gold_answer.extend(value if isinstance(value, list) else [value])
        else:
            raise ValueError("Each query sample must contain an answer field")
        if gold_answer is None:
            raise ValueError("Answer fields must not be null")
        answers = [gold_answer] if isinstance(gold_answer, str) else list(gold_answer)
        answers.extend(sample.get("answer_aliases", []))
        gold_answers.append(list(dict.fromkeys(answers)))
    return gold_answers


def parse_args():
    parser = argparse.ArgumentParser(description="HippoRAG retrieval and QA experiments")
    parser.add_argument("--dataset", default="musique", help="Dataset name under reproduce/dataset")
    parser.add_argument("--rag_type", choices=["hipporag", "standard"], default="hipporag", help="Retrieval method; standard reproduces the DPR-style dense baseline")
    parser.add_argument("--llm_base_url", default="https://api.openai.com/v1", help="OpenAI-compatible LLM base URL")
    parser.add_argument("--llm_name", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--embedding_name", default="nvidia/NV-Embed-v2", help="Embedding model name")
    parser.add_argument("--embedding_provider", choices=["openai", "transformers", "vllm", "gritlm", "nvembed", "contriever", "cohere"], help="Explicit embedding provider; otherwise inferred from the model name")
    parser.add_argument("--azure_endpoint", help="Azure OpenAI resource endpoint")
    parser.add_argument("--azure_api_version", help="Azure OpenAI API version for chat completions")
    parser.add_argument("--azure_chat_deployment", help="Azure OpenAI chat deployment name; defaults to --llm_name")
    parser.add_argument("--azure_embedding_endpoint", help="Azure OpenAI resource endpoint for embeddings")
    parser.add_argument("--azure_embedding_api_version", help="Azure OpenAI embeddings API version; defaults to --azure_api_version")
    parser.add_argument("--azure_embedding_deployment", help="Azure OpenAI embedding deployment name; defaults to --embedding_name")
    parser.add_argument("--embedding_batch_size", type=int, default=8, help="Embedding batch size")
    parser.add_argument("--force_index_from_scratch", default="false", help="Rebuild graph state while reusing compatible stored embeddings/OpenIE")
    parser.add_argument("--force_openie_from_scratch", default="false", help="Regenerate OpenIE only in a fresh/empty index directory")
    parser.add_argument("--openie_mode", choices=["online", "offline", "Transformers-offline"], default="online", help="OpenIE execution mode")
    parser.add_argument("--save_dir", default="outputs", help="Output directory prefix; custom values retain the legacy _<dataset> suffix")
    return parser.parse_args()


def main():
    args = parse_args()
    if Path(args.dataset).name != args.dataset:
        raise ValueError("dataset must be a name, not a path")
    save_dir = str(PROJECT_ROOT / "outputs" / args.dataset) if args.save_dir == "outputs" else f"{args.save_dir}_{args.dataset}"
    dataset_dir = PROJECT_ROOT / "reproduce" / "dataset"
    with open(dataset_dir / f"{args.dataset}_corpus.json", encoding="utf-8") as file:
        corpus = json.load(file)
    with open(dataset_dir / f"{args.dataset}.json", encoding="utf-8") as file:
        samples = json.load(file)

    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    queries = [sample["question"] for sample in samples]
    gold_answers = get_gold_answers(samples)
    try:
        gold_docs = get_gold_docs(samples, args.dataset)
    except (ValueError, KeyError):
        logging.warning("Retrieval evaluation is disabled because supporting documents are unavailable")
        gold_docs = None

    config = BaseConfig(
        save_dir=save_dir,
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        azure_endpoint=args.azure_endpoint,
        azure_api_version=args.azure_api_version,
        azure_chat_deployment=args.azure_chat_deployment,
        azure_embedding_endpoint=args.azure_embedding_endpoint,
        azure_embedding_api_version=args.azure_embedding_api_version,
        azure_embedding_deployment=args.azure_embedding_deployment,
        dataset=args.dataset,
        embedding_model_name=args.embedding_name,
        embedding_provider=args.embedding_provider,
        force_index_from_scratch=string_to_bool(args.force_index_from_scratch),
        force_openie_from_scratch=string_to_bool(args.force_openie_from_scratch),
        rerank_dspy_file_path=str(PROJECT_ROOT / "src" / "hipporag" / "prompts" / "dspy_prompts" / "filter_llama3.3-70B-Instruct.json"),
        retrieval_top_k=200,
        linking_top_k=5,
        qa_top_k=5,
        embedding_batch_size=args.embedding_batch_size,
        openie_mode=args.openie_mode,
    )
    logging.basicConfig(level=logging.INFO)
    rag_class = HippoRAG if args.rag_type == "hipporag" else StandardRAG
    with rag_class(global_config=config) as rag:
        rag.index(docs)
        rag.rag_qa(queries=queries, gold_docs=gold_docs, gold_answers=gold_answers)


if __name__ == "__main__":
    main()
