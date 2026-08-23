import argparse

from _shared import run_demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HippoRAG with Azure OpenAI")
    parser.add_argument("--azure_endpoint", required=True)
    parser.add_argument("--azure_api_version")
    parser.add_argument("--azure_chat_deployment")
    parser.add_argument("--azure_embedding_endpoint", required=True)
    parser.add_argument("--azure_embedding_api_version")
    parser.add_argument("--azure_embedding_deployment")
    args = parser.parse_args()
    run_demo(
        save_dir="outputs/azure",
        llm_model_name="gpt-4o-mini",
        embedding_model_name="text-embedding-3-small",
        azure_endpoint=args.azure_endpoint,
        azure_api_version=args.azure_api_version,
        azure_chat_deployment=args.azure_chat_deployment,
        azure_embedding_endpoint=args.azure_embedding_endpoint,
        azure_embedding_api_version=args.azure_embedding_api_version,
        azure_embedding_deployment=args.azure_embedding_deployment,
    )
