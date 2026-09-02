from _shared import run_lifecycle


if __name__ == "__main__":
    run_lifecycle(
        save_dir="outputs/orcarouter_test",
        llm_model_name="orcarouter/anthropic/claude-opus-4.8",
        embedding_model_name="text-embedding-3-small",
    )
