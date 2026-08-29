from hipporag.llm import _get_llm_class
from hipporag.utils.config_utils import BaseConfig


def main():
    config = BaseConfig(
        llm_name="orcarouter/anthropic/claude-opus-4.8",
        save_dir="outputs/orcarouter",
    )
    llm = _get_llm_class(config)
    message, metadata, cached = llm.infer([{"role": "user", "content": "Reply with exactly: HippoRAG OrcaRouter test passed"}])
    print(message)
    print({"metadata": metadata, "cached": cached})


if __name__ == "__main__":
    main()
