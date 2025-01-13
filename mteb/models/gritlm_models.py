from __future__ import annotations

import logging
from functools import partial

from mteb.model_meta import ModelMeta

from .instruct_wrapper import instruct_wrapper

logger = logging.getLogger(__name__)


def gritlm_instruction(instruction: str = "", prompt_type=None) -> str:
    return (
        "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"
    )


GRITLM_CITATION = """
@misc{muennighoff2024generative,
      title={Generative Representational Instruction Tuning}, 
      author={Niklas Muennighoff and Hongjin Su and Liang Wang and Nan Yang and Furu Wei and Tao Yu and Amanpreet Singh and Douwe Kiela},
      year={2024},
      eprint={2402.09906},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""


gritlm7b = ModelMeta(
    loader=partial(  # type: ignore
        instruct_wrapper,
        model_name_or_path="GritLM/GritLM-7B",
        instruction_template=gritlm_instruction,
        mode="embedding",
        torch_dtype="auto",
    ),
    name="GritLM/GritLM-7B",
    languages=["eng_Latn", "fra_Latn", "deu_Latn", "ita_Latn", "spa_Latn"],
    open_weights=True,
    revision="13f00a0e36500c80ce12870ea513846a066004af",
    release_date="2024-02-15",
    training_datasets={"GritLM/tulu2": ["train"]},
    n_parameters=7_240_000_000,
    embed_dim=4096,
    license="apache-2.0",
    max_tokens=4096,
    reference="https://huggingface.co/GritLM/GritLM-7B",
    similarity_fn_name="cosine",
    framework=["GritLM", "PyTorch"],
    use_instructions=True,
    citation=GRITLM_CITATION,
)
gritlm8x7b = ModelMeta(
    loader=partial(  # type: ignore
        instruct_wrapper,
        model_name_or_path="GritLM/GritLM-8x7B",
        instruction_template=gritlm_instruction,
        mode="embedding",
        torch_dtype="auto",
    ),
    name="GritLM/GritLM-8x7B",
    languages=["eng_Latn", "fra_Latn", "deu_Latn", "ita_Latn", "spa_Latn"],
    training_datasets={"GritLM/tulu2": ["train"]},
    open_weights=True,
    revision="7f089b13e3345510281733ca1e6ff871b5b4bc76",
    release_date="2024-02-15",
    n_parameters=57_920_000_000,
    embed_dim=4096,
    license="apache-2.0",
    max_tokens=4096,
    reference="https://huggingface.co/GritLM/GritLM-8x7B",
    similarity_fn_name="cosine",
    framework=["GritLM", "PyTorch"],
    use_instructions=True,
    citation=GRITLM_CITATION,
)
