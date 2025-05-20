from __future__ import annotations

import logging
from functools import partial

from mteb.evaluation.evaluators.RetrievalEvaluator import ModelWithIndex
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper
from mteb.requires_package import requires_package

logger = logging.getLogger(__name__)


class ColBERTWrapper(Wrapper, ModelWithIndex):
    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        model_prompts: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        """Wrapper for ColBERT models.

        Args:
            model_name: The ColBERT model to load from HuggingFace Hub.
            revision: The revision of the model to use.
            model_prompts: A dictionary mapping task names to prompt names.
                First priority is given to the composed prompt of task name + prompt type (query or passage), then to the specific task prompt,
                then to the composed prompt of task type + prompt type, then to the specific task type prompt,
                and finally to the specific prompt type.
            **kwargs: Additional arguments to pass to the model.
        """
        requires_package(self, "pylate", model_name, "pip install mteb[pylate]")
        from pylate import models as colbert_model

        self.model_name = model_name
        self.model = colbert_model.ColBERT(self.model_name, revision=revision, **kwargs)
        if (
            model_prompts is None
            and hasattr(self.model, "prompts")
            and len(self.model.prompts) > 0
        ):
            try:
                model_prompts = self.validate_task_to_prompt_name(self.model.prompts)
            except ValueError:
                model_prompts = None
        elif model_prompts is not None and hasattr(self.model, "prompts"):
            logger.info(f"Model prompts will be overwritten with {model_prompts}")
            self.model.prompts = model_prompts
        self.model_prompts = self.validate_task_to_prompt_name(model_prompts)

    def search_index(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str | list[str]],
        top_k: int,
        return_sorted: bool = False,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        """Search the index."""
        pass

    def build_index(self, corpus: dict[str, dict[str, str]], **kwargs):
        """Build the index."""
        from pylate import indexes as pylate_indexes
        from pylate import retrieve as pylate_retrieve

        self.index = pylate_indexes.Voyager(
            index_folder="pylate-index",
            index_name="index",
            override=True,
        )

        self.retriever = pylate_retrieve.ColBERT(index=self.index)

        doc_ids, docs = zip(*corpus.items())

        documents_embeddings = self.model.encode(
            docs,
            prompt_name=prompt_name,
            is_query=False,
            **kwargs,
        )

        # Add the documents ids and embeddings to the index
        self.index.add_documents(
            documents_ids=doc_ids,
            documents_embeddings=documents_embeddings,
        )

    def encode(self):
        prompt_name = self.get_prompt_name(task_metadata, prompt_type)
        if prompt_name:
            logger.info(
                f"Using prompt_name={prompt_name} for task={task_metadata.name} prompt_type={prompt_type}"
            )
        else:
            logger.info(
                f"No model prompts found for task={task_metadata.name} prompt_type={prompt_type}"
            )
        logger.info(f"Encoding {len(inputs)} sentences.")


colbert_v2 = ModelMeta(
    loader=partial(
        ColBERTWrapper,
        model_name="colbert-ir/colbertv2.0",
    ),
    name="colbert-ir/colbertv2.0",
    languages=["eng-Latn"],
    open_weights=True,
    revision="c1e84128e85ef755c096a95bdb06b47793b13acf",
    public_training_code=None,
    public_training_data=None,
    release_date="2024-09-21",
    n_parameters=110 * 1e6,
    memory_usage_mb=418,
    max_tokens=180,  # Reduced for Benchmarking - see ColBERT paper
    embed_dim=None,  # Bag of Embeddings (128) for each token
    license="mit",
    similarity_fn_name="max_sim",
    framework=["PyLate", "ColBERT"],
    reference="https://huggingface.co/colbert-ir/colbertv2.0",
    use_instructions=False,
    adapted_from=None,
    superseded_by=None,
    training_datasets={
        "MSMARCO": ["train"],  # dev?
        "mMARCO-NL": ["train"],  # translation not trained on
    },
)

jina_colbert_v2 = ModelMeta(
    loader=partial(
        ColBERTWrapper,
        model_name="jinaai/jina-colbert-v2",
        query_prefix="[QueryMarker]",
        document_prefix="[DocumentMarker]",
        attend_to_expansion_tokens=True,
        trust_remote_code=True,
    ),
    name="jinaai/jina-colbert-v2",
    languages=[  # list of languages the model has been evaluated on
        "ara-Arab",  # Arabic
        "ben-Beng",  # Bengali
        "deu-Latn",  # German
        "spa-Latn",  # Spanish
        "eng-Latn",  # English
        "fas-Arab",  # Persian
        "fin-Latn",  # Finnish
        "fra-Latn",  # French
        "hin-Deva",  # Hindi
        "ind-Latn",  # Indonesian
        "jpn-Jpan",  # Japanese
        "kor-Kore",  # Korean
        "rus-Cyrl",  # Russian
        "swa-Latn",  # Swahili
        "tel-Telu",  # Telugu
        "tha-Thai",  # Thai
        "yor-Latn",  # Yoruba
        "zho-Hans",  # Chinese (Simplified)
        "nld-Latn",  # Dutch
        "ita-Latn",  # Italian
        "por-Latn",  # Portuguese
        "vie-Latn",  # Vietnamese
    ],
    open_weights=True,
    revision="4cf816e5e2b03167b132a3c847a9ecd48ba708e1",
    public_training_code=None,
    public_training_data=None,
    release_date="2024-08-16",
    n_parameters=559 * 1e6,
    memory_usage_mb=1067,
    max_tokens=8192,
    embed_dim=None,  # Bag of Embeddings (128) for each token
    license="cc-by-nc-4.0",
    similarity_fn_name="max_sim",
    framework=["PyLate", "ColBERT"],
    reference="https://huggingface.co/jinaai/jina-colbert-v2",
    use_instructions=False,
    adapted_from=None,
    superseded_by=None,
    training_datasets={
        "MSMARCO": ["train"],
        "mMARCO-NL": ["train"],  # translation not trained on
        "DuRetrieval": [],
        "MIRACL": ["train"],
    },
)
