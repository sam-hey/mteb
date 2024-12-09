from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np
import torch

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta

from .wrapper import Wrapper

logger = logging.getLogger(__name__)


class ColBERTWrapper(Wrapper):
    def __init__(
        self,
        model_name: str,
        **kwargs,
    ) -> None:
        """Wrapper for ColBERT models.

        Args:
            model_name: The ColBERT model to load from HuggingFace Hub.
            **kwargs: Additional arguments to pass to the model.
        """
        try:
            from pylate import models as colbert_model
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "To use the ColBERT models `pylate` is required. Please install it with `pip install mteb[pylate]`."
            ) from e

        self.model_name = model_name
        self.static_model = colbert_model.ColBERT(self.model_name, **kwargs)

    def encode(
        self,
        sentences: Sequence[str],
        prompt_type: PromptType,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            prompt_type: PromptType.query or PromptType.passage.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """
        if "task_name" in kwargs:
            kwargs.pop("task_name")
        if "prompt_type" in kwargs:
            kwargs.pop("prompt_type")

        pred = self.static_model.encode(
            sentences,
            is_query=True if prompt_type == PromptType.query else False,
            **kwargs,
        )

        pred = torch.nn.utils.rnn.pad_sequence(pred, batch_first=True, padding_value=0)

        return pred.cpu().numpy()


colbert_v2 = ModelMeta(
    loader=partial(
        ColBERTWrapper,
        model_name="colbert-ir/colbertv2.0",
    ),
    name="colbert-ir/colbertv2.0",
    languages=["eng_Latn"],
    open_weights=True,
    revision="c1e84128e85ef755c096a95bdb06b47793b13acf",
    public_training_code=True,
    release_date="2024-09-21",
    n_parameters=110 * 1e6,
    max_tokens=512,
    embed_dim=None,  # Bag of Embeddings
    license="mit",
    similarity_fn_name="max_sim",
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/colbert-ir/colbertv2.0",
    use_instructions=False,
    adapted_from=None,
    superseded_by=None,
)
