from __future__ import annotations

import heapq
import logging

import torch

from mteb.encoder_interface import PromptType

logger = logging.getLogger(__name__)
import abc

from .utils import (
    cos_sim,
)


class SearchUtils(abc.ABC):
    def _maybe_create_index(self):
        if hasattr(self.model.model, "create_index"):
            self.model.model.create_index()
            logger.info("Index created for the model.")

    def _prepare_queries(self, queries, instructions):
        queries = [queries[qid] for qid in queries]
        if instructions:
            queries = [
                f"{query} {instructions.get(query, '')}".strip() for query in queries
            ]
        return queries

    def _encode_queries(self, queries, task_name):
        if isinstance(queries[0], list):
            return self.encode_conversations(
                model=self.model,
                conversations=queries,
                task_name=task_name,
                **self.encode_kwargs,
            )
        return self.model.encode(
            queries,
            task_name=task_name,
            prompt_type=PromptType.query,
            **self.encode_kwargs,
        )

    def _prepare_corpus(self, corpus):
        logger.info("Sorting Corpus by document length (Longest first)...")
        corpus_ids = sorted(corpus, reverse=True)
        corpus_list = [corpus[cid] for cid in corpus_ids]
        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        return corpus_ids, corpus_list

    def _get_corpus_embeddings(
        self,
        corpus_list,
        corpus_ids,
        start_idx,
        end_idx,
        task_name,
        request_qid,
        batch_num,
    ):
        if (
            self.save_corpus_embeddings
            and request_qid
            and len(self.corpus_embeddings[request_qid])
        ):
            return torch.tensor(self.corpus_embeddings[request_qid][batch_num])
        embeddings = self.model.encode(
            corpus_list[start_idx:end_idx],
            task_name=task_name,
            prompt_type=PromptType.passage,
            request_qid=request_qid,
            **self.encode_kwargs,
        )
        if self.save_corpus_embeddings and request_qid:
            self.corpus_embeddings[request_qid].append(embeddings)
        return embeddings

    def _maybe_add_to_index(self, embeddings, ids_slice):
        if hasattr(self.model.model, "add_to_index"):
            self.model.model.add_to_index(embeddings, ids_slice)
            return True
        return False

    def _update_results_heap(
        self,
        query_embeddings,
        corpus_embeddings,
        result_heaps,
        query_ids,
        corpus_ids,
        start_idx,
        top_k,
        return_sorted,
    ):
        similarity_fn = getattr(self.model, "similarity", cos_sim)
        similarity_scores = similarity_fn(query_embeddings, corpus_embeddings)

        is_nan = torch.isnan(similarity_scores)
        if is_nan.sum() > 0:
            logger.warning(
                f"Found {is_nan.sum()} NaN values in the similarity scores. Replacing with -1."
            )
            similarity_scores[is_nan] = -1

        top_k_vals, top_k_idxs = torch.topk(
            similarity_scores,
            min(top_k + 1, similarity_scores.shape[1]),
            dim=1,
            largest=True,
            sorted=return_sorted,
        )

        for i, qid in enumerate(query_ids):
            for idx, score in zip(top_k_idxs[i].tolist(), top_k_vals[i].tolist()):
                cid = corpus_ids[start_idx + idx]
                if len(result_heaps[qid]) < top_k:
                    heapq.heappush(result_heaps[qid], (score, cid))
                else:
                    heapq.heappushpop(result_heaps[qid], (score, cid))
        return result_heaps

    def _should_retrieve_from_index(self):
        return all(
            [
                hasattr(self.model.model, "add_to_index"),
                hasattr(self.model.model, "retrieve_from_index"),
            ]
        )

    def _retrieve_from_index(self, query_embeddings, query_ids, top_k):
        results = self.model.model.retrieve_from_index(query_embeddings, top_k)
        return {
            query_ids[i]: [(item["score"], item["id"]) for item in results[i]]
            for i in range(len(query_embeddings))
        }

    def _finalize_results(self, result_heaps):
        for qid, scored_docs in result_heaps.items():
            for score, cid in scored_docs:
                self.results[qid][cid] = score
