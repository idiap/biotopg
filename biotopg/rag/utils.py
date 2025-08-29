# Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: MIT

from typing import Any, List

import numpy as np


def mmr(doc_embeddings, relevance_scores, lambda_param=0.5, k=10):
    """
    Perform Maximal Marginal Relevance (MMR) re-ranking.

    Args:
        doc_embeddings (np.ndarray): Embeddings of the retrieved documents.
        relevance_scores (list): Precomputed relevance scores (e.g., cosine similarities).
        lambda_param (float): Trade-off parameter between relevance and diversity (0 <= lambda <= 1).
        k (int): Number of results to return.

    Returns:
        list: Indices of the re-ranked documents.
    """
    selected = []
    unselected = list(range(len(doc_embeddings)))
    k = min(k, len(unselected))

    for _ in range(k):
        mmr_scores = []
        for i in unselected:
            # Relevance term
            relevance = relevance_scores[i]
            # Diversity term
            diversity = max(
                [np.dot(doc_embeddings[i], doc_embeddings[j]) for j in selected] or [0]
            )
            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
            mmr_scores.append((mmr_score, i))

        # Select the document with the highest MMR score
        best_idx = max(mmr_scores, key=lambda x: x[0])[1]
        selected.append(best_idx)
        unselected.remove(best_idx)

    return selected
