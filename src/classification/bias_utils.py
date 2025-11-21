"""
Utility functions to bias classifiers toward detecting GEO sources.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.utils import shuffle


def oversample_positive(
    X: np.ndarray,
    y: np.ndarray,
    metadata: List[Dict[str, Any]],
    texts: List[str],
    positive_label: int,
    weight: float,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], List[str]]:
    """
    Oversample positive samples to emphasize GEO detection.
    """
    if weight <= 1.0:
        return X, y, metadata, texts
    pos_indices = np.where(y == positive_label)[0]
    if len(pos_indices) == 0:
        return X, y, metadata, texts
    repeat_count = int(weight) - 1
    if repeat_count <= 0:
        return X, y, metadata, texts
    X_pos = np.repeat(X[pos_indices], repeat_count, axis=0)
    y_pos = np.repeat(y[pos_indices], repeat_count, axis=0)
    metadata_pos = []
    texts_pos = []
    for idx in pos_indices:
        for _ in range(repeat_count):
            metadata_pos.append(metadata[idx].copy())
            texts_pos.append(texts[idx])
    X_aug = np.concatenate([X, X_pos], axis=0)
    y_aug = np.concatenate([y, y_pos], axis=0)
    metadata_aug = metadata + metadata_pos
    texts_aug = texts + texts_pos
    X_aug, y_aug, metadata_aug, texts_aug = shuffle(
        X_aug, y_aug, metadata_aug, texts_aug, random_state=random_state
    )
    return X_aug, y_aug, metadata_aug, texts_aug


def entry_argmax_predictions(
    geo_scores: np.ndarray,
    metadata: List[Dict[str, Any]]
) -> np.ndarray:
    """
    Force exactly one positive prediction per entry based on highest GEO score.
    """
    predictions = np.zeros(len(geo_scores), dtype=int)
    entry_map: Dict[int, List[int]] = {}
    for idx, meta in enumerate(metadata):
        entry_idx = meta['entry_idx']
        entry_map.setdefault(entry_idx, []).append(idx)
    for indices in entry_map.values():
        if not indices:
            continue
        best_idx = max(indices, key=lambda i: geo_scores[i])
        predictions[best_idx] = 1
    return predictions


def calculate_ranking_accuracy(
    geo_scores: np.ndarray,
    metadata: List[Dict[str, Any]],
    y: np.ndarray,
    positive_label: int
) -> float:
    """
    Compute the percentage of entries where the true GEO source ranks highest.
    """
    entry_map: Dict[int, List[int]] = {}
    for idx, meta in enumerate(metadata):
        entry_idx = meta['entry_idx']
        entry_map.setdefault(entry_idx, []).append(idx)
    correct = 0
    total = 0
    for indices in entry_map.values():
        if not indices:
            continue
        total += 1
        best_idx = max(indices, key=lambda i: geo_scores[i])
        if y[best_idx] == positive_label:
            correct += 1
    return correct / total if total > 0 else 0.0

