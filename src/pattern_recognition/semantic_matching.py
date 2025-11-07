"""
Semantic Matching Baseline for GEO Detection

This module implements a semantic matching system that calculates a score (S_GEO)
indicating how structurally or semantically similar a suspect document is to
known adversarial Generative Engine Optimization (GEO) patterns, using vector
embeddings and cosine similarity.

For backward compatibility, this module imports from similarity_scores.
For rule-based pattern detection, see pattern_detectors.
"""

# Import from similarity_scores for backward compatibility
from .similarity_scores import (
    GEO_PATTERNS,
    DEFAULT_MODEL_NAME,
    build_geo_vector_store,
    calculate_semantic_geo_score,
    SemanticGEODetector
)

__all__ = [
    'GEO_PATTERNS',
    'DEFAULT_MODEL_NAME',
    'build_geo_vector_store',
    'calculate_semantic_geo_score',
    'SemanticGEODetector'
]
