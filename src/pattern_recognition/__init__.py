"""
Pattern Recognition Module for GEO Detection

This module provides both semantic similarity-based and rule-based pattern
detection capabilities to identify adversarial Generative Engine Optimization (GEO) 
patterns in text content.
"""

# Import from semantic_matching (which imports from similarity_scores for backward compatibility)
from .semantic_matching import (
    GEO_PATTERNS,
    calculate_semantic_geo_score,
    build_geo_vector_store,
    SemanticGEODetector
)

# Import rule-based pattern detectors
from .pattern_detectors import (
    score_geo_struct_001,
    score_geo_struct_002,
    score_geo_struct_003,
    score_geo_semantic_004,
    score_geo_semantic_005,
    calculate_rule_based_geo_score,
    calculate_all_pattern_scores,
    RuleBasedGEODetector
)

__all__ = [
    # Semantic similarity-based detection
    'GEO_PATTERNS',
    'calculate_semantic_geo_score',
    'build_geo_vector_store',
    'SemanticGEODetector',
    # Rule-based pattern detection
    'score_geo_struct_001',
    'score_geo_struct_002',
    'score_geo_struct_003',
    'score_geo_semantic_004',
    'score_geo_semantic_005',
    'calculate_rule_based_geo_score',
    'calculate_all_pattern_scores',
    'RuleBasedGEODetector'
]

