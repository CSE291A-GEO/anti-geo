"""
Excessiveness features to measure how 'excessive' semantic patterns are.
"""

import numpy as np
from typing import Optional


def extract_excessiveness_features(pattern_scores: np.ndarray) -> np.ndarray:
    """
    Extract features measuring how 'excessive' the semantic patterns are.
    High scores might be natural (well-structured content) vs artificial (GEO optimization).
    
    Args:
        pattern_scores: Array of 5 semantic pattern scores
        
    Returns:
        numpy array of excessiveness features
    """
    if len(pattern_scores) == 0:
        return np.zeros(6)
    
    excessiveness_features = []
    
    # Feature 1: Pattern score variance (artificial patterns are more uniform)
    excessiveness_features.append(np.std(pattern_scores))
    
    # Feature 2: Maximum pattern score (very high single pattern = suspicious)
    excessiveness_features.append(np.max(pattern_scores))
    
    # Feature 3: Number of patterns above threshold
    excessiveness_features.append(np.sum(pattern_scores > 0.3))
    excessiveness_features.append(np.sum(pattern_scores > 0.5))
    
    # Feature 4: Pattern score sum (total GEO-like signal)
    excessiveness_features.append(np.sum(pattern_scores))
    
    # Feature 5: Pattern score ratio (max / mean) - indicates concentration
    mean_score = np.mean(pattern_scores)
    if mean_score > 0:
        excessiveness_features.append(np.max(pattern_scores) / mean_score)
    else:
        excessiveness_features.append(0.0)
    
    return np.array(excessiveness_features)


def get_num_features() -> int:
    """Get number of excessiveness features."""
    return 6

