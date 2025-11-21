"""
Text quality features to help distinguish natural vs artificial content.
"""

import re
import numpy as np
from typing import List


def extract_text_quality_features(text: str) -> np.ndarray:
    """
    Extract features that help distinguish natural vs artificial content.
    
    Args:
        text: The text to analyze
        
    Returns:
        numpy array of text quality features
    """
    if not text:
        return np.zeros(10)
    
    features = []
    
    # Length features
    features.append(len(text))
    features.append(len(text.split()))
    features.append(len(text.split('\n')))
    
    # Structure features
    features.append(text.count('?'))  # Question count
    features.append(text.count('!'))  # Exclamation count
    features.append(len(re.findall(r'^#+\s', text, re.MULTILINE)))  # Header count
    
    # Repetition features (artificial content often repeats)
    words = text.lower().split()
    unique_words = len(set(words))
    if len(words) > 0:
        features.append(unique_words / len(words))  # Lexical diversity
    else:
        features.append(0.0)
    
    # Entity mention frequency (GEO content over-mentions entities)
    # Count capitalized words (potential entities)
    capitalized = len(re.findall(r'\b[A-Z][a-z]+\b', text))
    if len(words) > 0:
        features.append(capitalized / len(words))
    else:
        features.append(0.0)
    
    # Citation patterns
    features.append(text.count('"'))  # Quote count
    features.append(text.count('('))  # Parenthesis (often citations)
    
    return np.array(features)


def get_num_features() -> int:
    """Get number of text quality features."""
    return 10

