"""
Helper module for extracting semantic matching pattern scores as features.
"""

import numpy as np
from typing import Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# GEO Pattern IDs (from similarity_scores.py)
GEO_PATTERN_IDS = [
    "GEO_STRUCT_001",  # Excessive Q&A blocks
    "GEO_STRUCT_002",  # Over-Chunking/Simplification
    "GEO_STRUCT_003",  # Title/Header Stuffing
    "GEO_SEMANTIC_004",  # Entity Over-Attribution
    "GEO_SEMANTIC_005",  # Unnatural Citation Embedding
]

# Pattern descriptions (simplified for embedding)
GEO_PATTERN_DESCRIPTIONS = [
    "Excessive Q&A blocks: Using a high volume of Q&A-style headings with overly short, simplistic answers, often repeating the brand/entity name in every question or answer for forced keyword stuffing.",
    "Over-Chunking/Simplification: Breaking down high-quality content into an excessive number of short, self-contained bullet points or unnaturally simple paragraphs to facilitate easy extraction by the LLM.",
    "Title/Header Stuffing: Repeating the target entity or long-tail keyword in every successive header beyond what is semantically natural for human readability.",
    "Entity Over-Attribution: Injecting verbose, repetitive entity definitions many times in a short span, often in list format, purely to anchor the LLM to the entity.",
    "Unnatural Citation Embedding: Embedding high-precision, specific statistics or quotes in a non-contextual, easy-to-extract manner, intended to be a single, quotable data point for the LLM's synthesis.",
]


class SemanticFeatureExtractor:
    """
    Extracts individual semantic matching pattern scores as features.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the semantic feature extractor.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.pattern_vectors = None
        self._build_pattern_vectors()
    
    def _build_pattern_vectors(self):
        """Build embeddings for all GEO patterns."""
        # Embed each pattern description
        pattern_texts = GEO_PATTERN_DESCRIPTIONS
        self.pattern_vectors = self.model.encode(
            pattern_texts, 
            convert_to_numpy=True
        )
    
    def extract_pattern_scores(self, text: str) -> np.ndarray:
        """
        Extract individual semantic matching scores for each GEO pattern.
        
        Args:
            text: The text to score against GEO patterns
            
        Returns:
            numpy array of shape (num_patterns,) with similarity scores for each pattern
        """
        if not text:
            return np.zeros(len(GEO_PATTERN_IDS))
        
        try:
            # Embed the text
            text_vector = self.model.encode([text], convert_to_numpy=True)
            
            # Calculate cosine similarity with each pattern
            similarities = cosine_similarity(text_vector, self.pattern_vectors)[0]
            
            return similarities
        except Exception as e:
            print(f"    Warning: Failed to extract semantic pattern scores: {str(e)[:50]}")
            return np.zeros(len(GEO_PATTERN_IDS))
    
    def get_feature_names(self) -> list:
        """Get names of semantic pattern features."""
        return [f"semantic_pattern_{pattern_id}" for pattern_id in GEO_PATTERN_IDS]
    
    def get_num_features(self) -> int:
        """Get number of semantic pattern features."""
        return len(GEO_PATTERN_IDS)

