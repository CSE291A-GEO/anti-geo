# Embedding and Scoring Methodology

This document explains how the semantic matching system embeds GEO patterns and calculates detection scores.

## Overview

The system uses **semantic embeddings** and **cosine similarity** to detect Generative Engine Optimization (GEO) patterns in text. It transforms both known GEO patterns and suspect text into high-dimensional vector representations, then measures how similar they are.

## Embedding Process

### 1. Pattern Embedding (Vector Store Construction)

**Purpose**: Create semantic "fingerprints" for each known GEO pattern.

**Steps**:
1. **Pattern Text Construction**: For each GEO pattern, combine:
   - Pattern description (abstract definition)
   - Pattern examples (concrete demonstrations)
   
   Example:
   ```
   Description: "Excessive Q&A blocks with keyword stuffing"
   + Examples: "Q: What is X? A: X is... Q: How does X work? A: X uses..."
   ```

2. **Vector Encoding**: Use a SentenceTransformer model (`all-MiniLM-L6-v2` by default) to encode each pattern text into a 384-dimensional vector.
   - The model converts semantic meaning into numerical representations
   - Similar meanings result in vectors that are close together in vector space

3. **Vector Store**: Store all pattern embeddings in a matrix:
   - Shape: `(num_patterns, 384)`
   - Each row represents one GEO pattern's semantic fingerprint

### 2. Suspect Text Embedding

**Purpose**: Convert the text being analyzed into the same vector space as the patterns.

**Steps**:
1. Take the suspect text (e.g., a source's `cleaned_text`)
2. Encode it using the same SentenceTransformer model
3. Result: A 384-dimensional vector representing the semantic meaning of the suspect text

## Scoring Process

### Cosine Similarity Calculation

**Purpose**: Measure how semantically similar the suspect text is to each GEO pattern.

**How it works**:
1. **Calculate Similarity**: For each pattern embedding, compute cosine similarity with the suspect text embedding:
   ```python
   cosine_similarity(suspect_vector, pattern_vector)
   ```

2. **Cosine Similarity Interpretation**:
   - **1.0**: Identical semantic meaning (vectors point in the same direction)
   - **0.0**: Completely unrelated (vectors are orthogonal)
   - **Higher values** = More semantically similar
   - **Lower values** = Less semantically similar

3. **Pattern Matching**: Compare suspect text against all patterns and get similarity scores for each

### S_GEO_Max Score

**Definition**: The maximum cosine similarity score across all GEO patterns.

**Calculation**:
```python
S_GEO_Max = max(similarity(suspect_text, pattern_i) for all patterns i)
```

**Interpretation**:
- **S_GEO_Max ≥ 0.75**: High likelihood of GEO-optimized content
- **S_GEO_Max 0.5 - 0.75**: Moderate GEO indicators
- **S_GEO_Max < 0.5**: Low GEO indicators, likely natural content

This score indicates how similar the suspect text is to **any** known GEO pattern, with higher scores suggesting stronger GEO characteristics.

## Technical Details

### Model: all-MiniLM-L6-v2
- **Embedding Dimension**: 384
- **Advantages**: Fast, lightweight, yet powerful for semantic similarity tasks
- **Normalization**: SentenceTransformer automatically normalizes vectors, making cosine similarity equivalent to dot product

### Why This Approach Works

1. **Semantic Understanding**: Unlike keyword matching, embeddings capture semantic meaning, detecting GEO patterns even when wording differs

2. **Pattern Recognition**: By combining descriptions and examples, patterns capture both abstract concepts and concrete manifestations

3. **Scalable**: Once patterns are embedded, scoring new text is fast (single encoding + matrix multiplication)

4. **Robust**: Works across different phrasings, synonyms, and variations of the same GEO technique

## Workflow Summary

```
1. Initialize Model (SentenceTransformer)
   ↓
2. Build Pattern Vector Store
   - Encode each pattern (description + examples) → 384-dim vector
   - Store in matrix: (num_patterns, 384)
   ↓
3. For Each Suspect Text:
   - Encode text → 384-dim vector
   - Calculate cosine similarity with all pattern vectors
   - Find maximum similarity → S_GEO_Max
   - Rank patterns by similarity → top_k matches
   ↓
4. Return Results
   - S_GEO_Max: Overall GEO-likelihood score
   - Top matches: Which patterns match best
```

## Example

**Pattern** (GEO_STRUCT_001):
- Description: "Excessive Q&A blocks with keyword stuffing"
- Examples: Multiple Q&A examples

**Suspect Text**:
```
"Q: What is the Alpha System? A: The Alpha System offers speed. 
 Q: How is the Alpha System different? A: The Alpha System uses technology."
```

**Process**:
1. Pattern embedded → vector_p
2. Suspect text embedded → vector_s
3. Cosine similarity calculated → 0.82
4. S_GEO_Max = 0.82 (high GEO likelihood)
5. Top match: GEO_STRUCT_001 with score 0.82

