# GEO Score Calculation and Featurization

## Overview

This document explains how GEO scores are calculated for baseline means and how they relate to model featurization.

## 1. GEO Score Calculation

### 1.1 What is a GEO Score?

The GEO score (also called `s_geo_max`) is a semantic similarity score that measures how similar a piece of text is to known GEO (Generative Engine Optimization) patterns. It's calculated using:

1. **Pattern Embeddings**: Pre-computed embeddings of 5 GEO patterns (descriptions + examples) using sentence-transformers (all-MiniLM-L6-v2, 384 dimensions)
2. **Text Embedding**: The suspect text is embedded into the same 384-dimensional vector space
3. **Cosine Similarity**: Cosine similarity is calculated between the text embedding and each pattern embedding
4. **S_GEO_Max**: The maximum similarity score across all patterns (range: 0.0 to 1.0)

**Code Location**: `src/pattern_recognition/similarity_scores.py`
- `SemanticGEODetector.score()` method
- `calculate_semantic_geo_score()` function

### 1.2 Calculating Baseline Means per Category

**Process** (in `categorize_and_demean_geo_scores.py`):

1. **Read TSV File**: Load `se_optimized_sources_with_content.tsv` which contains:
   - `source_url`: URL of the source
   - `clean_content`: The cleaned text content
   - Other metadata (query, se_rank, ge_rank)

2. **For Each Source**:
   ```python
   # Categorize the website
   category = categorize_website(url, content)
   
   # Calculate GEO score using SemanticGEODetector
   geo_score, _ = detector.score(content, top_k=3, parsed=False)
   # geo_score is a float between 0.0 and 1.0
   
   # Store in category list
   category_scores[category].append(float(geo_score))
   ```

3. **Calculate Statistics per Category**:
   ```python
   baseline_stats[category] = {
       'mean': np.mean(scores),      # Average GEO score for this category
       'std': np.std(scores),        # Standard deviation
       'count': len(scores),         # Number of sources in category
       'min': np.min(scores),
       'max': np.max(scores),
       'median': np.median(scores)
   }
   ```

**Result**: A dictionary mapping each category (E-commerce, Corporate, etc.) to its baseline GEO score statistics.

### 1.3 Demeaning Process

When processing datasets for experiments, sources are demeaned by category:

```python
# For each source in the dataset:
category = categorize_website(url, content)
original_geo_score = source.get('s_geo_max', 0.0)  # Get existing score or calculate it

# Get baseline mean for this category
baseline_mean = baseline_stats[category]['mean']

# Demean: subtract category baseline
demeaned_geo_score = original_geo_score - baseline_mean

# Store in dataset
source['s_geo_max'] = demeaned_geo_score
```

**Purpose**: Normalize GEO scores across categories. For example, if E-commerce sites naturally have higher GEO scores (mean=0.22) than Non-profit sites (mean=0.16), demeaning makes them comparable.

## 2. How GEO Scores Are Used in Featurization

### 2.1 Important: GEO Scores Are NOT Direct Features

**Key Finding**: The `s_geo_max` value stored in the dataset is **NOT used as a direct feature** in the classification models.

### 2.2 What Features Are Actually Used?

The classification models use the following features (from `extract_features()` method):

1. **Semantic Embeddings** (384 dimensions):
   - The `cleaned_text` is embedded using the same sentence-transformer model (all-MiniLM-L6-v2)
   - This creates a 384-dimensional vector representation of the text
   - **This is the primary feature**

2. **Optional Semantic Pattern Scores** (5 dimensions):
   - If `--use-semantic-features` is enabled
   - Individual similarity scores for each of the 5 GEO patterns
   - These are the same scores used to calculate `s_geo_max` (which is the max of these 5)

3. **Optional Additional Features**:
   - Excessiveness features (if enabled)
   - Text quality features (if enabled)
   - Relative features comparing sources within entries (if enabled)

### 2.3 Feature Extraction Process

**Code Location**: `src/classification/logistic_ordinal_classifier.py` (and similar for other models)

```python
def extract_features(self, cleaned_text: str, entry_features: Optional[List[np.ndarray]] = None):
    # 1. Generate semantic embedding (384-dim vector)
    base_features = self.embedding_model.encode(cleaned_text, convert_to_numpy=True)
    
    features_list = [base_features]
    
    # 2. Optionally add semantic pattern scores (5 individual scores)
    if self.use_semantic_features:
        semantic_scores = self.semantic_extractor.extract_pattern_scores(cleaned_text)
        features_list.append(semantic_scores)
        # s_geo_max would be max(semantic_scores), but we use all 5 scores
    
    # 3. Optionally add other features...
    
    return np.concatenate(features_list)  # Final feature vector
```

### 2.4 Why Demeaning Still Matters

Even though `s_geo_max` is not a direct feature, demeaning affects the experiment in these ways:

1. **Dataset Consistency**: The `s_geo_max` value in the dataset is updated to reflect demeaned scores, maintaining consistency in the data representation.

2. **Future Use**: If `s_geo_max` were to be used as a feature in future experiments, it would already be normalized.

3. **Analysis**: The demeaned scores can be used for analysis and comparison, even if not directly in features.

4. **Semantic Pattern Scores**: When `--use-semantic-features` is enabled, the individual pattern scores (which are used to calculate `s_geo_max`) are used as features. However, these are calculated fresh from the text during feature extraction, not from the stored `s_geo_max` value.

## 3. Summary

### GEO Score Calculation:
- **Method**: Semantic similarity using sentence-transformers and cosine similarity
- **Output**: Float between 0.0 and 1.0 (S_GEO_Max)
- **Baseline Calculation**: Mean GEO score per website category from TSV file
- **Demeaning**: Subtract category baseline mean from each source's GEO score

### Featurization:
- **Primary Feature**: 384-dimensional semantic embedding of the text (NOT the GEO score)
- **Optional Features**: Individual semantic pattern scores (5 dimensions), not the max
- **GEO Score Role**: Stored as metadata in dataset, but NOT used as a direct feature
- **Impact of Demeaning**: The stored `s_geo_max` value is updated, but since it's not used as a feature, demeaning doesn't directly affect model training

### Key Insight:
The models learn to detect GEO patterns from the **semantic embeddings** of the text itself, not from pre-computed GEO scores. The GEO scores are used for:
- Baseline calculation and demeaning
- Analysis and comparison
- Potentially as features in future experiments

But in the current implementation, the models rely on the raw semantic embeddings to learn GEO patterns, which is why demeaning the stored `s_geo_max` values has minimal impact on model performance.

