# Classification Features

This document describes the features used by all GEO detection classifiers in this directory.

## Feature Overview

All classification models in this directory use **semantic embeddings** as their primary features, with an optional enhancement of **semantic pattern scores**.

### Base Feature Set

**Semantic Embeddings (384 dimensions)**
- **Source**: SentenceTransformer model (`all-MiniLM-L6-v2`)
- **Extraction**: Text is embedded into a 384-dimensional dense vector
- **Purpose**: Captures semantic meaning and context of the text
- **Always included**: Yes, this is the base feature set for all models

### Optional Feature Enhancement

**Semantic Pattern Scores (5 dimensions)**
- **Source**: Cosine similarity scores between text and 5 known GEO patterns
- **Extraction**: Computed using `SemanticFeatureExtractor`
- **Purpose**: Explicit signals for specific GEO optimization patterns
- **Included when**: `--use-semantic-features` flag is enabled

## Feature Dimensions

| Configuration | Total Dimensions | Breakdown |
|---------------|------------------|-----------|
| **Baseline** | 384 | 384 (embeddings only) |
| **With Semantic Features** | 389 | 384 (embeddings) + 5 (pattern scores) |

## Feature Extraction

### 1. Semantic Embeddings

**Process:**
1. Input text is cleaned/preprocessed
2. Text is passed to SentenceTransformer model (`all-MiniLM-L6-v2`)
3. Model outputs a 384-dimensional dense vector
4. Each dimension represents a learned semantic representation

**Model Details:**
- **Model**: `all-MiniLM-L6-v2` (SentenceTransformer)
- **Dimensions**: 384
- **Type**: Dense vector embeddings
- **Normalization**: Features are standardized using `StandardScaler` before training

**What it captures:**
- Semantic meaning of words and phrases
- Contextual relationships
- Overall text semantics
- General linguistic patterns

### 2. Semantic Pattern Scores (Optional)

**Process:**
1. Five GEO pattern descriptions are pre-embedded
2. Input text is embedded using the same model
3. Cosine similarity is computed between text embedding and each pattern embedding
4. Five similarity scores are produced (one per pattern)

**Pattern Scores:**
1. `semantic_pattern_GEO_STRUCT_001` - Excessive Q&A blocks similarity
2. `semantic_pattern_GEO_STRUCT_002` - Over-Chunking/Simplification similarity
3. `semantic_pattern_GEO_STRUCT_003` - Title/Header Stuffing similarity
4. `semantic_pattern_GEO_SEMANTIC_004` - Entity Over-Attribution similarity
5. `semantic_pattern_GEO_SEMANTIC_005` - Unnatural Citation Embedding similarity

**Score Range:**
- Typically: 0.0 to 1.0 (cosine similarity)
- Higher values indicate stronger match to that specific pattern

## Feature Processing Pipeline

```
Input Text
    ↓
[Optional: Clean/Preprocess]
    ↓
SentenceTransformer Embedding (384 dims)
    ↓
[If --use-semantic-features:]
    ↓
Semantic Pattern Score Extraction (5 dims)
    ↓
Concatenate: [384 embeddings, 5 pattern scores] = 389 dims
    ↓
StandardScaler (normalize all features)
    ↓
Model Training/Prediction
```

## Feature Usage by Model

All models in this directory use the same feature extraction process:

### Models Using These Features:
1. **SVM Classifier** (`svm_classifier.py`)
2. **Logistic Ordinal Classifier** (`logistic_ordinal_classifier.py`)
3. **GBM Ordinal Classifier** (`gbm_ordinal_classifier.py`)
4. **Neural Ordinal Classifier** (`neural_ordinal_classifier.py`)
5. **RNN Ordinal Classifier** (`rnn_ordinal_classifier.py`)

### Feature Extraction Method

All models implement the `extract_features()` method:

```python
def extract_features(self, cleaned_text: str) -> np.ndarray:
    # 1. Generate base embedding (384 dims)
    base_features = self.embedding_model.encode(cleaned_text)
    
    # 2. Optionally add semantic pattern scores (5 dims)
    if self.use_semantic_features:
        semantic_scores = self.semantic_extractor.extract_pattern_scores(cleaned_text)
        return np.concatenate([base_features, semantic_scores])
    else:
        return base_features
```

## Feature Scaling

All features are standardized before model training:

- **Method**: `StandardScaler` from scikit-learn
- **Process**: 
  - Compute mean and standard deviation on training data
  - Transform: `(x - mean) / std`
- **Purpose**: Ensure all features are on similar scales for optimal model performance
- **Applied to**: Both embeddings and pattern scores (when enabled)

## Feature Names

### Baseline Features (384 dimensions)
- `embedding_dim_0` through `embedding_dim_383`
- Each represents a learned semantic dimension

### With Semantic Features (389 dimensions)
- `embedding_dim_0` through `embedding_dim_383` (same as above)
- `semantic_pattern_GEO_STRUCT_001`
- `semantic_pattern_GEO_STRUCT_002`
- `semantic_pattern_GEO_STRUCT_003`
- `semantic_pattern_GEO_SEMANTIC_004`
- `semantic_pattern_GEO_SEMANTIC_005`

## Feature Characteristics

### Embedding Features
- **Type**: Continuous values (typically normalized)
- **Distribution**: Learned from large text corpora
- **Interpretability**: Low (high-dimensional learned representations)
- **Information**: Rich semantic content

### Pattern Score Features
- **Type**: Continuous values (cosine similarity, 0.0-1.0)
- **Distribution**: Depends on text similarity to patterns
- **Interpretability**: High (direct similarity to known patterns)
- **Information**: Explicit GEO pattern signals

## Performance Considerations

### Feature Extraction Time
- **Embeddings**: ~5-10ms per text (SentenceTransformer encoding)
- **Pattern Scores**: ~5-10ms per text (additional similarity computations)
- **Total (with features)**: ~10-20ms per text

### Memory Usage
- **Embedding vectors**: 384 × 4 bytes = ~1.5 KB per sample
- **With pattern scores**: 389 × 4 bytes = ~1.6 KB per sample
- **Model storage**: Varies by model type

### Feature Quality
- **Embeddings**: High-quality semantic representations from pre-trained model
- **Pattern Scores**: Explicit signals that may help or hurt depending on model architecture
- **Combination**: Provides both learned and explicit pattern information

## Usage Examples

### Training with Baseline Features
```bash
python src/classification/svm_classifier.py \
    --opt-data optimization_dataset.json \
    --limit 1000 \
    --train-size 300 \
    --validation-size 700
# Uses: 384 embedding features only
```

### Training with Semantic Features
```bash
python src/classification/svm_classifier.py \
    --opt-data optimization_dataset.json \
    --limit 1000 \
    --train-size 300 \
    --validation-size 700 \
    --use-semantic-features
# Uses: 384 embedding features + 5 pattern scores = 389 features
```

## Feature Selection

### When to Use Baseline Features (384 dims)
- ✅ Fast training and prediction
- ✅ Deep neural networks (can learn patterns from embeddings)
- ✅ Tree-based models (may overfit with explicit features)
- ✅ Latency-critical applications

### When to Use Semantic Features (389 dims)
- ✅ Linear models (benefit from explicit pattern signals)
- ✅ Sequential models (RNN, LSTM)
- ✅ Limited-capacity models
- ✅ When interpretability of pattern matches is important

## Implementation Files

- **Feature Extraction**: `src/classification/semantic_features.py`
- **Embedding Model**: SentenceTransformer (`all-MiniLM-L6-v2`)
- **Scaling**: `StandardScaler` from scikit-learn
- **Usage**: All classifier files in `src/classification/`

## Related Documentation

- **Semantic Pattern Details**: See `SEMANTIC_FEATURES.md` for detailed pattern descriptions
- **Performance Comparison**: See `COMPARISON_RESULTS.md` for model performance with/without features
- **Pattern Definitions**: See `src/pattern_recognition/similarity_scores.py` for pattern source

