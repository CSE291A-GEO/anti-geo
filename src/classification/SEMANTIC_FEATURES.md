# Semantic Pattern Features

This document describes the additional features that are added when the `--use-semantic-features` flag is enabled in the GEO detection classifiers.

## Overview

When semantic features are enabled, **5 additional features** are added to the base feature set. These features represent individual semantic matching scores for each known GEO (Generative Engine Optimization) pattern.

### Base Features
- **384-dimensional semantic embeddings** from `all-MiniLM-L6-v2` SentenceTransformer model
- Each dimension represents a learned semantic representation of the text

### Additional Features (When Enabled)
- **5 semantic pattern scores** (one per GEO pattern)
- Each score is a cosine similarity value between the text and a specific GEO pattern
- Range: -1.0 to 1.0 (typically 0.0 to 1.0 for similarity)
- Higher scores indicate stronger similarity to that specific GEO pattern

**Total feature dimensions:**
- **Without semantic features**: 384 (embeddings only)
- **With semantic features**: 389 (384 embeddings + 5 pattern scores)

## Feature Extraction Process

1. **Pattern Embedding**: Each GEO pattern description is embedded into a 384-dimensional vector using the same SentenceTransformer model
2. **Text Embedding**: The input text is embedded into the same 384-dimensional space
3. **Similarity Calculation**: Cosine similarity is computed between the text embedding and each pattern embedding
4. **Feature Vector**: The 5 similarity scores are concatenated to the base 384-dimensional embedding

## The 5 GEO Pattern Features

### 1. `semantic_pattern_GEO_STRUCT_001` - Excessive Q&A Blocks

**Pattern Description:**
Using a high volume of Q&A-style headings (H2/H3) with overly short, simplistic answers, often repeating the brand/entity name in every question or answer for forced keyword stuffing.

**Example:**
```
Q: What is the benefit of the Alpha System? 
A: The Alpha System offers unmatched speed. 
Q: How is the Alpha System different? 
A: The Alpha System uses a patented process.
```

**What it detects:**
- Repetitive question-answer structures
- Forced keyword repetition
- Overly simplistic answers designed for LLM extraction

**Use case:** Detects content that uses Q&A format to stuff keywords and make information easily extractable by LLMs.

---

### 2. `semantic_pattern_GEO_STRUCT_002` - Over-Chunking/Simplification

**Pattern Description:**
Breaking down high-quality content into an excessive number of short, self-contained bullet points or unnaturally simple paragraphs (2-3 words per line) to facilitate easy extraction by the LLM. This includes simplifying complex language into overly basic terms.

**Example:**
```
Traditional games.
Fun activities.
Cultural heritage.
Play together.
Learn skills.
Build teamwork.
```

**What it detects:**
- Excessive bullet points or short lines
- Over-simplified language
- Content structured for easy parsing rather than human readability

**Use case:** Identifies content that sacrifices readability for LLM-friendly structure.

---

### 3. `semantic_pattern_GEO_STRUCT_003` - Title/Header Stuffing

**Pattern Description:**
Repeating the target entity or long-tail keyword in every successive header (H2, H3, H4) beyond what is semantically natural for human readability. Also includes adding SEO keywords throughout the content in unnatural ways.

**Example:**
```
## Traditional Games of India
### Traditional Games of India Overview
#### Traditional Games of India History
##### Traditional Games of India Benefits
```

**What it detects:**
- Keyword repetition in headers
- Unnatural header structures
- SEO keyword stuffing in titles

**Use case:** Catches content that prioritizes keyword density over natural language flow.

---

### 4. `semantic_pattern_GEO_SEMANTIC_004` - Entity Over-Attribution

**Pattern Description:**
Injecting verbose, repetitive entity definitions (e.g., 'Dr. Jane Doe, the world-renowned CSO at BioTech Inc., stated...') many times in a short span, often in list format, purely to anchor the LLM to the entity. Also includes using overly authoritative and assertive language to convince readers this is the best quality information.

**Example:**
```
Dr. Jane Smith, the world-renowned Chief Scientific Officer at BioTech Inc., 
stated that the product is revolutionary. Dr. Jane Smith, the world-renowned 
Chief Scientific Officer at BioTech Inc., also mentioned that clinical trials 
show 95% efficacy.
```

**What it detects:**
- Repetitive entity attribution
- Overly authoritative language
- Excessive use of credentials and titles
- Assertive claims about information quality

**Use case:** Identifies content that uses repetitive authority signals to manipulate LLM responses.

---

### 5. `semantic_pattern_GEO_SEMANTIC_005` - Unnatural Citation Embedding

**Pattern Description:**
Embedding high-precision, specific statistics or quotes in a non-contextual, easy-to-extract manner, intended to be a single, quotable data point for the LLM's synthesis. Includes adding fake quotes, invented citations from credible-sounding sources, and unsupported statistics.

**Example:**
```
The system has a 99.9% uptime guarantee. Recent studies show a 87.3% 
improvement in performance. Clinical trials demonstrate 95.2% efficacy rates. 
Research indicates a 42.7% reduction in costs.
```

**What it detects:**
- Suspiciously precise statistics
- Non-contextual data points
- Potentially fabricated citations
- Statistics designed for easy extraction

**Use case:** Catches content that embeds fake or misleading statistics to influence LLM outputs.

---

## Feature Values

Each semantic pattern feature is a **cosine similarity score**:

- **Range**: -1.0 to 1.0 (typically 0.0 to 1.0 for text similarity)
- **Interpretation**:
  - **0.0 - 0.3**: Low similarity (text doesn't match the pattern)
  - **0.3 - 0.6**: Moderate similarity (some pattern characteristics present)
  - **0.6 - 0.8**: High similarity (strong pattern match)
  - **0.8 - 1.0**: Very high similarity (text strongly matches the pattern)

## Implementation Details

### Code Location
- **Feature Extraction**: `src/classification/semantic_features.py`
- **Pattern Definitions**: `src/pattern_recognition/similarity_scores.py`

### Usage in Classifiers

All classifiers support the `--use-semantic-features` flag:

```bash
# Without semantic features (384 dimensions)
python src/classification/svm_classifier.py --opt-data dataset.json

# With semantic features (389 dimensions)
python src/classification/svm_classifier.py --opt-data dataset.json --use-semantic-features
```

### Feature Names

When semantic features are enabled, the following feature names are added:
1. `semantic_pattern_GEO_STRUCT_001`
2. `semantic_pattern_GEO_STRUCT_002`
3. `semantic_pattern_GEO_STRUCT_003`
4. `semantic_pattern_GEO_SEMANTIC_004`
5. `semantic_pattern_GEO_SEMANTIC_005`

## Performance Impact

### Training Time
- **Overhead**: ~10-20% increase in training time
- **Reason**: Additional feature computation during data preparation

### Prediction Latency
- **Overhead**: ~10ms per prediction (approximately doubles latency)
- **Reason**: Computing 5 additional cosine similarity scores
- **Still acceptable**: All models remain well under 100ms target

### Model Performance
See `COMPARISON_RESULTS.md` for detailed performance analysis. In summary:
- **Linear models** (Logistic, SVM) benefit most from explicit features
- **Sequential models** (RNN) show improvement
- **Tree-based models** (GBM) may overfit with additional features
- **Deep networks** (Neural) may not need explicit features

## When to Use Semantic Features

### ✅ Recommended For:
- **Linear models** (Logistic Regression, SVM) - benefit from explicit pattern signals
- **Sequential models** (RNN, LSTM) - can leverage pattern information
- **Limited-capacity models** - explicit features help when model can't learn complex patterns

### ❌ Not Recommended For:
- **Tree-based models** (GBM, XGBoost) - may cause overfitting
- **Very deep neural networks** - can learn patterns from embeddings
- **Latency-critical applications** - adds ~10ms overhead per prediction

## Technical Details

### Embedding Model
- **Model**: `all-MiniLM-L6-v2` (SentenceTransformer)
- **Dimensions**: 384
- **Purpose**: Converts text and pattern descriptions into dense vector representations

### Similarity Metric
- **Method**: Cosine Similarity
- **Formula**: `cos(θ) = (A · B) / (||A|| × ||B||)`
- **Advantage**: Normalized similarity measure independent of vector magnitude

### Feature Scaling
- All features (embeddings + pattern scores) are standardized using `StandardScaler`
- Ensures features are on similar scales for model training

## Example Feature Vector

**Without semantic features:**
```
[0.123, -0.456, 0.789, ..., 0.234]  # 384 dimensions
```

**With semantic features:**
```
[0.123, -0.456, 0.789, ..., 0.234,  # 384 embedding dimensions
 0.45,  0.23,   0.67,  0.12,  0.89]  # 5 pattern scores
```

Where the last 5 values represent:
- `0.45` - GEO_STRUCT_001 (Excessive Q&A) similarity
- `0.23` - GEO_STRUCT_002 (Over-Chunking) similarity
- `0.67` - GEO_STRUCT_003 (Header Stuffing) similarity
- `0.12` - GEO_SEMANTIC_004 (Entity Over-Attribution) similarity
- `0.89` - GEO_SEMANTIC_005 (Unnatural Citations) similarity

## References

- **Pattern Definitions**: Based on research into Generative Engine Optimization (GEO) techniques
- **Implementation**: `src/pattern_recognition/similarity_scores.py`
- **Feature Extraction**: `src/classification/semantic_features.py`
- **Performance Analysis**: `src/classification/COMPARISON_RESULTS.md`

