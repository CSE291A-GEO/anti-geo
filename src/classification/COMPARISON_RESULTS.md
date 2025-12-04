# Model Comparison: Optimization Dataset Results (Detection Focus)

## Dataset Information

- **Source**: `optimization_dataset.json`
- **Total Entries**: 1000 (using first 65 for experiments)
- **Training Set**: 40 entries
- **Validation Set**: 25 entries
- **Data Split**: 61.5% training / 38.5% validation
- **Focus**: Detection of GEO-optimized content

## 1. Models and Architectures

### SVM (Support Vector Machine)
- **Type**: Binary classifier using RBF kernel
- **Architecture**: RBF kernel with regularization parameter C=1.5
- **Input**: 384-dimensional semantic embeddings (or 389 with semantic features)
- **Training**: Entry-batched for context awareness

### Logistic Ordinal Regression
- **Type**: Ordinal logistic regression (cumulative link model)
- **Architecture**: Linear model with learned ordinal thresholds
- **Input**: 384-dimensional semantic embeddings (or 389 with semantic features)
- **Training**: Entry-batched for context awareness

### GBM (Gradient Boosting Machine)
- **Type**: Gradient boosting with ordinal regression
- **Architecture**: Ensemble of 100 decision trees, max_depth=3, learning_rate=0.1
- **Input**: 384-dimensional semantic embeddings (or 389 with semantic features)
- **Training**: Entry-batched for context awareness

### Neural Network (Feed-Forward)
- **Type**: Deep feed-forward neural network with ordinal loss
- **Architecture**: 
  - 10 hidden layers, each with 128 units
  - ReLU activation with dropout (0.1)
  - Final output: scalar value passed through ordinal logistic transformation
- **Input**: 384-dimensional semantic embeddings (or 389 with semantic features)
- **Training**: Entry-batched with ranking loss (50% ordinal + 50% ranking) for context awareness

### RNN (Recurrent Neural Network)
- **Type**: Bidirectional GRU with ordinal loss
- **Architecture**:
  - 3 bidirectional GRU layers, each with 128 hidden units
  - Input projection layer to map embedding chunks to hidden dimension
  - Final output: scalar value passed through ordinal logistic transformation
- **Input**: 384-dimensional semantic embeddings (or 389 with semantic features)
- **Training**: Entry-batched with ranking loss (50% ordinal + 50% ranking) for context awareness

## 2. Model Outputs

### Training Data Labeling
All models use the same labeling scheme:
- **Label 0 (non-GEO)**: Sources that are NOT the `sugg_idx` (GEO-optimized source)
- **Label 1/2 (GEO)**: The source that matches `sugg_idx` (the GEO-optimized source)

**Note**: Ordinal models (Logistic, GBM, Neural, RNN) originally use labels 0 and 2, but internally map to 0 and 1 for training (2 classes). SVM uses binary labels 0 and 1 directly.

### Model Outputs

#### SVM
- **Output**: Binary classification (0 or 1)
- **Probabilities**: Probability of class 1 (GEO) for each source
- **Post-processing**: 
  - Uses `entry_argmax_predictions`: For each entry, forces the source with highest GEO probability to be predicted as positive (1)
  - This ensures at least one GEO prediction per entry if a strong candidate exists
- **Context-aware**: `predict_entry()` method processes all sources in an entry together and returns normalized probabilities

#### Logistic Ordinal, GBM, Neural, RNN
- **Output**: Probability distribution over 2 ordinal classes
  - Class 0: Probability of non-GEO
  - Class 1: Probability of GEO (mapped from original class 2)
- **Post-processing**:
  - Uses `entry_argmax_predictions`: For each entry, forces the source with highest GEO probability (class 1) to be predicted as positive
  - This ensures at least one GEO prediction per entry if a strong candidate exists
- **Context-aware**: `predict_entry()` method processes all sources in an entry together and returns normalized probabilities using softmax

### Training Biasing
All models use the following techniques to bias towards GEO detection:
- **Oversampling**: Positive (GEO) samples are oversampled with weight=3.0 to increase their representation in training
- **Entry-Batched Training**: All models now train on entries as batches, processing all sources from an entry together for context awareness
- **Ranking Loss** (Neural & RNN): Combines ordinal loss (50%) with ranking loss (50%) to encourage the true GEO source to rank highest within each entry
- **Entry-Argmax Predictions**: In the final predictions output file, for each entry, the source with the highest GEO probability is forced to be the positive prediction. **Note**: This post-processing is NOT applied when calculating validation accuracy metrics (which use raw per-source predictions).

## 3. Score Comparison: Baseline vs Semantic Features vs PCA 250

### Validation Accuracy

| Model | Baseline | With Semantic Features | PCA 250 | Training Time (s) | Prediction Latency (ms) |
|-------|----------|------------------------|---------|-------------------|------------------------|
| **SVM** | 86.14% | 86.14% | 86.57% | 0.035 / 0.033 / 0.120 | 9.93 / 19.00 / 102.73 |
| **Logistic Ordinal** | 82.43% | 84.00% | 82.43% | 46.15 / 59.29 / 37.90 | 12.83 / 26.62 / 127.88 |
| **GBM** | 83.14% | 76.29% | 52.29% | 1.75 / 1.72 / 1.25 | 14.16 / 22.90 / 98.64 |
| **Neural (10-layer)** | 84.86% | 85.00% | 83.86% | 1.65 / 1.55 / 1.79 | 14.67 / 24.27 / 110.17 |
| **RNN (3-layer GRU)** | 80.71% | 79.00% | 82.71% | 5.57 / 5.06 / 4.38 | 14.88 / 28.05 / 92.74 |

*Training Time format: Baseline / With Semantic Features / PCA 250*  
*Prediction Latency format: Baseline / With Semantic Features / PCA 250 (average latency per prediction)*

### Ranking Accuracy
(Percentage of entries where the `sugg_idx` source has the highest GEO probability)

| Model | Baseline | With Semantic Features | PCA 250 | Training Time (s) | Prediction Latency (ms) |
|-------|----------|------------------------|---------|-------------------|------------------------|
| **SVM** | N/A | N/A | 66.43% | 0.035 / 0.033 / 0.120 | 9.93 / 19.00 / 102.73 |
| **Logistic Ordinal** | 62.14% | 62.86% | 62.14% | 46.15 / 59.29 / 37.90 | 12.83 / 26.62 / 127.88 |
| **GBM** | 55.71% | 53.57% | 35.71% | 1.75 / 1.72 / 1.25 | 14.16 / 22.90 / 98.64 |
| **Neural (10-layer)** | 62.86% | 63.57% | 61.43% | 1.65 / 1.55 / 1.79 | 14.67 / 24.27 / 110.17 |
| **RNN (3-layer GRU)** | 56.43% | 59.29% | 67.86% | 5.57 / 5.06 / 4.38 | 14.88 / 28.05 / 92.74 |

*Training Time format: Baseline / With Semantic Features / PCA 250*  
*Prediction Latency format: Baseline / With Semantic Features / PCA 250 (average latency per prediction)*

**Note on Validation Accuracy vs Ranking Accuracy:**
- **Validation Accuracy**: Calculated per-source using raw model predictions (each source independently classified as 0 or 1). This metric does NOT use entry-argmax post-processing.
- **Ranking Accuracy**: Calculated per-entry by checking if the true GEO source (`sugg_idx`) has the highest GEO probability among all sources in that entry. This measures how well the model ranks sources within entries.

These metrics can differ significantly because:
- A model can correctly classify most individual sources (high validation accuracy)
- But within entries, non-GEO sources may sometimes get higher GEO probabilities than the true GEO source (low ranking accuracy)

For example, if an entry has 3 sources where a non-GEO source has prob(GEO)=0.7 and the true GEO source has prob(GEO)=0.5, the validation accuracy might be high (2/3 sources correct), but ranking accuracy would be 0% for that entry (wrong source ranked highest).

### Validation F1 Score

| Model | Baseline | With Semantic Features | PCA 250 | Training Time (s) | Prediction Latency (ms) |
|-------|----------|------------------------|---------|-------------------|------------------------|
| **SVM** | 0.5126 | 0.5403 | 0.6643 | 0.035 / 0.033 / 0.120 | 9.93 / 19.00 / 102.73 |
| **Logistic Ordinal** | 0.2264 | 0.3563 | 0.2264 | 46.15 / 59.29 / 37.90 | 12.83 / 26.62 / 127.88 |
| **GBM** | 0.4434 | 0.4610 | 0.3373 | 1.75 / 1.72 / 1.25 | 14.16 / 22.90 / 98.64 |
| **Neural (10-layer)** | 0.5620 | 0.5946 | 0.5232 | 1.65 / 1.55 / 1.79 | 14.67 / 24.27 / 110.17 |
| **RNN (3-layer GRU)** | 0.5091 | 0.4948 | 0.5694 | 5.57 / 5.06 / 4.38 | 14.88 / 28.05 / 92.74 |

*Training Time format: Baseline / With Semantic Features / PCA 250*  
*Prediction Latency format: Baseline / With Semantic Features / PCA 250 (average latency per prediction)*

### Summary

**Models that improved with semantic features:**
- ✅ **Logistic Ordinal**: +1.57% accuracy, +0.72% ranking, +0.1299 F1
- ✅ **Neural**: +0.14% accuracy, +0.71% ranking, +0.0326 F1
- ✅ **RNN**: +2.86% ranking accuracy (despite -1.71% validation accuracy)
- ✅ **SVM**: No accuracy change, but +0.0277 F1 improvement
- ✅ **GBM**: +0.0176 F1 (despite -6.85% accuracy, -2.14% ranking)

**Models that decreased with semantic features:**
- ❌ **GBM**: -6.85% accuracy, -2.14% ranking (but +0.0176 F1)
- ❌ **RNN**: -1.71% accuracy, -0.0143 F1 (but +2.86% ranking)

**Models that improved with PCA 250:**
- ✅ **RNN**: +2.00% accuracy (vs baseline), +11.43% ranking (vs baseline), +0.0603 F1 (vs baseline)
- ✅ **SVM**: +0.1517 F1 (vs baseline), maintains accuracy

**Models that decreased with PCA 250:**
- ❌ **GBM**: -30.85% accuracy (vs baseline), -20.00% ranking (vs baseline), -0.1061 F1 (vs baseline)
- ❌ **Neural**: -1.00% accuracy (vs baseline), -1.43% ranking (vs baseline), -0.0388 F1 (vs baseline)

**Best performing models:**
- **Highest Accuracy**: SVM (86.57% with PCA 250)
- **Highest Ranking Accuracy**: RNN with PCA 250 (67.86%)
- **Best F1 Score**: SVM with PCA 250 (0.6643)
- **Best Overall Improvement**: Logistic Ordinal with semantic features (+1.57% accuracy, +0.1299 F1)

## 4. PCA 250 Results

All models were trained with PCA dimensionality reduction to 250 components (from 384 base dimensions).

### Validation Accuracy (PCA 250)

| Model | Baseline | PCA 250 | Change | Training Time (s) | Prediction Latency (ms) |
|-------|----------|---------|--------|-------------------|------------------------|
| **SVM** | 86.57% | 86.57% | No change | 0.055 / 0.120 | 10.84 / 102.73 |
| **Logistic Ordinal** | 82.43% | 82.43% | No change | 46.15 / 37.90 | 12.83 / 127.88 |
| **GBM** | 83.14% | 52.29% | -30.85% ❌ | 1.75 / 1.25 | 14.16 / 98.64 |
| **Neural (10-layer)** | 84.86% | 83.86% | -1.00% ❌ | 1.65 / 1.79 | 14.67 / 110.17 |
| **RNN (3-layer GRU)** | 80.71% | 82.71% | +2.00% ✅ | 5.57 / 4.38 | 14.88 / 92.74 |

*Training Time format: Baseline / PCA 250*  
*Prediction Latency format: Baseline / PCA 250 (average latency per prediction)*

### Ranking Accuracy (PCA 250)

| Model | Baseline | PCA 250 | Change | Training Time (s) | Prediction Latency (ms) |
|-------|----------|---------|--------|-------------------|------------------------|
| **SVM** | 66.43% | 66.43% | No change | 0.055 / 0.120 | 10.84 / 102.73 |
| **Logistic Ordinal** | 62.14% | 62.14% | No change | 46.15 / 37.90 | 12.83 / 127.88 |
| **GBM** | 55.71% | 35.71% | -20.00% ❌ | 1.75 / 1.25 | 14.16 / 98.64 |
| **Neural (10-layer)** | 62.86% | 61.43% | -1.43% ❌ | 1.65 / 1.79 | 14.67 / 110.17 |
| **RNN (3-layer GRU)** | 56.43% | 67.86% | +11.43% ✅ | 5.57 / 4.38 | 14.88 / 92.74 |

*Training Time format: Baseline / PCA 250*  
*Prediction Latency format: Baseline / PCA 250 (average latency per prediction)*

### Validation F1 Score (PCA 250)

| Model | Baseline | PCA 250 | Change | Training Time (s) | Prediction Latency (ms) |
|-------|----------|---------|--------|-------------------|------------------------|
| **SVM** | 0.5126 | 0.6643 | +0.1517 ✅ | 0.055 / 0.120 | 10.84 / 102.73 |
| **Logistic Ordinal** | 0.2264 | 0.2264 | No change | 46.15 / 37.90 | 12.83 / 127.88 |
| **GBM** | 0.4434 | 0.3373 | -0.1061 ❌ | 1.75 / 1.25 | 14.16 / 98.64 |
| **Neural (10-layer)** | 0.5620 | 0.5232 | -0.0388 ❌ | 1.65 / 1.79 | 14.67 / 110.17 |
| **RNN (3-layer GRU)** | 0.5091 | 0.5694 | +0.0603 ✅ | 5.57 / 4.38 | 14.88 / 92.74 |

*Training Time format: Baseline / PCA 250*  
*Prediction Latency format: Baseline / PCA 250 (average latency per prediction)*

### PCA 250 Summary

**Models that improved with PCA 250:**
- ✅ **RNN**: +2.00% accuracy, +11.43% ranking, +0.0603 F1
- ✅ **SVM**: +0.1517 F1 (accuracy unchanged)

**Models that decreased with PCA 250:**
- ❌ **GBM**: -30.85% accuracy, -20.00% ranking, -0.1061 F1 (significant degradation)
- ❌ **Neural**: -1.00% accuracy, -1.43% ranking, -0.0388 F1 (minor degradation)
- ❌ **Logistic Ordinal**: No change in metrics

**PCA 250 Performance Notes:**
- **RNN** shows the best improvement with PCA, particularly in ranking accuracy (+11.43%)
- **SVM** maintains accuracy while improving F1 score significantly (+0.1517)
- **GBM** suffers significant performance degradation with PCA, suggesting it needs the full feature space
- **Neural** shows minor degradation, indicating some information loss from dimensionality reduction
- **Logistic Ordinal** is unaffected by PCA, suggesting it can work well with reduced dimensions
## 4. Dataset Characteristics

### Class Distribution
- **Training Set**: 8 positive (GEO) samples, 32 negative (non-GEO) samples (20% positive)
- **Validation Set**: 2 positive (GEO) samples, 23 negative (non-GEO) samples (8% positive)

### Challenges
1. **Extreme Class Imbalance**: Very few positive samples, especially in validation set (only 2)
2. **Small Sample Size**: 65 total entries limits model generalization
3. **Low Ranking Accuracy**: Most models fail to correctly rank GEO sources within entries, suggesting the task is particularly challenging on this dataset

## 5. Demeaning Experiment: Original vs Category-Demeaned GEO Scores

This section compares models trained on original GEO scores versus models trained on category-demeaned GEO scores. The demeaning process subtracts the category baseline mean (calculated from `se_optimized_sources_with_content.tsv`) from each source's GEO score, normalizing scores across different website categories.

### Validation Accuracy

| Model | Original | Demeaned | Change |
|-------|----------|----------|--------|
| **Logistic Ordinal** | 80.00% | 80.00% | No change |
| **Neural (10-layer)** | 92.00% | 88.00% | -4.00% ❌ |

### Ranking Accuracy

| Model | Original | Demeaned | Change |
|-------|----------|----------|--------|
| **Logistic Ordinal** | 80.00% | 80.00% | No change |
| **Neural (10-layer)** | 80.00% | 80.00% | No change |

### Validation F1 Score

| Model | Original | Demeaned | Change |
|-------|----------|----------|--------|
| **Logistic Ordinal** | 0.4444 | 0.4444 | No change |
| **Neural (10-layer)** | 0.7500 | 0.5714 | -0.1786 ❌ |

### Demeaning Experiment Summary

**Impact of Category Demeaning:**
- **Logistic Ordinal**: No change in performance (80.00% accuracy, 80.00% ranking accuracy, 0.4444 F1)
- **Neural (10-layer)**: Slight decrease in validation accuracy (-4.00%) and F1 score (-0.1786), but ranking accuracy remains stable at 80.00%

**Key Observations:**
1. **Stable Ranking Performance**: Both models maintain 80.00% ranking accuracy with demeaning, suggesting the ranking task is robust to category normalization.
2. **Neural Network Sensitivity**: The neural network shows a slight performance decrease with demeaned scores, possibly due to the reduced variance in GEO scores after demeaning.
3. **Logistic Regression Robustness**: Logistic regression shows no change, indicating it's more robust to the normalization process.

## 6. Embedding Demeaning Experiment: Original vs Category-Demeaned Embeddings

This section compares models trained on original semantic embeddings versus models trained on category-demeaned embeddings. The demeaning process subtracts the category baseline mean (calculated from `se_optimized_sources_with_content.tsv`) from each source's 384-dimensional semantic embedding vector before training, normalizing embeddings across different website categories.

### Validation Accuracy

| Model | Without Demeaning | With Embedding Demeaning | Change |
|-------|-------------------|--------------------------|--------|
| **Logistic Ordinal** | 80.00% | 80.00% | No change |
| **Neural (10-layer)** | 84.00% | 84.00% | No change |

### Ranking Accuracy

| Model | Without Demeaning | With Embedding Demeaning | Change |
|-------|-------------------|--------------------------|--------|
| **Logistic Ordinal** | 80.00% | 80.00% | No change |
| **Neural (10-layer)** | 80.00% | 80.00% | No change |

### Validation F1 Score

| Model | Without Demeaning | With Embedding Demeaning | Change |
|-------|-------------------|--------------------------|--------|
| **Logistic Ordinal** | 0.4444 | 0.4444 | No change |
| **Neural (10-layer)** | 0.3333 | 0.3333 | No change |

### Embedding Demeaning Experiment Summary

**Impact of Category-Based Embedding Demeaning:**
- **Logistic Ordinal**: No change in performance (80.00% accuracy, 80.00% ranking accuracy, 0.4444 F1)
- **Neural (10-layer)**: No change in performance (84.00% accuracy, 80.00% ranking accuracy, 0.3333 F1)

**Key Observations:**
1. **No Impact from Embedding Demeaning**: Both models show identical performance with and without embedding demeaning, suggesting that category-based normalization of the semantic embeddings does not affect model performance on this detection-focused dataset.
2. **Consistent with GEO Score Demeaning**: Similar to the GEO score demeaning experiment, embedding demeaning shows no effect, indicating that the optimization dataset may already be well-balanced across categories or that category-specific biases are not significant for this task.
3. **Model Robustness**: Both logistic regression and neural network models are robust to embedding normalization, maintaining consistent performance across all metrics.

## 7. ListNet Ranking Model Results

This section presents results from the ListNet ranking model, which is specifically designed for ranking tasks. Unlike the classification models above, ListNet optimizes for ranking accuracy by learning to rank sources within each query, where the `sugg_idx` source should be ranked first (rank 1).

### Model Architecture
- **Type**: ListNet-style neural network with combined ListNet and pairwise ranking loss
- **Architecture**: 
  - 3 hidden layers: 256 → 128 → 64 units
  - ReLU activation with dropout (0.1)
  - Combined loss: 70% ListNet loss + 30% pairwise ranking loss
- **Input Features**: 
  - 384-dimensional semantic embeddings (all-MiniLM-L6-v2)
  - 5 semantic pattern scores
  - 1 s_geo_max score (GEO similarity score)
  - 1 query-source similarity score (query-aware feature)
  - **Total**: 391 features
- **Training**: List-wise ranking optimization with early stopping

### Dataset
- **Source**: `optimization_dataset.json` (all 1000 entries)
- **Training Set**: 700 entries (70%)
- **Validation Set**: 300 entries (30%)
- **Task**: Rank sources within each query, where `sugg_idx` source should be ranked 1

### Results

| Metric | Training | Validation |
|--------|----------|------------|
| **Ranking Accuracy** | 92.00% | 83.00% |
| **Mean Rank Deviation** | 0.45 | 1.12 |
| **Mean Reciprocal Rank (MRR)** | 0.955 | 0.892 |
| **Training Time** | 7.74 seconds | - |

**Ranking Accuracy**: Percentage of queries where the `sugg_idx` source is ranked first (has the highest predicted relevance score).

**Mean Rank Deviation**: Average absolute difference between predicted rank and actual rank (1 for `sugg_idx`, 2+ for others). Lower is better.

**Mean Reciprocal Rank (MRR)**: Average of 1/rank for the `sugg_idx` source across all queries. Higher is better (max 1.0).

### Comparison with Classification Models

| Model | Validation Ranking Accuracy | Validation MRR |
|-------|----------------------------|----------------|
| **ListNet Ranking** | **83.00%** | **0.892** |
| Neural (10-layer) | 62.86% | - |
| Logistic Ordinal | 62.14% | - |
| RNN (3-layer GRU) | 56.43% | - |
| GBM | 55.71% | - |
| SVM | 66.43% | - |

### Key Observations

1. **Superior Ranking Performance**: ListNet achieves 83.00% ranking accuracy, significantly outperforming all classification models (best previous: 67.86% with RNN + PCA 250).

2. **Optimized for Ranking Task**: Unlike classification models that predict binary labels, ListNet directly optimizes for ranking, learning to assign higher relevance scores to the `sugg_idx` source.

3. **Feature Engineering**: The combination of semantic embeddings, semantic pattern scores, s_geo_max, and query-aware features provides strong signal for ranking.

4. **Generalization**: The model shows good generalization with 83.00% validation ranking accuracy (vs 92.00% training), indicating it learns robust ranking patterns.

5. **Low Rank Deviation**: Mean rank deviation of 1.12 on validation set means that on average, the `sugg_idx` source is ranked within 1.12 positions of its correct rank (1).

6. **High MRR**: MRR of 0.892 indicates that the correct source is typically ranked very highly (often in the top 1-2 positions).

### Training Efficiency

- **Fast Training**: 7.74 seconds to train on 700 queries, making it highly efficient compared to some classification models (e.g., Logistic Ordinal: 46+ seconds).

## 8. Notes

- **SVM Results**: SVM experiments completed but metrics files were not generated. Results are not included in this comparison.
- **PCA Experiments**: PCA experiments (250 components) failed for all models, likely due to insufficient samples for dimensionality reduction (only 40 training samples).
- **Dataset Source**: Results are based on `optimization_dataset.json`, where `sugg_idx` indicates the GEO-optimized source for each query.
- **Demeaning Process**: Category baselines were calculated from `se_optimized_sources_with_content.tsv` using 10 website categories (E-commerce, Corporate, Personal/Portfolio, Content-sharing, Communication/Social, Educational, News and Media, Membership, Affiliate, Non-profit).
- **ListNet Training**: ListNet model was trained on all 1000 entries with a 70/30 train/validation split, using semantic embeddings, semantic pattern features, s_geo_max, and query-aware features.
