# Model Comparison: Combined GEO Dataset Results

## Dataset Information

- **Source**: Combined dataset from `scraped_data.jsonl` (labeled as GEO) and `se_optimized_sources_with_content.tsv` (labeled as non-GEO)
- **Total Entries**: 65
- **Training Set**: 45 entries
- **Validation Set**: 20 entries
- **Data Split**: 69.2% training / 30.8% validation
- **Labeling Strategy**: Sources from `scraped_data.jsonl` are labeled as GEO (1), sources from `se_optimized_sources_with_content.tsv` are labeled as non-GEO (0)
- **Feature Set**: 384-dimensional semantic embeddings only (no optional 5 GEO features)

## 1. Models and Architectures

### SVM (Support Vector Machine)
- **Type**: Binary classifier using RBF kernel
- **Architecture**: RBF kernel with regularization parameter C=1.5
- **Input**: 384-dimensional semantic embeddings
- **Training**: Entry-batched for context awareness

### Logistic Ordinal Regression
- **Type**: Ordinal logistic regression (cumulative link model)
- **Architecture**: Linear model with learned ordinal thresholds
- **Input**: 384-dimensional semantic embeddings
- **Training**: Entry-batched for context awareness

### Neural Network (Feed-Forward)
- **Type**: Deep feed-forward neural network with ordinal loss
- **Architecture**: 
  - 4 hidden layers, each with 128 units
  - ReLU activation with dropout (0.1)
  - Final output: scalar value passed through ordinal logistic transformation
- **Input**: 384-dimensional semantic embeddings
- **Training**: Entry-batched with ranking loss for context awareness

### ListNet Ranking Model
- **Type**: ListNet-style learning-to-rank neural network
- **Architecture**:
  - 3 hidden layers: 256, 128, 64 units
  - ReLU activation
  - Final output: scalar ranking score
- **Input**: 389-dimensional feature vector (384 embeddings + 5 semantic pattern scores)
- **Training**: ListNet loss (70% ListNet + 30% pairwise loss) with early stopping
- **Note**: Trained without GEO score features and without query-aware features

## 2. Model Outputs

### Training Data Labeling
All classification models use the same labeling scheme:
- **Label 0 (non-GEO)**: Sources from `se_optimized_sources_with_content.tsv`
- **Label 1 (GEO)**: Sources from `scraped_data.jsonl`

### Model Outputs

#### SVM, Logistic Ordinal, Neural Network
- **Output**: Binary classification (0 or 1) or probability distribution over classes
- **Post-processing**: 
  - Uses `entry_argmax_predictions`: For each entry, forces the source with highest GEO probability to be predicted as positive (1)
  - This ensures at least one GEO prediction per entry if a strong candidate exists
- **Context-aware**: `predict_entry()` method processes all sources in an entry together and returns normalized probabilities

#### ListNet
- **Output**: Scalar ranking score for each source
- **Post-processing**: Sources are ranked by their scores within each query/entry
- **Context-aware**: Processes all sources in an entry together to produce relative rankings

## 3. Classification Results

### Validation Accuracy

| Model | Train Accuracy | Validation Accuracy | Train Precision | Train Recall | Train F1 | Validation Precision | Validation Recall | Validation F1 |
|-------|----------------|---------------------|-----------------|--------------|----------|---------------------|-------------------|---------------|
| **SVM** | 84.31% | 75.00% | 0.6667 | 0.2222 | 0.3333 | 0.0000 | 0.0000 | 0.0000 |
| **Logistic Ordinal** | 97.78% | 90.00% | 1.0000 | 0.6667 | 0.8000 | 0.0000 | 0.0000 | 0.0000 |
| **Neural Network** | 97.78% | 90.00% | 0.7500 | 1.0000 | 0.8571 | 0.0000 | 0.0000 | 0.0000 |

### Ranking Accuracy
(Percentage of entries where the GEO source has the highest GEO probability/score)

| Model | Train Ranking Accuracy | Validation Ranking Accuracy |
|-------|------------------------|----------------------------|
| **SVM** | 66.67% | 0.00% |
| **Logistic Ordinal** | 100.00% | 0.00% |
| **Neural Network** | 100.00% | 0.00% |

### Class Distribution

| Split | Positive (GEO) | Negative (non-GEO) | Total |
|-------|----------------|-------------------|-------|
| **Training** | 9 | 42 | 51 |
| **Validation** | 2 | 18 | 20 |

**Note on Validation Metrics:**
- All models achieve 0.00% validation precision, recall, and F1 for the GEO class, indicating they fail to predict any positive samples in the validation set
- This is likely due to extreme class imbalance (only 2 positive samples in validation set of 20)
- Validation accuracy is high (75-90%) because models correctly predict most negative samples, but fail to identify the 2 positive samples

## 4. Ranking Results (ListNet)

### Ranking Scheme
- **Rank 1 (GEO)**: All sources from `scraped_data.jsonl` (identified by `se_rank = -1` or `label = 1`)
- **Rank 2 (non-GEO)**: All sources from `se_optimized_sources_with_content.tsv` (identified by `ge_rank = -1` or `label = 0`)
- **Ranking Accuracy**: Percentage of queries where all GEO sources (rank 1) are ranked above all non-GEO sources (rank 2)

### Ranking Metrics

| Metric | Training | Validation |
|--------|----------|------------|
| **Ranking Accuracy** | 26.67% | 25.00% |
| **Mean Rank Deviation** | 9.13 | 8.52 |
| **Mean Reciprocal Rank (MRR)** | 0.4297 | 0.4540 |

### Training Details
- **Training Time**: 1.96 seconds
- **Total Train Queries**: 45
- **Total Validation Queries**: 20
- **Early Stopping**: Stopped at epoch 17 (out of 50 max epochs)
- **GEO Sources**: 548 sources assigned rank 1
- **Non-GEO Sources**: 511 sources assigned rank 2

**Note on Ranking Metrics:**
- **Ranking Accuracy**: Percentage of queries where all GEO sources (rank 1) are correctly ranked above all non-GEO sources (rank 2)
- **Mean Rank Deviation**: Average absolute difference between predicted rank and true rank (lower is better). Higher values expected with binary ranking (1 vs 2) compared to granular ranking.
- **Mean Reciprocal Rank (MRR)**: Average of 1/rank for the first relevant (GEO) item (higher is better, max 1.0)

## 5. Summary and Key Observations

### Classification Models

**Best Performing Models:**
- **Highest Validation Accuracy**: Logistic Ordinal and Neural Network (tied at 90.00%)
- **Best Training Performance**: Logistic Ordinal and Neural Network (tied at 97.78% train accuracy)
- **Best Ranking on Training Set**: Logistic Ordinal and Neural Network (tied at 100.00% train ranking accuracy)

**Key Challenges:**
1. **Extreme Class Imbalance**: Only 2 positive (GEO) samples in validation set of 20 (10% positive rate)
2. **Validation F1 Collapse**: All models achieve 0.00% validation F1, indicating they fail to predict any positive samples in the validation set
3. **Ranking Accuracy Gap**: While training ranking accuracy is high (66.67-100%), validation ranking accuracy is 0.00% for all classification models
4. **Small Dataset**: With only 65 total entries, models have limited data for generalization

**Model-Specific Observations:**
- **SVM**: Lower training accuracy (84.31%) but more conservative predictions, resulting in 0 positive predictions on validation set
- **Logistic Ordinal**: Perfect training ranking accuracy (100%) but fails on validation ranking (0%)
- **Neural Network**: Similar performance to Logistic Ordinal, with perfect training ranking but 0% validation ranking

### Ranking Model (ListNet)

**Performance:**
- **Validation Ranking Accuracy**: 25.00% (best among all models for binary ranking task)
- **Validation MRR**: 0.4540 (indicates GEO sources are often ranked in top positions on average)
- **Mean Rank Deviation**: 8.52 (with binary ranking scheme where GEO=1 and non-GEO=2)

**Key Observations:**
1. **Binary Ranking Task**: With the new ranking scheme (all GEO sources = rank 1, all non-GEO = rank 2), ranking accuracy measures whether all GEO sources are correctly ranked above all non-GEO sources within each query
2. **Ranking Performance**: ListNet achieves 25% validation ranking accuracy, significantly better than classification models (0%) for this binary ranking task
3. **Generalization**: Validation MRR (0.4540) is slightly higher than training MRR (0.4297), suggesting reasonable generalization
4. **Ranking Task Suitability**: ListNet is specifically designed for ranking tasks and shows better performance than classification models adapted for ranking
5. **Task Difficulty**: The binary ranking task (all GEO above all non-GEO) is more challenging than single-source ranking, explaining the lower accuracy compared to the previous granular ranking scheme

### Overall Conclusions

1. **Classification vs Ranking**: Classification models (SVM, Logistic, Neural) achieve high validation accuracy (75-90%) but fail at ranking (0% validation ranking accuracy). ListNet, designed for ranking, achieves 25% validation ranking accuracy with the binary ranking scheme (all GEO sources = rank 1, all non-GEO = rank 2).

2. **Class Imbalance Impact**: The extreme class imbalance (only 2 positive samples in validation) causes all classification models to fail to predict any positive samples, resulting in 0.00% F1 scores despite high overall accuracy.

3. **Training vs Validation Gap**: Large gap between training and validation ranking accuracy for classification models suggests overfitting, while ListNet shows better generalization.

4. **Dataset Size**: With only 65 entries, models face significant challenges. Results should be interpreted with caution due to the small sample size.

5. **Feature Set**: Using only 384-dimensional semantic embeddings (without optional 5 GEO features) provides a baseline for comparison. Future work could explore adding GEO-specific features.

## 6. Notes

- **Dataset Source**: Combined dataset created by matching queries between `scraped_data.jsonl` and `se_optimized_sources_with_content.tsv`
- **Ranking Scheme**: All GEO sources (from `scraped_data.jsonl`) are assigned rank 1, all non-GEO sources (from `se_optimized_sources_with_content.tsv`) are assigned rank 2
- **Text Truncation**: Source text fields were truncated to 10,000 characters to reduce file size and improve loading speed
- **Feature Engineering**: All models use 384-dimensional semantic embeddings generated by `all-MiniLM-L6-v2` SentenceTransformer
- **Training Configuration**: 
  - SVM: C=1.5, RBF kernel
  - Logistic: C=1.0 regularization
  - Neural: 4 layers, 128 units each, 50 epochs, learning rate 0.001
  - ListNet: 3 layers (256-128-64), 50 epochs max, learning rate 0.001, early stopping enabled
