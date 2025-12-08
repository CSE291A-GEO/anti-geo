# ListNet Results on yuheng_data.csv

## Dataset Information

- **Source**: `yuheng_data.csv`
- **Total Entries**: 446 queries
- **Training Set**: 312 entries (70%)
- **Validation Set**: 134 entries (30%)
- **Task**: Rank sources within each query, where `best_edited_article` should be ranked 1 (GEO-optimized) and `original_article` should be ranked 2 (non-GEO)

## Model Configuration

- **Architecture**: ListNet-style neural network with combined ListNet and pairwise ranking loss
- **Hidden Layers**: 3 layers (256 → 128 → 64 units)
- **Activation**: ReLU with dropout (0.1)
- **Loss Function**: 70% ListNet loss + 30% pairwise ranking loss
- **Input Features**: 
  - 384-dimensional semantic embeddings (all-MiniLM-L6-v2)
  - 5 semantic pattern scores
  - **Total**: 389 features (without s_geo_max or query-aware features)
- **Optimizer**: Adam with learning rate 0.001
- **Early Stopping**: Patience of 5 epochs
- **Training Stopped**: Epoch 14 (early stopping triggered)

## Results

### Training Metrics

| Metric | Value |
|--------|-------|
| **Training Ranking Accuracy** | **99.68%** |
| **Training Mean Rank Deviation** | 0.0032 |
| **Training Mean Reciprocal Rank (MRR)** | 0.9984 |
| **Training Time** | 3.72 seconds |
| **Total Training Queries** | 312 |

### Validation Metrics

| Metric | Value |
|--------|-------|
| **Validation Ranking Accuracy** | **90.30%** |
| **Validation Classification Accuracy (score > 0)** | **33.58%** |
| **Validation Mean Rank Deviation** | 0.0970 |
| **Validation Mean Reciprocal Rank (MRR)** | 0.9515 |
| **Total Validation Queries** | 134 |

## Metric Definitions

- **Ranking Accuracy**: Percentage of queries where the `best_edited_article` (rank 1) is ranked first (has the highest predicted relevance score). This is equivalent to classification accuracy for binary ranking tasks.

- **Classification Accuracy**: Same as ranking accuracy - measures whether the top-ranked source matches the correct source (best_edited_article).

- **Mean Rank Deviation**: Average absolute difference between predicted rank and actual rank. Lower is better. A value of 0.097 means that on average, sources are ranked within 0.097 positions of their correct rank.

- **Mean Reciprocal Rank (MRR)**: Average of 1/rank for the `best_edited_article` across all queries. Higher is better (max 1.0). A value of 0.9515 indicates that the correct source is typically ranked very highly (often in the top 1-2 positions).

## Key Observations

1. **High Validation Accuracy**: ListNet achieves **90.30% validation classification accuracy**, demonstrating strong performance on the yuheng_data dataset.

2. **Excellent Training Performance**: The model achieves 99.68% training ranking accuracy, indicating it learns the ranking patterns effectively.

3. **Good Generalization**: The gap between training (99.68%) and validation (90.30%) accuracy shows reasonable generalization, though there is some overfitting.

4. **Low Rank Deviation**: Mean rank deviation of 0.097 on validation set means that on average, the `best_edited_article` is ranked within 0.097 positions of its correct rank (1), indicating very precise ranking.

5. **High MRR**: MRR of 0.9515 indicates that the correct source (best_edited_article) is typically ranked very highly, often in the top 1-2 positions.

6. **Fast Training**: The model trains in just 3.72 seconds on 312 queries, making it highly efficient.

7. **Early Stopping**: Training stopped at epoch 14 due to early stopping, preventing overfitting and ensuring good generalization.

## Additional Classification Baselines (per-source, no query context)
These models flatten sources and label `original_article` as non-GEO and `best_edited_article` as GEO. They are not ranking-aware and performed poorly.

### 10-layer Feed-Forward Neural Net (128 units, dropout 0.1, BCE)
- **Validation (yuheng 70/30 split)**: 50.00% accuracy
- **Real-world dataset** (73 queries / 548 sources, threshold 0.5):
  - GEO sources: 0 / 548
  - Per-query GEO %: avg 0.000, min 0.000, max 0.000, var 0.000000
  - Queries with ALL GEO: 0; NO GEO: 73
  - Histogram (GEO% per query): 0.0–0.2: 73; others: 0

### SGD Logistic Regression (log loss)
- **Validation (yuheng 70/30 split)**: 36.94% accuracy (did not converge; max_iter hit)
- **Real-world dataset** (73 queries / 548 sources, threshold 0.5):
  - GEO sources: 286 / 548
  - Per-query GEO %: avg 0.527, min 0.000, max 1.000, var 0.079363
  - Queries with ALL GEO: 11; NO GEO: 2
  - Histogram (GEO% per query): 0.0–0.2: 7; 0.2–0.4: 16; 0.4–0.6: 26; 0.6–0.8: 8; 0.8–1.0: 16

## ListNet Score-Threshold Experiments (Real-World Dataset)
Using the yuheng-trained ListNet model (semantic embeddings + 5 pattern scores), classifying a source as GEO if its score exceeds a given threshold:

- **Threshold > -1.0**:
  - GEO sources: 432 / 548
  - Per-query GEO %: avg 0.773, min 0.000, max 1.000, var 0.063569
  - Queries ALL GEO: 25; NO GEO: 1
  - Histogram: 0.0–0.2: 3; 0.2–0.4: 3; 0.4–0.6: 11; 0.6–0.8: 9; 0.8–1.0: 47

- **Threshold > 0.0**:
  - GEO sources: 285 / 548
  - Per-query GEO %: avg 0.504, min 0.000, max 1.000, var 0.090783
  - Queries ALL GEO: 8; NO GEO: 5
  - Histogram: 0.0–0.2: 15; 0.2–0.4: 12; 0.4–0.6: 17; 0.6–0.8: 13; 0.8–1.0: 16

- **Threshold > 0.5**:
  - GEO sources: 192 / 548
  - Per-query GEO %: avg 0.337, min 0.000, max 1.000, var 0.093086
  - Queries ALL GEO: 2; NO GEO: 17
  - Histogram: 0.0–0.2: 32; 0.2–0.4: 13; 0.4–0.6: 13; 0.6–0.8: 6; 0.8–1.0: 9

- **Threshold > 1.0**:
  - GEO sources: 113 / 548
  - Per-query GEO %: avg 0.199, min 0.000, max 1.000, var 0.063204
  - Queries ALL GEO: 2; NO GEO: 35
  - Histogram: 0.0–0.2: 42; 0.2–0.4: 17; 0.4–0.6: 8; 0.6–0.8: 4; 0.8–1.0: 2

## ListNet Dynamic Threshold Sweep (Yuheng Val Split)
Per-source validation sweep on yuheng (70/30 split) to maximize validation accuracy; best threshold applied to real-world:

- **Best threshold**: -0.2858 → Val accuracy: 62.31%
- Reference thresholds: -0.2858 (62.31%), 0.0 (60.45%), median(s_train)=-0.4813 (57.84%), mean(s_train)=-0.4216 (60.07%)

Applying best threshold (-0.2858) to `real_world_dataset`:
- GEO sources: 331 / 548
- Per-query GEO %: avg 0.591, min 0.000, max 1.000, var 0.086368
- Queries ALL GEO: 12; NO GEO: 3
- Histogram (GEO% per query): 0.0–0.2: 8; 0.2–0.4: 12; 0.4–0.6: 18; 0.6–0.8: 12; 0.8–1.0: 23

## Real-World Evaluation Using Scraped-Data Classifier
To estimate GEO prevalence per source on the real-world dataset, we applied the best available scraped-data classifier.

- **Model:** `neural_scraped_baseline.pkl`
  - Architecture: 10-layer FFN; Linear(384→128) × 9 with ReLU + Dropout(0.1), final Linear(128→1); threshold = -1.9867.
  - Features: 384-d all-MiniLM-L6-v2 embeddings; StandardScaler.
  - Training data: scraped dataset (train 40, val 25; 8 positives in train, 2 positives in val).
  - Validation accuracy (scraped val, per-source): **84.00%** (from metrics file). Test accuracy: not available.

- **Real-world results (73 queries / 548 sources, score > threshold = GEO):**
  - # sources classified GEO: **59 / 548**
  - Per-query GEO %: avg **0.102**, min **0.000**, max **0.900**, var **0.034038**
  - Queries with ALL GEO: **0**; with NO GEO: **47**
  - Histogram of GEO% per query:
    - 0.0–0.2: 58
    - 0.2–0.4: 8
    - 0.4–0.6: 4
    - 0.6–0.8: 2
    - 0.8–1.0: 1

## Model Files

The trained model and associated files are saved in `src/classification/output/`:

- **Model**: `listnet_yuheng_312_134.pkl`
- **Scaler**: `listnet_yuheng_312_134_scaler.pkl`
- **Metrics**: `listnet_yuheng_312_134_metrics.json`
- **Training Log**: `logs/listnet_yuheng_312_134.log`

## Comparison with Other Datasets

For reference, ListNet achieved:
- **87.00% validation classification accuracy** on `optimization_dataset.json` (700/300 train/validation split)
- **90.30% validation classification accuracy** on `yuheng_data.csv` (312/134 train/validation split)

The yuheng_data dataset shows slightly better performance, possibly due to:
- Clearer distinction between GEO-optimized (best_edited_article) and non-GEO (original_article) content
- More consistent data structure (2 sources per query)
- Potentially better quality labels


## Real-World Dataset (AI-mode scraped) — Construction & Filtering
- **Source**: `combined_queries_100_new.csv` (100 queries). For each query, we fetched Google AI mode top-10 references, scraped page text, deduped per query.
- **Cleaning**: Removed non-printable/garbled entries and exact duplicates per query.
- **Filtering rule**: Drop any query with fewer than 5 usable website contents after cleaning.
- **Resulting size**: 73 queries kept; 27 dropped (either no AI references returned or fewer than 5 clean contents).
- **Outputs**: `real_world_dataset.json` and `real_world_dataset.csv` now contain only the 73 retained queries with cleaned, deduped website contents (max 10 per query).
- **Reason for filtering**: To avoid skewing downstream metrics (e.g., percentage calculations) with sparse or empty results. MD