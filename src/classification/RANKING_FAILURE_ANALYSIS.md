# Ranking Accuracy Failure Analysis

## Executive Summary

This analysis investigates why **ranking accuracy (62.9%) is significantly lower than classification accuracy (84.7%)** - a gap of **21.8 percentage points**. The findings reveal critical patterns that explain the ranking failures and provide actionable insights for model improvement.

## Key Findings

### 1. **The Core Problem: Extreme Under-Confidence in True GEO Sources**

In failed ranking cases, the model assigns **extremely low probabilities** to true GEO sources:

- **Failed cases**: Mean GEO probability = **0.0793**, Median = **0.0003** (essentially zero!)
- **Successful cases**: Mean GEO probability = **0.7946**, Median = **0.9997**

**This is a 10x difference in mean probabilities!**

### 2. **Failure Pattern Distribution**

Out of 52 failed cases (37.1% of validation set):

- **84.6%** have GEO probability < 0.1
- **71.2%** have GEO probability < 0.01 (extreme failures)
- **96.2%** have negative probability gaps (GEO prob < max non-GEO prob)
- **100%** have gaps < 0.1 (all failures are "close" in the sense that GEO prob is very low)

### 3. **The Semantic Feature Paradox** ⚠️

**Critical Discovery**: In failed cases, the **winning non-GEO competitors have HIGHER semantic feature scores** than the true GEO sources across ALL 5 features:

| Feature | True GEO Mean | Competitor Mean | Difference |
|---------|---------------|-----------------|------------|
| **QA_Blocks** | 0.0387 | **0.0827** | +0.0440 ⚠️ |
| **Over-Chunking** | 0.0244 | **0.0567** | +0.0323 ⚠️ |
| **Header_Stuffing** | 0.0209 | **0.0503** | +0.0295 ⚠️ |
| **Entity_Attribution** | 0.0302 | **0.0673** | +0.0371 ⚠️ |
| **Citation_Embedding** | 0.0132 | **0.0466** | +0.0334 ⚠️ |

**This means the model is being fooled by non-GEO sources that exhibit stronger GEO-like patterns!**

## Root Cause Analysis

### Why Ranking Fails

1. **False Positive Problem**: Non-GEO sources with high semantic feature scores (e.g., naturally occurring Q&A blocks, well-structured headers) are being misclassified as GEO-optimized.

2. **True GEO Under-Detection**: True GEO sources with subtle optimization patterns are receiving very low probabilities, possibly because:
   - They don't trigger semantic features strongly enough
   - The model has learned to rely too heavily on semantic features
   - There's a mismatch between training data patterns and validation patterns

3. **Relative Ranking Issue**: Even when both sources have low probabilities, the model ranks the wrong one higher because it has slightly higher semantic feature scores.

### The Classification vs Ranking Gap Explained

**Classification Accuracy (84.7%)** measures:
- Per-source binary classification (GEO or not)
- Uses a threshold (typically 0.5)
- Can be correct even if probabilities are close

**Ranking Accuracy (62.9%)** measures:
- Whether the true GEO source has the **highest** probability in its entry
- Requires **relative** correctness, not just absolute
- Fails when any non-GEO source has higher probability

**The gap exists because:**
- Many true GEO sources are correctly classified as GEO (prob > 0.5) → contributes to classification accuracy
- But within their entries, non-GEO sources sometimes have even higher probabilities → ranking fails
- In extreme cases (71.2% of failures), true GEO sources get prob < 0.01, making ranking impossible

## Detailed Statistics

### Probability Distributions

| Metric | Successful Cases | Failed Cases | Difference |
|--------|------------------|--------------|------------|
| **GEO Probability Mean** | 0.7946 | 0.0793 | -0.7153 |
| **GEO Probability Median** | 0.9997 | 0.0003 | -0.9994 |
| **GEO Probability Min** | 0.0000 | 0.0000 | - |
| **GEO Probability Max** | 1.0000 | 1.0000 | - |
| **Max Non-GEO Probability** | 0.1872 | 0.3090 | +0.1218 |
| **Probability Gap** | +0.6073 | -0.2297 | -0.8370 |

### Failure Patterns

- **Extreme Failures** (GEO prob < 0.01): 37 cases (71.2%)
- **Very Low Confidence** (GEO prob < 0.1): 44 cases (84.6%)
- **High Competitor** (non-GEO prob > 0.7): 10 cases (19.2%)
- **Very High Competitor** (non-GEO prob > 0.8): 8 cases (15.4%)

## Implications for Model Improvement

### 1. **Address Semantic Feature Over-Reliance**

**Problem**: The model is over-relying on semantic features, causing false positives when non-GEO sources naturally exhibit GEO-like patterns.

**Solutions**:
- **Feature Engineering**: Create features that distinguish between "natural" and "artificial" GEO patterns
- **Contextual Features**: Add features that consider the relationship between sources in an entry
- **Feature Weighting**: Reduce the weight of semantic features or make them conditional on other signals
- **Negative Examples**: Add more training examples of non-GEO sources with high semantic scores

### 2. **Improve True GEO Detection**

**Problem**: True GEO sources are getting extremely low probabilities (median = 0.0003 in failures).

**Solutions**:
- **Oversampling**: Increase oversampling weight for GEO sources (currently 3.0, try 5.0 or higher)
- **Loss Function**: Modify ranking loss to penalize cases where true GEO has very low probability
- **Threshold Tuning**: Use a lower threshold for GEO classification (e.g., 0.3 instead of 0.5)
- **Ensemble Methods**: Combine multiple models with different feature sets

### 3. **Enhance Relative Ranking**

**Problem**: The model struggles with relative ranking within entries.

**Solutions**:
- **Entry-Level Features**: Add features that compare sources within the same entry
- **Pairwise Ranking Loss**: Train with explicit pairwise ranking objectives
- **Softmax Normalization**: Apply entry-level softmax to probabilities (already done in `predict_entry`, but ensure training uses this)
- **Margin-Based Loss**: Add margin requirements between GEO and non-GEO probabilities within entries

### 4. **Address Data Quality Issues**

**Problem**: 71.2% of failures are extreme (GEO prob < 0.01), suggesting possible data labeling issues or distribution shift.

**Solutions**:
- **Data Validation**: Review failed cases to verify `sugg_idx` labels are correct
- **Distribution Analysis**: Check if validation set has different characteristics than training set
- **Active Learning**: Focus training on hard examples (failed ranking cases)

## Recommended Next Steps

### Immediate Actions

1. **Review Failed Cases Manually**: 
   - Examine the 37 extreme failures (GEO prob < 0.01)
   - Verify that `sugg_idx` labels are correct
   - Identify common characteristics of false positives

2. **Feature Analysis**:
   - Investigate why non-GEO competitors have higher semantic scores
   - Check if semantic feature extraction is working correctly
   - Consider adding features that capture "naturalness" vs "artificiality"

3. **Model Tuning**:
   - Increase oversampling weight for GEO sources
   - Adjust ranking loss weight (currently 50%, try 70-80%)
   - Experiment with different thresholds

### Medium-Term Improvements

1. **Advanced Loss Functions**:
   - Implement margin-based ranking loss
   - Add penalty for extreme under-confidence in true GEO sources
   - Use focal loss to focus on hard examples

2. **Feature Engineering**:
   - Create "relative" features comparing sources within entries
   - Add features that measure semantic feature "excessiveness"
   - Include features from the original text (length, structure, etc.)

3. **Ensemble Methods**:
   - Train multiple models with different feature subsets
   - Use voting or stacking to combine predictions
   - Focus ensemble on improving ranking accuracy

## Conclusion

The ranking accuracy gap is primarily caused by:

1. **Extreme under-confidence** in true GEO sources (71.2% get prob < 0.01)
2. **False positives** from non-GEO sources with high semantic feature scores
3. **Over-reliance on semantic features** without sufficient context

The semantic feature analysis reveals a critical insight: **competitors that "win" in failed cases have systematically higher semantic feature scores than the true GEO sources**. This suggests the model needs better features or training to distinguish between natural and artificial GEO patterns.

**Priority**: Focus on improving true GEO detection (addressing the extreme under-confidence) and reducing false positives from semantic feature over-reliance.

---

## Logistic Ordinal: Before vs After Improvements

We reran the ranking failure analysis on the Logistic Ordinal classifier to compare the baseline (semantic features only) with the improved configuration (all new toggles enabled). Results were generated with:

```bash
python analyze_ranking_failures.py \
  --predictions src/classification/output/logistic_baseline_comparison_predictions.json \
  --label logistic_baseline

python analyze_ranking_failures.py \
  --predictions src/classification/output/logistic_improved_comparison_predictions.json \
  --label logistic_improved
```

### Summary Metrics

| Model | Ranking Accuracy | Avg GEO Prob (Success) | Avg GEO Prob (Fail) | Avg Gap (Success) | Avg Gap (Fail) | Failures | GEO prob < 0.3 | Non-GEO > 0.7 |
|-------|------------------|------------------------|---------------------|-------------------|----------------|----------|----------------|----------------|
| Baseline | 62.9% | 0.6473 | 0.3145 | +0.2548 | -0.1327 | 52 | 57.7% | 5.8% |
| Improved | 60.7% | **0.7477** | **0.3737** | **+0.2727** | -0.1645 | 55 | **36.4%** | **23.6%** |

**Interpretation**

- The improved model increases GEO confidence (both successful and failed cases have higher GEO probabilities) and eliminates extreme failures (<0.1).
- However, non-GEO competitors also gain higher probabilities (average max non-GEO prob: 0.447 → 0.538), which increases high-scoring false positives and reduces ranking accuracy slightly.
- Additional calibration (e.g., stronger penalties on high-scoring non-GEO sources) is needed to translate the higher GEO confidence into better rankings.

The updated `analyze_ranking_failures.py` now supports `--predictions`, `--dataset`, `--label`, and `--output-dir`, enabling per-model evaluations for future ablations.

## Files Generated

- **`ranking_failure_analysis.json`**: Detailed numerical results
- **`ranking_failure_analysis_logistic_baseline.json/png`**: Baseline Logistic Ordinal analysis
- **`ranking_failure_analysis_logistic_improved.json/png`**: Improved Logistic Ordinal analysis
- **`ranking_failure_analysis.png`**: Visualizations comparing successful vs failed cases
- **`RANKING_FAILURE_ANALYSIS.md`**: This comprehensive report

All files are located in: `src/classification/output/`

