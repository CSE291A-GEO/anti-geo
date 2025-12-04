# Suggestions to Improve Ranking Accuracy for GEO-Focused Data

Current performance: **40.40% validation ranking accuracy**

## Priority 1: Add GEO-Specific Features (Highest Impact) ⭐⭐⭐

### 1.1 Add `s_geo_max` as a Feature
**Impact**: Very High | **Effort**: Low

The `s_geo_max` score is a direct measure of GEO optimization. This is likely the strongest single feature.

**Implementation**:
- Modify `extract_features()` to accept and include `s_geo_max`
- Add it to the feature vector: `features = np.concatenate([features, [s_geo_max]])`
- Update `RankingDataset` to pass `s_geo_max` from source data

**Expected improvement**: +5-10% ranking accuracy

### 1.2 Enable Semantic Pattern Features
**Impact**: High | **Effort**: Low

The 5 semantic pattern scores capture specific GEO optimization patterns.

**Implementation**:
- Set `use_semantic_features=True` in classifier initialization
- This adds 5 dimensions to the feature vector (384 → 389)

**Expected improvement**: +3-7% ranking accuracy

### 1.3 Add Text Quality Features
**Impact**: Medium | **Effort**: Medium

Text quality features (readability, structure, etc.) can help identify well-optimized content.

**Implementation**:
- Import `extract_text_quality_features` from `text_quality_features.py`
- Add to feature extraction: `features = np.concatenate([features, text_quality_features])`

**Expected improvement**: +2-5% ranking accuracy

## Priority 2: Improve Model Architecture ⭐⭐

### 2.1 Increase Model Capacity
**Impact**: Medium | **Effort**: Low

Current: `[128, 64]` hidden layers. Try deeper/wider networks.

**Suggestions**:
- `[256, 128, 64]` - Deeper network
- `[512, 256, 128]` - Wider and deeper
- `[128, 128, 128, 64]` - More layers

**Expected improvement**: +2-5% ranking accuracy

### 2.2 Add Query-Aware Features
**Impact**: High | **Effort**: Medium

Make features query-dependent by encoding the query and computing query-source similarity.

**Implementation**:
- Encode query text using the same embedding model
- Compute cosine similarity between query and source embeddings
- Add as a feature: `query_source_similarity`

**Expected improvement**: +5-10% ranking accuracy

### 2.3 Add Attention Mechanism
**Impact**: Medium-High | **Effort**: High

Use attention to focus on important parts of the source text relative to the query.

**Implementation**:
- Add a query-source attention layer
- Weight source features by query relevance

**Expected improvement**: +3-8% ranking accuracy

## Priority 3: Enhance Loss Function ⭐⭐

### 3.1 Adjust Loss Weighting
**Impact**: Medium | **Effort**: Low

Current: 50% ListNet, 50% pairwise. Try different ratios.

**Suggestions**:
- `listnet_weight=0.7` - More focus on top-1 accuracy
- `listnet_weight=0.3` - More focus on pairwise ordering
- Try pure ListNet (`listnet_weight=1.0`) or pure pairwise (`listnet_weight=0.0`)

**Expected improvement**: +1-3% ranking accuracy

### 3.2 Use LambdaRank Loss
**Impact**: High | **Effort**: Medium

LambdaRank considers position in ranking (top positions matter more).

**Implementation**:
- Replace or combine with LambdaRank loss
- Weights errors by position (higher weight for top positions)

**Expected improvement**: +3-7% ranking accuracy

### 3.3 Add Margin Scaling
**Impact**: Medium | **Effort**: Low

Scale pairwise loss margin by rank difference (larger margin for larger rank gaps).

**Implementation**:
- In `PairwiseRankingLoss`, scale margin by `abs(rank_i - rank_j)`
- Larger rank differences → larger required margin

**Expected improvement**: +2-4% ranking accuracy

## Priority 4: Training Improvements ⭐

### 4.1 Increase Training Epochs
**Impact**: Low-Medium | **Effort**: Low

Current: Early stopping at epoch 11. Try more epochs with better patience.

**Suggestions**:
- `epochs=100`, `patience=20`
- Use learning rate scheduling

**Expected improvement**: +1-3% ranking accuracy

### 4.2 Learning Rate Scheduling
**Impact**: Medium | **Effort**: Low

Use learning rate decay to fine-tune in later epochs.

**Implementation**:
- `torch.optim.lr_scheduler.ReduceLROnPlateau`
- Reduce LR when validation loss plateaus

**Expected improvement**: +2-4% ranking accuracy

### 4.3 Better Data Splitting
**Impact**: Low | **Effort**: Low

Ensure validation set is representative and not just sequential.

**Implementation**:
- Use stratified splitting by query characteristics
- Shuffle before splitting

**Expected improvement**: +1-2% ranking accuracy

## Priority 5: Feature Engineering ⭐

### 5.1 Add Relative Features
**Impact**: Medium | **Effort**: Medium

Add features relative to other sources in the same query.

**Examples**:
- Position in query (1st, 2nd, 3rd, etc.)
- Rank difference from best source
- Average rank of other sources

**Expected improvement**: +2-5% ranking accuracy

### 5.2 Category-Based Features
**Impact**: Low-Medium | **Effort**: Low

Use category information more effectively.

**Implementation**:
- One-hot encode category
- Or use category embeddings

**Expected improvement**: +1-3% ranking accuracy

### 5.3 Query-Source Interaction Features
**Impact**: Medium | **Effort**: Medium

Compute features that capture query-source relationship.

**Examples**:
- Query-source semantic similarity
- Keyword overlap
- Topic alignment

**Expected improvement**: +3-6% ranking accuracy

## Recommended Implementation Order

1. **Quick Wins** (1-2 hours):
   - Add `s_geo_max` feature
   - Enable semantic pattern features
   - Try different loss weightings

2. **Medium Effort** (4-6 hours):
   - Increase model capacity
   - Add query-aware features
   - Implement LambdaRank loss

3. **Advanced** (1-2 days):
   - Add attention mechanism
   - Implement relative features
   - Add text quality features

## Expected Combined Improvement

If implementing Priority 1 (all features) + Priority 2.2 (query-aware) + Priority 3.2 (LambdaRank):
- **Expected ranking accuracy**: 55-65% (from 40.40%)
- **Improvement**: +15-25 percentage points

## Testing Strategy

1. Implement one change at a time
2. Measure improvement on validation set
3. Keep changes that improve performance
4. Combine best-performing changes
5. Final evaluation on held-out test set

