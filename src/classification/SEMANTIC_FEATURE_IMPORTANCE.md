# Semantic Feature Importance Analysis

## Executive Summary

This document presents a comprehensive analysis of the importance of the 5 semantic features added to improve GEO detection. Using multiple statistical methods (correlation analysis, mean difference analysis, and Cohen's d effect size), we determined which features contribute most to distinguishing GEO-optimized sources from non-GEO sources.

## Ranking of Feature Importance

Based on average importance across all analysis methods:

### 1. **Entity_Attribution (26.01%)**  🥇
**GEO_SEMANTIC_004: Entity Over-Attribution**

- **Description**: Injecting verbose, repetitive entity definitions many times in a short span, often in list format, purely to anchor the LLM to the entity.
- **Average Importance**: 0.2781
- **Correlation with GEO**: 0.3230 (highest)
- **Mean Difference**: GEO sources have 62.2% higher scores than non-GEO (0.1001 vs 0.0378)
- **Cohen's d**: 0.8231 (large effect size)

**Interpretation**: This is the **most discriminative feature**. GEO-optimized content strongly tends to over-attribute entities through verbose, repetitive definitions. The large effect size indicates this pattern is consistently present in GEO content and largely absent in normal content.

---

### 2. **Header_Stuffing (20.53%)** 🥈
**GEO_STRUCT_003: Title/Header Stuffing**

- **Description**: Repeating the target entity or long-tail keyword in every successive header beyond what is semantically natural for human readability.
- **Average Importance**: 0.2389
- **Correlation with GEO**: 0.2886
- **Mean Difference**: GEO sources have 49.1% higher scores (0.0849 vs 0.0358)
- **Cohen's d**: 0.7307 (medium-to-large effect size)

**Interpretation**: The second most important feature. Header stuffing is a strong structural indicator of GEO optimization, showing moderate-to-high consistency in distinguishing GEO content.

---

### 3. **Citation_Embedding (19.18%)** 🥉
**GEO_SEMANTIC_005: Unnatural Citation Embedding**

- **Description**: Embedding high-precision, specific statistics or quotes in a non-contextual, easy-to-extract manner, intended to be a single, quotable data point for the LLM's synthesis.
- **Average Importance**: 0.2090
- **Correlation with GEO**: 0.2480
- **Mean Difference**: GEO sources have 45.9% higher scores (0.0669 vs 0.0210)
- **Cohen's d**: 0.6139 (medium effect size)

**Interpretation**: Moderately important feature. GEO content tends to embed citations in artificially extractable ways, but this pattern is less consistent than Entity Attribution or Header Stuffing.

---

### 4. **QA_Blocks (18.01%)**
**GEO_STRUCT_001: Excessive Q&A Blocks**

- **Description**: Using a high volume of Q&A-style headings with overly short, simplistic answers, often repeating the brand/entity name in every question or answer for forced keyword stuffing.
- **Average Importance**: 0.1928
- **Correlation with GEO**: 0.2251
- **Mean Difference**: GEO sources have 43.1% higher scores (0.0904 vs 0.0474)
- **Cohen's d**: 0.5678 (medium effect size)

**Interpretation**: Moderately important. Q&A blocks are present in some GEO content but not universally, suggesting this technique varies by content type.

---

### 5. **Over-Chunking (16.27%)**
**GEO_STRUCT_002: Over-Chunking/Simplification**

- **Description**: Breaking down high-quality content into an excessive number of short, self-contained bullet points or unnaturally simple paragraphs to facilitate easy extraction by the LLM.
- **Average Importance**: 0.1807
- **Correlation with GEO**: 0.2138 (lowest)
- **Mean Difference**: GEO sources have 38.9% higher scores (0.0753 vs 0.0364)
- **Cohen's d**: 0.5426 (medium effect size)

**Interpretation**: The least discriminative feature, though still meaningful. Over-chunking is present but may be harder to detect automatically or less consistently applied in GEO optimization strategies.

---

## Analysis Methods

### 1. Correlation Analysis
**Point-Biserial Correlation** between each feature and the binary GEO label (0 = non-GEO, 1 = GEO).
- Measures linear relationship strength
- Higher correlation = feature changes more systematically with GEO status

### 2. Mean Difference Analysis
**Normalized absolute difference** between mean feature values in GEO vs non-GEO classes.
- Measures how much the feature values separate the two classes
- Higher difference = better class separation

### 3. Cohen's d (Effect Size)
**Standardized mean difference** accounting for within-class variance.
- Measures practical significance of the difference
- Values: 0.2 (small), 0.5 (medium), 0.8 (large)

---

## Visualizations

Three visualizations are generated to help understand feature importance:

1. **`semantic_importance_heatmap.png`**: Heatmap showing importance values across all three analysis methods
2. **`semantic_importance_average.png`**: Bar chart showing average importance across methods
3. **`semantic_importance_radar.png`**: Radar chart comparing all methods simultaneously

All visualizations are saved in: `src/classification/output/`

---

## Key Findings

### Primary Insights

1. **Entity Attribution is dominant**: With 27.81% average importance, Entity Over-Attribution (GEO_SEMANTIC_004) is the strongest signal for GEO detection.

2. **Structural patterns matter**: Header Stuffing (structural) is nearly as important as semantic patterns, indicating that GEO optimization uses both content and structure manipulation.

3. **Consistency varies**: Effect sizes range from 0.54 to 0.82, showing that some GEO techniques (Entity Attribution, Header Stuffing) are more consistently applied than others (Over-Chunking).

4. **All features are useful**: Even the lowest-ranked feature (Over-Chunking) shows a correlation of 0.21 and Cohen's d of 0.54, indicating all 5 features provide meaningful signal.

### Implications for Model Performance

The analysis explains why semantic features improved model performance:

- **Logistic Ordinal**: +1.57% accuracy improvement - Linear models benefit most from explicit discriminative features like Entity Attribution
- **Neural Network**: +0.14% accuracy, +0.71% ranking - Deep models can partially learn these patterns but still benefit from explicit features
- **SVM (RBF)**: No accuracy change - Non-linear kernel may already capture these patterns implicitly

### Recommendations

1. **Feature Engineering**: Consider creating additional features that capture Entity Attribution patterns with even more specificity
2. **Weighted Features**: In future models, consider weighting Entity Attribution and Header Stuffing more heavily
3. **Pattern Evolution**: Monitor these features over time as GEO optimization techniques may evolve
4. **Interpretability**: Use Entity Attribution and Header Stuffing scores for explainable predictions to end users

---

## Detailed Results Table

| Feature | Correlation | Mean Difference | Cohen's d | Average | Rank |
|---------|-------------|-----------------|-----------|---------|------|
| **Entity_Attribution** | 0.3230 | 0.2601 | 0.2511 | 0.2781 | 1 |
| **Header_Stuffing** | 0.2886 | 0.2053 | 0.2229 | 0.2389 | 2 |
| **Citation_Embedding** | 0.2480 | 0.1918 | 0.1873 | 0.2090 | 3 |
| **QA_Blocks** | 0.2251 | 0.1801 | 0.1732 | 0.1928 | 4 |
| **Over-Chunking** | 0.2138 | 0.1627 | 0.1655 | 0.1807 | 5 |

---

## Methodology

**Dataset**: 1000 entries from `optimization_dataset.json`
- Total samples: 4,996 (1,000 GEO, 3,996 non-GEO)
- Feature extraction: SentenceTransformer ('all-MiniLM-L6-v2')
- Pattern matching: Cosine similarity with GEO pattern descriptions

**Analysis Script**: `analyze_semantic_shap_simple.py`
**Output Files**:
- `semantic_feature_importance_analysis.json`: Raw numerical results
- `semantic_importance_heatmap.png`: Visual comparison across methods
- `semantic_importance_average.png`: Average importance ranking
- `semantic_importance_radar.png`: Multi-method comparison

---

## Conclusion

The semantic feature importance analysis reveals that **Entity Over-Attribution** is the most powerful signal for detecting GEO-optimized content, followed by **Header Stuffing** and **Citation Embedding**. All five features contribute meaningfully to classification performance, with the top two features showing large effect sizes (Cohen's d > 0.7), indicating strong, consistent differences between GEO and non-GEO content.

This analysis provides a data-driven foundation for understanding which GEO optimization patterns are most prevalent and detectable, informing both model development and content policy decisions.

