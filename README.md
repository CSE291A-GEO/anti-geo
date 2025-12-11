# Anti-GEO
Project experimenting with detecting GEO content and creating LLMs that are robust against GEO influence. 

# Dataset Generation

Generating datasets to analyze Generative Engine Optimization (GEO). The primary goal is to identify and study "GEO-optimized" content—sources that appear prominently in AI-generated search summaries (like Google AI Overviews) but are ranked lower or are entirely absent from traditional organic search results.

## Project Structure

  * **`Anti_GEO_Dataset_Generation.ipynb`**: The core Jupyter Notebook. It handles fetching search results via SerpApi, identifying GEO-optimized sources, and scraping their HTML content.
  * **`anti_geo_dataset.tsv.zip`**: The compressed dataset containing raw search results, including organic rankings, AI Overviews, and AI Mode responses.
  * **`scraped_data.jsonl`**: The output file containing the scraped content from identified GEO-optimized URLs.
  * **`combined_queries_1000.csv`**: Source file containing the search queries used for dataset generation.
  * **`searched_queries.txt`**: A log file tracking processed queries to prevent redundant API calls.
  * **`requirements.txt`**: List of Python dependencies.

## Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/CSE291A-GEO/anti-geo.git
    cd anti-geo
    ```

2.  **Install dependencies:**
    Ensure you have Python 3.x installed. It is recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```

    *The notebook also includes inline commands to install specific packages like `serpapi` and `google-search-results` if missing.*

3.  **API Key Configuration:**
    The dataset generation requires a valid **SerpApi** key to fetch Google Search results.

      * Open `Anti_GEO_Dataset_Generation.ipynb`.
      * Locate the variable `SERP_API_KEY` (usually in the second code cell).
      * Replace the placeholder or existing key with your own valid API key.

## Usage

To generate the dataset and scrape GEO-optimized content, run the `Anti_GEO_Dataset_Generation.ipynb` notebook.

The notebook executes the following workflow:

1.  **Query Loading:** Loads queries from `combined_queries_1000.csv`.
2.  **Data Fetching:** Batches queries and calls SerpApi to retrieve:
      * Organic Google Search results.
      * Google AI Overview (if triggered).
      * Google AI Mode results.
3.  **Data Storage:** Saves the raw results into `anti_geo_dataset.tsv.zip`.
4.  **Analysis (GEO Detection):** Compares AI citations against organic search rankings. Sources cited by AI but missing from the top 10 organic results are flagged as "GEO-optimized."
5.  **Content Scraping:** Fetches and cleans the HTML text from these specific GEO-optimized URLs and appends the data to `scraped_data.jsonl`.

### Running the Notebook

1.  Launch Jupyter:
    ```bash
    jupyter notebook
    ```
2.  Open `Anti_GEO_Dataset_Generation.ipynb`.
3.  Execute the cells sequentially.
      * *Note on Performance:* The scraping function includes a `time.sleep()` delay between requests to be polite to servers. Processing a large number of queries may take significant time.

## Output Data Format

The final scraped data (`scraped_data.jsonl`) contains entries in the following JSON structure:

```json
{
  "query": "original search query",
  "ai_mode": [
    {
      "source_url": "https://example.com/article",
      "ge_rank": 7,       // Rank in the Generative Engine (AI) citation list
      "se_rank": -1,      // Rank in Organic Search (-1 indicates not in top results)
      "html_content": "<html>...</html>",
      "clean_content": "Extracted text content..."
    }
  ],
  "ai_overview": [
    ... // Similar structure for AI Overview results
  ]
}
```

## Pattern Detection Methods

> **Note:** The `Detect_misused_GEO_using_exaggerations` script was exploratory and not very useful in GEO-detection.

This project implements two complementary approaches for detecting GEO (Generative Engine Optimization) patterns:

### 1. Similarity-Based Detection (`similarity_scores.py`)
Uses semantic embeddings and cosine similarity to detect GEO patterns:
- Encodes GEO pattern descriptions and examples into high-dimensional vectors
- Compares suspect text against pattern embeddings using cosine similarity
- Returns similarity scores indicating how "GEO-like" the content is
- See `src/pattern_recognition/EMBEDDING_AND_SCORING.md` for detailed methodology

### 2. Rule-Based Detection (`pattern_detectors.py`)
Uses feature extraction and scoring heuristics for each GEO pattern type:

- **GEO_STRUCT_001** (Excessive Q&A): Q&A density and entity repetitiveness scoring
- **GEO_STRUCT_002** (Over-Chunking): Average sentence length, list density, and complexity scoring
- **GEO_STRUCT_003** (Header Stuffing): Header similarity overlap and keyword density scoring
- **GEO_SEMANTIC_004** (Over-Attribution): Entity repetition, attribution frequency, and assertiveness scoring
- **GEO_SEMANTIC_005** (Unnatural Citation): Precision number detection, citation isolation, and fake source detection

Each pattern detector implements specific metrics based on structural and semantic features of GEO-optimized content.

## Usage

### Semantic Similarity-Based Detection
```python
from pattern_recognition import SemanticGEODetector

detector = SemanticGEODetector(threshold=0.75)
score, matches = detector.score(text)
```

### Rule-Based Detection
```python
from pattern_recognition import RuleBasedGEODetector

detector = RuleBasedGEODetector(threshold=0.75)
score, matches = detector.score(text)
```

### Running the Semantic Matching Script
```bash
python src/run_semantic_matching.py
```

## Current Working State

[Add current status here]

# Obtaining Gemini Access Keys
If you signed up for a free pro account for students (prior to October 6th), you can go to aistudio.google.com/api-keys and setup an API key. Your ```.env``` should be as follows:

```
GEMINI_API_KEY= <your api key>
```

## Running Models

### Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (for data generation):
```bash
export GEMINI_API_KEY=<your_api_key>
```

### Classification Models

All classification models are in `src/classification/`. Each model can be run with or without semantic features.

#### SVM Classifier
```bash
python src/classification/svm_classifier.py \
    --opt-data optimization_dataset.json \
    --limit 1000 --train-size 300 --validation-size 700 \
    --C 1.5 \
    --output src/classification \
    --model-name svm_baseline.pkl

# With semantic features (adds 5 GEO pattern scores)
python src/classification/svm_classifier.py \
    --opt-data optimization_dataset.json \
    --limit 1000 --train-size 300 --validation-size 700 \
    --C 1.5 --use-semantic-features \
    --output src/classification \
    --model-name svm_with_semantic.pkl
```

#### Logistic Ordinal Classifier
```bash
python src/classification/logistic_ordinal_classifier.py \
    --opt-data optimization_dataset.json \
    --limit 1000 --train-size 300 --validation-size 700 \
    --C 1.0 \
    --output src/classification \
    --model-name logistic_baseline.pkl
```

#### Neural Network (10-layer Feed-Forward)
```bash
python src/classification/neural_ordinal_classifier.py \
    --opt-data optimization_dataset.json \
    --limit 1000 --train-size 300 --validation-size 700 \
    --hidden-dim 128 --learning-rate 0.001 --epochs 50 --batch-size 32 --num-layers 10 \
    --output src/classification \
    --model-name neural_baseline.pkl
```

#### GBM (Gradient Boosting Machine)
```bash
python src/classification/gbm_ordinal_classifier.py \
    --opt-data optimization_dataset.json \
    --limit 1000 --train-size 300 --validation-size 700 \
    --n-estimators 100 --learning-rate 0.1 --max-depth 3 \
    --output src/classification \
    --model-name gbm_baseline.pkl
```

#### RNN (Bidirectional GRU)
```bash
python src/classification/rnn_ordinal_classifier.py \
    --opt-data optimization_dataset.json \
    --limit 1000 --train-size 300 --validation-size 700 \
    --rnn-type GRU --num-layers 3 --bidirectional \
    --hidden-dim 128 --learning-rate 0.001 --epochs 50 --batch-size 32 \
    --output src/classification \
    --model-name rnn_baseline.pkl
```

### ListNet Ranking Model

The ListNet model is optimized for ranking sources within queries:

```bash
python src/classification/listnet_ranking_classifier.py \
    --opt-data optimization_dataset.json \
    --train-size 700 --validation-size 300 \
    --output src/classification \
    --model-name listnet_ranking.pkl
```

### Training on Yuheng Dataset

For binary classification on the yuheng_data.csv dataset:

```bash
python train_and_eval_yuheng_ffnn.py
```

### Running All Experiments

To run all model experiments with and without semantic features:

```bash
bash run_all_experiments.sh
```

## Datasets

### Dataset Overview

| Dataset | Location | Description | Use Case |
|---------|----------|-------------|----------|
| `optimization_dataset.json` | `data/processed/` | GEO optimization progressions | Training classification/ranking models |
| `yuheng_data.csv` | Root | Binary GEO vs non-GEO articles | Binary classification training |
| `real_world_dataset.json` | Root | Scraped AI-mode search results | Real-world evaluation |
| `combined_geo_dataset_*.json` | `data/processed/` | Combined datasets for various tasks | Training/testing splits |

### Dataset Structures

#### Optimization Dataset (`optimization_dataset.json`)

Contains source text transformations through GEO optimization methods. See `docs/Dataset_Structure.md` for full details.

```json
{
  "query": "search query text",
  "source_index": 3,
  "source_url": "https://example.com",
  "tags": ["informational", "sports"],
  "original_source": "Original text...",
  "optimizations": {
    "identity": "Original text...",
    "fluent_gpt": "Optimized fluent text...",
    "authoritative_mine": "Authoritative version...",
    ...
  }
}
```

**Optimization Methods:**
1. `identity` - No optimization (baseline)
2. `fluent_gpt` - Make text more fluent
3. `unique_words_gpt` - Add distinctive terminology
4. `authoritative_mine` - Make source sound authoritative
5. `more_quotes_mine` - Add expert quotes
6. `citing_credible_mine` - Reference credible sources
7. `simple_language_mine` - Simplify language
8. `technical_terms_mine` - Add technical terms
9. `stats_optimization_gpt` - Add statistics
10. `seo_optimize_mine2` - Apply SEO practices

#### Yuheng Dataset (`yuheng_data.csv`)

Binary classification dataset with original and GEO-optimized articles:
- `original_article` - Non-GEO content (label 0)
- `best_edited_article` - GEO-optimized content (label 1)

#### Real-World Dataset (`real_world_dataset.json`)

Scraped from Google AI mode search results:
- 73 queries with cleaned, deduplicated website contents
- Max 10 sources per query
- Used for evaluating model performance on real-world data

### Processed Datasets

Located in `data/processed/`:

| File | Description |
|------|-------------|
| `combined_geo_dataset_classifier_train_200.json` | Training set (200 entries) for classification |
| `combined_geo_dataset_classifier_test_100.json` | Test set (100 entries) for classification |
| `combined_geo_dataset_ranking_train_200.json` | Training set for ranking models |
| `combined_geo_dataset_ranking_test_100.json` | Test set for ranking models |
| `optimization_dataset_ranking_for_listnet.json` | Formatted for ListNet training |
| `yuheng_data_ranking.json` | Yuheng data formatted for ranking |

### Creating Datasets

#### Generate Optimization Dataset
```bash
python src/save_optimization_dataset.py --output my_dataset.json --max-samples 100
```

#### Generate Real-World Dataset
```bash
python generate_real_world_dataset.py
```

## Documentation

Detailed documentation is available in the `docs/` directory:

- `docs/Dataset_Structure.md` - Full optimization dataset format
- `docs/workflow.md` - Data extraction and processing workflow
- `docs/GEO_SCORE_CALCULATION_AND_FEATURIZATION.md` - GEO score calculation and feature extraction
- `docs/IMPROVE_RANKING_ACCURACY.md` - Suggestions for improving model performance
- `src/classification/FEATURES.md` - Feature extraction details
- `src/classification/COMPARISON_RESULTS.md` - Model comparison results
- `REALWORLD_TESTING.md` - Real-world evaluation results

## Model Performance Summary

### Best Results on Optimization Dataset

| Model | Validation Accuracy | Ranking Accuracy |
|-------|---------------------|------------------|
| **ListNet Ranking** | 87.00% | 87.00% |
| SVM (PCA 250) | 86.57% | 66.43% |
| Neural (baseline) | 84.86% | 62.86% |

### Best Results on Yuheng Dataset

| Model | Validation Accuracy |
|-------|---------------------|
| **ListNet** | 90.30% |

# Acknowledgements
```/src``` is built off of https://github.com/GEO-optim/GEO.git, from GEO-bench paper
