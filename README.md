# anti-geo
Project experimenting with detecting GEO content and creating LLMs that are robust against GEO influence. 

## Pattern Detection Methods

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

# Acknowledgements
```/src``` is built off of https://github.com/GEO-optim/GEO.git, from GEO-bench paper
