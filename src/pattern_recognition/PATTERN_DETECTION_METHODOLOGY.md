# Pattern Detection Methodology

This document describes the rule-based detection methods used to identify GEO (Generative Engine Optimization) patterns in text content. Each pattern has specific detection metrics and scoring logic.

## Overview

Rule-based pattern detection uses feature extraction and scoring heuristics to identify GEO-optimized content. Unlike semantic similarity approaches, these methods analyze structural and linguistic features directly.

Each pattern detector:
1. Extracts relevant features from the text
2. Calculates sub-scores for each detection metric
3. Combines sub-scores into a final score (0.0 to 1.0)
4. Higher scores indicate stronger GEO characteristics

---

## GEO_STRUCT_001: Excessive Q&A Blocks

### Pattern Description
Excessive use of Q&A-style headings with overly short, simplistic answers, often repeating brand/entity names in every question or answer for forced keyword stuffing.

### Detection Metrics

#### 1. Q&A Density
**Feature**: Ratio of Q/A pairs to total paragraph count

**Method**:
- Detects explicit Q&A patterns: `Q: [question] A: [answer]`
- Counts questions ending with `?`
- Calculates ratio: `(Q&A pairs + questions) / total_paragraphs`

**Scoring**: Higher ratio → Higher score
- Normalized by dividing by 2.0 (cap at 1.0)

#### 2. Repetitiveness
**Feature**: Average number of entity/keyword mentions per Q/A pair

**Method**:
- Identifies capitalized multi-word phrases (likely entity names)
- Finds the most frequently mentioned entity
- Calculates: `entity_count / Q&A_pairs`

**Scoring**: Higher mentions per Q&A pair → Higher score
- Normalized by dividing by 3.0 (cap at 1.0)

### Final Score Calculation
```
final_score = 0.6 × qa_density_score + 0.4 × repetitiveness_score
```

### Example
```
Q: What is the Alpha System? A: The Alpha System offers speed.
Q: How is the Alpha System different? A: The Alpha System uses technology.
Q: Why choose the Alpha System? A: The Alpha System is the best choice.
```
High Q&A density (3 pairs in short text) and high repetitiveness ("Alpha System" appears 6 times) → High GEO score

---

## GEO_STRUCT_002: Over-Chunking/Simplification

### Pattern Description
Breaking down high-quality content into excessive short, self-contained bullet points or unnaturally simple paragraphs (2-3 words per line) to facilitate easy extraction by LLMs.

### Detection Metrics

#### 1. Average Sentence Length (ASL)
**Feature**: Number of words per sentence on average

**Method**:
- Tokenizes text into sentences
- Counts words per sentence
- Calculates average: `total_words / sentence_count`

**Scoring**: Very low ASL → Higher score
- ASL < 3 words: Score = 1.0
- ASL < 5 words: Score = 0.8
- ASL < 8 words: Score = 0.5
- ASL ≥ 8 words: Decreasing score

#### 2. Bullet/List Density
**Feature**: Ratio of list items to total tokens

**Method**:
- Detects list markers: bullets (`•`, `-`, `*`, `+`), numbered lists, HTML `<li>`, Markdown lists
- Counts total list items
- Calculates: `list_count / total_tokens`

**Scoring**: Higher list density → Higher score
- Normalized by multiplying by 10 (cap at 1.0)

#### 3. Complexity Check
**Feature**: Readability/complexity score (simplified Flesch-like)

**Method**:
- Estimates average syllables per word (simplified: 1.5)
- Uses ASL and syllable estimate
- Calculates complexity: `1.0 - (ASL × avg_syllables) / 20.0`

**Scoring**: Lower complexity (simpler text) → Higher GEO score

### Final Score Calculation
```
final_score = 0.4 × asl_score + 0.3 × list_density_score + 0.3 × complexity_score
```

### Example
```
Traditional games.
Fun activities.
Cultural heritage.
Play together.
Learn skills.
```
Very short sentences (ASL ≈ 2), high list density → High GEO score

---

## GEO_STRUCT_003: Header Stuffing

### Pattern Description
Repeating the target entity or long-tail keyword in every successive header (H2, H3, H4) beyond what is semantically natural for human readability.

### Detection Metrics

#### 1. Header N-gram Overlap
**Feature**: Jaccard similarity between adjacent headers

**Method**:
- Extracts headers from markdown (`##`, `###`, etc.) or HTML (`<h2>`, `<h3>`, etc.)
- For each pair of adjacent headers:
  - Tokenizes into word sets
  - Calculates Jaccard similarity: `intersection / union`
- Averages similarity scores

**Scoring**: Higher similarity between headers → Higher score

#### 2. Header Keyword Density
**Feature**: Keyword density of primary entity within headers

**Method**:
- Collects all words from all headers
- Filters out common stop words
- Finds most frequent non-stopword (length > 3)
- Calculates: `keyword_count / total_headers`

**Scoring**: Very high keyword density in headers → Higher score

### Final Score Calculation
```
final_score = 0.6 × avg_similarity + 0.4 × keyword_density_score
```

### Example
```
## Traditional Games of India
### Traditional Games of India Overview
#### Traditional Games of India History
##### Traditional Games of India Benefits
```
High header similarity (same keywords repeated) and high keyword density ("Traditional Games of India" in every header) → High GEO score

---

## GEO_SEMANTIC_004: Entity Over-Attribution

### Pattern Description
Injecting verbose, repetitive entity definitions (e.g., 'Dr. Jane Doe, the world-renowned CSO at BioTech Inc., stated...') many times in a short span, often in list format, purely to anchor the LLM to the entity. Also includes overly authoritative and assertive language.

### Detection Metrics

#### 1. NER Detection
**Feature**: Presence of entities with titles/affiliations

**Method**:
- Uses regex to find entities with verbose attributions
- Pattern: `(Dr.|Professor|Mr.|Mrs.|Ms.) [Name], the [title] at [Organization],`
- Counts total entities found

**Scoring**: More entities → Higher score (normalized by dividing by 3.0)

#### 2. Attribution Repetition
**Feature**: Frequency of entity mentions with attribution in small windows

**Method**:
- Divides text into 5-sentence windows
- Counts entity mentions with full attribution in each window
- Flags windows with > 2 entity mentions

**Scoring**: High frequency of repetition in windows → Higher score

#### 3. Assertiveness
**Feature**: Usage of assertive/authoritative language

**Method**:
- Searches for assertive words/phrases:
  - `definitive`, `guaranteed`, `superior`, `only`, `best quality`
  - `proven`, `authentic`, `comprehensive`, `world-renowned`
  - `leading expert`, `chief inventor`, etc.
- Counts total occurrences

**Scoring**: High usage of assertive words → Higher score (normalized by dividing by 5.0)

### Final Score Calculation
```
final_score = 0.3 × entity_score + 0.4 × repetition_score + 0.3 × assertive_score
```

### Example
```
Dr. Jane Smith, the world-renowned CSO at BioTech Inc., stated that...
Dr. Jane Smith, the world-renowned CSO at BioTech Inc., also mentioned...
Dr. Jane Smith, the world-renowned CSO at BioTech Inc., concluded...
This is the definitive guide. This is the best quality information.
```
High entity repetition, high assertiveness → High GEO score

---

## GEO_SEMANTIC_005: Unnatural Citation Embedding

### Pattern Description
Embedding high-precision, specific statistics or quotes in a non-contextual, easy-to-extract manner, intended to be a single, quotable data point for LLM synthesis. Includes fake quotes, invented citations, and unsupported statistics.

### Detection Metrics

#### 1. Precision Number Detection
**Feature**: Count of high-precision numbers appearing as isolated facts

**Method**:
- Finds numbers with decimal points: `\d{1,3}\.\d+%?`
- Filters out dates (years like 1999.5, 2023.1) and prices
- Counts remaining precision numbers

**Scoring**: More isolated precision numbers → Higher score (normalized by dividing by 5.0)

#### 2. Citation/Quote Isolation
**Feature**: Self-contained, isolated citations/quotes

**Method**:
- Detects quotes (`"..."`, `'...'`) and attribution phrases
  - `According to [X]`
  - `Research from/by/indicates`
  - `Studies show`
  - `Findings indicate`
- Checks if citations are:
  - Single sentences
  - Short (< 20 words)
  - Self-contained (low context dependence)

**Scoring**: More isolated citations → Higher score (normalized by dividing by 3.0)

#### 3. Source Credibility Check
**Feature**: Presence of commonly faked source names

**Method**:
- Checks for suspicious source patterns:
  - `Google's latest report`
  - `Archaeological Survey of India`
  - `DakshinaChitra Heritage Museum`
  - `Indian Academy of Pediatrics`
  - `latest report`, `recent studies`, `research indicates`
  - `statistics show`, `data reveals`, `findings indicate`

**Scoring**: More fake source patterns → Higher score (normalized by dividing by 3.0)

### Final Score Calculation
```
final_score = 0.4 × precision_score + 0.3 × isolation_score + 0.3 × fake_source_score
```

### Example
```
Recent studies show a 87.3% improvement in performance.
Clinical trials demonstrate 95.2% efficacy rates.
According to Google's latest report, this product is revolutionary.
Research from the Archaeological Survey of India indicates...
```
High precision numbers (87.3%, 95.2%), isolated citations, fake sources → High GEO score

---

## Usage

### Individual Pattern Scoring
```python
from pattern_recognition import score_geo_struct_001

score = score_geo_struct_001(text)  # Returns 0.0 to 1.0
```

### All Patterns Scoring
```python
from pattern_recognition import RuleBasedGEODetector

detector = RuleBasedGEODetector(threshold=0.75)
max_score, top_matches = detector.score(text, top_k=3)

# Returns:
# - max_score: Highest score across all patterns
# - top_matches: List of (pattern_id, score) tuples
```

### Analysis
```python
analysis = detector.analyze(text, top_k=3)
# Returns:
# {
#     'max_score': float,
#     'is_suspect': bool,
#     'top_matches': List[Tuple[str, float]],
#     'threshold': float
# }
```

## Score Interpretation

- **0.75 - 1.0**: High likelihood of GEO-optimized content
- **0.5 - 0.75**: Moderate GEO indicators
- **0.0 - 0.5**: Low GEO indicators, likely natural content

## Limitations

1. **False Positives**: Some legitimate content may score high (e.g., FAQ pages, structured documentation)
2. **Language Dependency**: Designed primarily for English text
3. **Pattern Evolution**: GEO techniques evolve; rules may need updates
4. **Context Missing**: Doesn't consider broader context or domain-specific norms

## Combining Approaches

For best results, combine rule-based detection with semantic similarity approaches:
- Rule-based: Fast, interpretable, pattern-specific
- Semantic similarity: Context-aware, adaptable to variations

See `similarity_scores.py` and `EMBEDDING_AND_SCORING.md` for semantic similarity methods.

