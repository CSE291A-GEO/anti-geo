# Optimization Dataset Format

## Overview

The optimization dataset stores the progression of source text transformations through each GEO optimization method.

## Dataset Structure

```json
[
  {
    "query": "mention the names of any 3 famous folklore sports in karnataka state",
    "source_index": 3,
    "source_url": "https://www.example.com/karnataka-sports",
    "tags": ["informational", "sports", "non-technical"],
    "original_source": "Traditional games have always been part of Karnataka's rich culture...",
    "optimizations": {
      "identity": "Traditional games have always been part of Karnataka's rich culture...",
      "fluent_gpt": "Karnataka boasts a rich cultural heritage of traditional games...",
      "unique_words_gpt": "Karnataka showcases distinctive folklore sports including...",
      "authoritative_mine": "According to cultural experts, Karnataka's traditional games...",
      "more_quotes_mine": "As Dr. Smith noted, 'Karnataka's folklore sports are unique'...",
      "citing_credible_mine": "Research from the University of Karnataka shows...",
      "simple_language_mine": "Karnataka has many fun traditional games...",
      "technical_terms_mine": "Karnataka's ludological traditions encompass...",
      "stats_optimization_gpt": "Over 50 traditional games are documented in Karnataka...",
      "seo_optimize_mine2": "Famous Folklore Sports in Karnataka | Traditional Games Guide..."
    }
  },
  ...
]
```

## Fields

### Record-Level Fields

- **query**: The user query/question
- **source_index**: Index of the source being optimized (0-4 typically)
- **source_url**: Original URL of the source
- **tags**: Query classification tags (genre, complexity, domain, etc.)
- **original_source**: Unmodified source text from dataset
- **optimizations**: Dictionary mapping method name → optimized text

### Optimization Methods

1. **identity**: No optimization (baseline)
2. **fluent_gpt**: Make text more fluent and readable
3. **unique_words_gpt**: Add unique/distinctive terminology
4. **authoritative_mine**: Make source sound more authoritative
5. **more_quotes_mine**: Add expert quotes and citations
6. **citing_credible_mine**: Reference credible external sources
7. **simple_language_mine**: Simplify language for accessibility
8. **technical_terms_mine**: Add technical/domain-specific terms
9. **stats_optimization_gpt**: Add statistics and data
10. **seo_optimize_mine2**: Apply SEO best practices

## Usage Examples

### Creating the Dataset

```bash
# Process first 10 samples
python src/save_optimization_dataset.py --output my_dataset.json --max-samples 10

# Process entire test set
python src/save_optimization_dataset.py --output full_dataset.json

# Process train set
python src/save_optimization_dataset.py --output train_dataset.json --split train
```

### Loading and Analyzing

```python
from save_optimization_dataset import load_optimization_dataset, get_optimization_by_method

# Load dataset
records = load_optimization_dataset('optimization_dataset.json')

# Get all optimizations for a specific method
fluent_opts = get_optimization_by_method(records, 'fluent_gpt')

# Analyze a single record
record = records[0]
print(f"Query: {record['query']}")
print(f"Original: {record['original_source'][:100]}...")

for method, optimized in record['optimizations'].items():
    if optimized:
        print(f"\n{method}:")
        print(f"  {optimized[:100]}...")
```

### Comparing Methods

```python
from save_optimization_dataset import compare_methods, load_optimization_dataset

records = load_optimization_dataset('optimization_dataset.json')

# Compare two optimization strategies
compare_methods(records, 'more_quotes_mine', 'authoritative_mine')
```

### Export to CSV for Analysis

```python
import pandas as pd
import json

# Load dataset
with open('optimization_dataset.json', 'r') as f:
    records = json.load(f)

# Flatten to DataFrame
rows = []
for record in records:
    for method, optimized_text in record['optimizations'].items():
        rows.append({
            'query': record['query'],
            'source_index': record['source_index'],
            'source_url': record['source_url'],
            'method': method,
            'original_length': len(record['original_source']),
            'optimized_length': len(optimized_text) if optimized_text else 0,
            'original_text': record['original_source'],
            'optimized_text': optimized_text
        })

df = pd.DataFrame(rows)
df.to_csv('optimizations_flat.csv', index=False)
```

### Analyze Length Changes

```python
import json
import numpy as np

with open('optimization_dataset.json', 'r') as f:
    records = json.load(f)

# Calculate length changes per method
for method in ['fluent_gpt', 'authoritative_mine', 'seo_optimize_mine2']:
    lengths = []
    for record in records:
        orig_len = len(record['original_source'])
        opt_text = record['optimizations'].get(method)
        if opt_text:
            opt_len = len(opt_text)
            change = (opt_len - orig_len) / orig_len * 100
            lengths.append(change)
    
    print(f"{method}:")
    print(f"  Mean length change: {np.mean(lengths):.1f}%")
    print(f"  Std dev: {np.std(lengths):.1f}%")
```

## Use Cases

### 1. Training Anti-GEO Detectors

Use the dataset to train models that detect which GEO method was applied:

```python
# Features: optimized text
# Labels: GEO method name
# Task: Multi-class classification
```

### 2. Analyzing GEO Effectiveness

Compare before/after optimizations to understand what works:

```python
# Measure: citation counts, visibility scores
# By method, query type, domain
```

### 3. Building Better GEO Methods

Study successful optimizations to create improved methods:

```python
# Extract patterns from high-performing optimizations
# Combine multiple methods
```

### 4. Creating Benchmark Datasets

Use as ground truth for evaluation:

```python
# Original = clean text
# Optimized = GEO-manipulated text
# Test detection accuracy
```

## Output File Size

Approximate sizes (depends on source length):

- 10 samples: ~500 KB
- 100 samples: ~5 MB
- 1000 samples (full test set): ~50 MB
- 10000 samples (full dataset): ~500 MB

## Notes

- Processing is slow because it calls Ollama for each optimization method
- Each sample takes ~10-30 seconds (10 methods × 1-3 sec each)
- Use `--max-samples` to create a small test dataset first
- Cache is used to avoid re-optimizing the same text
- Failed optimizations are stored as `null`

