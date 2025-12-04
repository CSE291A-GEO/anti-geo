# GEO Score Demeaning by Website Category

This script categorizes websites and demeans GEO scores based on website category baselines.

## Prerequisites

Make sure you have all dependencies installed:
```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- sentence-transformers
- scikit-learn
- All other packages from requirements.txt

## Usage

```bash
python src/classification/categorize_and_demean_geo_scores.py \
    --tsv se_optimized_sources_with_content.tsv \
    --jsonl scraped_data.jsonl \
    --output demeaned_scraped_data.jsonl \
    --report GEO_DEMEANING_ANALYSIS_REPORT.md
```

## What it does

1. **Reads baseline data** from `se_optimized_sources_with_content.tsv`
   - Categorizes each website into one of 10 categories
   - Calculates GEO scores for all sources
   - Computes baseline statistics (mean, std, median, etc.) per category

2. **Processes scraped data** from `scraped_data.jsonl`
   - Categorizes each source within queries
   - Calculates original GEO scores
   - Demeans scores by subtracting category baseline mean
   - Saves processed data with original and demeaned scores

3. **Generates comprehensive report** with:
   - Baseline statistics per category
   - Category distribution in scraped data
   - Impact analysis of demeaning
   - Key findings and recommendations

## Website Categories

The script uses 10 predefined categories:
- E-commerce
- Corporate
- Personal/Portfolio
- Content-sharing
- Communication/Social
- Educational
- News and Media
- Membership
- Affiliate
- Non-profit

## Output Files

- `demeaned_scraped_data.jsonl`: Processed data with original and demeaned GEO scores
- `GEO_DEMEANING_ANALYSIS_REPORT.md`: Comprehensive analysis report

## Notes

- Sources without meaningful content (< 50 characters) are skipped
- Sources that can't be categorized are marked as "Unknown"
- The demeaning process subtracts the category baseline mean from each GEO score

