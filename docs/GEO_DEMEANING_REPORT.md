# GEO Score Demeaning by Website Category - Comprehensive Report

## Executive Summary

This report analyzes the impact of demeaning GEO scores by website category.
Total sources processed: 290
Number of categories: 10

## Methodology

1. **Baseline Calculation**: Calculated mean GEO scores per category from `se_optimized_sources_with_content.tsv`
2. **Categorization**: Each source in `scraped_data.jsonl` was categorized based on URL patterns and content
3. **Demeaning**: GEO scores were demeaned by subtracting the category baseline mean
4. **Classification**: Classification experiments were run with demeaned scores

## Category Definitions

- **E-commerce**: Sells products or services directly to consumers.
- **Corporate**: Provides information about a company, its services, and its brand.
- **Personal/Portfolio**: Showcases an individual's work, creative projects, or professional skills.
- **Content-sharing**: Allows users to share and exchange user-generated content like photos, videos, or music.
- **Communication/Social**: Facilitates interaction and discussion among users, such as social networking sites.
- **Educational**: Offers courses, learning platforms, or knowledge bases.
- **News and Media**: Publishes news, articles, and other media content.
- **Membership**: Provides exclusive content or services to users who register or pay a fee.
- **Affiliate**: Generates revenue by promoting and linking to products or services from other companies.
- **Non-profit**: Supports the mission of a charitable or non-profit organization.

## Baseline GEO Scores by Category (from TSV file)

| Category | Count | Mean | Std | Median | Min | Max |
|----------|-------|------|-----|--------|-----|-----|
| Communication/Social | 826 | 0.1813 | 0.0793 | 0.1784 | -0.0226 | 0.4211 |
| E-commerce | 428 | 0.2163 | 0.0840 | 0.2300 | 0.0129 | 0.4989 |
| Educational | 256 | 0.1830 | 0.0780 | 0.1758 | 0.0207 | 0.3864 |
| Unknown | 204 | 0.2334 | 0.0670 | 0.2709 | 0.0197 | 0.3156 |
| Non-profit | 180 | 0.1577 | 0.0687 | 0.1515 | 0.0157 | 0.3794 |
| News and Media | 123 | 0.1719 | 0.0705 | 0.1604 | 0.0025 | 0.3217 |
| Corporate | 114 | 0.2092 | 0.0764 | 0.2114 | 0.0411 | 0.4106 |
| Affiliate | 97 | 0.2105 | 0.0803 | 0.2139 | -0.0079 | 0.3618 |
| Content-sharing | 89 | 0.1854 | 0.0887 | 0.1717 | 0.0371 | 0.4798 |
| Personal/Portfolio | 49 | 0.1597 | 0.0758 | 0.1579 | -0.0312 | 0.3303 |
| Membership | 17 | 0.1733 | 0.0810 | 0.1716 | 0.0195 | 0.3315 |

## Category Distribution in Scraped Data

| Category | Count | Percentage |
|----------|-------|------------|
| Affiliate | 4 | 1.4% |
| Communication/Social | 37 | 12.8% |
| Content-sharing | 14 | 4.8% |
| Corporate | 30 | 10.3% |
| E-commerce | 78 | 26.9% |
| Educational | 57 | 19.7% |
| News and Media | 20 | 6.9% |
| Non-profit | 32 | 11.0% |
| Personal/Portfolio | 14 | 4.8% |
| Unknown | 4 | 1.4% |

## GEO Score Statistics by Category (Original Scores)

| Category | Count | Mean | Std | Median | Min | Max |
|----------|-------|------|-----|--------|-----|-----|
| Affiliate | 4 | 0.1295 | 0.0984 | 0.0745 | 0.0690 | 0.2998 |
| Communication/Social | 37 | 0.1987 | 0.0970 | 0.2166 | -0.0237 | 0.4459 |
| Content-sharing | 14 | 0.1760 | 0.0847 | 0.1920 | 0.0583 | 0.2931 |
| Corporate | 30 | 0.2150 | 0.0804 | 0.2254 | 0.0433 | 0.3295 |
| E-commerce | 78 | 0.2172 | 0.0614 | 0.2149 | 0.0972 | 0.3409 |
| Educational | 57 | 0.1886 | 0.0693 | 0.1936 | 0.0255 | 0.2950 |
| News and Media | 20 | 0.1643 | 0.1005 | 0.1578 | -0.0039 | 0.3749 |
| Non-profit | 32 | 0.1594 | 0.0638 | 0.1686 | 0.0181 | 0.3084 |
| Personal/Portfolio | 14 | 0.2199 | 0.0864 | 0.2585 | 0.0604 | 0.3128 |
| Unknown | 4 | 0.1416 | 0.0883 | 0.1120 | 0.0542 | 0.2883 |

## GEO Score Statistics by Category (Demeaned Scores)

| Category | Count | Mean | Std | Median | Min | Max |
|----------|-------|------|-----|--------|-----|-----|
| Affiliate | 4 | -0.0811 | 0.0984 | -0.1360 | -0.1415 | 0.0892 |
| Communication/Social | 37 | 0.0174 | 0.0970 | 0.0353 | -0.2050 | 0.2647 |
| Content-sharing | 14 | -0.0093 | 0.0847 | 0.0066 | -0.1271 | 0.1077 |
| Corporate | 30 | 0.0058 | 0.0804 | 0.0162 | -0.1658 | 0.1203 |
| E-commerce | 78 | 0.0009 | 0.0614 | -0.0014 | -0.1191 | 0.1246 |
| Educational | 57 | 0.0055 | 0.0693 | 0.0106 | -0.1576 | 0.1120 |
| News and Media | 20 | -0.0076 | 0.1005 | -0.0140 | -0.1757 | 0.2031 |
| Non-profit | 32 | 0.0017 | 0.0638 | 0.0109 | -0.1396 | 0.1507 |
| Personal/Portfolio | 14 | 0.0602 | 0.0864 | 0.0987 | -0.0993 | 0.1530 |
| Unknown | 4 | -0.0918 | 0.0883 | -0.1214 | -0.1792 | 0.0549 |

## Impact of Demeaning

### Key Findings

1. **Overall Mean GEO Score**:
   - Original: 0.1948
   - Demeaned: 0.0039
   - Change: -0.1910

2. **Overall Standard Deviation**:
   - Original: 0.0803
   - Demeaned: 0.0796
   - Change: -0.0007

3. **Category-Specific Effects**:

   - **Affiliate**: Original mean = 0.1295, Baseline = 0.2105, Demeaned mean = -0.0811
   - **Communication/Social**: Original mean = 0.1987, Baseline = 0.1813, Demeaned mean = 0.0174
   - **Content-sharing**: Original mean = 0.1760, Baseline = 0.1854, Demeaned mean = -0.0093
   - **Corporate**: Original mean = 0.2150, Baseline = 0.2092, Demeaned mean = 0.0058
   - **E-commerce**: Original mean = 0.2172, Baseline = 0.2163, Demeaned mean = 0.0009
   - **Educational**: Original mean = 0.1886, Baseline = 0.1830, Demeaned mean = 0.0055
   - **News and Media**: Original mean = 0.1643, Baseline = 0.1719, Demeaned mean = -0.0076
   - **Non-profit**: Original mean = 0.1594, Baseline = 0.1577, Demeaned mean = 0.0017
   - **Personal/Portfolio**: Original mean = 0.2199, Baseline = 0.1597, Demeaned mean = 0.0602
   - **Unknown**: Original mean = 0.1416, Baseline = 0.2334, Demeaned mean = -0.0918

## Recommendations

1. **Category-Specific Baselines**: Different website categories have different baseline GEO scores. Demeaning helps normalize scores across categories.

2. **Classification Impact**: Demeaned scores should improve classification by reducing category bias.

3. **Further Analysis**: Compare classification performance with and without demeaning to quantify the impact.

4. **Category Refinement**: Consider refining category definitions based on classification results.

---
*Report generated on 1764786459.9405382*