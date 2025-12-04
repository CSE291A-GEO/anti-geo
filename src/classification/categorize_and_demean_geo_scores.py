"""
Categorize websites and demean GEO scores by website category.

This script:
1. Reads se_optimized_sources_with_content.tsv and categorizes websites
2. Calculates baseline GEO scores per category
3. Processes scraped_data.jsonl and categorizes sources
4. Demeans GEO scores by category before classification
5. Generates a comprehensive report
"""

import json
import csv
import sys
import os
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
from urllib.parse import urlparse
import re

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pattern_recognition.similarity_scores import SemanticGEODetector


# Website categories as defined by user
CATEGORIES = {
    'E-commerce': {
        'keywords': ['shop', 'store', 'buy', 'cart', 'checkout', 'product', 'purchase', 'price', 'sale', 'deal'],
        'domains': ['amazon', 'ebay', 'etsy', 'shopify', 'woocommerce', 'bigcommerce'],
        'description': 'Sells products or services directly to consumers.'
    },
    'Corporate': {
        'keywords': ['about us', 'our company', 'our services', 'contact us', 'careers', 'team', 'mission', 'vision'],
        'domains': ['corp', 'company', 'inc', 'llc', 'ltd'],
        'description': 'Provides information about a company, its services, and its brand.'
    },
    'Personal/Portfolio': {
        'keywords': ['portfolio', 'my work', 'about me', 'resume', 'cv', 'projects', 'blog', 'personal'],
        'domains': [],
        'description': "Showcases an individual's work, creative projects, or professional skills."
    },
    'Content-sharing': {
        'keywords': ['share', 'upload', 'gallery', 'photos', 'videos', 'media', 'community', 'submit'],
        'domains': ['flickr', 'imgur', 'deviantart', 'behance', 'dribbble', 'pinterest'],
        'description': 'Allows users to share and exchange user-generated content like photos, videos, or music.'
    },
    'Communication/Social': {
        'keywords': ['social', 'network', 'connect', 'friends', 'follow', 'message', 'chat', 'forum', 'discussion'],
        'domains': ['facebook', 'twitter', 'instagram', 'linkedin', 'reddit', 'discord', 'slack'],
        'description': 'Facilitates interaction and discussion among users, such as social networking sites.'
    },
    'Educational': {
        'keywords': ['course', 'learn', 'education', 'tutorial', 'lesson', 'academy', 'university', 'college', 'school', 'training'],
        'domains': ['edu', 'coursera', 'udemy', 'khan', 'edx', 'udacity'],
        'description': 'Offers courses, learning platforms, or knowledge bases.'
    },
    'News and Media': {
        'keywords': ['news', 'article', 'journalism', 'report', 'breaking', 'headline', 'media', 'press', 'publication'],
        'domains': ['news', 'bbc', 'cnn', 'reuters', 'nytimes', 'washingtonpost', 'theguardian'],
        'description': 'Publishes news, articles, and other media content.'
    },
    'Membership': {
        'keywords': ['membership', 'subscribe', 'premium', 'exclusive', 'member', 'join', 'sign up', 'register', 'login'],
        'domains': ['patreon', 'onlyfans', 'substack', 'medium'],
        'description': 'Provides exclusive content or services to users who register or pay a fee.'
    },
    'Affiliate': {
        'keywords': ['affiliate', 'commission', 'referral', 'partner', 'sponsored', 'ad', 'advertisement', 'promo code'],
        'domains': [],
        'description': 'Generates revenue by promoting and linking to products or services from other companies.'
    },
    'Non-profit': {
        'keywords': ['non-profit', 'nonprofit', 'charity', 'donate', 'donation', 'foundation', 'mission', 'cause', 'volunteer'],
        'domains': ['org', 'foundation'],
        'description': 'Supports the mission of a charitable or non-profit organization.'
    }
}


def categorize_website(url: str, content: str = '') -> str:
    """
    Categorize a website based on URL and content.
    
    Args:
        url: The website URL
        content: Optional content text for additional context
        
    Returns:
        Category name or 'Unknown' if no match
    """
    url_lower = url.lower()
    content_lower = content.lower() if content else ''
    combined_text = f"{url_lower} {content_lower}"
    
    # Score each category
    category_scores = {}
    
    for category, criteria in CATEGORIES.items():
        score = 0.0
        
        # Check domain matches (strong signal)
        domain = urlparse(url).netloc.lower()
        for domain_keyword in criteria['domains']:
            if domain_keyword in domain:
                score += 3.0
        
        # Check URL keywords (medium signal)
        for keyword in criteria['keywords']:
            if keyword in url_lower:
                score += 2.0
            if keyword in content_lower:
                score += 1.0
        
        # Check for .org domain for non-profit
        if category == 'Non-profit' and domain.endswith('.org'):
            score += 2.0
        
        # Check for .edu domain for educational
        if category == 'Educational' and domain.endswith('.edu'):
            score += 2.0
        
        category_scores[category] = score
    
    # Return category with highest score, or 'Unknown' if all scores are 0
    if max(category_scores.values()) > 0:
        return max(category_scores.items(), key=lambda x: x[1])[0]
    else:
        return 'Unknown'


def read_tsv_file(tsv_path: str) -> List[Dict[str, Any]]:
    """
    Read the TSV file and return list of records.
    
    Args:
        tsv_path: Path to the TSV file
        
    Returns:
        List of dictionaries with query, source_url, se_rank, ge_rank, clean_content
    """
    records = []
    print(f"Reading TSV file: {tsv_path}")
    
    # Read file and remove NUL characters
    with open(tsv_path, 'rb') as f:
        content = f.read()
        # Remove NUL bytes
        content = content.replace(b'\x00', b'')
    
    # Decode and parse
    text = content.decode('utf-8', errors='ignore')
    lines = text.split('\n')
    
    # Get header
    if not lines:
        print("  Warning: TSV file is empty")
        return records
    
    header = lines[0].split('\t')
    print(f"  Header columns: {header[:5]}...")
    
    # Parse rows
    for line_num, line in enumerate(lines[1:], 2):
        if not line.strip():
            continue
        try:
            values = line.split('\t')
            if len(values) >= len(header):
                row = dict(zip(header, values))
                records.append({
                    'query': row.get('query', ''),
                    'source_url': row.get('source_url', ''),
                    'se_rank': row.get('se_rank', ''),
                    'ge_rank': row.get('ge_rank', ''),
                    'clean_content': row.get('clean_content', '')
                })
        except Exception as e:
            if line_num <= 10:  # Only print first few errors
                print(f"  Warning: Skipping line {line_num}: {e}")
            continue
    
    print(f"  Read {len(records)} records from TSV file")
    return records


def read_jsonl_file(jsonl_path: str) -> List[Dict[str, Any]]:
    """
    Read the JSONL file and return list of entries.
    
    Args:
        jsonl_path: Path to the JSONL file
        
    Returns:
        List of dictionaries with query and ai_mode sources
    """
    entries = []
    print(f"Reading JSONL file: {jsonl_path}")
    
    with open(jsonl_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            try:
                if line.strip():
                    entry = json.loads(line)
                    entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"  Warning: Skipping invalid JSON on line {line_num}: {e}")
                continue
    
    print(f"  Read {len(entries)} entries from JSONL file")
    return entries


def calculate_baseline_geo_scores(records: List[Dict[str, Any]], detector: SemanticGEODetector) -> Dict[str, Dict[str, float]]:
    """
    Calculate baseline GEO scores per category from TSV file.
    
    Args:
        records: List of records from TSV file
        detector: GEO detector instance
        
    Returns:
        Dictionary mapping category to statistics (mean, std, count, min, max)
    """
    print("\n" + "="*80)
    print("CALCULATING BASELINE GEO SCORES BY CATEGORY")
    print("="*80)
    
    # Group records by category
    category_scores = defaultdict(list)
    category_urls = defaultdict(list)
    
    print("\nCategorizing websites...")
    categorized_count = 0
    unknown_count = 0
    
    for i, record in enumerate(records):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(records)} records...")
        
        url = record['source_url']
        content = record.get('clean_content', '')
        
        category = categorize_website(url, content)
        category_urls[category].append(url)
        
        if category == 'Unknown':
            unknown_count += 1
        else:
            categorized_count += 1
        
        # Calculate GEO score
        if content and len(content.strip()) > 50:  # Only score if we have meaningful content
            try:
                geo_score, _ = detector.score(content, top_k=3, parsed=False)
                category_scores[category].append(geo_score)
            except Exception as e:
                print(f"  Warning: Failed to calculate GEO score for {url}: {e}")
                continue
    
    print(f"\nCategorization complete:")
    print(f"  Categorized: {categorized_count}")
    print(f"  Unknown: {unknown_count}")
    
    # Calculate statistics per category
    baseline_stats = {}
    
    print("\nCalculating baseline statistics per category...")
    for category in sorted(CATEGORIES.keys()) + ['Unknown']:
        scores = category_scores[category]
        if scores:
            import numpy as np
            scores_array = np.array(scores)
            baseline_stats[category] = {
                'mean': float(np.mean(scores_array)),
                'std': float(np.std(scores_array)),
                'median': float(np.median(scores_array)),
                'count': len(scores),
                'min': float(np.min(scores_array)),
                'max': float(np.max(scores_array)),
                'urls': len(category_urls[category])
            }
        else:
            baseline_stats[category] = {
                'mean': 0.0,
                'std': 0.0,
                'median': 0.0,
                'count': 0,
                'min': 0.0,
                'max': 0.0,
                'urls': len(category_urls[category])
            }
    
    return baseline_stats


def process_scraped_data(entries: List[Dict[str, Any]], baseline_stats: Dict[str, Dict[str, float]], 
                        detector: SemanticGEODetector) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Process scraped_data.jsonl entries, categorize sources, and demean GEO scores.
    
    Args:
        entries: List of entries from JSONL file
        baseline_stats: Baseline statistics per category
        detector: GEO detector instance
        
    Returns:
        Tuple of (processed_entries, analysis_stats)
    """
    print("\n" + "="*80)
    print("PROCESSING SCRAPED DATA AND DEMEANING GEO SCORES")
    print("="*80)
    
    processed_entries = []
    analysis_stats = {
        'total_queries': len(entries),
        'total_sources': 0,
        'categorized_sources': 0,
        'unknown_sources': 0,
        'sources_with_content': 0,
        'category_distribution': Counter(),
        'geo_score_changes': {
            'before_demeaning': [],
            'after_demeaning': []
        },
        'per_category_changes': defaultdict(list)
    }
    
    for entry_idx, entry in enumerate(entries):
        if (entry_idx + 1) % 10 == 0:
            print(f"  Processing entry {entry_idx + 1}/{len(entries)}...")
        
        query = entry.get('query', '')
        sources = entry.get('ai_mode', [])
        
        processed_entry = {
            'query': query,
            'sources': []
        }
        
        for source in sources:
            analysis_stats['total_sources'] += 1
            
            url = source.get('source_url', '')
            content = source.get('clean_content', '')
            
            if not content or len(content.strip()) < 50:
                # Skip sources without meaningful content
                processed_entry['sources'].append({
                    **source,
                    'category': 'Unknown',
                    'geo_score_original': None,
                    'geo_score_demeaned': None,
                    'baseline_mean': None
                })
                continue
            
            analysis_stats['sources_with_content'] += 1
            
            # Categorize source
            category = categorize_website(url, content)
            analysis_stats['category_distribution'][category] += 1
            
            if category == 'Unknown':
                analysis_stats['unknown_sources'] += 1
            else:
                analysis_stats['categorized_sources'] += 1
            
            # Calculate original GEO score
            try:
                geo_score_original, _ = detector.score(content, top_k=3, parsed=False)
            except Exception as e:
                print(f"  Warning: Failed to calculate GEO score for {url}: {e}")
                geo_score_original = 0.0
            
            # Get baseline mean for this category
            baseline_mean = baseline_stats.get(category, {}).get('mean', 0.0)
            
            # Demean the GEO score
            geo_score_demeaned = geo_score_original - baseline_mean
            
            # Store statistics
            analysis_stats['geo_score_changes']['before_demeaning'].append(geo_score_original)
            analysis_stats['geo_score_changes']['after_demeaning'].append(geo_score_demeaned)
            analysis_stats['per_category_changes'][category].append({
                'original': geo_score_original,
                'demeaned': geo_score_demeaned,
                'baseline': baseline_mean
            })
            
            # Add processed source to entry
            processed_entry['sources'].append({
                **source,
                'category': category,
                'geo_score_original': float(geo_score_original),
                'geo_score_demeaned': float(geo_score_demeaned),
                'baseline_mean': float(baseline_mean)
            })
        
        processed_entries.append(processed_entry)
    
    return processed_entries, analysis_stats


def generate_report(baseline_stats: Dict[str, Dict[str, float]], 
                   analysis_stats: Dict[str, Any],
                   output_path: str):
    """
    Generate a comprehensive report of findings.
    
    Args:
        baseline_stats: Baseline statistics per category
        analysis_stats: Analysis statistics from processed data
        output_path: Path to save the report
    """
    print("\n" + "="*80)
    print("GENERATING REPORT")
    print("="*80)
    
    import numpy as np
    
    report_lines = []
    report_lines.append("# GEO Score Demeaning Analysis Report")
    report_lines.append("")
    report_lines.append("## Executive Summary")
    report_lines.append("")
    report_lines.append("This report analyzes GEO scores across different website categories and examines")
    report_lines.append("the impact of demeaning scores by category baseline. The analysis uses:")
    report_lines.append("- Baseline data from `se_optimized_sources_with_content.tsv`")
    report_lines.append("- Test data from `scraped_data.jsonl`")
    report_lines.append("")
    
    # Baseline Statistics Section
    report_lines.append("## 1. Baseline GEO Scores by Category")
    report_lines.append("")
    report_lines.append("Baseline statistics calculated from `se_optimized_sources_with_content.tsv`:")
    report_lines.append("")
    report_lines.append("| Category | Count | Mean | Std Dev | Median | Min | Max | URLs |")
    report_lines.append("|----------|-------|------|---------|--------|-----|-----|------|")
    
    for category in sorted(CATEGORIES.keys()) + ['Unknown']:
        stats = baseline_stats[category]
        report_lines.append(
            f"| {category} | {stats['count']} | {stats['mean']:.4f} | "
            f"{stats['std']:.4f} | {stats['median']:.4f} | {stats['min']:.4f} | "
            f"{stats['max']:.4f} | {stats['urls']} |"
        )
    
    report_lines.append("")
    report_lines.append("### Key Observations:")
    report_lines.append("")
    
    # Find categories with highest/lowest baseline scores
    categories_with_scores = [(cat, stats) for cat, stats in baseline_stats.items() 
                             if stats['count'] > 0]
    categories_with_scores.sort(key=lambda x: x[1]['mean'], reverse=True)
    
    if categories_with_scores:
        highest = categories_with_scores[0]
        lowest = categories_with_scores[-1]
        report_lines.append(f"- **Highest baseline GEO score**: {highest[0]} (mean: {highest[1]['mean']:.4f})")
        report_lines.append(f"- **Lowest baseline GEO score**: {lowest[0]} (mean: {lowest[1]['mean']:.4f})")
        report_lines.append("")
    
    # Scraped Data Analysis Section
    report_lines.append("## 2. Scraped Data Analysis")
    report_lines.append("")
    report_lines.append("Statistics from `scraped_data.jsonl`:")
    report_lines.append("")
    report_lines.append(f"- **Total queries**: {analysis_stats['total_queries']}")
    report_lines.append(f"- **Total sources**: {analysis_stats['total_sources']}")
    report_lines.append(f"- **Sources with content**: {analysis_stats['sources_with_content']}")
    report_lines.append(f"- **Categorized sources**: {analysis_stats['categorized_sources']}")
    report_lines.append(f"- **Unknown sources**: {analysis_stats['unknown_sources']}")
    report_lines.append("")
    
    # Category Distribution
    report_lines.append("### Category Distribution in Scraped Data")
    report_lines.append("")
    report_lines.append("| Category | Count | Percentage |")
    report_lines.append("|----------|-------|------------|")
    
    total_categorized = sum(analysis_stats['category_distribution'].values())
    for category, count in analysis_stats['category_distribution'].most_common():
        percentage = (count / total_categorized * 100) if total_categorized > 0 else 0
        report_lines.append(f"| {category} | {count} | {percentage:.1f}% |")
    
    report_lines.append("")
    
    # GEO Score Changes
    report_lines.append("## 3. Impact of Demeaning")
    report_lines.append("")
    
    before_scores = np.array(analysis_stats['geo_score_changes']['before_demeaning'])
    after_scores = np.array(analysis_stats['geo_score_changes']['after_demeaning'])
    
    if len(before_scores) > 0:
        report_lines.append("### Overall Statistics")
        report_lines.append("")
        report_lines.append("| Metric | Before Demeaning | After Demeaning | Change |")
        report_lines.append("|--------|-------------------|-----------------|--------|")
        report_lines.append(
            f"| Mean | {np.mean(before_scores):.4f} | {np.mean(after_scores):.4f} | "
            f"{np.mean(after_scores) - np.mean(before_scores):.4f} |"
        )
        report_lines.append(
            f"| Std Dev | {np.std(before_scores):.4f} | {np.std(after_scores):.4f} | "
            f"{np.std(after_scores) - np.std(before_scores):.4f} |"
        )
        report_lines.append(
            f"| Median | {np.median(before_scores):.4f} | {np.median(after_scores):.4f} | "
            f"{np.median(after_scores) - np.median(before_scores):.4f} |"
        )
        report_lines.append(
            f"| Min | {np.min(before_scores):.4f} | {np.min(after_scores):.4f} | "
            f"{np.min(after_scores) - np.min(before_scores):.4f} |"
        )
        report_lines.append(
            f"| Max | {np.max(before_scores):.4f} | {np.max(after_scores):.4f} | "
            f"{np.max(after_scores) - np.max(before_scores):.4f} |"
        )
        report_lines.append("")
        
        # Per-category impact
        report_lines.append("### Impact by Category")
        report_lines.append("")
        report_lines.append("| Category | Count | Mean Original | Mean Demeaned | Mean Change |")
        report_lines.append("|----------|-------|---------------|---------------|-------------|")
        
        for category in sorted(analysis_stats['per_category_changes'].keys()):
            changes = analysis_stats['per_category_changes'][category]
            if changes:
                orig_means = [c['original'] for c in changes]
                demeaned_means = [c['demeaned'] for c in changes]
                mean_orig = np.mean(orig_means)
                mean_demeaned = np.mean(demeaned_means)
                mean_change = mean_demeaned - mean_orig
                
                report_lines.append(
                    f"| {category} | {len(changes)} | {mean_orig:.4f} | "
                    f"{mean_demeaned:.4f} | {mean_change:+.4f} |"
                )
        
        report_lines.append("")
    
    # Findings and Recommendations
    report_lines.append("## 4. Key Findings")
    report_lines.append("")
    
    # Calculate some insights
    if len(before_scores) > 0:
        # Find categories with largest changes
        category_changes = {}
        for category, changes in analysis_stats['per_category_changes'].items():
            if changes:
                orig_means = [c['original'] for c in changes]
                demeaned_means = [c['demeaned'] for c in changes]
                mean_change = np.mean(demeaned_means) - np.mean(orig_means)
                category_changes[category] = mean_change
        
        if category_changes:
            largest_increase = max(category_changes.items(), key=lambda x: x[1])
            largest_decrease = min(category_changes.items(), key=lambda x: x[1])
            
            report_lines.append("### Score Distribution Changes")
            report_lines.append("")
            report_lines.append(f"- **Largest mean increase after demeaning**: {largest_increase[0]} ({largest_increase[1]:+.4f})")
            report_lines.append(f"- **Largest mean decrease after demeaning**: {largest_decrease[0]} ({largest_decrease[1]:+.4f})")
            report_lines.append("")
            
            # Variance analysis
            before_var = np.var(before_scores)
            after_var = np.var(after_scores)
            var_change = after_var - before_var
            
            report_lines.append("### Variance Analysis")
            report_lines.append("")
            report_lines.append(f"- **Variance before demeaning**: {before_var:.4f}")
            report_lines.append(f"- **Variance after demeaning**: {after_var:.4f}")
            report_lines.append(f"- **Variance change**: {var_change:+.4f}")
            if var_change < 0:
                report_lines.append("  - Demeaning reduced variance, suggesting category-specific baselines help normalize scores")
            else:
                report_lines.append("  - Demeaning increased variance, suggesting category differences are significant")
            report_lines.append("")
    
    report_lines.append("### Category-Specific Insights")
    report_lines.append("")
    
    # Analyze each category
    for category in sorted(CATEGORIES.keys()):
        baseline = baseline_stats.get(category, {})
        changes = analysis_stats['per_category_changes'].get(category, [])
        
        if baseline.get('count', 0) > 0 and changes:
            orig_means = [c['original'] for c in changes]
            mean_orig = np.mean(orig_means)
            baseline_mean = baseline['mean']
            
            report_lines.append(f"#### {category}")
            report_lines.append("")
            report_lines.append(f"- **Baseline mean**: {baseline_mean:.4f} (from {baseline['count']} samples)")
            report_lines.append(f"- **Scraped data mean (original)**: {mean_orig:.4f} (from {len(changes)} samples)")
            report_lines.append(f"- **Difference**: {mean_orig - baseline_mean:+.4f}")
            report_lines.append(f"- **Description**: {CATEGORIES[category]['description']}")
            report_lines.append("")
    
    # Recommendations
    report_lines.append("## 5. Recommendations")
    report_lines.append("")
    report_lines.append("Based on the analysis:")
    report_lines.append("")
    report_lines.append("1. **Category-specific baselines**: Different website categories show different baseline GEO scores.")
    report_lines.append("   Demeaning helps normalize scores relative to category norms.")
    report_lines.append("")
    report_lines.append("2. **Classification improvement**: Using demeaned scores should improve GEO detection")
    report_lines.append("   by reducing false positives from categories that naturally have higher GEO scores.")
    report_lines.append("")
    report_lines.append("3. **Category coverage**: Some categories may need more samples for reliable baselines.")
    report_lines.append("   Consider collecting more data for underrepresented categories.")
    report_lines.append("")
    report_lines.append("4. **Unknown category handling**: Sources categorized as 'Unknown' should be handled")
    report_lines.append("   with care, as they may represent edge cases or new website types.")
    report_lines.append("")
    
    # Write report
    report_text = '\n'.join(report_lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"Report saved to: {output_path}")
    print(f"Report length: {len(report_text)} characters")


def main():
    """Main function to run the analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Categorize websites and demean GEO scores by category')
    parser.add_argument('--tsv', type=str, 
                       default='se_optimized_sources_with_content.tsv',
                       help='Path to TSV file with baseline data')
    parser.add_argument('--jsonl', type=str,
                       default='scraped_data.jsonl',
                       help='Path to JSONL file with scraped data')
    parser.add_argument('--output', type=str,
                       default='demeaned_scraped_data.jsonl',
                       help='Path to output file with demeaned scores')
    parser.add_argument('--report', type=str,
                       default='GEO_DEMEANING_ANALYSIS_REPORT.md',
                       help='Path to output report file')
    
    args = parser.parse_args()
    
    # Resolve paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    tsv_path = os.path.join(project_root, args.tsv)
    jsonl_path = os.path.join(project_root, args.jsonl)
    output_path = os.path.join(project_root, args.output)
    report_path = os.path.join(project_root, 'src', 'classification', args.report)
    
    print("="*80)
    print("GEO SCORE DEMEANING ANALYSIS")
    print("="*80)
    print(f"TSV file: {tsv_path}")
    print(f"JSONL file: {jsonl_path}")
    print(f"Output file: {output_path}")
    print(f"Report file: {report_path}")
    print("="*80)
    
    # Initialize GEO detector
    print("\nInitializing GEO detector...")
    detector = SemanticGEODetector(use_parsed=False)
    print("  GEO detector ready")
    
    # Step 1: Read TSV file and calculate baseline scores
    tsv_records = read_tsv_file(tsv_path)
    baseline_stats = calculate_baseline_geo_scores(tsv_records, detector)
    
    # Step 2: Read JSONL file
    jsonl_entries = read_jsonl_file(jsonl_path)
    
    # Step 3: Process scraped data and demean scores
    processed_entries, analysis_stats = process_scraped_data(
        jsonl_entries, baseline_stats, detector
    )
    
    # Step 4: Save processed data
    print(f"\nSaving processed data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in processed_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"  Saved {len(processed_entries)} entries")
    
    # Step 5: Generate report
    generate_report(baseline_stats, analysis_stats, report_path)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"✓ Processed {len(processed_entries)} queries")
    print(f"✓ Categorized {analysis_stats['categorized_sources']} sources")
    print(f"✓ Generated report: {report_path}")
    print("="*80)


if __name__ == '__main__':
    main()

