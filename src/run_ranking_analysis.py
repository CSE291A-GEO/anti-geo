#!/usr/bin/env python3
"""
Run GEO Score Analysis

This script compares GEO scores (semantic similarity) between the original dataset
and the optimized dataset, identifying the source with the greatest relative increase
in GEO score as the GEO-optimized source.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from pattern_recognition.ranking_analyzer import RankingAnalyzer
from pattern_recognition.semantic_matching import SemanticGEODetector
from datasets import load_dataset
from dotenv import load_dotenv


def main():
    """Main function to run GEO score analysis."""
    load_dotenv()
    
    project_root = Path(__file__).parent
    
    # Paths
    optimized_dataset_path = project_root / 'optimization_dataset.json'
    
    print("="*80)
    print("GEO SCORE ANALYSIS")
    print("="*80)
    print()
    
    # Initialize detector
    print("Initializing SemanticGEODetector...")
    detector = SemanticGEODetector(threshold=0.75)
    print("Detector ready!")
    print()
    
    # Initialize analyzer
    analyzer = RankingAnalyzer(
        original_dataset_path=None,  # Load from HuggingFace
        optimized_dataset_path=str(optimized_dataset_path),
        detector=detector
    )
    
    print("Loading datasets...")
    print(f"  Original dataset: Loading from HuggingFace (GEO-Optim/geo-bench)")
    print(f"  Optimized dataset: {optimized_dataset_path}")
    print()
    
    # Analyze all entries
    print("Analyzing GEO score changes (using parsed=False for speed)...")
    print("-"*80)
    
    results = analyzer.analyze_all_entries(limit=None)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        return
    
    # Print summary
    print()
    print("="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total entries analyzed: {results['total_entries']}")
    print(f"Valid entries: {results['valid_entries']}")
    print(f"Correct identifications: {results['correct_identifications']}")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    avg_ranking = results.get('average_geo_source_ranking')
    if avg_ranking is not None:
        print(f"Average ranking of actual GEO source: {avg_ranking:.2f} (out of 5)")
    else:
        print(f"Average ranking of actual GEO source: N/A")
    print()
    
    # Show detailed statistics
    print("="*80)
    print("DETAILED STATISTICS")
    print("="*80)
    
    # Count relative change magnitudes
    relative_change_buckets = {
        '0-10%': 0,  # 0% to 10% increase
        '10-50%': 0,  # 10% to 50% increase
        '50-100%': 0,  # 50% to 100% increase
        '100%+': 0,  # 100%+ increase
        'negative': 0  # Decreased
    }
    
    correct_by_change = {
        '0-10%': 0,
        '10-50%': 0,
        '50-100%': 0,
        '100%+': 0,
        'negative': 0
    }
    
    for result in results['results']:
        if 'error' not in result:
            score_changes = result['score_changes']
            identified_url = result.get('identified_geo_source_url')
            
            if identified_url and identified_url in score_changes:
                change_info = score_changes[identified_url]
                relative_change = change_info.get('relative_change')
                
                if relative_change is not None and relative_change != float('inf'):
                    if relative_change < 0:
                        relative_change_buckets['negative'] += 1
                        if result.get('correct_identification'):
                            correct_by_change['negative'] += 1
                    elif relative_change < 0.1:
                        relative_change_buckets['0-10%'] += 1
                        if result.get('correct_identification'):
                            correct_by_change['0-10%'] += 1
                    elif relative_change < 0.5:
                        relative_change_buckets['10-50%'] += 1
                        if result.get('correct_identification'):
                            correct_by_change['10-50%'] += 1
                    elif relative_change < 1.0:
                        relative_change_buckets['50-100%'] += 1
                        if result.get('correct_identification'):
                            correct_by_change['50-100%'] += 1
                    else:
                        relative_change_buckets['100%+'] += 1
                        if result.get('correct_identification'):
                            correct_by_change['100%+'] += 1
                elif relative_change == float('inf'):
                    # Original was 0, optimized is positive
                    relative_change_buckets['100%+'] += 1
                    if result.get('correct_identification'):
                        correct_by_change['100%+'] += 1
    
    print("Relative GEO score change distribution (for identified sources):")
    for change_type, count in relative_change_buckets.items():
        correct = correct_by_change.get(change_type, 0)
        accuracy = (correct / count * 100) if count > 0 else 0.0
        print(f"  {change_type} increase: {count} entries (accuracy: {accuracy:.1f}%)")
    print()
    
    # Show examples
    print("="*80)
    print("SAMPLE RESULTS")
    print("="*80)
    
    # Show first 10 results
    print("\nFirst 10 entries:")
    print("-"*80)
    for i, result in enumerate(results['results'][:10]):
        if 'error' not in result:
            print(f"\nEntry {i+1}:")
            print(f"  Query: {result['query'][:70]}...")
            print(f"  Identified GEO source: {result['identified_geo_source_url']}")
            print(f"  Known GEO source (sugg_idx={result['known_geo_source_idx']}, ranking={result.get('known_geo_source_ranking')}): {result['known_geo_source_url']}")
            print(f"  Correct: {'✓' if result['correct_identification'] else '✗'}")
            
            # Show GEO score changes for identified source
            identified_url = result.get('identified_geo_source_url')
            if identified_url and identified_url in result['score_changes']:
                change_info = result['score_changes'][identified_url]
                orig_score = change_info.get('original_geo_score')
                opt_score = change_info.get('optimized_geo_score')
                abs_change = change_info.get('absolute_change')
                rel_change = change_info.get('relative_change')
                orig_pos = change_info.get('original_position')
                opt_pos = change_info.get('optimized_position')
                if orig_score is not None and opt_score is not None:
                    if rel_change == float('inf'):
                        print(f"  GEO score: {orig_score:.4f} → {opt_score:.4f} (relative: ∞)")
                    else:
                        print(f"  GEO score: {orig_score:.4f} → {opt_score:.4f} (absolute: {abs_change:+.4f}, relative: {rel_change*100:+.1f}%)")
                    print(f"  Position: {orig_pos} → {opt_pos}")
            
            # Show GEO score change for actual GEO source
            actual_url = result.get('known_geo_source_url')
            if actual_url and actual_url in result['score_changes']:
                change_info = result['score_changes'][actual_url]
                actual_rel_change = change_info.get('relative_change')
                actual_abs_change = change_info.get('absolute_change')
                if actual_rel_change is not None:
                    if actual_rel_change == float('inf'):
                        print(f"  Actual GEO source relative change: ∞")
                    else:
                        print(f"  Actual GEO source relative change: {actual_rel_change*100:+.1f}% (absolute: {actual_abs_change:+.4f})")
    
    # Show incorrect identifications
    incorrect_results = [r for r in results['results'] if 'error' not in r and not r.get('correct_identification', False)]
    if incorrect_results:
        print()
        print("="*80)
        print("INCORRECT IDENTIFICATIONS (Sample)")
        print("="*80)
        for i, result in enumerate(incorrect_results[:5]):
            print(f"\nEntry {i+1}:")
            print(f"  Query: {result['query'][:70]}...")
            print(f"  Identified: {result['identified_geo_source_url']}")
            print(f"  Actual (sugg_idx={result['known_geo_source_idx']}): {result['known_geo_source_url']}")
            
            # Show GEO score changes for both
            score_changes = result['score_changes']
            identified_url = result.get('identified_geo_source_url')
            actual_url = result.get('known_geo_source_url')
            
            if identified_url and identified_url in score_changes:
                change_info = score_changes[identified_url]
                rel_change = change_info.get('relative_change')
                abs_change = change_info.get('absolute_change')
                orig_score = change_info.get('original_geo_score')
                opt_score = change_info.get('optimized_geo_score')
                if orig_score is not None and opt_score is not None:
                    if rel_change == float('inf'):
                        print(f"  Identified source: {orig_score:.4f} → {opt_score:.4f} (relative: ∞)")
                    else:
                        print(f"  Identified source: {orig_score:.4f} → {opt_score:.4f} (relative: {rel_change*100:+.1f}%, absolute: {abs_change:+.4f})")
            
            if actual_url and actual_url in score_changes:
                change_info = score_changes[actual_url]
                rel_change = change_info.get('relative_change')
                abs_change = change_info.get('absolute_change')
                orig_score = change_info.get('original_geo_score')
                opt_score = change_info.get('optimized_geo_score')
                if orig_score is not None and opt_score is not None:
                    if rel_change == float('inf'):
                        print(f"  Actual GEO source: {orig_score:.4f} → {opt_score:.4f} (relative: ∞)")
                    else:
                        print(f"  Actual GEO source: {orig_score:.4f} → {opt_score:.4f} (relative: {rel_change*100:+.1f}%, absolute: {abs_change:+.4f})")
    
    # Save results
    output_path = project_root / 'ranking_analysis.json'
    print()
    print("="*80)
    print(f"Saving results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Results saved successfully!")
    print()


if __name__ == '__main__':
    main()

