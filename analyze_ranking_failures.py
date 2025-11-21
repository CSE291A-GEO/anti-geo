#!/usr/bin/env python3
"""
Analyze ranking accuracy failures to identify patterns.

This script examines entries where the true GEO source (sugg_idx) does NOT
have the highest GEO probability, and compares them to successful cases.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import argparse
import sys
sys.path.append('src/classification')
from semantic_features import SemanticFeatureExtractor


def load_predictions(predictions_path: Path) -> List[Dict]:
    """Load predictions JSON file."""
    with open(predictions_path, 'r') as f:
        data = json.load(f)
    return data.get('entries', [])


def analyze_ranking_accuracy(entries: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Separate entries into successful and failed ranking cases."""
    successful = []
    failed = []
    
    for entry in entries:
        sources = entry['sources']
        sugg_idx = entry['sugg_idx']
        
        # Find true GEO source
        geo_source = next((s for s in sources if s['source_idx'] == sugg_idx), None)
        if not geo_source:
            continue
        
        # Find source with highest GEO probability
        max_prob_source = max(sources, key=lambda s: s['geo_probability'])
        
        # Check if ranking is correct
        if max_prob_source['source_idx'] == sugg_idx:
            successful.append(entry)
        else:
            failed.append(entry)
    
    return successful, failed


def analyze_probability_distributions(successful: List[Dict], failed: List[Dict]) -> Dict:
    """Analyze probability distributions in successful vs failed cases."""
    print("\n" + "="*80)
    print("Probability Distribution Analysis")
    print("="*80)
    
    successful_geo_probs = []
    failed_geo_probs = []
    successful_max_non_geo = []
    failed_max_non_geo = []
    successful_gaps = []
    failed_gaps = []
    
    for entry in successful:
        sources = entry['sources']
        sugg_idx = entry['sugg_idx']
        
        geo_source = next(s for s in sources if s['source_idx'] == sugg_idx)
        non_geo_sources = [s for s in sources if s['source_idx'] != sugg_idx]
        
        successful_geo_probs.append(geo_source['geo_probability'])
        if non_geo_sources:
            max_non_geo = max(s['geo_probability'] for s in non_geo_sources)
            successful_max_non_geo.append(max_non_geo)
            successful_gaps.append(geo_source['geo_probability'] - max_non_geo)
    
    for entry in failed:
        sources = entry['sources']
        sugg_idx = entry['sugg_idx']
        
        geo_source = next(s for s in sources if s['source_idx'] == sugg_idx)
        non_geo_sources = [s for s in sources if s['source_idx'] != sugg_idx]
        
        failed_geo_probs.append(geo_source['geo_probability'])
        if non_geo_sources:
            max_non_geo = max(s['geo_probability'] for s in non_geo_sources)
            failed_max_non_geo.append(max_non_geo)
            failed_gaps.append(geo_source['geo_probability'] - max_non_geo)
    
    results = {
        'successful': {
            'geo_prob_mean': float(np.mean(successful_geo_probs)),
            'geo_prob_std': float(np.std(successful_geo_probs)),
            'geo_prob_median': float(np.median(successful_geo_probs)),
            'geo_prob_min': float(np.min(successful_geo_probs)),
            'geo_prob_max': float(np.max(successful_geo_probs)),
            'max_non_geo_mean': float(np.mean(successful_max_non_geo)),
            'gap_mean': float(np.mean(successful_gaps)),
            'gap_std': float(np.std(successful_gaps)),
        },
        'failed': {
            'geo_prob_mean': float(np.mean(failed_geo_probs)),
            'geo_prob_std': float(np.std(failed_geo_probs)),
            'geo_prob_median': float(np.median(failed_geo_probs)),
            'geo_prob_min': float(np.min(failed_geo_probs)),
            'geo_prob_max': float(np.max(failed_geo_probs)),
            'max_non_geo_mean': float(np.mean(failed_max_non_geo)),
            'gap_mean': float(np.mean(failed_gaps)),
            'gap_std': float(np.std(failed_gaps)),
        }
    }
    
    print("\nTrue GEO Source Probabilities:")
    print(f"  Successful: mean={results['successful']['geo_prob_mean']:.4f}, "
          f"median={results['successful']['geo_prob_median']:.4f}, "
          f"min={results['successful']['geo_prob_min']:.4f}, "
          f"max={results['successful']['geo_prob_max']:.4f}")
    print(f"  Failed:     mean={results['failed']['geo_prob_mean']:.4f}, "
          f"median={results['failed']['geo_prob_median']:.4f}, "
          f"min={results['failed']['geo_prob_min']:.4f}, "
          f"max={results['failed']['geo_prob_max']:.4f}")
    
    print("\nMax Non-GEO Source Probabilities in Entry:")
    print(f"  Successful: mean={results['successful']['max_non_geo_mean']:.4f}")
    print(f"  Failed:     mean={results['failed']['max_non_geo_mean']:.4f}")
    
    print("\nProbability Gap (GEO prob - max non-GEO prob):")
    print(f"  Successful: mean={results['successful']['gap_mean']:.4f}, "
          f"std={results['successful']['gap_std']:.4f}")
    print(f"  Failed:     mean={results['failed']['gap_mean']:.4f}, "
          f"std={results['failed']['gap_std']:.4f}")
    
    return results


def analyze_failed_case_patterns(failed: List[Dict]) -> Dict:
    """Identify specific patterns in failed cases."""
    print("\n" + "="*80)
    print("Failed Case Pattern Analysis")
    print("="*80)
    
    patterns = {
        'geo_prob_below_0_1': 0,
        'geo_prob_below_0_2': 0,
        'geo_prob_below_0_3': 0,
        'geo_prob_below_0_4': 0,
        'geo_prob_below_0_5': 0,
        'high_non_geo_competitor': 0,  # Non-GEO source with prob > 0.7
        'very_high_non_geo_competitor': 0,  # Non-GEO source with prob > 0.8
        'close_competition': 0,  # Gap < 0.1
        'very_close_competition': 0,  # Gap < 0.05
        'negative_gap': 0,  # GEO prob < max non-GEO prob
        'extreme_failures': [],  # GEO prob < 0.01
    }
    
    for entry in failed:
        sources = entry['sources']
        sugg_idx = entry['sugg_idx']
        
        geo_source = next(s for s in sources if s['source_idx'] == sugg_idx)
        non_geo_sources = [s for s in sources if s['source_idx'] != sugg_idx]
        
        geo_prob = geo_source['geo_probability']
        
        if non_geo_sources:
            max_non_geo = max(s['geo_probability'] for s in non_geo_sources)
            gap = geo_prob - max_non_geo
            
            # Pattern detection
            if geo_prob < 0.1:
                patterns['geo_prob_below_0_1'] += 1
            if geo_prob < 0.2:
                patterns['geo_prob_below_0_2'] += 1
            if geo_prob < 0.3:
                patterns['geo_prob_below_0_3'] += 1
            if geo_prob < 0.4:
                patterns['geo_prob_below_0_4'] += 1
            if geo_prob < 0.5:
                patterns['geo_prob_below_0_5'] += 1
            
            if geo_prob < 0.01:
                patterns['extreme_failures'].append({
                    'entry_idx': entry['entry_idx'],
                    'geo_prob': float(geo_prob),
                    'max_non_geo': float(max_non_geo),
                    'gap': float(gap)
                })
            
            if max_non_geo > 0.7:
                patterns['high_non_geo_competitor'] += 1
            if max_non_geo > 0.8:
                patterns['very_high_non_geo_competitor'] += 1
            
            if gap < 0.1:
                patterns['close_competition'] += 1
            if gap < 0.05:
                patterns['very_close_competition'] += 1
            if gap < 0:
                patterns['negative_gap'] += 1
    
    total_failed = len(failed)
    
    print(f"\nPatterns in {total_failed} Failed Cases:")
    print(f"  GEO prob < 0.1: {patterns['geo_prob_below_0_1']} ({100*patterns['geo_prob_below_0_1']/total_failed:.1f}%)")
    print(f"  GEO prob < 0.2: {patterns['geo_prob_below_0_2']} ({100*patterns['geo_prob_below_0_2']/total_failed:.1f}%)")
    print(f"  GEO prob < 0.3: {patterns['geo_prob_below_0_3']} ({100*patterns['geo_prob_below_0_3']/total_failed:.1f}%)")
    print(f"  GEO prob < 0.4: {patterns['geo_prob_below_0_4']} ({100*patterns['geo_prob_below_0_4']/total_failed:.1f}%)")
    print(f"  GEO prob < 0.5: {patterns['geo_prob_below_0_5']} ({100*patterns['geo_prob_below_0_5']/total_failed:.1f}%)")
    print(f"  Non-GEO competitor > 0.7: {patterns['high_non_geo_competitor']} ({100*patterns['high_non_geo_competitor']/total_failed:.1f}%)")
    print(f"  Non-GEO competitor > 0.8: {patterns['very_high_non_geo_competitor']} ({100*patterns['very_high_non_geo_competitor']/total_failed:.1f}%)")
    print(f"  Gap < 0.1 (close): {patterns['close_competition']} ({100*patterns['close_competition']/total_failed:.1f}%)")
    print(f"  Gap < 0.05 (very close): {patterns['very_close_competition']} ({100*patterns['very_close_competition']/total_failed:.1f}%)")
    print(f"  Negative gap: {patterns['negative_gap']} ({100*patterns['negative_gap']/total_failed:.1f}%)")
    print(f"  Extreme failures (GEO prob < 0.01): {len(patterns['extreme_failures'])} ({100*len(patterns['extreme_failures'])/total_failed:.1f}%)")
    
    return patterns


def analyze_semantic_features_in_failures(failed_entries: List[Dict], dataset_path: Path):
    """Analyze semantic feature patterns in failed cases."""
    print("\n" + "="*80)
    print("Semantic Feature Analysis in Failed Cases")
    print("="*80)
    
    # Load original dataset
    with open(dataset_path, 'r') as f:
        all_data = json.load(f)
    
    failed_entry_indices = {e['entry_idx'] for e in failed_entries}
    failed_data = [entry for i, entry in enumerate(all_data) if i in failed_entry_indices]
    
    print(f"Loaded {len(failed_data)} failed entries from original dataset")
    
    semantic_extractor = SemanticFeatureExtractor()
    
    # Extract semantic features
    geo_semantic_features = []
    non_geo_semantic_features = []
    competitor_semantic_features = []  # The non-GEO source that "won"
    
    for entry in failed_data:
        entry_idx = entry.get('entry_idx', all_data.index(entry) if entry in all_data else -1)
        if entry_idx == -1:
            # Try to find by matching structure
            for i, e in enumerate(all_data):
                if e.get('sugg_idx') == entry.get('sugg_idx') and len(e.get('sources', [])) == len(entry.get('sources', [])):
                    entry_idx = i
                    break
        
        sugg_idx = entry.get('sugg_idx')
        sources = entry.get('sources', [])
        
        # Find the competitor (non-GEO source with highest prob)
        entry_predictions = next((e for e in failed_entries if e['entry_idx'] == entry_idx), None)
        if not entry_predictions:
            continue
        
        max_prob_source = max(entry_predictions['sources'], key=lambda s: s['geo_probability'])
        competitor_idx = max_prob_source['source_idx']
        
        for source_idx, source in enumerate(sources):
            cleaned_text = source.get('cleaned_text', '')
            if not cleaned_text:
                continue
            
            semantic_scores = semantic_extractor.extract_pattern_scores(cleaned_text)
            
            if source_idx == sugg_idx:
                geo_semantic_features.append(semantic_scores)
            elif source_idx == competitor_idx:
                competitor_semantic_features.append(semantic_scores)
            else:
                non_geo_semantic_features.append(semantic_scores)
    
    if not geo_semantic_features or not competitor_semantic_features:
        print("Could not extract semantic features")
        return {}
    
    geo_features = np.array(geo_semantic_features)
    competitor_features = np.array(competitor_semantic_features)
    
    feature_names = [
        "QA_Blocks", "Over-Chunking", "Header_Stuffing",
        "Entity_Attribution", "Citation_Embedding"
    ]
    
    print("\nSemantic Feature Means in Failed Cases:")
    print("  True GEO sources vs Winning Non-GEO competitors:")
    for i, name in enumerate(feature_names):
        geo_mean = geo_features[:, i].mean()
        comp_mean = competitor_features[:, i].mean()
        diff = comp_mean - geo_mean
        print(f"    {name}:")
        print(f"      GEO: {geo_mean:.4f}, Competitor: {comp_mean:.4f}, Diff: {diff:.4f}")
        if diff > 0.01:
            print(f"      ⚠️  Competitor has HIGHER {name} score!")
    
    return {
        'geo_means': geo_features.mean(axis=0).tolist(),
        'competitor_means': competitor_features.mean(axis=0).tolist(),
        'feature_names': feature_names
    }


def visualize_failures(prob_results, patterns, output_dir: Path, label: str):
    """Create visualizations of failure patterns."""
    print("\n" + "="*80)
    print("Creating Visualizations")
    print("="*80)
    
    # Plot 1: Probability distributions comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # GEO probabilities comparison
    axes[0].bar(['Successful', 'Failed'],
                [prob_results['successful']['geo_prob_mean'],
                 prob_results['failed']['geo_prob_mean']],
                yerr=[prob_results['successful']['geo_prob_std'],
                      prob_results['failed']['geo_prob_std']],
                capsize=5, alpha=0.7, color=['green', 'red'])
    axes[0].set_ylabel('GEO Probability')
    axes[0].set_title('Average True GEO Source Probability')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim([0, 1])
    
    # Probability gaps
    axes[1].bar(['Successful', 'Failed'],
                [prob_results['successful']['gap_mean'],
                 prob_results['failed']['gap_mean']],
                yerr=[prob_results['successful']['gap_std'],
                      prob_results['failed']['gap_std']],
                capsize=5, alpha=0.7, color=['green', 'red'])
    axes[1].set_ylabel('Probability Gap (GEO - Max Non-GEO)')
    axes[1].set_title('Average Probability Gap')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Max non-GEO probabilities
    axes[2].bar(['Successful', 'Failed'],
                [prob_results['successful']['max_non_geo_mean'],
                 prob_results['failed']['max_non_geo_mean']],
                capsize=5, alpha=0.7, color=['green', 'red'])
    axes[2].set_ylabel('Max Non-GEO Probability')
    axes[2].set_title('Average Max Non-GEO Competitor Probability')
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].set_ylim([0, 1])
    
    plt.tight_layout()
    plot_path = output_dir / f'ranking_failure_analysis_{label}.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Saved: {plot_path}")
    plt.close()


def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(description="Analyze ranking failures.")
    parser.add_argument('--predictions', type=str,
                        default='src/classification/output/neural_with_semantic_predictions.json',
                        help='Path to predictions JSON file')
    parser.add_argument('--dataset', type=str, default='optimization_dataset.json',
                        help='Path to optimization dataset JSON file')
    parser.add_argument('--label', type=str, default='neural_with_semantic',
                        help='Label to use for outputs')
    parser.add_argument('--output-dir', type=str, default='src/classification/output',
                        help='Directory to save analysis outputs')
    args = parser.parse_args()
    
    project_root = Path(__file__).parent
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    predictions_path = Path(args.predictions)
    if not predictions_path.is_absolute():
        predictions_path = project_root / args.predictions
    predictions_path = predictions_path.resolve()
    
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = project_root / args.dataset
    dataset_path = dataset_path.resolve()
    
    print("="*80)
    print(f"Ranking Failure Analysis: {args.label}")
    print("="*80)
    print(f"Loading predictions from: {predictions_path}")
    
    entries = load_predictions(predictions_path)
    print(f"Loaded {len(entries)} entries")
    
    # Separate successful and failed
    successful, failed = analyze_ranking_accuracy(entries)
    
    print(f"\n{'='*80}")
    print(f"Results:")
    print(f"  Successful rankings: {len(successful)} ({100*len(successful)/len(entries):.1f}%)")
    print(f"  Failed rankings: {len(failed)} ({100*len(failed)/len(entries):.1f}%)")
    print(f"{'='*80}")
    
    # Analysis 1: Probability distributions
    prob_results = analyze_probability_distributions(successful, failed)
    
    # Analysis 2: Failed case patterns
    patterns = analyze_failed_case_patterns(failed)
    
    # Analysis 3: Semantic features in failures
    if dataset_path.exists():
        semantic_results = analyze_semantic_features_in_failures(failed, dataset_path)
    else:
        print(f"\nDataset not found at {dataset_path}, skipping semantic feature analysis")
        semantic_results = {}
    
    # Visualize
    visualize_failures(prob_results, patterns, output_dir, args.label)
    
    # Save results
    results = {
        'summary': {
            'total_entries': len(entries),
            'successful': len(successful),
            'failed': len(failed),
            'ranking_accuracy': len(successful) / len(entries)
        },
        'probability_analysis': prob_results,
        'failure_patterns': patterns,
        'semantic_features': semantic_results
    }
    
    results_path = output_dir / f'ranking_failure_analysis_{args.label}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nResults saved to: {results_path}")
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


if __name__ == '__main__':
    main()

