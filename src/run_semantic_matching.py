#!/usr/bin/env python3
"""
Semantic Matching Run Script

This script processes all entries from optimization_dataset.json and calculates 
GEO-detection scores for each source's cleaned_text. Sources are ranked by their
GEO scores, helping identify which sources contain GEO-optimized content.

The script toggles parsed mode (parsed=True/False), which employs Gemini 2.0 Flash to extract
only relevant GEO patterns from the text before semantic matching if parsed=True, improving detection
accuracy by focusing on pattern-specific content.

The script tracks how well the detector ranks the actual GEO-optimized source
(identified by sugg_idx) and reports statistics including maximum discrepancy.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from pattern_recognition.semantic_matching import SemanticGEODetector
from dotenv import load_dotenv
import time


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load the optimization dataset from JSON file."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return list(data.values())
    else:
        raise ValueError(f"Unexpected dataset format: {type(data)}")


def score_sources(
    entry: Dict[str, Any],
    detector: SemanticGEODetector
) -> List[Dict[str, Any]]:
    """
    Score all sources in an entry and return ranked results.
    
    Args:
        entry: Dataset entry containing sources
        detector: Initialized SemanticGEODetector
    
    Returns:
        List of source dictionaries with added GEO scores, ranked by score (highest first)
    """
    if 'sources' not in entry or not entry['sources']:
        return []
    
    scored_sources = []
    
    for idx, source in enumerate(entry['sources']):
        # Get the cleaned_text, fallback to raw_text if not available
        text = source.get('cleaned_text', '')
        if not text:
            text = source.get('raw_text', '')
        
        if not text:
            # Skip sources without text
            continue
        
        # Calculate GEO score with parsed mode enabled (uses Gemini to extract relevant patterns)
        s_geo_max, top_matches = detector.score(text, top_k=3, parsed=True)
        
        # Create scored source entry
        scored_source = {
            'source_index': idx,
            'url': source.get('url', 'N/A'),
            's_geo_max': float(s_geo_max),
            'top_matches': [(pid, float(score)) for pid, score in top_matches],
            'text_length': len(text),
            'text_preview': text[:200] + '...' if len(text) > 200 else text
        }
        
        scored_sources.append(scored_source)
    
    # Sort by S_GEO_Max score (highest first - most GEO-like)
    scored_sources.sort(key=lambda x: x['s_geo_max'], reverse=True)
    
    return scored_sources


def main():
    load_dotenv()

    """Main function to run semantic matching on all dataset entries."""
    # Get project root and dataset path
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / 'optimization_dataset.json'
    
    if not dataset_path.exists():
        print(f"Error: Dataset file not found: {dataset_path}")
        return
    
    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_dataset(str(dataset_path))
    print(f"Loaded {len(dataset)} entries")
    
    # Initialize detector
    print(f"\nInitializing SemanticGEODetector...")
    detector = SemanticGEODetector(threshold=0.75)
    print("Detector ready!\n")
    
    # Tracking variables
    max_discrepancy = 0
    total_discrepancy = 0
    perfect_matches = 0  # Count when actual GEO source is ranked #1
    total_entries = 0
    max_discrepancy_entry = None  # Store entry info with max discrepancy
    all_entry_results = []  # Store results for each entry
    
    # Process all entries
    for entry_idx, entry in enumerate(dataset):
        # to avoid rate limiting, sleep for 25 seconds
        time.sleep(25)

        # stop at 50 entries
        if total_entries >= 50:
            break

        query = entry.get('query', 'N/A')
        num_sources = len(entry.get('sources', []))
        sugg_idx = entry.get('sugg_idx', None)
        
        if sugg_idx is None:
            continue  # Skip entries without sugg_idx
        
        total_entries += 1
        
        # Print entry information and source details
        print(f"\n{'='*80}")
        print(f"Entry {entry_idx + 1}/{len(dataset)}")
        print(f"{'='*80}")
        print(f"Query: {query}")
        print(f"Entry Index: {entry_idx}")
        print(f"sugg_idx (Actual GEO Source Index): {sugg_idx}")
        print(f"Total Sources: {num_sources}")
        
        # Print all sources with their original indices and URLs
        print(f"\nSources in Original Order:")
        print(f"{'-'*80}")
        for orig_idx, source in enumerate(entry.get('sources', [])):
            url = source.get('url', 'N/A')
            marker = " ⭐ [ACTUAL GEO]" if orig_idx == sugg_idx else ""
            print(f"  Index {orig_idx}: {url}{marker}")
        
        # Score all sources
        scored_sources = score_sources(entry, detector)
        
        if not scored_sources:
            print(f"Warning: No sources were scored for this entry")
            continue  # Skip entries with no scored sources
        
        # Print ranked sources with their original indices
        print(f"\nRanked Sources (by GEO Score, highest first):")
        print(f"{'-'*80}")
        for rank, scored_source in enumerate(scored_sources, 1):
            orig_idx = scored_source['source_index']
            is_actual_geo = (orig_idx == sugg_idx)
            marker = " ⭐ [ACTUAL GEO]" if is_actual_geo else ""
            print(f"  Rank {rank}{marker}:")
            print(f"    Original Index: {orig_idx}")
            print(f"    URL: {scored_source['url']}")
            print(f"    GEO Score: {scored_source['s_geo_max']:.4f}")
        
        # Find the ranking position of the actual GEO-optimized source
        actual_geo_ranking = None
        actual_geo_source_info = None
        
        for rank, scored_source in enumerate(scored_sources, 1):
            if scored_source['source_index'] == sugg_idx:
                actual_geo_ranking = rank
                actual_geo_source_info = scored_source
                break
        
        if actual_geo_ranking is None:
            print(f"Warning: Source at sugg_idx={sugg_idx} not found in scored sources")
            continue  # Skip if actual GEO source not found in scored sources
        
        # Calculate discrepancy: absolute value of (ranking - 1)
        discrepancy = actual_geo_ranking - 1  # 0 = perfect (rank 1), 1 = rank 2, etc.
        abs_discrepancy = abs(discrepancy)
        
        print(f"\nEvaluation:")
        print(f"  Actual GEO Source (index {sugg_idx}): Rank {actual_geo_ranking} / {len(scored_sources)}")
        print(f"  Discrepancy: {abs_discrepancy} (absolute value of ranking - 1 = |{actual_geo_ranking} - 1|)")
        if len(scored_sources) > 0:
            print(f"  Actual GEO Score: {actual_geo_source_info['s_geo_max']:.4f}")
            print(f"  Top Score: {scored_sources[0]['s_geo_max']:.4f}")
            print(f"  Score Difference: {scored_sources[0]['s_geo_max'] - actual_geo_source_info['s_geo_max']:.4f}")
        
        # Update tracking
        if abs_discrepancy > max_discrepancy:
            max_discrepancy = abs_discrepancy
            max_discrepancy_entry = {
                'entry_idx': entry_idx,
                'query': query,
                'sugg_idx': sugg_idx,
                'actual_ranking': actual_geo_ranking,
                'total_sources': len(scored_sources),
                'actual_score': actual_geo_source_info['s_geo_max'],
                'top_score': scored_sources[0]['s_geo_max'],
                'url': actual_geo_source_info['url']
            }
        
        total_discrepancy += abs_discrepancy
        
        if discrepancy == 0:
            perfect_matches += 1
        
        # Store entry results
        entry_result = {
            'entry_idx': entry_idx,
            'query': query,
            'sugg_idx': sugg_idx,
            'total_sources': num_sources,
            'scored_sources_count': len(scored_sources),
            'actual_geo_ranking': actual_geo_ranking,
            'discrepancy': abs_discrepancy,
            'actual_geo_score': actual_geo_source_info['s_geo_max'],
            'top_score': scored_sources[0]['s_geo_max'] if scored_sources else 0.0,
            'score_difference': (scored_sources[0]['s_geo_max'] - actual_geo_source_info['s_geo_max']) if scored_sources else 0.0,
            'actual_geo_url': actual_geo_source_info['url'],
            'ranked_sources': [
                {
                    'rank': rank,
                    'original_index': scored_source['source_index'],
                    'url': scored_source['url'],
                    'score': scored_source['s_geo_max'],
                    'is_actual_geo': (scored_source['source_index'] == sugg_idx)
                }
                for rank, scored_source in enumerate(scored_sources, 1)
            ]
        }
        all_entry_results.append(entry_result)

        

        # break # TODO: Remove this?

    
    # Calculate final statistics
    avg_discrepancy = total_discrepancy / total_entries if total_entries > 0 else 0.0
    perfect_match_percentage = (perfect_matches / total_entries * 100) if total_entries > 0 else 0.0
    
    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total Entries Processed: {total_entries}")
    print(f"Perfect Matches (Rank #1): {perfect_matches} ({perfect_match_percentage:.1f}%)")
    print(f"Max Discrepancy: {max_discrepancy}")
    print(f"Average Discrepancy: {avg_discrepancy:.2f}")
    
    # Print entry with max discrepancy
    if max_discrepancy_entry:
        print(f"\n{'='*80}")
        print("ENTRY WITH MAXIMUM DISCREPANCY")
        print(f"{'='*80}")
        print(f"Entry Index: {max_discrepancy_entry['entry_idx']}")
        print(f"Query: {max_discrepancy_entry['query']}")
        print(f"Actual GEO Source Index: {max_discrepancy_entry['sugg_idx']}")
        print(f"Ranking: {max_discrepancy_entry['actual_ranking']} / {max_discrepancy_entry['total_sources']}")
        print(f"Discrepancy: {max_discrepancy_entry['actual_ranking'] - 1}")
        print(f"URL: {max_discrepancy_entry['url']}")
        print(f"Actual Score: {max_discrepancy_entry['actual_score']:.4f}")
        print(f"Top Score: {max_discrepancy_entry['top_score']:.4f}")
        print(f"Score Difference: {max_discrepancy_entry['top_score'] - max_discrepancy_entry['actual_score']:.4f}")
    
    # Prepare results dictionary for JSON output
    results = {
        'summary': {
            'total_entries_processed': total_entries,
            'perfect_matches': perfect_matches,
            'perfect_match_percentage': perfect_match_percentage,
            'max_discrepancy': max_discrepancy,
            'average_discrepancy': avg_discrepancy,
            'total_discrepancy': total_discrepancy
        },
        'max_discrepancy_entry': max_discrepancy_entry,
        'entry_results': all_entry_results
    }
    
    # Save results to JSON file
    output_path = project_root / 'parsed_semantic_matching.json'
    print(f"\n{'='*80}")
    print(f"Saving results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved successfully!")


if __name__ == '__main__':
    main()

