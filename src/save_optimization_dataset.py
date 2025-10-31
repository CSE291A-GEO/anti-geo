"""
Save optimization progression dataset showing source transformations through each GEO method.

This creates a dataset with:
1. Original source text
2. Unoptimized baseline (identity)
3. optimize_1 (fluent_gpt)
4. optimize_2 (unique_words_gpt)
... and so on for each optimization method

Usage:
    python save_optimization_dataset.py
"""

import json
import os
from typing import List, Dict
from datasets import load_dataset
from geo_functions import *

# Map of GEO method names to functions (same as run_geo.py)
def identity(summary, source=None):
    return summary

GEO_METHODS = {
    'identity': identity,
    'fluent_gpt': fluent_optimization_gpt,
    'unique_words_gpt': unique_words_optimization_gpt,
    'authoritative_mine': authoritative_optimization_mine,
    'more_quotes_mine': more_quotes_mine,
    'citing_credible_mine': citing_credible_sources_mine,
    'simple_language_mine': simple_language_mine,
    'technical_terms_mine': technical_terms_mine,
    'stats_optimization_gpt': stats_optimization_mine,
    'seo_optimize_mine2': seo_optimize_mine2,
}


if __name__ == '__main__':
    # Same data intake as run_geo.py
    dataset = load_dataset("GEO-Optim/geo-bench", 'test')
    
    optimization_records = []
    
    print("="*80)
    print("CREATING OPTIMIZATION DATASET")
    print("="*80)
    print()
    
    for i, k in enumerate(dataset['test']):
        query = k['query']
        sugg_idx = int(k['sugg_idx'])
        
        # Extract sources from dataset (same as run_geo.py)
        dataset_sources = [
            s.get('cleaned_text', s.get('raw_text', '')) 
            for s in k['sources']
        ]
        
        # Get the target source to optimize
        original_text = dataset_sources[sugg_idx]
        
        print(f"[{i+1}/{len(dataset['test'])}] Query: {query[:60]}...")
        print(f"  Optimizing source [{sugg_idx}]: {k['sources'][sugg_idx].get('url', 'N/A')}")
        
        # Create record with original source
        record = {
            'query': query,
            'source_index': sugg_idx,
            'source_url': k['sources'][sugg_idx].get('url', ''),
            'tags': k.get('tags', []),
            'original_source': original_text,
            'optimizations': {}
        }
        
        # Apply each GEO method to the target source
        for method_name, method_func in GEO_METHODS.items():
            try:
                print(f"    {method_name}...", end=' ', flush=True)
                optimized_text = method_func(original_text)
                record['optimizations'][method_name] = optimized_text
                print("✓")
            except Exception as e:
                print(f"✗ ({str(e)[:50]})")
                record['optimizations'][method_name] = None
        
        optimization_records.append(record)
        print()
        
        # TODO: Remove after debugging - process only first query
        break
    
    # Save to file
    output_file = 'optimization_dataset.json'
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(optimization_records, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(optimization_records)} records")
    
    # Print statistics
    print()
    print("="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Total records: {len(optimization_records)}")
    print(f"Optimization methods per record: {len(GEO_METHODS)}")
    
    # Calculate success rates
    method_success = {method: 0 for method in GEO_METHODS}
    for record in optimization_records:
        for method, result in record['optimizations'].items():
            if result is not None:
                method_success[method] += 1
    
    print()
    print("Success rates:")
    for method, count in method_success.items():
        rate = (count / len(optimization_records)) * 100 if optimization_records else 0
        print(f"  {method:30s} {count}/{len(optimization_records)} ({rate:.1f}%)")
    
    print()
    print(f"✓ Dataset saved to: {output_file}")
    print()
    print("Example record structure:")
    if optimization_records:
        print(json.dumps({
            'query': optimization_records[0]['query'],
            'source_index': optimization_records[0]['source_index'],
            'original_source': optimization_records[0]['original_source'][:100] + '...',
            'optimizations': {
                method: (text[:100] + '...' if text else None)
                for method, text in list(optimization_records[0]['optimizations'].items())[:3]
            }
        }, indent=2))

