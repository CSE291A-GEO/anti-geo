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

# Ordered dictionary - adversarial method comes FIRST
GEO_METHODS = {
    'adversarial_geo_mine': adversarial_geo_mine,  # ADVERSARIAL: Negative baseline with deceptive practices
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
    import time as time_module
    start_time = time_module.time()
    
    # Same data intake as run_geo.py
    dataset = load_dataset("GEO-Optim/geo-bench", 'test')
    
    # Output file setup - we'll append to this incrementally
    output_file = 'optimization_dataset.json'
    
    # Check if file exists and load existing records
    if os.path.exists(output_file):
        print(f"Loading existing dataset from {output_file}...")
        with open(output_file, 'r', encoding='utf-8') as f:
            optimization_records = json.load(f)
        print(f"  Loaded {len(optimization_records)} existing records")
        starting_index = len(optimization_records)
    else:
        optimization_records = []
        starting_index = 0
    
    print("="*80)
    print("CREATING OPTIMIZATION DATASET")
    print("="*80)
    print(f"Total queries in dataset: {len(dataset['test'])}")
    print(f"Already processed: {starting_index}")
    print(f"Remaining to process: {len(dataset['test']) - starting_index}")
    print(f"Optimization methods: {len(GEO_METHODS)}")
    print(f"Methods: {', '.join(GEO_METHODS.keys())}")
    print()
    print("Starting optimization process...")
    print("="*80)
    print()
    
    for i, k in enumerate(dataset['test']):
        # Skip already processed queries
        if i < starting_index:
            continue
            
        query = k['query']
        sugg_idx = int(k['sugg_idx'])
        
        # Extract sources from dataset (same as run_geo.py)
        dataset_sources = [
            s.get('cleaned_text', s.get('raw_text', '')) 
            for s in k['sources']
        ]
        
        # Get the target source to optimize
        original_text = dataset_sources[sugg_idx]
        
        query_start_time = time_module.time()
        print(f"[{i+1}/{len(dataset['test'])}] Query: {query[:60]}...")
        print(f"  Target source index: [{sugg_idx}]")
        print(f"  Source URL: {k['sources'][sugg_idx].get('url', 'N/A')}")
        print(f"  Tags: {', '.join(k.get('tags', [])[:3])}{'...' if len(k.get('tags', [])) > 3 else ''}")
        
        # Apply ALL 10 GEO methods SEQUENTIALLY to the target source
        # Each optimization builds on the previous one
        current_text = original_text
        method_num = 0
        total_methods = len(GEO_METHODS)
        
        print(f"  Applying {total_methods} optimizations sequentially...")
        
        for method_name, method_func in GEO_METHODS.items():
            method_num += 1
            try:
                print(f"    [{method_num}/{total_methods}] Applying: {method_name}...", flush=True)
                print(f"        Input length: {len(current_text)} chars", flush=True)
                
                optimized_text = method_func(current_text)
                
                if optimized_text and len(optimized_text) > 100:  # Sanity check
                    output_len = len(optimized_text)
                    change_pct = ((output_len - len(current_text)) / len(current_text)) * 100
                    print(f"        Output length: {output_len} chars ({change_pct:+.1f}%)", flush=True)
                    print(f"        Status: ✓ Success", flush=True)
                    # Update current_text for next optimization
                    current_text = optimized_text
                else:
                    print(f"        Status: ✗ Empty or too short, keeping previous version", flush=True)
            except Exception as e:
                print(f"        Status: ✗ Error: {str(e)[:80]}, keeping previous version", flush=True)
        
        # Create ONE record with the final optimized text (after all 10 optimizations)
        final_optimized_text = current_text
        
        print(f"  Final optimization: {len(final_optimized_text)} chars (original: {len(original_text)} chars)")
        print(f"  Total change: {((len(final_optimized_text) - len(original_text)) / len(original_text)) * 100:+.1f}%")
        
        record = {
            'query': query,
            'sugg_idx': sugg_idx,
            'tags': k.get('tags', []),
            'sources': []
        }
        
        # Copy all sources with their original structure
        for src_idx, source in enumerate(k['sources']):
            source_copy = {
                'url': source.get('url', ''),
                'raw_text': source.get('raw_text', ''),
                'cleaned_text': source.get('cleaned_text', '')
            }
            
            # Replace cleaned_text of target source with FINAL optimized version
            if src_idx == sugg_idx:
                source_copy['cleaned_text'] = final_optimized_text
            
            record['sources'].append(source_copy)
        
        optimization_records.append(record)
        
        # Save immediately after each query is processed
        print(f"  Saving progress to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(optimization_records, f, indent=2, ensure_ascii=False)
        
        query_elapsed = time_module.time() - query_start_time
        print(f"  Query completed in {query_elapsed:.1f}s")
        print(f"  Total records so far: {len(optimization_records)}/{len(dataset['test'])}")
        print()
        
        # TODO: Remove after debugging - process only first query
        # break
    
    # Final summary
    total_elapsed = time_module.time() - start_time
    
    print()
    print("="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"Output file: {output_file}")
    print(f"Total records: {len(optimization_records)}")
    print(f"✓ All records saved incrementally")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    
    # Print statistics
    print()
    print("="*80)
    print("STATISTICS")
    print("="*80)
    
    print(f"Total queries processed: {len(optimization_records)}")
    print(f"Total dataset records created: {len(optimization_records)}")
    print(f"Optimizations per record: {len(GEO_METHODS)} (applied sequentially)")
    print()
    print("Optimization pipeline:")
    for idx, method_name in enumerate(GEO_METHODS.keys(), 1):
        print(f"  {idx}. {method_name}")
    
    print()
    print(f"✓ Dataset saved to: {output_file}")
    print()
    print("Example record structure:")
    if optimization_records:
        example_record = optimization_records[0]
        
        # Find the optimized source (at sugg_idx)
        optimized_source = example_record['sources'][example_record['sugg_idx']]
        
        print(json.dumps({
            'query': example_record['query'][:80] + '...',
            'sugg_idx': example_record['sugg_idx'],
            'tags': example_record['tags'][:3],
            'num_sources': len(example_record['sources']),
            'optimized_source_preview': {
                'url': optimized_source['url'][:50] + '...',
                'cleaned_text_preview': optimized_source['cleaned_text'][:150] + '...',
                'optimizations_applied': list(GEO_METHODS.keys())
            }
        }, indent=2))

