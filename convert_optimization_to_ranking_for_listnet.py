"""
Convert optimization_dataset.json to ranking format for ListNet.
Assigns rank 1 to sugg_idx source, rank 2 to all others.
"""

import json
import sys
from pathlib import Path

def convert_to_ranking_format(input_path: str, output_path: str):
    """Convert optimization dataset to ranking format."""
    print(f"Loading data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(data, dict):
        entries = list(data.values())
    else:
        entries = data
    
    print(f"Processing {len(entries)} entries...")
    
    ranking_entries = []
    for i, entry in enumerate(entries):
        query = entry.get('query', '')
        sugg_idx = entry.get('sugg_idx', -1)
        sources = entry.get('sources', [])
        
        if sugg_idx < 0 or sugg_idx >= len(sources):
            print(f"Warning: Entry {i} has invalid sugg_idx {sugg_idx}, skipping...")
            continue
        
        # Create ranking entry
        ranking_entry = {
            'query': query,
            'entry_idx': i,
            'sources': []
        }
        
        # Assign ranks: sugg_idx gets rank 1, others get rank 2
        for j, source in enumerate(sources):
            rank = 1 if j == sugg_idx else 2
            ranking_source = {
                **source,
                'source_idx': j,
                'ge_rank': rank,  # Use ge_rank field for ListNet
                'se_rank': rank   # Also set se_rank as fallback
            }
            ranking_entry['sources'].append(ranking_source)
        
        ranking_entries.append(ranking_entry)
    
    print(f"Converted {len(ranking_entries)} entries to ranking format")
    
    # Save
    print(f"Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ranking_entries, f, indent=2, ensure_ascii=False)
    
    print("Conversion complete!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_optimization_to_ranking_for_listnet.py <input_json> <output_json>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    convert_to_ranking_format(input_path, output_path)

