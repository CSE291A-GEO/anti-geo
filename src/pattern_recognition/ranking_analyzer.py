"""
GEO Score Analyzer for GEO Detection

This module implements a GEO score-based detection method that compares semantic
similarity GEO scores between the original dataset (from HuggingFace) and the
optimized dataset. The source with the greatest relative increase in GEO score
is identified as the GEO-optimized source.

GEO scores are calculated using semantic similarity matching (parsed=False for speed).
The source with the greatest relative increase in GEO score (optimized_score - original_score) / original_score
is identified as the GEO-optimized source.

The hypothesis is that GEO optimization increases the semantic similarity to known GEO patterns,
and the source with the largest relative increase is the GEO-optimized one.
"""

from typing import List, Dict, Any, Tuple, Optional
from datasets import load_dataset
import json
from pathlib import Path
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pattern_recognition.semantic_matching import SemanticGEODetector


class RankingAnalyzer:
    """
    Analyzer that compares GEO scores between original and optimized datasets.
    """
    
    def __init__(
        self, 
        original_dataset_path: Optional[str] = None, 
        optimized_dataset_path: Optional[str] = None,
        detector: Optional[SemanticGEODetector] = None
    ):
        """
        Initialize the ranking analyzer.
        
        Args:
            original_dataset_path: Path to original dataset JSON file. If None, loads from HuggingFace.
            optimized_dataset_path: Path to optimized dataset JSON file. If None, uses default path.
            detector: SemanticGEODetector instance. If None, creates a new one.
        """
        self.original_dataset_path = original_dataset_path
        self.optimized_dataset_path = optimized_dataset_path
        self.detector = detector if detector is not None else SemanticGEODetector(threshold=0.75)
        
    def load_original_dataset(self):
        """Load the original dataset from HuggingFace or file."""
        if self.original_dataset_path:
            with open(self.original_dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return list(data.values())
            else:
                return data
        else:
            # Load from HuggingFace
            dataset = load_dataset("GEO-Optim/geo-bench", 'test')
            return dataset['test']
    
    def load_optimized_dataset(self) -> List[Dict[str, Any]]:
        """Load the optimized dataset from JSON file."""
        if self.optimized_dataset_path:
            dataset_path = Path(self.optimized_dataset_path)
        else:
            # Default path
            dataset_path = Path(__file__).parent.parent.parent / 'optimization_dataset.json'
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Optimized dataset not found at {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return list(data.values())
        else:
            return data
    
    def get_geo_scores(self, entry: Dict[str, Any], use_cleaned_text: bool = True) -> Dict[str, float]:
        """
        Calculate GEO scores for each source using semantic similarity matching.
        
        Args:
            entry: Dataset entry with sources list
            use_cleaned_text: If True, use cleaned_text; otherwise use raw_text
            
        Returns:
            Dictionary mapping source URL to its GEO score
        """
        geo_scores = {}
        sources = entry.get('sources', [])
        
        for idx, source in enumerate(sources):
            url = source.get('url', '')
            if not url:
                continue
            
            # Get text to score
            if use_cleaned_text:
                text = source.get('cleaned_text', '')
                if not text:
                    text = source.get('raw_text', '')
            else:
                text = source.get('raw_text', '')
                if not text:
                    text = source.get('cleaned_text', '')
            
            if not text:
                continue
            
            # Calculate GEO score using semantic matching (parsed=False for speed)
            try:
                s_geo_max, _ = self.detector.score(text, top_k=3, parsed=False)
                geo_scores[url] = float(s_geo_max)
            except Exception as e:
                # If scoring fails, skip this source
                print(f"    Warning: Failed to score source {url[:50]}...: {str(e)[:50]}")
                continue
        
        return geo_scores
    
    def calculate_geo_score_changes(
        self,
        original_entry: Dict[str, Any],
        optimized_entry: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate the GEO score change for each source between original and optimized datasets.
        
        Args:
            original_entry: Entry from original dataset
            optimized_entry: Entry from optimized dataset
            
        Returns:
            Dictionary mapping source URL to GEO score change information
        """
        # Get GEO scores for original dataset (use cleaned_text)
        original_scores = self.get_geo_scores(original_entry, use_cleaned_text=True)
        
        # Get GEO scores for optimized dataset (use cleaned_text which has the optimized content)
        optimized_scores = self.get_geo_scores(optimized_entry, use_cleaned_text=True)
        
        # Get source positions for reference
        original_sources = original_entry.get('sources', [])
        optimized_sources = optimized_entry.get('sources', [])
        
        original_positions = {}
        for idx, source in enumerate(original_sources):
            url = source.get('url', '')
            if url:
                original_positions[url] = idx
        
        optimized_positions = {}
        for idx, source in enumerate(optimized_sources):
            url = source.get('url', '')
            if url:
                optimized_positions[url] = idx
        
        score_changes = {}
        
        # Get all unique URLs from both datasets
        all_urls = set(original_scores.keys()) | set(optimized_scores.keys())
        
        for url in all_urls:
            orig_score = original_scores.get(url)
            opt_score = optimized_scores.get(url)
            orig_pos = original_positions.get(url)
            opt_pos = optimized_positions.get(url)
            
            if orig_score is None:
                # Source appeared in optimized but not original
                score_changes[url] = {
                    'original_geo_score': None,
                    'optimized_geo_score': opt_score,
                    'absolute_change': None,
                    'relative_change': None,  # New source
                    'original_position': None,
                    'optimized_position': opt_pos
                }
            elif opt_score is None:
                # Source appeared in original but not optimized
                score_changes[url] = {
                    'original_geo_score': orig_score,
                    'optimized_geo_score': None,
                    'absolute_change': None,
                    'relative_change': None,  # Removed source
                    'original_position': orig_pos,
                    'optimized_position': None
                }
            else:
                # Source exists in both
                absolute_change = opt_score - orig_score
                
                # Calculate relative change: (optimized - original) / original
                # Handle division by zero
                if orig_score > 0:
                    relative_change = absolute_change / orig_score
                elif absolute_change > 0:
                    # If original was 0 but optimized is positive, use a large relative change
                    relative_change = float('inf')
                else:
                    # If both are 0, relative change is 0
                    relative_change = 0.0
                
                score_changes[url] = {
                    'original_geo_score': orig_score,
                    'optimized_geo_score': opt_score,
                    'absolute_change': absolute_change,  # Positive = increased GEO score
                    'relative_change': relative_change,  # Relative increase/decrease
                    'original_position': orig_pos,
                    'optimized_position': opt_pos
                }
        
        return score_changes
    
    def identify_geo_source(
        self,
        original_entry: Dict[str, Any],
        optimized_entry: Dict[str, Any]
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Identify the GEO-optimized source by finding the one with the greatest relative increase in GEO score.
        
        Args:
            original_entry: Entry from original dataset
            optimized_entry: Entry from optimized dataset
            
        Returns:
            Tuple of (geo_source_url, score_changes_dict)
            Returns (None, score_changes) if no clear GEO source identified
        """
        score_changes = self.calculate_geo_score_changes(original_entry, optimized_entry)
        
        # Find source with maximum relative increase in GEO score
        max_relative_change = float('-inf')
        geo_source_url = None
        
        for url, change_info in score_changes.items():
            relative_change = change_info.get('relative_change')
            # Only consider positive relative changes (increased GEO score)
            if relative_change is not None and relative_change != float('inf'):
                if relative_change > max_relative_change:
                    max_relative_change = relative_change
                    geo_source_url = url
            elif relative_change == float('inf'):
                # If relative change is infinity (original was 0, optimized is positive), use this
                max_relative_change = float('inf')
                geo_source_url = url
        
        # If no positive changes, return None
        if max_relative_change <= 0 and max_relative_change != float('inf'):
            geo_source_url = None
        
        return geo_source_url, score_changes
    
    def analyze_entry(
        self,
        original_entry: Dict[str, Any],
        optimized_entry: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze a single entry pair and return detailed attribution score analysis.
        
        Args:
            original_entry: Entry from original dataset
            optimized_entry: Entry from optimized dataset
            
        Returns:
            Dictionary with analysis results
        """
        # Verify entries match
        if original_entry.get('query') != optimized_entry.get('query'):
            return {
                'error': 'Query mismatch',
                'original_query': original_entry.get('query'),
                'optimized_query': optimized_entry.get('query')
            }
        
        geo_source_url, score_changes = self.identify_geo_source(original_entry, optimized_entry)
        
        # Get sugg_idx from optimized entry (the known GEO source)
        known_geo_idx = optimized_entry.get('sugg_idx')
        known_geo_url = None
        known_geo_ranking = None
        if known_geo_idx is not None:
            sources = optimized_entry.get('sources', [])
            if 0 <= known_geo_idx < len(sources):
                known_geo_url = sources[known_geo_idx].get('url', '')
                # Ranking is 1-indexed (1 = first, 2 = second, etc.)
                known_geo_ranking = known_geo_idx + 1
        
        # Check if our identified GEO source matches the known one
        correct_identification = (geo_source_url == known_geo_url) if geo_source_url and known_geo_url else False
        
        # Get GEO score change for identified source
        identified_relative_change = None
        identified_absolute_change = None
        if geo_source_url and geo_source_url in score_changes:
            change_info = score_changes[geo_source_url]
            identified_relative_change = change_info.get('relative_change')
            identified_absolute_change = change_info.get('absolute_change')
        
        # Get GEO score change for actual GEO source
        actual_relative_change = None
        actual_absolute_change = None
        if known_geo_url and known_geo_url in score_changes:
            change_info = score_changes[known_geo_url]
            actual_relative_change = change_info.get('relative_change')
            actual_absolute_change = change_info.get('absolute_change')
        
        return {
            'query': original_entry.get('query'),
            'identified_geo_source_url': geo_source_url,
            'identified_relative_change': identified_relative_change,
            'identified_absolute_change': identified_absolute_change,
            'known_geo_source_url': known_geo_url,
            'known_geo_source_idx': known_geo_idx,
            'known_geo_source_ranking': known_geo_ranking,  # 1-indexed ranking (1-5)
            'actual_relative_change': actual_relative_change,
            'actual_absolute_change': actual_absolute_change,
            'correct_identification': correct_identification,
            'score_changes': score_changes,
            'num_sources_original': len(original_entry.get('sources', [])),
            'num_sources_optimized': len(optimized_entry.get('sources', []))
        }
    
    def analyze_all_entries(
        self,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze all entries in the datasets.
        
        Args:
            limit: Optional limit on number of entries to analyze
            
        Returns:
            Dictionary with overall analysis results
        """
        original_dataset = self.load_original_dataset()
        optimized_dataset = self.load_optimized_dataset()
        
        if limit:
            original_dataset = original_dataset[:limit]
            optimized_dataset = optimized_dataset[:limit]
        
        if len(original_dataset) != len(optimized_dataset):
            return {
                'error': f'Dataset length mismatch: original={len(original_dataset)}, optimized={len(optimized_dataset)}'
            }
        
        results = []
        correct_identifications = 0
        total_entries = 0
        geo_source_rankings = []  # Track rankings of actual GEO sources (1-indexed)
        
        for idx, (orig_entry, opt_entry) in enumerate(zip(original_dataset, optimized_dataset)):
            if (idx + 1) % 50 == 0:
                print(f"  Processing entry {idx + 1}/{len(original_dataset)}... "
                      f"({correct_identifications}/{total_entries} correct so far)")
            
            analysis = self.analyze_entry(orig_entry, opt_entry)
            
            if 'error' not in analysis:
                total_entries += 1
                if analysis.get('correct_identification', False):
                    correct_identifications += 1
                
                # Track ranking of actual GEO source
                known_geo_ranking = analysis.get('known_geo_source_ranking')
                if known_geo_ranking is not None:
                    geo_source_rankings.append(known_geo_ranking)
            
            results.append(analysis)
        
        accuracy = (correct_identifications / total_entries * 100) if total_entries > 0 else 0.0
        avg_geo_ranking = (sum(geo_source_rankings) / len(geo_source_rankings)) if geo_source_rankings else None
        
        return {
            'total_entries': len(results),
            'valid_entries': total_entries,
            'correct_identifications': correct_identifications,
            'accuracy': accuracy,
            'average_geo_source_ranking': avg_geo_ranking,
            'geo_source_rankings': geo_source_rankings,
            'results': results
        }


def main():
    """Main function to run GEO score analysis."""
    import argparse
    from dotenv import load_dotenv
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Analyze GEO score changes between original and optimized datasets')
    parser.add_argument('--original', type=str, help='Path to original dataset JSON file')
    parser.add_argument('--optimized', type=str, help='Path to optimized dataset JSON file')
    parser.add_argument('--limit', type=int, help='Limit number of entries to analyze')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Initialize detector
    print("Initializing SemanticGEODetector...")
    detector = SemanticGEODetector(threshold=0.75)
    print("Detector ready!")
    print()
    
    analyzer = RankingAnalyzer(
        original_dataset_path=args.original,
        optimized_dataset_path=args.optimized,
        detector=detector
    )
    
    print("="*80)
    print("GEO SCORE ANALYSIS")
    print("="*80)
    print()
    print("Analyzing GEO score changes between original and optimized datasets...")
    print("(Using parsed=False for speed)")
    print()
    
    results = analyzer.analyze_all_entries(limit=args.limit)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        return
    
    print(f"Total entries analyzed: {results['total_entries']}")
    print(f"Valid entries: {results['valid_entries']}")
    print(f"Correct identifications: {results['correct_identifications']}")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    avg_ranking = results.get('average_geo_source_ranking')
    if avg_ranking is not None:
        print(f"Average ranking of actual GEO source: {avg_ranking:.2f} (out of 5)")
    print()
    
    # Show some examples
    print("Sample results (first 10):")
    print("-"*80)
    for i, result in enumerate(results['results'][:10]):
        if 'error' not in result:
            print(f"\nEntry {i+1}:")
            print(f"  Query: {result['query'][:60]}...")
            print(f"  Identified GEO source: {result['identified_geo_source_url']}")
            print(f"  Known GEO source: {result['known_geo_source_url']}")
            print(f"  Correct: {'✓' if result['correct_identification'] else '✗'}")
            
            # Show GEO score changes
            score_changes = result['score_changes']
            print(f"  GEO score changes:")
            for url, change_info in list(score_changes.items())[:3]:
                orig_score = change_info.get('original_geo_score')
                opt_score = change_info.get('optimized_geo_score')
                rel_change = change_info.get('relative_change')
                if orig_score is not None and opt_score is not None:
                    if rel_change == float('inf'):
                        print(f"    {url[:50]}...: {orig_score:.4f} → {opt_score:.4f} (relative: ∞)")
                    elif rel_change is not None:
                        print(f"    {url[:50]}...: {orig_score:.4f} → {opt_score:.4f} (relative: {rel_change*100:+.1f}%)")
    
    # Save results if output path specified
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")
    else:
        # Default output path
        output_path = Path(__file__).parent.parent.parent / 'ranking_analysis.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

