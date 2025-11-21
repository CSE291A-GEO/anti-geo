#!/usr/bin/env python3
"""
Compare Logistic Ordinal classifier before and after improvements.

This script runs the baseline model and the improved model with all enhancements,
then creates a comparison document.
"""

import subprocess
import json
import time
from pathlib import Path
from typing import Dict, Any


def run_experiment(name: str, cmd: list) -> Dict[str, Any]:
    """Run an experiment and return results."""
    print(f"\n{'='*80}")
    print(f"Running: {name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"✓ Completed in {elapsed:.2f}s")
        return {'success': True, 'time': elapsed}
    else:
        print(f"✗ Failed after {elapsed:.2f}s")
        print(f"Error: {result.stderr[:500]}")
        return {'success': False, 'time': elapsed, 'error': result.stderr[:500]}


def load_metrics(metrics_path: Path) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    if not metrics_path.exists():
        return None
    with open(metrics_path, 'r') as f:
        return json.load(f)


def main():
    """Run before/after comparison."""
    project_root = Path(__file__).parent
    output_dir = project_root / "src" / "classification" / "output"
    
    experiments = [
        {
            "name": "Baseline (with semantic features)",
            "cmd": [
                "conda", "run", "-n", "fifteenAI", "python",
                "src/classification/logistic_ordinal_classifier.py",
                "--opt-data", "optimization_dataset.json",
                "--limit", "1000",
                "--train-size", "300",
                "--validation-size", "700",
                "--C", "1.0",
                "--use-semantic-features",
                "--output", "src/classification/output",
                "--model-name", "logistic_baseline_comparison.pkl"
            ]
        },
        {
            "name": "Improved (all enhancements)",
            "cmd": [
                "conda", "run", "-n", "fifteenAI", "python",
                "src/classification/logistic_ordinal_classifier.py",
                "--opt-data", "optimization_dataset.json",
                "--limit", "1000",
                "--train-size", "300",
                "--validation-size", "700",
                "--C", "1.0",
                "--use-semantic-features",
                "--use-excessiveness-features",
                "--use-relative-features",
                "--use-text-quality-features",
                "--enhanced-oversampling-weight", "5.0",
                "--use-extreme-penalty",
                "--use-margin-loss",
                "--margin", "0.2",
                "--extreme-penalty-weight", "2.0",
                "--output", "src/classification/output",
                "--model-name", "logistic_improved_comparison.pkl"
            ]
        }
    ]
    
    results = {}
    for exp in experiments:
        run_result = run_experiment(exp["name"], exp["cmd"])
        results[exp["name"]] = run_result
        
        if run_result['success']:
            # Load metrics
            model_name = exp["cmd"][exp["cmd"].index("--model-name") + 1]
            metrics_path = output_dir / f"{Path(model_name).stem}_metrics.json"
            metrics = load_metrics(metrics_path)
            if metrics:
                results[exp["name"]]['metrics'] = metrics
    
    # Create comparison document
    create_comparison_doc(results, output_dir)
    
    print("\n" + "="*80)
    print("Comparison Complete!")
    print("="*80)


def create_comparison_doc(results: Dict[str, Any], output_dir: Path):
    """Create comparison markdown document."""
    baseline = results.get("Baseline (with semantic features)", {})
    improved = results.get("Improved (all enhancements)", {})
    
    baseline_metrics = baseline.get('metrics', {})
    improved_metrics = improved.get('metrics', {})
    
    doc = f"""# Logistic Ordinal Classifier: Before vs After Improvements

## Overview

This document compares the baseline Logistic Ordinal classifier (with semantic features) against the improved version with all enhancements enabled.

## Improvements Implemented

### 1. Enhanced Features
- **Excessiveness Features**: 6 features measuring how 'excessive' semantic patterns are
- **Relative Features**: Features comparing sources within entries (10 features: max ratio + mean ratio for each of 5 semantic patterns)
- **Text Quality Features**: 10 features measuring text structure, repetition, and citation patterns

### 2. Enhanced Training
- **Higher Oversampling Weight**: Increased from 3.0 to 5.0 to emphasize GEO examples
- **Extreme Failure Penalty**: Penalizes cases where true GEO source gets probability < 0.1
- **Margin-Based Ranking Loss**: Enforces minimum gap (0.2) between GEO and non-GEO probabilities

## Results Comparison

### Validation Accuracy

| Metric | Baseline | Improved | Change |
|--------|----------|----------|--------|
| **Validation Accuracy** | {baseline_metrics.get('validation_accuracy', 'N/A'):.4f} | {improved_metrics.get('validation_accuracy', 'N/A'):.4f} | {improved_metrics.get('validation_accuracy', 0) - baseline_metrics.get('validation_accuracy', 0):.4f} |
| **Ranking Accuracy** | {baseline_metrics.get('validation_ranking_accuracy', 'N/A'):.4f} | {improved_metrics.get('validation_ranking_accuracy', 'N/A'):.4f} | {improved_metrics.get('validation_ranking_accuracy', 0) - baseline_metrics.get('validation_ranking_accuracy', 0):.4f} |
| **F1 Score** | {baseline_metrics.get('validation_f1', 'N/A'):.4f} | {improved_metrics.get('validation_f1', 'N/A'):.4f} | {improved_metrics.get('validation_f1', 0) - baseline_metrics.get('validation_f1', 0):.4f} |

### Detailed Metrics

| Metric | Baseline | Improved | Change |
|--------|----------|----------|--------|
| **Precision** | {baseline_metrics.get('validation_precision', 'N/A'):.4f} | {improved_metrics.get('validation_precision', 'N/A'):.4f} | {improved_metrics.get('validation_precision', 0) - baseline_metrics.get('validation_precision', 0):.4f} |
| **Recall** | {baseline_metrics.get('validation_recall', 'N/A'):.4f} | {improved_metrics.get('validation_recall', 'N/A'):.4f} | {improved_metrics.get('validation_recall', 0) - baseline_metrics.get('validation_recall', 0):.4f} |

### Performance Metrics

| Metric | Baseline | Improved | Change |
|--------|----------|----------|--------|
| **Training Time (s)** | {baseline_metrics.get('training_time_seconds', 'N/A'):.2f} | {improved_metrics.get('training_time_seconds', 'N/A'):.2f} | {improved_metrics.get('training_time_seconds', 0) - baseline_metrics.get('training_time_seconds', 0):.2f} |
| **Prediction Latency (ms)** | {baseline_metrics.get('avg_prediction_latency_ms', 'N/A'):.2f} | {improved_metrics.get('avg_prediction_latency_ms', 'N/A'):.2f} | {improved_metrics.get('avg_prediction_latency_ms', 0) - baseline_metrics.get('avg_prediction_latency_ms', 0):.2f} |

## Feature Counts

| Feature Type | Baseline | Improved |
|--------------|----------|----------|
| **Base Embeddings** | 384 | 384 |
| **Semantic Features** | 5 | 5 |
| **Excessiveness Features** | 0 | 6 |
| **Relative Features** | 0 | 10 |
| **Text Quality Features** | 0 | 10 |
| **Total Features** | 389 | 415 |

## Key Findings

### Ranking Accuracy Improvement

The ranking accuracy gap (difference between classification and ranking accuracy) was:
- **Baseline**: {baseline_metrics.get('validation_accuracy', 0) - baseline_metrics.get('validation_ranking_accuracy', 0):.2%} ({baseline_metrics.get('validation_accuracy', 0):.2%} - {baseline_metrics.get('validation_ranking_accuracy', 0):.2%})
- **Improved**: {improved_metrics.get('validation_accuracy', 0) - improved_metrics.get('validation_ranking_accuracy', 0):.2%} ({improved_metrics.get('validation_accuracy', 0):.2%} - {improved_metrics.get('validation_ranking_accuracy', 0):.2%})
- **Change**: {((improved_metrics.get('validation_accuracy', 0) - improved_metrics.get('validation_ranking_accuracy', 0)) - (baseline_metrics.get('validation_accuracy', 0) - baseline_metrics.get('validation_ranking_accuracy', 0))):.2%}

### Extreme Failure Reduction

Based on the ranking failure analysis, the improvements target:
- **Extreme failures** (GEO prob < 0.01): Reduced through extreme penalty
- **False positives**: Reduced through excessiveness and relative features
- **Close competitions**: Reduced through margin-based loss

## Configuration

### Baseline Configuration
```bash
--use-semantic-features
```

### Improved Configuration
```bash
--use-semantic-features
--use-excessiveness-features
--use-relative-features
--use-text-quality-features
--enhanced-oversampling-weight 5.0
--use-extreme-penalty
--use-margin-loss
--margin 0.2
--extreme-penalty-weight 2.0
```

## Ablation Study Support

All improvements are togglable via command-line arguments, enabling future ablation studies:

- `--use-excessiveness-features`: Toggle excessiveness features
- `--use-relative-features`: Toggle relative features
- `--use-text-quality-features`: Toggle text quality features
- `--enhanced-oversampling-weight`: Adjust oversampling weight
- `--use-extreme-penalty`: Toggle extreme failure penalty
- `--use-margin-loss`: Toggle margin-based ranking loss
- `--margin`: Adjust margin value
- `--extreme-penalty-weight`: Adjust penalty weight

## Conclusion

The improved model addresses the key issues identified in the ranking failure analysis:
1. **Extreme under-confidence**: Addressed through extreme penalty and higher oversampling
2. **False positives**: Addressed through excessiveness and relative features
3. **Relative ranking**: Addressed through margin-based loss

Expected improvements:
- Ranking accuracy should increase significantly
- Extreme failures (GEO prob < 0.01) should decrease
- False positive rate should decrease

"""
    
    output_path = output_dir / "LOGISTIC_IMPROVEMENTS_COMPARISON.md"
    with open(output_path, 'w') as f:
        f.write(doc)
    
    print(f"\nComparison document saved to: {output_path}")


if __name__ == '__main__':
    main()

