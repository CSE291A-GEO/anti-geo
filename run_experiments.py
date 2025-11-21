#!/usr/bin/env python3
"""
Run all model experiments with and without semantic features.
"""

import subprocess
import json
from pathlib import Path
import time

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"✓ Completed in {elapsed:.2f}s")
        return True
    else:
        print(f"✗ Failed after {elapsed:.2f}s")
        print(f"Error: {result.stderr[:500]}")
        return False

def main():
    base_dir = Path(__file__).parent
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    experiments = [
        # SVM
        {
            "name": "SVM Baseline",
            "cmd": [
                "conda", "run", "-n", "fifteenAI", "python",
                "src/classification/svm_classifier.py",
                "--opt-data", "optimization_dataset.json",
                "--limit", "1000", "--train-size", "300", "--validation-size", "700",
                "--C", "1.5",
                "--output", "src/classification/output",
                "--model-name", "svm_baseline.pkl"
            ]
        },
        {
            "name": "SVM With Semantic Features",
            "cmd": [
                "conda", "run", "-n", "fifteenAI", "python",
                "src/classification/svm_classifier.py",
                "--opt-data", "optimization_dataset.json",
                "--limit", "1000", "--train-size", "300", "--validation-size", "700",
                "--C", "1.5", "--use-semantic-features",
                "--output", "src/classification/output",
                "--model-name", "svm_with_semantic.pkl"
            ]
        },
        # Logistic Ordinal
        {
            "name": "Logistic Ordinal Baseline",
            "cmd": [
                "conda", "run", "-n", "fifteenAI", "python",
                "src/classification/logistic_ordinal_classifier.py",
                "--opt-data", "optimization_dataset.json",
                "--limit", "1000", "--train-size", "300", "--validation-size", "700",
                "--C", "1.0",
                "--output", "src/classification/output",
                "--model-name", "logistic_baseline.pkl"
            ]
        },
        {
            "name": "Logistic Ordinal With Semantic Features",
            "cmd": [
                "conda", "run", "-n", "fifteenAI", "python",
                "src/classification/logistic_ordinal_classifier.py",
                "--opt-data", "optimization_dataset.json",
                "--limit", "1000", "--train-size", "300", "--validation-size", "700",
                "--C", "1.0", "--use-semantic-features",
                "--output", "src/classification/output",
                "--model-name", "logistic_with_semantic.pkl"
            ]
        },
        # GBM
        {
            "name": "GBM Baseline",
            "cmd": [
                "conda", "run", "-n", "fifteenAI", "python",
                "src/classification/gbm_ordinal_classifier.py",
                "--opt-data", "optimization_dataset.json",
                "--limit", "1000", "--train-size", "300", "--validation-size", "700",
                "--n-estimators", "100", "--learning-rate", "0.1", "--max-depth", "3",
                "--output", "src/classification/output",
                "--model-name", "gbm_baseline.pkl"
            ]
        },
        {
            "name": "GBM With Semantic Features",
            "cmd": [
                "conda", "run", "-n", "fifteenAI", "python",
                "src/classification/gbm_ordinal_classifier.py",
                "--opt-data", "optimization_dataset.json",
                "--limit", "1000", "--train-size", "300", "--validation-size", "700",
                "--n-estimators", "100", "--learning-rate", "0.1", "--max-depth", "3", "--use-semantic-features",
                "--output", "src/classification/output",
                "--model-name", "gbm_with_semantic.pkl"
            ]
        },
        # Neural
        {
            "name": "Neural Baseline",
            "cmd": [
                "conda", "run", "-n", "fifteenAI", "python",
                "src/classification/neural_ordinal_classifier.py",
                "--opt-data", "optimization_dataset.json",
                "--limit", "1000", "--train-size", "300", "--validation-size", "700",
                "--hidden-dim", "128", "--learning-rate", "0.001", "--epochs", "50", "--batch-size", "32", "--num-layers", "10",
                "--output", "src/classification/output",
                "--model-name", "neural_baseline.pkl"
            ]
        },
        {
            "name": "Neural With Semantic Features",
            "cmd": [
                "conda", "run", "-n", "fifteenAI", "python",
                "src/classification/neural_ordinal_classifier.py",
                "--opt-data", "optimization_dataset.json",
                "--limit", "1000", "--train-size", "300", "--validation-size", "700",
                "--hidden-dim", "128", "--learning-rate", "0.001", "--epochs", "50", "--batch-size", "32", "--num-layers", "10", "--use-semantic-features",
                "--output", "src/classification/output",
                "--model-name", "neural_with_semantic.pkl"
            ]
        },
        # RNN
        {
            "name": "RNN Baseline",
            "cmd": [
                "conda", "run", "-n", "fifteenAI", "python",
                "src/classification/rnn_ordinal_classifier.py",
                "--opt-data", "optimization_dataset.json",
                "--limit", "1000", "--train-size", "300", "--validation-size", "700",
                "--rnn-type", "GRU", "--num-layers", "3", "--bidirectional",
                "--hidden-dim", "128", "--learning-rate", "0.001", "--epochs", "50", "--batch-size", "32",
                "--output", "src/classification/output",
                "--model-name", "rnn_baseline.pkl"
            ]
        },
        {
            "name": "RNN With Semantic Features",
            "cmd": [
                "conda", "run", "-n", "fifteenAI", "python",
                "src/classification/rnn_ordinal_classifier.py",
                "--opt-data", "optimization_dataset.json",
                "--limit", "1000", "--train-size", "300", "--validation-size", "700",
                "--rnn-type", "GRU", "--num-layers", "3", "--bidirectional", "--use-semantic-features",
                "--hidden-dim", "128", "--learning-rate", "0.001", "--epochs", "50", "--batch-size", "32",
                "--output", "src/classification/output",
                "--model-name", "rnn_with_semantic.pkl"
            ]
        },
    ]
    
    # Add PCA experiments for all models (250 components only)
    pca_components = [250]
    models = [
        {
            "name_prefix": "SVM",
            "script": "src/classification/svm_classifier.py",
            "base_args": ["--opt-data", "optimization_dataset.json", "--limit", "1000", 
                         "--train-size", "300", "--validation-size", "700", "--C", "1.5"]
        },
        {
            "name_prefix": "Logistic Ordinal",
            "script": "src/classification/logistic_ordinal_classifier.py",
            "base_args": ["--opt-data", "optimization_dataset.json", "--limit", "1000",
                         "--train-size", "300", "--validation-size", "700", "--C", "1.0"]
        },
        {
            "name_prefix": "GBM",
            "script": "src/classification/gbm_ordinal_classifier.py",
            "base_args": ["--opt-data", "optimization_dataset.json", "--limit", "1000",
                         "--train-size", "300", "--validation-size", "700",
                         "--n-estimators", "100", "--learning-rate", "0.1", "--max-depth", "3"]
        },
        {
            "name_prefix": "Neural",
            "script": "src/classification/neural_ordinal_classifier.py",
            "base_args": ["--opt-data", "optimization_dataset.json", "--limit", "1000",
                         "--train-size", "300", "--validation-size", "700",
                         "--hidden-dim", "128", "--learning-rate", "0.001", "--epochs", "50",
                         "--batch-size", "32", "--num-layers", "10"]
        },
        {
            "name_prefix": "RNN",
            "script": "src/classification/rnn_ordinal_classifier.py",
            "base_args": ["--opt-data", "optimization_dataset.json", "--limit", "1000",
                         "--train-size", "300", "--validation-size", "700",
                         "--rnn-type", "GRU", "--num-layers", "3", "--bidirectional",
                         "--hidden-dim", "128", "--learning-rate", "0.001", "--epochs", "50",
                         "--batch-size", "32"]
        },
    ]
    
    for model in models:
        for n_components in pca_components:
            experiments.append({
                "name": f"{model['name_prefix']} PCA {n_components}",
                "cmd": [
                    "conda", "run", "-n", "fifteenAI", "python",
                    model["script"],
                    *model["base_args"],
                    "--pca-components", str(n_components),
                    "--output", "src/classification/output",
                    "--model-name", f"{model['name_prefix'].lower().replace(' ', '_')}_pca_{n_components}.pkl"
                ]
            })
            experiments.append({
                "name": f"{model['name_prefix']} PCA {n_components} With Semantic",
                "cmd": [
                    "conda", "run", "-n", "fifteenAI", "python",
                    model["script"],
                    *model["base_args"],
                    "--use-semantic-features",
                    "--pca-components", str(n_components),
                    "--output", "src/classification/output",
                    "--model-name", f"{model['name_prefix'].lower().replace(' ', '_')}_pca_{n_components}_semantic.pkl"
                ]
            })
    
    results = []
    for exp in experiments:
        success = run_command(exp["cmd"], exp["name"])
        results.append({
            "name": exp["name"],
            "success": success
        })
    
    print("\n" + "="*80)
    print("Experiment Summary")
    print("="*80)
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"{status} {r['name']}")
    
    print(f"\nCompleted {sum(r['success'] for r in results)}/{len(results)} experiments")

if __name__ == "__main__":
    main()

