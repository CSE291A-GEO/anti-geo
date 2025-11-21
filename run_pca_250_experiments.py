#!/usr/bin/env python3
"""
Run PCA 250 experiments for all 5 models.
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
    
    # PCA 250 experiments for all 5 models
    experiments = [
        {
            "name": "SVM PCA 250",
            "cmd": [
                "conda", "run", "-n", "fifteenAI", "python",
                "src/classification/svm_classifier.py",
                "--opt-data", "optimization_dataset.json",
                "--limit", "1000", "--train-size", "300", "--validation-size", "700",
                "--C", "1.5",
                "--pca-components", "250",
                "--output", "src/classification/output",
                "--model-name", "svm_pca_250.pkl"
            ]
        },
        {
            "name": "Logistic Ordinal PCA 250",
            "cmd": [
                "conda", "run", "-n", "fifteenAI", "python",
                "src/classification/logistic_ordinal_classifier.py",
                "--opt-data", "optimization_dataset.json",
                "--limit", "1000", "--train-size", "300", "--validation-size", "700",
                "--C", "1.0",
                "--pca-components", "250",
                "--output", "src/classification/output",
                "--model-name", "logistic_ordinal_pca_250.pkl"
            ]
        },
        {
            "name": "GBM PCA 250",
            "cmd": [
                "conda", "run", "-n", "fifteenAI", "python",
                "src/classification/gbm_ordinal_classifier.py",
                "--opt-data", "optimization_dataset.json",
                "--limit", "1000", "--train-size", "300", "--validation-size", "700",
                "--n-estimators", "100", "--learning-rate", "0.1", "--max-depth", "3",
                "--pca-components", "250",
                "--output", "src/classification/output",
                "--model-name", "gbm_pca_250.pkl"
            ]
        },
        {
            "name": "Neural PCA 250",
            "cmd": [
                "conda", "run", "-n", "fifteenAI", "python",
                "src/classification/neural_ordinal_classifier.py",
                "--opt-data", "optimization_dataset.json",
                "--limit", "1000", "--train-size", "300", "--validation-size", "700",
                "--hidden-dim", "128", "--learning-rate", "0.001", "--epochs", "50", "--batch-size", "32", "--num-layers", "10",
                "--pca-components", "250",
                "--output", "src/classification/output",
                "--model-name", "neural_pca_250.pkl"
            ]
        },
        {
            "name": "RNN PCA 250",
            "cmd": [
                "conda", "run", "-n", "fifteenAI", "python",
                "src/classification/rnn_ordinal_classifier.py",
                "--opt-data", "optimization_dataset.json",
                "--limit", "1000", "--train-size", "300", "--validation-size", "700",
                "--rnn-type", "GRU", "--num-layers", "3", "--bidirectional",
                "--hidden-dim", "128", "--learning-rate", "0.001", "--epochs", "50", "--batch-size", "32",
                "--pca-components", "250",
                "--output", "src/classification/output",
                "--model-name", "rnn_pca_250.pkl"
            ]
        },
    ]
    
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

