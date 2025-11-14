#!/usr/bin/env python3
"""
Train SVM Classifier for GEO Detection

This script trains an SVM classifier on the first 300 examples from the
semantic matching results to classify GEO-optimized sources.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from classification.svm_classifier import train_svm_classifier


def main():
    """Main function to train the SVM classifier."""
    project_root = Path(__file__).parent.parent
    
    # Paths
    optimization_dataset_path = project_root / 'optimization_dataset.json'
    output_dir = Path(__file__).parent / 'classification'
    
    print("="*80)
    print("SVM GEO DETECTION CLASSIFIER TRAINING")
    print("="*80)
    print()
    print(f"Using optimization dataset: {optimization_dataset_path}")
    print(f"  - First 1000 entries")
    print(f"  - 300 samples for training")
    print(f"  - 700 samples for validation")
    print(f"Features: Semantic embeddings only (from cleaned_text)")
    print(f"Labels: 1 if source matches sugg_idx (GEO), 0 otherwise")
    print(f"Output directory: {output_dir}")
    print()
    
    # Train the classifier
    detector = train_svm_classifier(
        optimization_dataset_path=str(optimization_dataset_path),
        output_dir=str(output_dir),
        limit=1000,
        train_size=300,
        validation_size=700,
        C=1.5,
        model_name='svm_geo_detector.pkl'
    )
    
    print()
    print("="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Model saved to: {output_dir / 'svm_geo_detector.pkl'}")
    print(f"Scaler saved to: {output_dir / 'svm_geo_detector_scaler.pkl'}")
    print(f"Metrics saved to: {output_dir / 'svm_geo_detector_metrics.json'}")


if __name__ == '__main__':
    main()

