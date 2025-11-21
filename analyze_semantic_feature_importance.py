#!/usr/bin/env python3
"""
Analyze semantic feature importance using SHAP values.

This script loads models trained with semantic features and uses SHAP
to determine which of the 5 semantic pattern features contribute most
to GEO detection predictions.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# Import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("ERROR: SHAP not installed. Install with: pip install shap")
    SHAP_AVAILABLE = False
    exit(1)

# Import semantic feature extractor
import sys
sys.path.append('src/classification')
from semantic_features import SemanticFeatureExtractor

# Feature names for the 5 semantic patterns
SEMANTIC_FEATURE_NAMES = [
    "QA_Blocks (GEO_STRUCT_001)",
    "Over-Chunking (GEO_STRUCT_002)",
    "Header_Stuffing (GEO_STRUCT_003)",
    "Entity_Attribution (GEO_SEMANTIC_004)",
    "Citation_Embedding (GEO_SEMANTIC_005)",
]


def load_dataset(dataset_path: str, limit: int = 1000, 
                validation_start: int = 300, validation_size: int = 700):
    """Load and prepare dataset with semantic features."""
    print(f"Loading dataset from: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if limit:
        data = data[:limit]
    
    print(f"Loaded {len(data)} entries")
    
    # Initialize models
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    semantic_extractor = SemanticFeatureExtractor('all-MiniLM-L6-v2')
    
    # Extract features
    X_list = []
    y_list = []
    
    print("Extracting features with semantic patterns...")
    for entry_idx, entry in enumerate(data):
        if entry_idx % 100 == 0:
            print(f"  Processed {entry_idx}/{len(data)} entries")
        
        sugg_idx = entry.get('sugg_idx', None)
        sources = entry.get('sources', [])
        
        for source_idx, source in enumerate(sources):
            cleaned_text = source.get('cleaned_text', '')
            
            # Base embedding
            embedding = embedding_model.encode(cleaned_text, convert_to_numpy=True)
            
            # Semantic pattern scores
            pattern_scores = semantic_extractor.extract_pattern_scores(cleaned_text)
            
            # Combine features
            features = np.concatenate([embedding, pattern_scores])
            X_list.append(features)
            
            # Label
            label = 1 if source_idx == sugg_idx else 0
            y_list.append(label)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"Feature shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Positive samples: {y.sum()}, Negative samples: {(y == 0).sum()}")
    
    # Split into train/validation
    X_train = X[:validation_start]
    y_train = y[:validation_start]
    X_val = X[validation_start:validation_start + validation_size]
    y_val = y[validation_start:validation_start + validation_size]
    
    return X_train, y_train, X_val, y_val


def analyze_svm_shap(model_path: str, scaler_path: str, X_val: np.ndarray, 
                     y_val: np.ndarray, pca_path: str = None) -> Dict:
    """Analyze SVM model using SHAP KernelExplainer."""
    print("\n" + "="*80)
    print("Analyzing SVM Model")
    print("="*80)
    
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Apply preprocessing
    X_scaled = scaler.transform(X_val)
    
    # Apply PCA if provided
    if pca_path and Path(pca_path).exists():
        pca = joblib.load(pca_path)
        X_scaled = pca.transform(X_scaled)
        print(f"Applied PCA: {X_scaled.shape[1]} components")
        # Can't get individual feature SHAP values after PCA
        return {"error": "PCA dimensionality reduction applied - cannot extract individual semantic feature importance"}
    
    # Extract only semantic features (last 5 columns)
    X_semantic = X_scaled[:, -5:]
    
    # Sample for SHAP (KernelExplainer is slow)
    sample_size = min(100, len(X_semantic))
    X_sample = X_semantic[:sample_size]
    
    print(f"Computing SHAP values for {sample_size} samples...")
    
    # Create a prediction function for just semantic features
    def predict_fn(semantic_features):
        # Reconstruct full feature vector with zeros for embedding features
        full_features = np.zeros((semantic_features.shape[0], X_scaled.shape[1]))
        full_features[:, -5:] = semantic_features
        full_features[:, :-5] = X_scaled[:semantic_features.shape[0], :-5]
        return model.predict_proba(full_features)[:, 1]
    
    # SHAP KernelExplainer
    explainer = shap.KernelExplainer(predict_fn, X_sample)
    shap_values = explainer.shap_values(X_sample, nsamples=100)
    
    # Calculate mean absolute SHAP values
    mean_shap = np.abs(shap_values).mean(axis=0)
    
    results = {
        'model': 'SVM',
        'shap_values': shap_values.tolist(),
        'mean_abs_shap': mean_shap.tolist(),
        'feature_importance': dict(zip(SEMANTIC_FEATURE_NAMES, mean_shap)),
        'sample_size': sample_size
    }
    
    return results


def analyze_logistic_shap(model_path: str, scaler_path: str, X_val: np.ndarray,
                          y_val: np.ndarray, pca_path: str = None) -> Dict:
    """Analyze Logistic Ordinal model using SHAP."""
    print("\n" + "="*80)
    print("Analyzing Logistic Ordinal Model")
    print("="*80)
    
    # Load model
    import sys
    sys.path.append('src/classification')
    from logistic_ordinal_classifier import LogisticOrdinalGEODetector
    
    detector = LogisticOrdinalGEODetector()
    detector.load_model(model_path, scaler_path, pca_path)
    
    # Check if PCA was applied
    if hasattr(detector, 'pca') and detector.pca is not None:
        print("PCA was applied - cannot extract individual semantic feature importance")
        return {"error": "PCA dimensionality reduction applied"}
    
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X_val)
    X_semantic = X_scaled[:, -5:]
    
    sample_size = min(100, len(X_semantic))
    X_sample = X_semantic[:sample_size]
    
    print(f"Computing SHAP values for {sample_size} samples...")
    
    def predict_fn(semantic_features):
        full_features = np.zeros((semantic_features.shape[0], X_scaled.shape[1]))
        full_features[:, -5:] = semantic_features
        full_features[:, :-5] = X_scaled[:semantic_features.shape[0], :-5]
        probs = detector.model.predict_proba(full_features)
        return probs[:, 1]  # GEO probability
    
    explainer = shap.KernelExplainer(predict_fn, X_sample)
    shap_values = explainer.shap_values(X_sample, nsamples=100)
    
    mean_shap = np.abs(shap_values).mean(axis=0)
    
    results = {
        'model': 'Logistic_Ordinal',
        'shap_values': shap_values.tolist(),
        'mean_abs_shap': mean_shap.tolist(),
        'feature_importance': dict(zip(SEMANTIC_FEATURE_NAMES, mean_shap)),
        'sample_size': sample_size
    }
    
    return results


def analyze_gbm_shap(model_path: str, scaler_path: str, X_val: np.ndarray,
                     y_val: np.ndarray, pca_path: str = None) -> Dict:
    """Analyze GBM model using SHAP TreeExplainer."""
    print("\n" + "="*80)
    print("Analyzing GBM Model")
    print("="*80)
    
    # Load model
    import sys
    sys.path.append('src/classification')
    from gbm_ordinal_classifier import GBMOrdinalGEODetector
    
    detector = GBMOrdinalGEODetector()
    detector.load_model(model_path, scaler_path, pca_path)
    
    # Check if PCA was applied
    if hasattr(detector, 'pca') and detector.pca is not None:
        print("PCA was applied - cannot extract individual semantic feature importance")
        return {"error": "PCA dimensionality reduction applied"}
    
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X_val)
    
    # For tree models, we can get feature importance directly
    # But let's also use SHAP for consistency
    
    # Use TreeExplainer if using XGBoost
    try:
        explainer = shap.TreeExplainer(detector.model.base_model)
        shap_values = explainer.shap_values(X_scaled)
        
        # Extract semantic features (last 5)
        semantic_shap = shap_values[:, -5:]
        mean_shap = np.abs(semantic_shap).mean(axis=0)
        
        results = {
            'model': 'GBM',
            'shap_values': semantic_shap.tolist(),
            'mean_abs_shap': mean_shap.tolist(),
            'feature_importance': dict(zip(SEMANTIC_FEATURE_NAMES, mean_shap)),
            'sample_size': len(X_scaled)
        }
    except Exception as e:
        print(f"TreeExplainer failed: {e}")
        print("Falling back to KernelExplainer...")
        
        X_semantic = X_scaled[:, -5:]
        sample_size = min(100, len(X_semantic))
        X_sample = X_semantic[:sample_size]
        
        def predict_fn(semantic_features):
            full_features = np.zeros((semantic_features.shape[0], X_scaled.shape[1]))
            full_features[:, -5:] = semantic_features
            full_features[:, :-5] = X_scaled[:semantic_features.shape[0], :-5]
            probs = detector.model.predict_proba(full_features)
            return probs[:, 1]
        
        explainer = shap.KernelExplainer(predict_fn, X_sample)
        shap_values = explainer.shap_values(X_sample, nsamples=100)
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        results = {
            'model': 'GBM',
            'shap_values': shap_values.tolist(),
            'mean_abs_shap': mean_shap.tolist(),
            'feature_importance': dict(zip(SEMANTIC_FEATURE_NAMES, mean_shap)),
            'sample_size': sample_size
        }
    
    return results


def visualize_shap_importance(all_results: Dict, output_dir: Path):
    """Create visualization of SHAP feature importance across models."""
    print("\n" + "="*80)
    print("Creating Visualizations")
    print("="*80)
    
    # Prepare data for plotting
    models = []
    importances = []
    
    for model_name, results in all_results.items():
        if 'error' in results:
            continue
        
        models.append(results['model'])
        importances.append(results['mean_abs_shap'])
    
    if not models:
        print("No valid results to visualize")
        return
    
    # Create DataFrame
    df = pd.DataFrame(importances, columns=SEMANTIC_FEATURE_NAMES, index=models)
    
    # Normalize for comparison (0-1 scale per model)
    df_normalized = df.div(df.sum(axis=1), axis=0)
    
    # Plot 1: Heatmap of normalized importance
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_normalized, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'Normalized SHAP Importance'})
    plt.title('Semantic Feature Importance Across Models (Normalized)', fontsize=14, fontweight='bold')
    plt.xlabel('Semantic Features', fontsize=12)
    plt.ylabel('Models', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'semantic_feature_importance_heatmap.png', dpi=300)
    print(f"Saved: {output_dir / 'semantic_feature_importance_heatmap.png'}")
    plt.close()
    
    # Plot 2: Bar chart comparing features
    fig, ax = plt.subplots(figsize=(14, 6))
    df_normalized.T.plot(kind='bar', ax=ax, width=0.8)
    plt.title('Semantic Feature Importance by Model', fontsize=14, fontweight='bold')
    plt.xlabel('Semantic Features', fontsize=12)
    plt.ylabel('Normalized SHAP Importance', fontsize=12)
    plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'semantic_feature_importance_bars.png', dpi=300)
    print(f"Saved: {output_dir / 'semantic_feature_importance_bars.png'}")
    plt.close()
    
    # Plot 3: Average importance across models
    avg_importance = df_normalized.mean(axis=0).sort_values(ascending=True)
    
    plt.figure(figsize=(10, 6))
    avg_importance.plot(kind='barh', color='steelblue')
    plt.title('Average Semantic Feature Importance Across All Models', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Average Normalized SHAP Importance', fontsize=12)
    plt.ylabel('Semantic Features', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'semantic_feature_importance_average.png', dpi=300)
    print(f"Saved: {output_dir / 'semantic_feature_importance_average.png'}")
    plt.close()
    
    # Create summary table
    print("\n" + "="*80)
    print("Feature Importance Rankings")
    print("="*80)
    print("\nNormalized SHAP Values (proportion of total importance):")
    print(df_normalized.to_string())
    
    print("\n\nAverage Importance Across Models:")
    for feat, imp in avg_importance.sort_values(ascending=False).items():
        print(f"  {feat}: {imp:.4f}")
    
    # Ranking
    print("\n\nFeature Rankings (1 = Most Important):")
    rankings = df_normalized.rank(axis=1, ascending=False)
    print(rankings.to_string())


def main():
    """Main analysis pipeline."""
    if not SHAP_AVAILABLE:
        print("SHAP is not installed. Please install with: pip install shap")
        return
    
    # Setup paths
    project_root = Path(__file__).parent
    dataset_path = project_root / "optimization_dataset.json"
    output_dir = project_root / "src" / "classification" / "output"
    
    # Load dataset
    X_train, y_train, X_val, y_val = load_dataset(
        str(dataset_path),
        limit=1000,
        validation_start=300,
        validation_size=700
    )
    
    print(f"\nValidation set: {X_val.shape[0]} samples")
    print(f"Semantic features: columns {X_val.shape[1]-5} to {X_val.shape[1]-1}")
    
    # Models to analyze (only those trained with semantic features)
    models_to_analyze = [
        {
            'name': 'SVM',
            'model_path': output_dir / 'svm_with_semantic.pkl',
            'scaler_path': output_dir / 'svm_with_semantic_scaler.pkl',
            'analyzer': analyze_svm_shap
        },
        {
            'name': 'Logistic_Ordinal',
            'model_path': output_dir / 'logistic_with_semantic.pkl',
            'scaler_path': output_dir / 'logistic_with_semantic_scaler.pkl',
            'analyzer': analyze_logistic_shap
        },
        {
            'name': 'GBM',
            'model_path': output_dir / 'gbm_with_semantic.pkl',
            'scaler_path': output_dir / 'gbm_with_semantic_scaler.pkl',
            'analyzer': analyze_gbm_shap
        },
    ]
    
    # Analyze each model
    all_results = {}
    for model_config in models_to_analyze:
        name = model_config['name']
        
        if not model_config['model_path'].exists():
            print(f"\nSkipping {name}: Model not found at {model_config['model_path']}")
            continue
        
        try:
            results = model_config['analyzer'](
                str(model_config['model_path']),
                str(model_config['scaler_path']),
                X_val,
                y_val
            )
            all_results[name] = results
        except Exception as e:
            print(f"\nError analyzing {name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save results
    results_path = output_dir / 'semantic_feature_shap_analysis.json'
    # Remove numpy arrays before saving
    json_results = {}
    for model, result in all_results.items():
        json_results[model] = {
            'model': result.get('model', model),
            'mean_abs_shap': result.get('mean_abs_shap', []),
            'feature_importance': result.get('feature_importance', {}),
            'sample_size': result.get('sample_size', 0),
            'error': result.get('error', None)
        }
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n\nResults saved to: {results_path}")
    
    # Create visualizations
    visualize_shap_importance(all_results, output_dir)
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


if __name__ == '__main__':
    main()

