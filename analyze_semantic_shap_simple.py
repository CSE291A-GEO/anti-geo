#!/usr/bin/env python3
"""
Simplified SHAP analysis for semantic features using model coefficients and correlations.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import joblib

# Import semantic feature extractor
import sys
sys.path.append('src/classification')
from semantic_features import SemanticFeatureExtractor

# Feature names for the 5 semantic patterns
SEMANTIC_FEATURE_NAMES = [
    "QA_Blocks",
    "Over-Chunking",
    "Header_Stuffing",
    "Entity_Attribution",
    "Citation_Embedding",
]

SEMANTIC_FEATURE_DESCRIPTIONS = [
    "Excessive Q&A blocks (GEO_STRUCT_001)",
    "Over-Chunking/Simplification (GEO_STRUCT_002)",
    "Title/Header Stuffing (GEO_STRUCT_003)",
    "Entity Over-Attribution (GEO_SEMANTIC_004)",
    "Unnatural Citation Embedding (GEO_SEMANTIC_005)",
]


def load_dataset_with_semantic(dataset_path: str, limit: int = 1000):
    """Load dataset and extract semantic features."""
    print(f"Loading dataset from: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if limit:
        data = data[:limit]
    
    # Initialize extractors
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    semantic_extractor = SemanticFeatureExtractor('all-MiniLM-L6-v2')
    
    # Extract features
    semantic_features_list = []
    labels = []
    
    print("Extracting semantic features...")
    for entry_idx, entry in enumerate(data):
        if entry_idx % 100 == 0:
            print(f"  Processed {entry_idx}/{len(data)} entries")
        
        sugg_idx = entry.get('sugg_idx', None)
        sources = entry.get('sources', [])
        
        for source_idx, source in enumerate(sources):
            cleaned_text = source.get('cleaned_text', '')
            
            # Only semantic pattern scores
            pattern_scores = semantic_extractor.extract_pattern_scores(cleaned_text)
            semantic_features_list.append(pattern_scores)
            
            # Label
            label = 1 if source_idx == sugg_idx else 0
            labels.append(label)
    
    semantic_features = np.array(semantic_features_list)
    y = np.array(labels)
    
    print(f"Semantic features shape: {semantic_features.shape}")
    print(f"Positive samples: {y.sum()}, Negative samples: {(y == 0).sum()}")
    
    return semantic_features, y


def analyze_with_correlation(X_semantic: np.ndarray, y: np.ndarray):
    """Analyze feature importance using correlation with labels."""
    print("\n" + "="*80)
    print("Correlation Analysis")
    print("="*80)
    
    # Point-biserial correlation for binary labels
    df = pd.DataFrame(X_semantic, columns=SEMANTIC_FEATURE_NAMES)
    df['label'] = y
    
    correlations = df.corr()['label'][:-1].abs().sort_values(ascending=False)
    
    print("\nAbsolute Correlation with GEO label:")
    for feat, corr in correlations.items():
        print(f"  {feat}: {corr:.4f}")
    
    return correlations


def analyze_svm_coefficients(model_path: str, scaler_path: str):
    """Analyze SVM model coefficients for last 5 features."""
    print("\n" + "="*80)
    print("SVM Coefficient Analysis")
    print("="*80)
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Get coefficients for all features
        if hasattr(model, 'coef_'):
            coef = model.coef_[0]
            # Extract last 5 features (semantic features)
            semantic_coef = coef[-5:]
            
            # Absolute values for importance
            importance = np.abs(semantic_coef)
            
            # Normalize to sum to 1
            importance_normalized = importance / importance.sum()
            
            print("\nSVM Coefficients (semantic features only):")
            for i, (name, val) in enumerate(zip(SEMANTIC_FEATURE_NAMES, semantic_coef)):
                print(f"  {name}: {val:.4f} (importance: {importance_normalized[i]:.4f})")
            
            return dict(zip(SEMANTIC_FEATURE_NAMES, importance_normalized))
        else:
            print("Model does not have coefficients (possibly RBF kernel)")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def analyze_logistic_coefficients(model_path: str, scaler_path: str):
    """Analyze Logistic model coefficients."""
    print("\n" + "="*80)
    print("Logistic Ordinal Coefficient Analysis")
    print("="*80)
    
    try:
        # Load just the scaler to understand feature structure
        scaler = joblib.load(scaler_path)
        
        # Load model manually to avoid import issues
        with open(model_path, 'rb') as f:
            import pickle
            model = pickle.load(f)
        
        # Get coefficients
        if hasattr(model, 'coef_'):
            coef = model.coef_
            # Extract last 5 features
            semantic_coef = coef[-5:]
            
            importance = np.abs(semantic_coef)
            importance_normalized = importance / importance.sum()
            
            print("\nLogistic Coefficients (semantic features only):")
            for i, (name, val) in enumerate(zip(SEMANTIC_FEATURE_NAMES, semantic_coef)):
                print(f"  {name}: {val:.4f} (importance: {importance_normalized[i]:.4f})")
            
            return dict(zip(SEMANTIC_FEATURE_NAMES, importance_normalized))
        else:
            print("Model does not have accessible coefficients")
            return None
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_variance_between_classes(X_semantic: np.ndarray, y: np.ndarray):
    """Analyze variance of each feature between GEO and non-GEO classes."""
    print("\n" + "="*80)
    print("Variance & Discriminative Power Analysis")
    print("="*80)
    
    geo_features = X_semantic[y == 1]
    non_geo_features = X_semantic[y == 0]
    
    # Mean values per class
    geo_means = geo_features.mean(axis=0)
    non_geo_means = non_geo_features.mean(axis=0)
    
    # Difference in means (discriminative power)
    mean_diff = np.abs(geo_means - non_geo_means)
    mean_diff_normalized = mean_diff / mean_diff.sum()
    
    # Standard deviations
    geo_std = geo_features.std(axis=0)
    non_geo_std = non_geo_features.std(axis=0)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((geo_std**2 + non_geo_std**2) / 2)
    cohens_d = mean_diff / pooled_std
    cohens_d_normalized = np.abs(cohens_d) / np.abs(cohens_d).sum()
    
    print("\nMean values by class:")
    for i, name in enumerate(SEMANTIC_FEATURE_NAMES):
        print(f"  {name}:")
        print(f"    GEO: {geo_means[i]:.4f}, Non-GEO: {non_geo_means[i]:.4f}")
        print(f"    Difference: {mean_diff[i]:.4f}, Cohen's d: {cohens_d[i]:.4f}")
    
    print("\nDiscriminative Power (normalized mean difference):")
    for name, val in zip(SEMANTIC_FEATURE_NAMES, mean_diff_normalized):
        print(f"  {name}: {val:.4f}")
    
    return {
        'mean_difference': dict(zip(SEMANTIC_FEATURE_NAMES, mean_diff_normalized)),
        'cohens_d': dict(zip(SEMANTIC_FEATURE_NAMES, cohens_d_normalized))
    }


def visualize_results(correlation_results, variance_results, svm_results, output_dir):
    """Create comprehensive visualizations."""
    print("\n" + "="*80)
    print("Creating Visualizations")
    print("="*80)
    
    # Prepare data
    methods = ['Correlation', 'Mean Difference', 'Cohen\'s d']
    data = {
        'Correlation': correlation_results,
        'Mean Difference': variance_results['mean_difference'],
        'Cohen\'s d': variance_results['cohens_d']
    }
    
    if svm_results:
        methods.append('SVM Coefficients')
        data['SVM Coefficients'] = svm_results
    
    # Create DataFrame
    df = pd.DataFrame(data, index=SEMANTIC_FEATURE_NAMES).T
    
    # Plot 1: Heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'Normalized Importance'})
    plt.title('Semantic Feature Importance Across Different Methods', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Semantic Features', fontsize=12)
    plt.ylabel('Analysis Method', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'semantic_importance_heatmap.png', dpi=300)
    print(f"Saved: {output_dir / 'semantic_importance_heatmap.png'}")
    plt.close()
    
    # Plot 2: Average importance
    avg_importance = df.mean(axis=0).sort_values(ascending=True)
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.RdYlGn(avg_importance / avg_importance.max())
    avg_importance.plot(kind='barh', color=colors)
    plt.title('Average Semantic Feature Importance Across All Methods', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Average Normalized Importance', fontsize=12)
    plt.ylabel('Semantic Features', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'semantic_importance_average.png', dpi=300)
    print(f"Saved: {output_dir / 'semantic_importance_average.png'}")
    plt.close()
    
    # Plot 3: Radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(SEMANTIC_FEATURE_NAMES), endpoint=False).tolist()
    angles += angles[:1]
    
    for method in methods:
        values = df.loc[method].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=method)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([name.replace('_', '\n') for name in SEMANTIC_FEATURE_NAMES], 
                       fontsize=10)
    ax.set_ylim(0, df.max().max() * 1.1)
    ax.set_title('Semantic Feature Importance Comparison\n(Radar Chart)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'semantic_importance_radar.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'semantic_importance_radar.png'}")
    plt.close()
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: Feature Importance Rankings")
    print("="*80)
    print("\nNormalized Importance Values:")
    print(df.to_string())
    
    print("\n\nAverage Importance Across Methods (Ranked):")
    for i, (feat, imp) in enumerate(avg_importance.sort_values(ascending=False).items(), 1):
        print(f"{i}. {feat}: {imp:.4f}")
    
    return df, avg_importance


def main():
    """Main analysis pipeline."""
    project_root = Path(__file__).parent
    dataset_path = project_root / "optimization_dataset.json"
    output_dir = project_root / "src" / "classification" / "output"
    
    # Load semantic features
    X_semantic, y = load_dataset_with_semantic(str(dataset_path), limit=1000)
    
    # Analysis 1: Correlation
    correlation_results = analyze_with_correlation(X_semantic, y)
    
    # Analysis 2: Variance & discriminative power
    variance_results = analyze_variance_between_classes(X_semantic, y)
    
    # Analysis 3: Model coefficients (SVM)
    svm_results = analyze_svm_coefficients(
        output_dir / 'svm_with_semantic.pkl',
        output_dir / 'svm_with_semantic_scaler.pkl'
    )
    
    # Visualize
    df, avg_importance = visualize_results(
        correlation_results,
        variance_results,
        svm_results,
        output_dir
    )
    
    # Save results (convert numpy types to Python native types)
    results = {
        'correlation': {k: float(v) for k, v in correlation_results.to_dict().items()},
        'mean_difference': {k: float(v) for k, v in variance_results['mean_difference'].items()},
        'cohens_d': {k: float(v) for k, v in variance_results['cohens_d'].items()},
        'average_importance': {k: float(v) for k, v in avg_importance.to_dict().items()},
        'feature_descriptions': dict(zip(SEMANTIC_FEATURE_NAMES, SEMANTIC_FEATURE_DESCRIPTIONS))
    }
    
    if svm_results:
        results['svm_coefficients'] = {k: float(v) for k, v in svm_results.items()}
    
    results_path = output_dir / 'semantic_feature_importance_analysis.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: {results_path}")
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


if __name__ == '__main__':
    main()

