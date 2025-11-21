"""
Logistic Regression with Ordinal Loss for GEO Detection

This module implements a Logistic Regression classifier with Ordinal Logistic Loss
for detecting GEO-optimized sources. The ordinal loss ensures that probability
distributions respect the order (low, medium, high GEO rankings).
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import joblib
from sentence_transformers import SentenceTransformer
from scipy.optimize import minimize
from scipy.special import expit, logit
try:
    from .semantic_features import SemanticFeatureExtractor
    from .bias_utils import (
        oversample_positive,
        entry_argmax_predictions,
        calculate_ranking_accuracy,
    )
except ImportError:
    from semantic_features import SemanticFeatureExtractor
    from bias_utils import (
        oversample_positive,
        entry_argmax_predictions,
        calculate_ranking_accuracy,
    )


def _group_by_entry(
    X: np.ndarray,
    y: np.ndarray,
    metadata: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Group samples by entry_idx.
    
    Returns:
        List of dicts, each containing:
        - 'entry_idx': The entry index
        - 'X': Feature matrix for all sources in this entry
        - 'y': Labels for all sources in this entry
        - 'indices': Original indices in the flat array
        - 'sugg_idx': The sugg_idx for this entry
    """
    entry_groups = {}
    for i, meta in enumerate(metadata):
        entry_idx = meta['entry_idx']
        if entry_idx not in entry_groups:
            entry_groups[entry_idx] = {
                'entry_idx': entry_idx,
                'indices': [],
                'sugg_idx': meta.get('sugg_idx', None)
            }
        entry_groups[entry_idx]['indices'].append(i)
    
    # Build grouped data
    grouped = []
    for entry_idx, group in entry_groups.items():
        indices = group['indices']
        grouped.append({
            'entry_idx': entry_idx,
            'X': X[indices],
            'y': y[indices],
            'indices': indices,
            'sugg_idx': group['sugg_idx']
        })
    
    return grouped


class OrdinalLogisticRegression:
    """
    Ordinal Logistic Regression using Cumulative Link Model.
    
    This implements the proportional odds model where we model cumulative probabilities
    P(y <= k) using a single scalar output and learned thresholds.
    """
    
    def __init__(self, n_classes: int = 3, max_iter: int = 1000, alpha: float = 0.01):
        """
        Initialize ordinal logistic regression.
        
        Args:
            n_classes: Number of ordinal classes (default: 3 for low, medium, high)
            max_iter: Maximum iterations for optimization
            alpha: L2 regularization parameter
        """
        self.n_classes = n_classes
        self.max_iter = max_iter
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
        self.thresholds_ = None  # Thresholds for cumulative probabilities
    
    def _ordinal_log_loss(self, params, X, y, n_features):
        """
        Compute ordinal logistic loss (negative log-likelihood).
        
        Args:
            params: Flattened parameters [coefficients, thresholds]
            X: Feature matrix
            y: Ordinal labels (0, 1, 2, ...)
            n_features: Number of features
            
        Returns:
            Negative log-likelihood
        """
        coef = params[:n_features].reshape(-1, 1)
        thresholds = params[n_features:]
        
        # Compute z = X @ coef (scalar output for each sample)
        z = X @ coef
        
        # Compute cumulative probabilities P(y <= k)
        # For k = 0, 1, ..., n_classes-2
        n_samples = len(y)
        log_likelihood = 0.0
        
        for i in range(n_samples):
            z_i = z[i, 0]
            y_i = int(y[i])
            
            if y_i == 0:
                # P(y <= 0) = sigmoid(threshold_0 - z_i)
                prob = expit(thresholds[0] - z_i)
                log_likelihood += np.log(prob + 1e-10)
            elif y_i == self.n_classes - 1:
                # P(y <= n_classes-1) = 1, so P(y = n_classes-1) = 1 - P(y <= n_classes-2)
                prob_prev = expit(thresholds[-1] - z_i)
                log_likelihood += np.log(1 - prob_prev + 1e-10)
            else:
                # P(y = k) = P(y <= k) - P(y <= k-1)
                prob_k = expit(thresholds[y_i] - z_i)
                prob_k_minus_1 = expit(thresholds[y_i - 1] - z_i)
                log_likelihood += np.log(prob_k - prob_k_minus_1 + 1e-10)
        
        # Add L2 regularization
        reg_term = self.alpha * np.sum(coef ** 2)
        
        return -(log_likelihood / n_samples) + reg_term
    
    def fit(self, X, y):
        """
        Fit the ordinal logistic regression model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Ordinal labels (n_samples,) with values in [0, n_classes-1]
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        # Start with small random coefficients and ordered thresholds
        np.random.seed(42)
        coef_init = np.random.randn(n_features) * 0.01
        # Thresholds must be in increasing order
        thresholds_init = np.linspace(-2, 2, self.n_classes - 1)
        
        params_init = np.concatenate([coef_init, thresholds_init])
        
        # Optimize
        result = minimize(
            self._ordinal_log_loss,
            params_init,
            args=(X, y, n_features),
            method='L-BFGS-B',
            options={'maxiter': self.max_iter}
        )
        
        # Extract parameters
        self.coef_ = result.x[:n_features].reshape(-1, 1)
        self.thresholds_ = result.x[n_features:]
        
        return self
    
    def predict_proba(self, X):
        """
        Predict probability distribution for each class.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Probability matrix (n_samples, n_classes) with probabilities for each class
        """
        if self.coef_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        n_samples = X.shape[0]
        z = X @ self.coef_  # (n_samples, 1)
        
        probs = np.zeros((n_samples, self.n_classes))
        
        for i in range(n_samples):
            z_i = z[i, 0]
            
            # Compute cumulative probabilities
            cum_probs = []
            for k in range(self.n_classes - 1):
                cum_probs.append(expit(self.thresholds_[k] - z_i))
            
            # Convert to individual class probabilities
            probs[i, 0] = cum_probs[0]
            for k in range(1, self.n_classes - 1):
                probs[i, k] = cum_probs[k] - cum_probs[k - 1]
            probs[i, self.n_classes - 1] = 1 - cum_probs[-1]
            
            # Normalize to ensure they sum to 1
            probs[i] = probs[i] / (probs[i].sum() + 1e-10)
        
        return probs
    
    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predicted class labels (n_samples,)
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


class LogisticOrdinalGEODetector:
    """
    Logistic Regression with Ordinal Loss for GEO detection.
    """
    
    def __init__(
        self, 
        model_path: Optional[Path] = None, 
        scaler_path: Optional[Path] = None,
        embedding_model_name: str = 'all-MiniLM-L6-v2',
        use_semantic_features: bool = False,
        pca_components: Optional[int] = None
    ):
        """
        Initialize the Logistic Ordinal detector.
        
        Args:
            model_path: Path to saved model (pickle file)
            scaler_path: Path to saved scaler (pickle file)
            embedding_model_name: Name of the sentence transformer model for embeddings
            use_semantic_features: If True, include individual semantic matching pattern scores as features
        """
        self.model = None
        self.scaler = StandardScaler()
        self.pca = None
        self.pca_components = pca_components
        if pca_components is not None:
            self.pca = PCA(n_components=pca_components)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.use_semantic_features = use_semantic_features
        
        # Initialize semantic feature extractor if needed
        self.semantic_extractor = None
        if use_semantic_features:
            self.semantic_extractor = SemanticFeatureExtractor(model_name=embedding_model_name)
        
        # Build feature names
        self.feature_names = [f'embedding_dim_{i}' for i in range(self.embedding_dim)]
        if use_semantic_features:
            self.feature_names.extend(self.semantic_extractor.get_feature_names())
        
        if model_path and model_path.exists():
            self.load_model(model_path, scaler_path)
    
    def extract_features(self, cleaned_text: str) -> np.ndarray:
        """
        Extract features from cleaned_text.
        
        Args:
            cleaned_text: The cleaned text to embed
            
        Returns:
            numpy array of features (embeddings + optionally semantic pattern scores)
        """
        if not cleaned_text:
            base_features = np.zeros(self.embedding_dim)
        else:
            try:
                base_features = self.embedding_model.encode(cleaned_text, convert_to_numpy=True)
            except Exception as e:
                print(f"    Warning: Failed to generate embedding: {str(e)[:50]}")
                base_features = np.zeros(self.embedding_dim)
        
        # Add semantic pattern scores if enabled
        if self.use_semantic_features and self.semantic_extractor:
            semantic_scores = self.semantic_extractor.extract_pattern_scores(cleaned_text)
            return np.concatenate([base_features, semantic_scores])
        else:
            return base_features
    
    def prepare_training_data(
        self, 
        optimization_dataset: List[Dict[str, Any]],
        limit: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], List[str]]:
        """
        Prepare training data from optimization dataset.
        
        For ordinal classification, we assign labels based on whether a source
        is the GEO source (sugg_idx) or not. We use a simple 3-class system:
        - 0: Non-GEO (not sugg_idx)
        - 1: Medium (could be used for ranking, but for now we use binary-like)
        - 2: High GEO (is sugg_idx)
        
        Args:
            optimization_dataset: List of entries from optimization dataset
            limit: Maximum number of entries to use
            
        Returns:
            Tuple of (X, y, entry_metadata) where:
            - X is feature matrix
            - y is ordinal labels (0=non-GEO, 2=GEO)
            - entry_metadata contains (entry_idx, source_idx, sugg_idx) for each sample
        """
        X = []
        y = []
        entry_metadata = []
        source_texts = []
        
        entries_used = 0
        for entry_idx, entry in enumerate(optimization_dataset[:limit]):
            sources = entry.get('sources', [])
            sugg_idx = entry.get('sugg_idx', None)
            
            if sugg_idx is None:
                continue
            
            # Extract features for each source in this entry
            for source_idx, source in enumerate(sources):
                cleaned_text = source.get('cleaned_text', '')
                if not cleaned_text:
                    continue
                
                # Label: 2 if this source is the GEO source (sugg_idx), 0 otherwise
                # Using 2 for GEO to make it the "high" class in ordinal ranking
                label = 2 if source_idx == sugg_idx else 0
                
                # Extract features
                features = self.extract_features(cleaned_text)
                
                X.append(features)
                y.append(label)
                entry_metadata.append({
                    'entry_idx': entry_idx,
                    'source_idx': source_idx,
                    'sugg_idx': sugg_idx
                })
                source_texts.append(cleaned_text)
            
            entries_used += 1
            if entries_used >= limit:
                break
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y, entry_metadata, source_texts
    
    def train(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        entry_metadata: List[Dict[str, Any]],
        source_texts: List[str],
        train_size: Optional[int] = None,
        validation_size: Optional[int] = None,
        test_size: Optional[float] = None,
        random_state: int = 42,
        C: float = 1.0
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Train the Logistic Ordinal classifier.
        
        Args:
            X: Feature matrix
            y: Ordinal labels
            entry_metadata: Metadata for each sample
            train_size: Number of samples to use for training
            validation_size: Number of samples to use for validation
            test_size: Proportion of data to use for testing
            random_state: Random seed
            C: Inverse regularization strength (1/alpha)
            
        Returns:
            Tuple of (metrics dict, latency info dict, predictions info dict)
        """
        # Split data
        if train_size is not None:
            if train_size >= len(X):
                raise ValueError(f"train_size ({train_size}) must be less than total samples ({len(X)})")
            X_train = X[:train_size]
            y_train = y[:train_size]
            metadata_train = entry_metadata[:train_size]
            
            if validation_size is not None:
                if train_size + validation_size > len(X):
                    raise ValueError(f"train_size + validation_size exceeds total samples")
                X_val = X[train_size:train_size + validation_size]
                y_val = y[train_size:train_size + validation_size]
                metadata_val = entry_metadata[train_size:train_size + validation_size]
                texts_val = source_texts[train_size:train_size + validation_size]
            else:
                X_val = X[train_size:]
                y_val = y[train_size:]
                metadata_val = entry_metadata[train_size:]
                texts_val = source_texts[train_size:]
        else:
            from sklearn.model_selection import train_test_split
            if test_size is None:
                test_size = 0.2
            (X_train, X_val, y_train, y_val,
             metadata_train, metadata_val,
             _texts_train, texts_val) = train_test_split(
                X, y, entry_metadata, source_texts,
                test_size=test_size, random_state=random_state, stratify=y
            )
        
        # Scale features
        training_start = time.perf_counter()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Apply PCA if enabled
        if self.pca is not None:
            X_train_scaled = self.pca.fit_transform(X_train_scaled)
            X_val_scaled = self.pca.transform(X_val_scaled)
            print(f"  Applied PCA: {X_train_scaled.shape[1]} components (explained variance: {self.pca.explained_variance_ratio_.sum():.4f})")
        
        # Train ordinal logistic regression
        print(f"Training Ordinal Logistic Regression classifier...")
        print(f"  Using entry-batched training for context awareness")
        alpha = 1.0 / C if C > 0 else 0.0
        # Use 2 classes since we only have labels 0 (non-GEO) and 2 (GEO)
        # We'll map label 2 -> 1 for the ordinal model
        self.model = OrdinalLogisticRegression(n_classes=2, max_iter=1000, alpha=alpha)
        # Map labels: 0 -> 0, 2 -> 1
        y_train_mapped = (y_train == 2).astype(int)
        y_val_mapped = (y_val == 2).astype(int)
        
        # Group training data by entry for context-aware training
        train_entry_groups = _group_by_entry(X_train_scaled, y_train_mapped, metadata_train)
        
        # For entry-batched training: collect all sources from entries and train
        X_train_batched = []
        y_train_batched = []
        for entry_group in train_entry_groups:
            X_train_batched.append(entry_group['X'])
            y_train_batched.extend(entry_group['y'])
        
        X_train_batched = np.vstack(X_train_batched)
        y_train_batched = np.array(y_train_batched)
        
        self.model.fit(X_train_batched, y_train_batched)
        training_time = time.perf_counter() - training_start
        
        # Evaluate
        y_train_pred_mapped = self.model.predict(X_train_scaled)
        y_val_pred_mapped = self.model.predict(X_val_scaled)
        # Map back: 0 -> 0, 1 -> 2
        y_train_pred = (y_train_pred_mapped * 2).astype(int)
        y_val_pred = (y_val_pred_mapped * 2).astype(int)
        
        # Binary accuracy (treating class 2 as positive, 0 as negative)
        y_train_binary = y_train_mapped
        y_val_binary = y_val_mapped
        y_train_pred_binary = y_train_pred_mapped
        y_val_pred_binary = y_val_pred_mapped
        
        train_accuracy = accuracy_score(y_train_binary, y_train_pred_binary)
        val_accuracy = accuracy_score(y_val_binary, y_val_pred_binary)
        
        train_precision = precision_score(y_train_binary, y_train_pred_binary, zero_division=0)
        train_recall = recall_score(y_train_binary, y_train_pred_binary, zero_division=0)
        train_f1 = f1_score(y_train_binary, y_train_pred_binary, zero_division=0)
        
        val_precision = precision_score(y_val_binary, y_val_pred_binary, zero_division=0)
        val_recall = recall_score(y_val_binary, y_val_pred_binary, zero_division=0)
        val_f1 = f1_score(y_val_binary, y_val_pred_binary, zero_division=0)
        
        # Calculate ranking accuracy: check if sugg_idx has highest probability
        train_ranking_accuracy = self._calculate_ranking_accuracy(
            X_train_scaled, metadata_train, y_train
        )
        val_ranking_accuracy = self._calculate_ranking_accuracy(
            X_val_scaled, metadata_val, y_val
        )
        
        metrics = {
            'train_accuracy': float(train_accuracy),
            'validation_accuracy': float(val_accuracy),
            'train_precision': float(train_precision),
            'train_recall': float(train_recall),
            'train_f1': float(train_f1),
            'validation_precision': float(val_precision),
            'validation_recall': float(val_recall),
            'validation_f1': float(val_f1),
            'train_ranking_accuracy': float(train_ranking_accuracy),
            'validation_ranking_accuracy': float(val_ranking_accuracy),
            'train_samples': len(X_train),
            'validation_samples': len(X_val),
            'positive_samples_train': int(np.sum(y_train_binary)),
            'negative_samples_train': int(len(y_train_binary) - np.sum(y_train_binary)),
            'positive_samples_validation': int(np.sum(y_val_binary)),
            'negative_samples_validation': int(len(y_val_binary) - np.sum(y_val_binary)),
            'training_time_seconds': float(training_time)
        }
        
        latency_info = self._measure_prediction_latency(texts_val, metadata_val)
        metrics.update({
            'avg_prediction_latency_ms': latency_info['stats']['avg_ms'],
            'p95_prediction_latency_ms': latency_info['stats']['p95_ms'],
            'max_prediction_latency_ms': latency_info['stats']['max_ms']
        })
        
        # Collect predictions and probabilities for all validation samples
        predictions_info = self._collect_predictions(X_val_scaled, metadata_val, y_val)
        
        print(f"\nTraining Results:")
        print(f"  Train Accuracy: {train_accuracy:.4f}")
        print(f"  Validation Accuracy: {val_accuracy:.4f}")
        print(f"  Train Ranking Accuracy: {train_ranking_accuracy:.4f}")
        print(f"  Validation Ranking Accuracy: {val_ranking_accuracy:.4f}")
        print(f"  Train Precision: {train_precision:.4f}")
        print(f"  Train Recall: {train_recall:.4f}")
        print(f"  Train F1: {train_f1:.4f}")
        print(f"  Validation Precision: {val_precision:.4f}")
        print(f"  Validation Recall: {val_recall:.4f}")
        print(f"  Validation F1: {val_f1:.4f}")
        
        return metrics, latency_info, predictions_info
    
    def _calculate_ranking_accuracy(
        self, 
        X: np.ndarray, 
        metadata: List[Dict[str, Any]], 
        y: np.ndarray
    ) -> float:
        """
        Calculate ranking accuracy: percentage of entries where sugg_idx has highest probability.
        
        Args:
            X: Feature matrix
            metadata: List of metadata dicts with entry_idx, source_idx, sugg_idx
            y: Labels (for grouping by entry)
            
        Returns:
            Ranking accuracy (0-1)
        """
        # Group samples by entry_idx
        entry_groups = {}
        for i, meta in enumerate(metadata):
            entry_idx = meta['entry_idx']
            if entry_idx not in entry_groups:
                entry_groups[entry_idx] = []
            entry_groups[entry_idx].append((i, meta))
        
        # Get probabilities for all samples
        probs = self.model.predict_proba(X)
        # For 2-class ordinal: class 0 = non-GEO, class 1 = GEO
        # Use probability of class 1 (GEO) as the score
        geo_scores = probs[:, 1]
        
        correct = 0
        total = 0
        
        for entry_idx, samples in entry_groups.items():
            if len(samples) < 2:
                continue  # Need at least 2 sources to rank
            
            total += 1
            # Find which sample is sugg_idx
            sugg_idx_in_entry = None
            scores = []
            indices = []
            
            for sample_idx, meta in samples:
                indices.append(sample_idx)
                scores.append(geo_scores[sample_idx])
                if meta['source_idx'] == meta['sugg_idx']:
                    sugg_idx_in_entry = len(scores) - 1
            
            if sugg_idx_in_entry is None:
                continue
            
            # Check if sugg_idx has the highest score
            max_score_idx = np.argmax(scores)
            if max_score_idx == sugg_idx_in_entry:
                correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def _measure_prediction_latency(
        self,
        texts: List[str],
        metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Measure latency for individual predictions.
        
        Args:
            texts: List of cleaned texts corresponding to samples
            metadata: Metadata aligned with texts
        
        Returns:
            Dict with per-sample records and summary stats
        """
        records = []
        latencies = []
        
        for idx, text in enumerate(texts):
            start = time.perf_counter()
            try:
                self.predict(text)
            except Exception as exc:
                latency_ms = None
                error = str(exc)
            else:
                latency_ms = (time.perf_counter() - start) * 1000
                error = None
            if latency_ms is not None:
                latencies.append(latency_ms)
            record = {
                'sample_index': idx,
                'entry_idx': metadata[idx]['entry_idx'],
                'source_idx': metadata[idx]['source_idx'],
                'sugg_idx': metadata[idx]['sugg_idx'],
                'latency_ms': latency_ms
            }
            if error:
                record['error'] = error
            records.append(record)
        
        if latencies:
            stats = {
                'count': len(latencies),
                'avg_ms': float(np.mean(latencies)),
                'median_ms': float(np.median(latencies)),
                'p95_ms': float(np.percentile(latencies, 95)),
                'max_ms': float(np.max(latencies)),
                'min_ms': float(np.min(latencies))
            }
        else:
            stats = {
                'count': 0,
                'avg_ms': 0.0,
                'median_ms': 0.0,
                'p95_ms': 0.0,
                'max_ms': 0.0,
                'min_ms': 0.0
            }
        
        return {'records': records, 'stats': stats}
    
    def _collect_predictions(
        self,
        X: np.ndarray,
        metadata: List[Dict[str, Any]],
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Collect predictions and probabilities for all samples.
        
        Args:
            X: Feature matrix (scaled)
            metadata: List of metadata dicts
            y: True labels
            
        Returns:
            Dict with predictions organized by entry
        """
        # Get predictions and probabilities
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Organize by entry
        entry_results = {}
        for i, meta in enumerate(metadata):
            entry_idx = meta['entry_idx']
            source_idx = meta['source_idx']
            sugg_idx = meta['sugg_idx']
            
            if entry_idx not in entry_results:
                entry_results[entry_idx] = {
                    'entry_idx': entry_idx,
                    'sugg_idx': sugg_idx,
                    'sources': []
                }
            
            is_geo_source = (source_idx == sugg_idx)
            # Map predictions: 0 -> 0, 1 -> 2
            predicted_class_mapped = int(predictions[i])
            predicted_class = predicted_class_mapped * 2
            is_predicted_geo = (predicted_class_mapped == 1)
            
            # For 2-class ordinal: probabilities[i, 0] = non-GEO, probabilities[i, 1] = GEO
            entry_results[entry_idx]['sources'].append({
                'source_idx': source_idx,
                'is_geo_source': is_geo_source,
                'predicted_class': predicted_class,
                'is_predicted_geo': is_predicted_geo,
                'probabilities': {
                    'class_0': float(probabilities[i, 0]),  # non-GEO
                    'class_1': 0.0,  # Not used in 2-class model
                    'class_2': float(probabilities[i, 1])  # GEO probability
                },
                'geo_probability': float(probabilities[i, 1]),
                'true_label': int(y[i])
            })
        
        # Sort sources by GEO probability (descending) for each entry
        for entry_idx in entry_results:
            entry_results[entry_idx]['sources'].sort(
                key=lambda x: x['geo_probability'], 
                reverse=True
            )
        
        return {
            'entries': list(entry_results.values()),
            'total_entries': len(entry_results),
            'total_sources': len(metadata)
        }
    
    def predict(self, cleaned_text: str) -> Tuple[int, np.ndarray]:
        """
        Predict if a source is GEO-optimized based on cleaned_text.
        
        Args:
            cleaned_text: The cleaned text to classify
            
        Returns:
            Tuple of (prediction, probabilities) where prediction is class index
            and probabilities is array of probabilities for each class
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a saved model.")
        
        features = self.extract_features(cleaned_text)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        if self.pca is not None:
            features_scaled = self.pca.transform(features_scaled)
        
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        return int(prediction), probabilities
    
    def predict_entry(self, sources: List[str]) -> List[Dict[str, Any]]:
        """
        Predict probability distributions for all sources in an entry, considering context.
        
        This method processes all sources together and returns context-aware probabilities
        that are normalized within the entry (using softmax on GEO probabilities).
        
        Args:
            sources: List of cleaned_text strings for all sources in an entry
            
        Returns:
            List of dictionaries, one per source, each containing:
            - 'source_idx': Index of the source in the input list
            - 'prediction': Class prediction (0 or 1, mapped from 0 or 2)
            - 'probabilities': Array of probabilities [P(non-GEO), P(GEO)]
            - 'geo_probability': Probability of GEO class (context-normalized)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a saved model.")
        
        if not sources:
            return []
        
        # Extract features for all sources
        features_list = []
        for source in sources:
            features = self.extract_features(source)
            features_list.append(features)
        
        X = np.array(features_list)
        X_scaled = self.scaler.transform(X)
        if self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        
        # Get raw predictions and probabilities
        predictions_mapped = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Extract GEO probabilities (class 1)
        geo_probs_raw = probabilities[:, 1]
        
        # Apply softmax normalization within entry to make probabilities context-aware
        exp_geo_probs = np.exp(geo_probs_raw)
        geo_probs_normalized = exp_geo_probs / np.sum(exp_geo_probs)
        
        # Build results
        results = []
        for i, source in enumerate(sources):
            # Reconstruct full probability distribution with normalized GEO prob
            non_geo_prob = 1.0 - geo_probs_normalized[i]
            
            results.append({
                'source_idx': i,
                'prediction': int(predictions_mapped[i]),
                'probabilities': np.array([non_geo_prob, geo_probs_normalized[i]]),
                'geo_probability': float(geo_probs_normalized[i])
            })
        
        return results
    
    def save_model(self, model_path: Path, scaler_path: Optional[Path] = None, pca_path: Optional[Path] = None):
        """Save the trained model, scaler, and PCA transformer."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_path = Path(model_path)
        if scaler_path is None:
            scaler_path = model_path.parent / f"{model_path.stem}_scaler{model_path.suffix}"
        else:
            scaler_path = Path(scaler_path)
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        if self.pca is not None:
            if pca_path is None:
                pca_path = model_path.parent / f"{model_path.stem}_pca{model_path.suffix}"
            else:
                pca_path = Path(pca_path)
            joblib.dump(self.pca, pca_path)
            print(f"PCA saved to: {pca_path}")
        
        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
    
    def load_model(self, model_path: Path, scaler_path: Optional[Path] = None, pca_path: Optional[Path] = None):
        """Load a saved model, scaler, and PCA transformer."""
        model_path = Path(model_path)
        if scaler_path is None:
            scaler_path = model_path.parent / f"{model_path.stem}_scaler{model_path.suffix}"
        else:
            scaler_path = Path(scaler_path)
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Try to load PCA if it exists
        if pca_path is None:
            pca_path = model_path.parent / f"{model_path.stem}_pca{model_path.suffix}"
        else:
            pca_path = Path(pca_path)
        
        if pca_path.exists():
            self.pca = joblib.load(pca_path)
            self.pca_components = self.pca.n_components
            print(f"PCA loaded from: {pca_path}")
        
        print(f"Model loaded from: {model_path}")
        print(f"Scaler loaded from: {scaler_path}")


def train_logistic_ordinal_classifier(
    optimization_dataset_path: str,
    output_dir: Optional[str] = None,
    limit: int = 1000,
    train_size: int = 300,
    validation_size: int = 700,
    C: float = 1.0,
    use_semantic_features: bool = False,
    model_name: str = 'logistic_ordinal_geo_detector.pkl',
    pca_components: Optional[int] = None
) -> LogisticOrdinalGEODetector:
    """
    Train a Logistic Ordinal classifier on optimization dataset.
    
    Args:
        optimization_dataset_path: Path to optimization dataset JSON file
        output_dir: Directory to save the trained model
        limit: Number of entries to use
        train_size: Number of samples to use for training
        validation_size: Number of samples to use for validation
        C: Inverse regularization strength
        model_name: Name of the model file to save
        
    Returns:
        Trained LogisticOrdinalGEODetector instance
    """
    # Load optimization dataset
    opt_file = Path(optimization_dataset_path)
    if not opt_file.exists():
        raise FileNotFoundError(f"Optimization dataset not found: {optimization_dataset_path}")
    
    print(f"Loading optimization dataset from: {opt_file}")
    with open(opt_file, 'r', encoding='utf-8') as f:
        opt_data = json.load(f)
    
    if isinstance(opt_data, list):
        optimization_dataset = opt_data
    elif isinstance(opt_data, dict):
        optimization_dataset = list(opt_data.values())
    else:
        raise ValueError(f"Unexpected dataset format: {type(opt_data)}")
    
    print(f"Loaded {len(optimization_dataset)} optimization entries")
    
    # Initialize detector
    print("\nInitializing embedding model...")
    detector = LogisticOrdinalGEODetector(use_semantic_features=use_semantic_features, pca_components=pca_components)
    print(f"Embedding model: {detector.embedding_model.get_sentence_embedding_dimension()} dimensions")
    if use_semantic_features:
        print(f"Using semantic pattern features: {detector.semantic_extractor.get_num_features()} additional features")
    if pca_components is not None:
        print(f"PCA enabled: {pca_components} components")
    
    # Prepare training data
    print(f"\nPreparing training data from first {limit} entries...")
    X, y, entry_metadata, source_texts = detector.prepare_training_data(
        optimization_dataset,
        limit=limit
    )
    print(f"Extracted {len(X)} samples ({np.sum(y == 2)} GEO, {np.sum(y == 0)} non-GEO)")
    feature_desc = "embeddings + semantic pattern scores" if use_semantic_features else "semantic embeddings only"
    print(f"Feature vector dimension: {X.shape[1]} ({feature_desc})")
    
    # Verify we have enough samples
    if len(X) < train_size + validation_size:
        print(f"Warning: Only {len(X)} samples available, but train_size={train_size} + validation_size={validation_size} = {train_size + validation_size}")
        print(f"Adjusting: using {train_size} for training, {len(X) - train_size} for validation")
        validation_size = len(X) - train_size
    
    print(f"\nSplitting data: {train_size} samples for training, {validation_size} samples for validation")
    print(f"Regularization C parameter: {C}")
    
    # Train
    metrics, latency_info, predictions_info = detector.train(
        X, y, entry_metadata, source_texts,
        train_size=train_size, 
        validation_size=validation_size, 
        C=C
    )
    
    # Save model
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / model_name
    detector.save_model(model_path)
    
    # Save training metrics
    model_name_path = Path(model_name)
    metrics_path = output_dir / f"{model_name_path.stem}_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    latency_path = output_dir / f"{model_name_path.stem}_prediction_latency.json"
    with open(latency_path, 'w', encoding='utf-8') as f:
        json.dump(latency_info, f, indent=2)
    predictions_path = output_dir / f"{model_name_path.stem}_predictions.json"
    with open(predictions_path, 'w', encoding='utf-8') as f:
        json.dump(predictions_info, f, indent=2)
    print(f"Training metrics saved to: {metrics_path}")
    print(f"Prediction latency details saved to: {latency_path}")
    print(f"Predictions and probabilities saved to: {predictions_path}")
    
    return detector


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Logistic Ordinal classifier for GEO detection')
    parser.add_argument('--opt-data', type=str, default='optimization_dataset.json',
                        help='Path to optimization dataset JSON file')
    parser.add_argument('--limit', type=int, default=1000,
                        help='Total number of entries to use')
    parser.add_argument('--train-size', type=int, default=300,
                        help='Number of samples to use for training')
    parser.add_argument('--validation-size', type=int, default=700,
                        help='Number of samples to use for validation')
    parser.add_argument('--C', type=float, default=1.0,
                        help='Inverse regularization strength C (default: 1.0)')
    parser.add_argument('--use-semantic-features', action='store_true',
                        help='Include individual semantic matching pattern scores as features')
    parser.add_argument('--pca-components', type=int, default=None,
                        help='Number of PCA components to keep (None = no PCA)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for saved model')
    parser.add_argument('--model-name', type=str, default='logistic_ordinal_geo_detector.pkl',
                        help='Name of the model file')
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    opt_data_path = project_root / args.opt_data
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).parent
    
    detector = train_logistic_ordinal_classifier(
        optimization_dataset_path=str(opt_data_path),
        output_dir=str(output_dir),
        limit=args.limit,
        train_size=args.train_size,
        validation_size=args.validation_size,
        C=args.C,
        use_semantic_features=args.use_semantic_features,
        model_name=args.model_name,
        pca_components=args.pca_components
    )
    
    print("\n" + "="*80)
    print("Logistic Ordinal Classifier Training Complete!")
    print("="*80)

