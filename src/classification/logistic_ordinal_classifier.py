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
    from .excessiveness_features import extract_excessiveness_features, get_num_features as get_excessiveness_num_features
    from .text_quality_features import extract_text_quality_features, get_num_features as get_text_quality_num_features
except ImportError:
    from semantic_features import SemanticFeatureExtractor
    from bias_utils import (
        oversample_positive,
        entry_argmax_predictions,
        calculate_ranking_accuracy,
    )
    from excessiveness_features import extract_excessiveness_features, get_num_features as get_excessiveness_num_features
    from text_quality_features import extract_text_quality_features, get_num_features as get_text_quality_num_features


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
    
    def _ordinal_log_loss(self, params, X, y, n_features, entry_groups=None, 
                         use_extreme_penalty=False, use_margin_loss=False, 
                         margin=0.2, extreme_penalty_weight=2.0):
        """
        Compute ordinal logistic loss (negative log-likelihood) with optional enhancements.
        
        Args:
            params: Flattened parameters [coefficients, thresholds]
            X: Feature matrix
            y: Ordinal labels (0, 1, 2, ...)
            n_features: Number of features
            entry_groups: List of entry groups for ranking loss (optional)
            use_extreme_penalty: If True, add penalty for extreme under-confidence
            use_margin_loss: If True, add margin-based ranking loss
            margin: Minimum gap between GEO and non-GEO probabilities
            extreme_penalty_weight: Weight for extreme failure penalty
            
        Returns:
            Negative log-likelihood + penalties
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
        base_loss = -(log_likelihood / n_samples) + reg_term
        
        # Add extreme failure penalty
        extreme_penalty = 0.0
        if use_extreme_penalty and entry_groups is not None:
            for group in entry_groups:
                sugg_idx = group.get('sugg_idx')
                if sugg_idx is None:
                    continue
                
                # Find GEO source in this group
                source_indices = group.get('source_indices', [])
                for i, idx in enumerate(group['indices']):
                    if i < len(source_indices) and source_indices[i] == sugg_idx:
                        if idx < len(z):
                            z_geo = z[idx, 0]
                            geo_prob = expit(z_geo - thresholds[-1])  # GEO probability (class 1)
                            
                            # Heavy penalty if GEO prob < 0.1
                            if geo_prob < 0.1:
                                extreme_penalty += extreme_penalty_weight * (0.1 - geo_prob)
                        break
        
        # Add margin-based ranking loss
        margin_loss = 0.0
        if use_margin_loss and entry_groups is not None:
            for group in entry_groups:
                sugg_idx = group.get('sugg_idx')
                if sugg_idx is None:
                    continue
                
                geo_z = None
                max_non_geo_z = -float('inf')
                
                source_indices = group.get('source_indices', [])
                for i, idx in enumerate(group['indices']):
                    if idx < len(z) and i < len(source_indices):
                        source_idx = source_indices[i]
                        z_val = z[idx, 0]
                        
                        if source_idx == sugg_idx:
                            geo_z = z_val
                        else:
                            max_non_geo_z = max(max_non_geo_z, z_val)
                
                if geo_z is not None and max_non_geo_z > -float('inf'):
                    gap = geo_z - max_non_geo_z
                    # Penalize if gap is too small
                    if gap < margin:
                        margin_loss += (margin - gap) ** 2
        
        total_loss = base_loss
        if use_extreme_penalty and entry_groups:
            total_loss += extreme_penalty / max(len(entry_groups), 1)
        if use_margin_loss and entry_groups:
            total_loss += 0.3 * margin_loss / max(len(entry_groups), 1)
        
        return total_loss
    
    def fit(self, X, y, entry_groups=None, use_extreme_penalty=False, 
            use_margin_loss=False, margin=0.2, extreme_penalty_weight=2.0):
        """
        Fit the ordinal logistic regression model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Ordinal labels (n_samples,) with values in [0, n_classes-1]
            entry_groups: List of entry groups for enhanced loss (optional)
            use_extreme_penalty: If True, add penalty for extreme under-confidence
            use_margin_loss: If True, add margin-based ranking loss
            margin: Minimum gap between GEO and non-GEO probabilities
            extreme_penalty_weight: Weight for extreme failure penalty
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        # Start with small random coefficients and ordered thresholds
        np.random.seed(42)
        coef_init = np.random.randn(n_features) * 0.01
        # Thresholds must be in increasing order
        thresholds_init = np.linspace(-2, 2, self.n_classes - 1)
        
        params_init = np.concatenate([coef_init, thresholds_init])
        
        # Store for loss function
        self._entry_groups = entry_groups
        self._use_extreme_penalty = use_extreme_penalty
        self._use_margin_loss = use_margin_loss
        self._margin = margin
        self._extreme_penalty_weight = extreme_penalty_weight
        
        # Optimize
        result = minimize(
            self._ordinal_log_loss,
            params_init,
            args=(X, y, n_features, entry_groups, use_extreme_penalty, 
                  use_margin_loss, margin, extreme_penalty_weight),
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
        pca_components: Optional[int] = None,
        # New toggles for improvements
        use_excessiveness_features: bool = False,
        use_relative_features: bool = False,
        use_text_quality_features: bool = False,
        enhanced_oversampling_weight: Optional[float] = None,  # None = use default (3.0), else use this value
        use_extreme_penalty: bool = False,
        use_margin_loss: bool = False,
        margin: float = 0.2,
        extreme_penalty_weight: float = 2.0,
        # Demeaning by category
        baseline_embeddings_path: Optional[str] = None
    ):
        """
        Initialize the Logistic Ordinal detector.
        
        Args:
            model_path: Path to saved model (pickle file)
            scaler_path: Path to saved scaler (pickle file)
            embedding_model_name: Name of the sentence transformer model for embeddings
            use_semantic_features: If True, include individual semantic matching pattern scores as features
            use_excessiveness_features: If True, add excessiveness features for semantic patterns
            use_relative_features: If True, add relative features comparing sources within entries
            use_text_quality_features: If True, add text quality features
            enhanced_oversampling_weight: Oversampling weight (None = default 3.0, else use this value)
            use_extreme_penalty: If True, add penalty for extreme under-confidence in loss
            use_margin_loss: If True, add margin-based ranking loss
            margin: Minimum gap between GEO and non-GEO probabilities
            extreme_penalty_weight: Weight for extreme failure penalty
            baseline_embeddings_path: Path to JSON file with category baseline embeddings for demeaning
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
        self.use_excessiveness_features = use_excessiveness_features
        self.use_relative_features = use_relative_features
        self.use_text_quality_features = use_text_quality_features
        self.enhanced_oversampling_weight = enhanced_oversampling_weight
        self.use_extreme_penalty = use_extreme_penalty
        self.use_margin_loss = use_margin_loss
        self.margin = margin
        self.extreme_penalty_weight = extreme_penalty_weight
        
        # Initialize semantic feature extractor if needed
        self.semantic_extractor = None
        if use_semantic_features:
            self.semantic_extractor = SemanticFeatureExtractor(model_name=embedding_model_name)
        
        # Load baseline embeddings for category-based demeaning
        self.baseline_embeddings = None
        if baseline_embeddings_path:
            try:
                with open(baseline_embeddings_path, 'r', encoding='utf-8') as f:
                    baseline_data = json.load(f)
                    # Convert lists back to numpy arrays
                    self.baseline_embeddings = {}
                    for category, data in baseline_data.items():
                        self.baseline_embeddings[category] = np.array(data['embedding_mean'])
                print(f"Loaded baseline embeddings for {len(self.baseline_embeddings)} categories")
            except Exception as e:
                print(f"Warning: Failed to load baseline embeddings from {baseline_embeddings_path}: {e}")
        
        # Build feature names
        self.feature_names = [f'embedding_dim_{i}' for i in range(self.embedding_dim)]
        if use_semantic_features:
            self.feature_names.extend(self.semantic_extractor.get_feature_names())
            if use_excessiveness_features:
                self.feature_names.extend([f'excessiveness_{i}' for i in range(get_excessiveness_num_features())])
        if use_text_quality_features:
            self.feature_names.extend([f'text_quality_{i}' for i in range(get_text_quality_num_features())])
        
        if model_path and model_path.exists():
            self.load_model(model_path, scaler_path)
    
    def extract_features(self, cleaned_text: str, entry_features: Optional[List[np.ndarray]] = None, 
                        s_geo_max: Optional[float] = None, category: Optional[str] = None) -> np.ndarray:
        """
        Extract features from cleaned_text.
        
        Args:
            cleaned_text: The cleaned text to embed
            entry_features: Optional list of features for other sources in the same entry (for relative features)
            s_geo_max: Optional GEO score (s_geo_max) to include as a feature
            category: Optional category name for demeaning embeddings
            
        Returns:
            numpy array of features (embeddings + optionally semantic pattern scores + s_geo_max + new features)
        """
        if not cleaned_text:
            base_features = np.zeros(self.embedding_dim)
        else:
            try:
                base_features = self.embedding_model.encode(cleaned_text, convert_to_numpy=True)
            except Exception as e:
                print(f"    Warning: Failed to generate embedding: {str(e)[:50]}")
                base_features = np.zeros(self.embedding_dim)
        
        # Demean embedding by category if baseline embeddings are loaded
        if self.baseline_embeddings is not None and category:
            category_mean = self.baseline_embeddings.get(category, self.baseline_embeddings.get('Unknown', np.zeros(self.embedding_dim)))
            base_features = base_features - category_mean
        
        features_list = [base_features]
        
        # Add s_geo_max as a feature if provided
        if s_geo_max is not None:
            features_list.append(np.array([float(s_geo_max)]))
        
        # Add semantic pattern scores if enabled
        if self.use_semantic_features and self.semantic_extractor:
            semantic_scores = self.semantic_extractor.extract_pattern_scores(cleaned_text)
            features_list.append(semantic_scores)
            
            # Add excessiveness features if enabled
            if self.use_excessiveness_features:
                excessiveness = extract_excessiveness_features(semantic_scores)
                features_list.append(excessiveness)
        
        # Add text quality features if enabled
        if self.use_text_quality_features:
            text_quality = extract_text_quality_features(cleaned_text)
            features_list.append(text_quality)
        
        # Add relative features if enabled and entry_features provided
        if self.use_relative_features and entry_features is not None and len(entry_features) > 0:
            current_features = np.concatenate(features_list)
            relative = self._extract_relative_features(current_features, entry_features)
            features_list.append(relative)
        
        return np.concatenate(features_list)
    
    def _extract_relative_features(self, source_features: np.ndarray, 
                                  entry_features: List[np.ndarray]) -> np.ndarray:
        """
        Extract features comparing this source to others in its entry.
        """
        if not entry_features:
            # Return zeros if no other sources
            num_semantic = 5 if self.use_semantic_features else 0
            return np.zeros(num_semantic * 2)  # 2 features per semantic pattern (max ratio, mean ratio)
        
        entry_features_array = np.array(entry_features)
        
        # Find semantic feature indices (after embedding dim)
        embedding_end = self.embedding_dim
        semantic_start = embedding_end
        semantic_end = semantic_start + (5 if self.use_semantic_features else 0)
        
        if self.use_semantic_features and semantic_end > semantic_start:
            semantic_scores = source_features[semantic_start:semantic_end]
            entry_semantic_scores = entry_features_array[:, semantic_start:semantic_end]
            
            relative_features = []
            
            for i in range(len(semantic_scores)):
                # This source's score relative to max in entry
                max_in_entry = np.max(entry_semantic_scores[:, i]) if len(entry_semantic_scores) > 0 else 0
                if max_in_entry > 0:
                    relative_features.append(semantic_scores[i] / max_in_entry)
                else:
                    relative_features.append(0.0)
                
                # This source's score relative to mean in entry
                mean_in_entry = np.mean(entry_semantic_scores[:, i]) if len(entry_semantic_scores) > 0 else 0
                if mean_in_entry > 0:
                    relative_features.append(semantic_scores[i] / mean_in_entry)
                else:
                    relative_features.append(0.0)
            
            return np.array(relative_features)
        else:
            return np.zeros(0)
    
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
            # First pass: collect all texts for relative features
            entry_texts = []
            for source_idx, source in enumerate(sources):
                cleaned_text = source.get('cleaned_text', '')
                if cleaned_text:
                    entry_texts.append((source_idx, cleaned_text))
            
            # Second pass: extract features with relative context
            for source_idx, cleaned_text in entry_texts:
                # Label: 2 if this source is the GEO source (sugg_idx), 0 otherwise
                # Using 2 for GEO to make it the "high" class in ordinal ranking
                label = 2 if source_idx == sugg_idx else 0
                
                # Get source data
                source = sources[source_idx]
                s_geo_max = source.get('s_geo_max', None)
                category = source.get('category', None)
                
                # Extract features for other sources in entry (for relative features)
                entry_features = None
                if self.use_relative_features:
                    entry_features = []
                    for other_idx, other_text in entry_texts:
                        if other_idx != source_idx:
                            other_source = sources[other_idx]
                            other_s_geo_max = other_source.get('s_geo_max', None)
                            other_category = other_source.get('category', None)
                            other_features = self.extract_features(other_text, entry_features=None, 
                                                                  s_geo_max=other_s_geo_max, category=other_category)
                            entry_features.append(other_features)
                
                # Extract features
                features = self.extract_features(cleaned_text, entry_features, s_geo_max=s_geo_max, category=category)
                
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
        
        # Prepare entry groups for enhanced loss (if enabled)
        enhanced_entry_groups = None
        if self.use_extreme_penalty or self.use_margin_loss:
            enhanced_entry_groups = []
            for group in train_entry_groups:
                enhanced_entry_groups.append({
                    'indices': group['indices'],
                    'sugg_idx': group['sugg_idx'],
                    'source_indices': [metadata_train[i]['source_idx'] for i in group['indices']]
                })
        
        # For entry-batched training: collect all sources from entries and train
        X_train_batched = []
        y_train_batched = []
        for entry_group in train_entry_groups:
            X_train_batched.append(entry_group['X'])
            y_train_batched.extend(entry_group['y'])
        
        X_train_batched = np.vstack(X_train_batched)
        y_train_batched = np.array(y_train_batched)
        
        # Use enhanced oversampling if enabled
        oversampling_weight = self.enhanced_oversampling_weight if self.enhanced_oversampling_weight is not None else 3.0
        if oversampling_weight > 1.0:
            X_train_batched, y_train_batched, metadata_train_batched, texts_train_batched = oversample_positive(
                X_train_batched, y_train_batched, 
                [metadata_train[i] for entry_group in train_entry_groups for i in entry_group['indices']],
                [source_texts[i] for entry_group in train_entry_groups for i in entry_group['indices']],
                positive_label=1,
                weight=oversampling_weight
            )
            # Re-group after oversampling
            train_entry_groups = _group_by_entry(X_train_batched, y_train_batched, metadata_train_batched)
            if enhanced_entry_groups:
                enhanced_entry_groups = []
                for group in train_entry_groups:
                    enhanced_entry_groups.append({
                        'indices': group['indices'],
                        'sugg_idx': group['sugg_idx'],
                        'source_indices': [metadata_train_batched[i]['source_idx'] for i in group['indices']]
                    })
        
        self.model.fit(
            X_train_batched, y_train_batched,
            entry_groups=enhanced_entry_groups,
            use_extreme_penalty=self.use_extreme_penalty,
            use_margin_loss=self.use_margin_loss,
            margin=self.margin,
            extreme_penalty_weight=self.extreme_penalty_weight
        )
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
        
        # Extract features for all sources (with relative context if enabled)
        features_list = []
        if self.use_relative_features:
            # First pass: extract all features
            all_features = []
            for source in sources:
                features = self.extract_features(source, entry_features=None)
                all_features.append(features)
            
            # Second pass: extract with relative context
            for i, source in enumerate(sources):
                entry_features = [all_features[j] for j in range(len(sources)) if j != i]
                features = self.extract_features(source, entry_features)
                features_list.append(features)
        else:
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
    pca_components: Optional[int] = None,
    # New improvement toggles
    use_excessiveness_features: bool = False,
    use_relative_features: bool = False,
    use_text_quality_features: bool = False,
    enhanced_oversampling_weight: Optional[float] = None,
    use_extreme_penalty: bool = False,
    use_margin_loss: bool = False,
    margin: float = 0.2,
    extreme_penalty_weight: float = 2.0,
    baseline_embeddings_path: Optional[str] = None
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
    detector = LogisticOrdinalGEODetector(
        use_semantic_features=use_semantic_features, 
        pca_components=pca_components,
        use_excessiveness_features=use_excessiveness_features,
        use_relative_features=use_relative_features,
        use_text_quality_features=use_text_quality_features,
        enhanced_oversampling_weight=enhanced_oversampling_weight,
        use_extreme_penalty=use_extreme_penalty,
        use_margin_loss=use_margin_loss,
        margin=margin,
        extreme_penalty_weight=extreme_penalty_weight,
        baseline_embeddings_path=baseline_embeddings_path
    )
    print(f"Embedding model: {detector.embedding_model.get_sentence_embedding_dimension()} dimensions")
    if use_semantic_features:
        print(f"Using semantic pattern features: {detector.semantic_extractor.get_num_features()} additional features")
        if use_excessiveness_features:
            print(f"Using excessiveness features: {get_excessiveness_num_features()} additional features")
    if use_text_quality_features:
        print(f"Using text quality features: {get_text_quality_num_features()} additional features")
    if use_relative_features:
        print(f"Using relative features: comparing sources within entries")
    if enhanced_oversampling_weight is not None:
        print(f"Enhanced oversampling weight: {enhanced_oversampling_weight}")
    if use_extreme_penalty:
        print(f"Extreme failure penalty enabled (weight: {extreme_penalty_weight})")
    if use_margin_loss:
        print(f"Margin-based ranking loss enabled (margin: {margin})")
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
    # New improvement toggles
    parser.add_argument('--use-excessiveness-features', action='store_true',
                        help='Add excessiveness features for semantic patterns')
    parser.add_argument('--use-relative-features', action='store_true',
                        help='Add relative features comparing sources within entries')
    parser.add_argument('--use-text-quality-features', action='store_true',
                        help='Add text quality features')
    parser.add_argument('--enhanced-oversampling-weight', type=float, default=None,
                        help='Oversampling weight for GEO samples (None = default 3.0)')
    parser.add_argument('--use-extreme-penalty', action='store_true',
                        help='Add penalty for extreme under-confidence in loss function')
    parser.add_argument('--use-margin-loss', action='store_true',
                        help='Add margin-based ranking loss')
    parser.add_argument('--margin', type=float, default=0.2,
                        help='Minimum gap between GEO and non-GEO probabilities (default: 0.2)')
    parser.add_argument('--extreme-penalty-weight', type=float, default=2.0,
                        help='Weight for extreme failure penalty (default: 2.0)')
    parser.add_argument('--baseline-embeddings', type=str, default=None,
                        help='Path to JSON file with category baseline embeddings for demeaning')
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    opt_data_path = project_root / args.opt_data
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).parent
    
    baseline_embeddings_path = None
    if args.baseline_embeddings:
        baseline_embeddings_path = str(project_root / args.baseline_embeddings) if not Path(args.baseline_embeddings).is_absolute() else args.baseline_embeddings
    
    detector = train_logistic_ordinal_classifier(
        optimization_dataset_path=str(opt_data_path),
        output_dir=str(output_dir),
        limit=args.limit,
        train_size=args.train_size,
        validation_size=args.validation_size,
        C=args.C,
        use_semantic_features=args.use_semantic_features,
        model_name=args.model_name,
        pca_components=args.pca_components,
        use_excessiveness_features=args.use_excessiveness_features,
        use_relative_features=args.use_relative_features,
        use_text_quality_features=args.use_text_quality_features,
        enhanced_oversampling_weight=args.enhanced_oversampling_weight,
        use_extreme_penalty=args.use_extreme_penalty,
        use_margin_loss=args.use_margin_loss,
        margin=args.margin,
        extreme_penalty_weight=args.extreme_penalty_weight,
        baseline_embeddings_path=baseline_embeddings_path
    )
    
    print("\n" + "="*80)
    print("Logistic Ordinal Classifier Training Complete!")
    print("="*80)

