"""
SVM-based GEO Detection Classifier

This module implements a Support Vector Machine (SVM) classifier for detecting
GEO-optimized sources based on features extracted from semantic matching scores
and other GEO detection metrics.
"""

import json
import pickle
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import joblib
from sentence_transformers import SentenceTransformer
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


class SVMGEODetector:
    """
    SVM-based classifier for GEO detection.
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
        Initialize the SVM detector.
        
        Args:
            model_path: Path to saved SVM model (pickle file)
            scaler_path: Path to saved scaler (pickle file)
            embedding_model_name: Name of the sentence transformer model for embeddings
            use_semantic_features: If True, include individual semantic matching pattern scores as features
            pca_components: Number of PCA components to keep (None = no PCA)
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
            # If no text provided, use zero vector
            base_features = np.zeros(self.embedding_dim)
        else:
            try:
                # Generate embedding for the cleaned text
                base_features = self.embedding_model.encode(cleaned_text, convert_to_numpy=True)
            except Exception as e:
                # If embedding fails, use zero vector
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
        Prepare training data directly from optimization dataset.
        
        Args:
            optimization_dataset: List of entries from optimization dataset
            limit: Maximum number of entries to use
            
        Returns:
            Tuple of (X, y, entry_metadata, source_texts)
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
                    # Skip sources without cleaned_text
                    continue
                
                # Label: 1 if this source is the GEO source (matches sugg_idx), 0 otherwise
                label = 1 if source_idx == sugg_idx else 0
                
                # Extract features (only embeddings)
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
        C: float = 1.5,
        positive_class_weight: float = 3.0
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Train the SVM classifier.
        
        Args:
            X: Feature matrix
            y: Labels
            train_size: Number of samples to use for training
            validation_size: Number of samples to use for validation (if None, uses remaining after train_size)
            test_size: Proportion of data to use for testing (only used if train_size is None)
            random_state: Random seed for reproducibility (only used if using proportion split)
            C: Regularization parameter. Higher values (e.g., 1.5) = harder margin (less tolerance for misclassifications)
            
        Returns:
            Dictionary with training metrics
        """
        # Split data - use fixed split if train_size is specified
        if train_size is not None:
            # Fixed split: first train_size samples for training, rest for validation
            if train_size >= len(X):
                raise ValueError(f"train_size ({train_size}) must be less than total samples ({len(X)})")
            X_train = X[:train_size]
            y_train = y[:train_size]
            metadata_train = entry_metadata[:train_size]
            texts_train = source_texts[:train_size]
            
            # Use remaining samples for validation
            if validation_size is not None:
                if train_size + validation_size > len(X):
                    raise ValueError(f"train_size ({train_size}) + validation_size ({validation_size}) exceeds total samples ({len(X)})")
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
            # Proportion-based split (for backward compatibility)
            from sklearn.model_selection import train_test_split
            if test_size is None:
                test_size = 0.2
            (X_train, X_val, y_train, y_val,
             metadata_train, metadata_val,
             texts_train, texts_val) = train_test_split(
                X, y, entry_metadata, source_texts,
                test_size=test_size, random_state=random_state, stratify=y
            )
        
        # Oversample positives in training data
        X_train, y_train, metadata_train, texts_train = oversample_positive(
            X_train, y_train, metadata_train, texts_train,
            positive_label=1, weight=positive_class_weight,
            random_state=random_state
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
        
        # Group training data by entry for context-aware training
        train_entry_groups = _group_by_entry(X_train_scaled, y_train, metadata_train)
        
        # Train SVM with entry-aware approach
        # Process entries in batches to make training context-aware
        print(f"Training SVM classifier with C={C} (soft margin)...")
        print(f"  Using entry-batched training for context awareness")
        class_weight = {0: 1.0, 1: positive_class_weight}
        self.model = SVC(
            kernel='rbf',
            probability=True,
            random_state=random_state,
            C=C,
            class_weight=class_weight
        )
        
        # For entry-batched training: collect all sources from entries and train
        # This ensures the model sees sources in context during training
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
        train_probs = self.model.predict_proba(X_train_scaled)
        val_probs = self.model.predict_proba(X_val_scaled)
        y_train_pred = entry_argmax_predictions(train_probs[:, 1], metadata_train)
        y_val_pred = entry_argmax_predictions(val_probs[:, 1], metadata_val)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        
        # Calculate precision, recall, and F1 scores
        train_precision = precision_score(y_train, y_train_pred, zero_division=0)
        train_recall = recall_score(y_train, y_train_pred, zero_division=0)
        train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
        
        val_precision = precision_score(y_val, y_val_pred, zero_division=0)
        val_recall = recall_score(y_val, y_val_pred, zero_division=0)
        val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
        
        train_ranking_accuracy = calculate_ranking_accuracy(
            train_probs[:, 1], metadata_train, y_train, positive_label=1
        )
        val_ranking_accuracy = calculate_ranking_accuracy(
            val_probs[:, 1], metadata_val, y_val, positive_label=1
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
            'positive_samples_train': int(np.sum(y_train)),
            'negative_samples_train': int(len(y_train) - np.sum(y_train)),
            'positive_samples_validation': int(np.sum(y_val)),
            'negative_samples_validation': int(len(y_val) - np.sum(y_val)),
            'training_time_seconds': float(training_time)
        }
        
        latency_info = self._measure_prediction_latency(texts_val, metadata_val)
        metrics.update({
            'avg_prediction_latency_ms': latency_info['stats']['avg_ms'],
            'p95_prediction_latency_ms': latency_info['stats']['p95_ms'],
            'max_prediction_latency_ms': latency_info['stats']['max_ms']
        })
        
        print(f"\nTraining Results:")
        print(f"  Train Accuracy: {train_accuracy:.4f}")
        print(f"  Validation Accuracy: {val_accuracy:.4f}")
        print(f"  Train Precision: {train_precision:.4f}")
        print(f"  Train Recall: {train_recall:.4f}")
        print(f"  Train F1: {train_f1:.4f}")
        print(f"  Validation Precision: {val_precision:.4f}")
        print(f"  Validation Recall: {val_recall:.4f}")
        print(f"  Validation F1: {val_f1:.4f}")
        print(f"  Train Ranking Accuracy: {train_ranking_accuracy:.4f}")
        print(f"  Validation Ranking Accuracy: {val_ranking_accuracy:.4f}")
        print(f"  Train Samples: {len(X_train)} (Positive: {metrics['positive_samples_train']}, Negative: {metrics['negative_samples_train']})")
        print(f"  Validation Samples: {len(X_val)} (Positive: {metrics['positive_samples_validation']}, Negative: {metrics['negative_samples_validation']})")
        
        print(f"\nValidation Set Classification Report:")
        print(classification_report(y_val, y_val_pred, target_names=['Non-GEO', 'GEO']))
        
        print(f"\nConfusion Matrix (Validation Set):")
        cm = confusion_matrix(y_val, y_val_pred)
        print(f"  True Negatives: {cm[0, 0]}, False Positives: {cm[0, 1]}")
        print(f"  False Negatives: {cm[1, 0]}, True Positives: {cm[1, 1]}")
        
        return metrics, latency_info
    
    def predict(self, cleaned_text: str) -> Tuple[int, float]:
        """
        Predict if a source is GEO-optimized based on cleaned_text.
        
        Args:
            cleaned_text: The cleaned text to classify
            
        Returns:
            Tuple of (prediction, probability) where prediction is 0 or 1
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a saved model.")
        
        features = self.extract_features(cleaned_text)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        if self.pca is not None:
            features_scaled = self.pca.transform(features_scaled)
        
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return int(prediction), float(probability[1])  # Return probability of GEO class
    
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
            - 'prediction': Binary prediction (0 or 1)
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
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Extract GEO probabilities (class 1)
        geo_probs_raw = probabilities[:, 1]
        
        # Apply softmax normalization within entry to make probabilities context-aware
        # This ensures probabilities are relative to other sources in the same entry
        exp_geo_probs = np.exp(geo_probs_raw)
        geo_probs_normalized = exp_geo_probs / np.sum(exp_geo_probs)
        
        # Build results
        results = []
        for i, source in enumerate(sources):
            # Reconstruct full probability distribution with normalized GEO prob
            # Keep non-GEO prob proportional, but ensure they sum to 1
            non_geo_prob = 1.0 - geo_probs_normalized[i]
            
            results.append({
                'source_idx': i,
                'prediction': int(predictions[i]),
                'probabilities': np.array([non_geo_prob, geo_probs_normalized[i]]),
                'geo_probability': float(geo_probs_normalized[i])
            })
        
        return results
    
    def _measure_prediction_latency(
        self,
        texts: List[str],
        metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Measure latency for each classification call.
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
    
    def save_model(self, model_path: Path, scaler_path: Optional[Path] = None, pca_path: Optional[Path] = None):
        """
        Save the trained model, scaler, and PCA transformer.
        
        Args:
            model_path: Path to save the model
            scaler_path: Path to save the scaler (if None, uses model_path with _scaler suffix)
            pca_path: Path to save the PCA transformer (if None and PCA is used, uses model_path with _pca suffix)
        """
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
        """
        Load a saved model, scaler, and PCA transformer.
        
        Args:
            model_path: Path to the saved model
            scaler_path: Path to the saved scaler (if None, uses model_path with _scaler suffix)
            pca_path: Path to the saved PCA transformer (if None, uses model_path with _pca suffix)
        """
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


def train_svm_classifier(
    optimization_dataset_path: str,
    output_dir: Optional[str] = None,
    limit: int = 1000,
    train_size: int = 300,
    validation_size: int = 700,
    C: float = 1.5,
    use_semantic_features: bool = False,
    positive_class_weight: float = 3.0,
    model_name: str = 'svm_geo_detector.pkl',
    pca_components: Optional[int] = None
) -> SVMGEODetector:
    """
    Train an SVM classifier on optimization dataset.
    
    Args:
        optimization_dataset_path: Path to optimization dataset JSON file
        output_dir: Directory to save the trained model (default: src/classification)
        limit: Number of entries to use (default: 1000)
        train_size: Number of samples to use for training (default: 300)
        validation_size: Number of samples to use for validation (default: 700)
        C: Regularization parameter for SVM (default: 1.5, higher = harder margin)
        use_semantic_features: If True, include individual semantic matching pattern scores as features
        model_name: Name of the model file to save
        
    Returns:
        Trained SVMGEODetector instance
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
    detector = SVMGEODetector(use_semantic_features=use_semantic_features, pca_components=pca_components)
    print(f"Embedding model: {detector.embedding_model.get_sentence_embedding_dimension()} dimensions")
    if pca_components is not None:
        print(f"PCA enabled: {pca_components} components")
    if use_semantic_features:
        print(f"Using semantic pattern features: {detector.semantic_extractor.get_num_features()} additional features")
    
    # Prepare training data directly from optimization dataset
    print(f"\nPreparing training data from first {limit} entries...")
    X, y, entry_metadata, source_texts = detector.prepare_training_data(
        optimization_dataset,
        limit=limit
    )
    print(f"Extracted {len(X)} samples ({np.sum(y)} positive, {len(y) - np.sum(y)} negative)")
    feature_desc = "embeddings + semantic pattern scores" if use_semantic_features else "semantic embeddings only"
    print(f"Feature vector dimension: {X.shape[1]} ({feature_desc})")
    
    # Verify we have enough samples
    if len(X) < train_size + validation_size:
        print(f"Warning: Only {len(X)} samples available, but train_size={train_size} + validation_size={validation_size} = {train_size + validation_size}")
        print(f"Adjusting: using {train_size} for training, {len(X) - train_size} for validation")
        validation_size = len(X) - train_size
    
    print(f"\nSplitting data: {train_size} samples for training, {validation_size} samples for validation")
    print(f"SVM C parameter: {C} (soft margin, higher = more restrictive)")
    
    # Train
    metrics, latency_info = detector.train(
        X, y, entry_metadata, source_texts,
        train_size=train_size,
        validation_size=validation_size,
        C=C,
        positive_class_weight=positive_class_weight
    )
    
    # Save model
    base_dir = Path(__file__).parent
    if output_dir is None:
        output_dir = base_dir / 'output'
    else:
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = base_dir / output_dir
    
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
    print(f"Training metrics saved to: {metrics_path}")
    print(f"Prediction latency details saved to: {latency_path}")
    
    return detector


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SVM classifier for GEO detection')
    parser.add_argument('--opt-data', type=str, default='optimization_dataset.json',
                        help='Path to optimization dataset JSON file')
    parser.add_argument('--limit', type=int, default=1000,
                        help='Total number of entries to use')
    parser.add_argument('--train-size', type=int, default=300,
                        help='Number of samples to use for training')
    parser.add_argument('--validation-size', type=int, default=700,
                        help='Number of samples to use for validation')
    parser.add_argument('--C', type=float, default=1.5,
                        help='SVM regularization parameter C (higher = harder margin, default: 1.5)')
    parser.add_argument('--use-semantic-features', action='store_true',
                        help='Include individual semantic matching pattern scores as features')
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory for saved model')
    parser.add_argument('--model-name', type=str, default='svm_geo_detector.pkl',
                        help='Name of the model file')
    parser.add_argument('--positive-weight', type=float, default=3.0,
                        help='Weight multiplier for GEO (positive) samples')
    parser.add_argument('--pca-components', type=int, default=None,
                        help='Number of PCA components to keep (None = no PCA)')
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    opt_data_path = project_root / args.opt_data
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).parent
    
    detector = train_svm_classifier(
        optimization_dataset_path=str(opt_data_path),
        output_dir=str(output_dir),
        limit=args.limit,
        train_size=args.train_size,
        validation_size=args.validation_size,
        C=args.C,
        use_semantic_features=args.use_semantic_features,
        positive_class_weight=args.positive_weight,
        model_name=args.model_name,
        pca_components=args.pca_components
    )
    
    print("\n" + "="*80)
    print("SVM Classifier Training Complete!")
    print("="*80)

