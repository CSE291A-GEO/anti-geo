"""
SVM-based GEO Detection Classifier

This module implements a Support Vector Machine (SVM) classifier for detecting
GEO-optimized sources based on features extracted from semantic matching scores
and other GEO detection metrics.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import joblib
from sentence_transformers import SentenceTransformer


class SVMGEODetector:
    """
    SVM-based classifier for GEO detection.
    """
    
    def __init__(
        self, 
        model_path: Optional[Path] = None, 
        scaler_path: Optional[Path] = None,
        embedding_model_name: str = 'all-MiniLM-L6-v2'
    ):
        """
        Initialize the SVM detector.
        
        Args:
            model_path: Path to saved SVM model (pickle file)
            scaler_path: Path to saved scaler (pickle file)
            embedding_model_name: Name of the sentence transformer model for embeddings
        """
        self.model = None
        self.scaler = StandardScaler()
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.feature_names = [f'embedding_dim_{i}' for i in range(self.embedding_dim)]
        
        if model_path and model_path.exists():
            self.load_model(model_path, scaler_path)
    
    def extract_features(self, cleaned_text: str) -> np.ndarray:
        """
        Extract features from cleaned_text using only semantic embeddings.
        
        Args:
            cleaned_text: The cleaned text to embed
            
        Returns:
            numpy array of embedding features (384 dimensions)
        """
        if not cleaned_text:
            # If no text provided, use zero vector
            return np.zeros(self.embedding_dim)
        
        try:
            # Generate embedding for the cleaned text
            embedding = self.embedding_model.encode(cleaned_text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            # If embedding fails, use zero vector
            print(f"    Warning: Failed to generate embedding: {str(e)[:50]}")
            return np.zeros(self.embedding_dim)
    
    def prepare_training_data(
        self, 
        optimization_dataset: List[Dict[str, Any]],
        limit: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data directly from optimization dataset.
        
        Args:
            optimization_dataset: List of entries from optimization dataset
            limit: Maximum number of entries to use
            
        Returns:
            Tuple of (X, y) where X is feature matrix and y is labels
        """
        X = []
        y = []
        
        entries_used = 0
        for entry in optimization_dataset[:limit]:
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
            
            entries_used += 1
            if entries_used >= limit:
                break
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y
    
    def train(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        train_size: Optional[int] = None,
        validation_size: Optional[int] = None,
        test_size: Optional[float] = None,
        random_state: int = 42,
        C: float = 1.5
    ) -> Dict[str, Any]:
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
            
            # Use remaining samples for validation
            if validation_size is not None:
                if train_size + validation_size > len(X):
                    raise ValueError(f"train_size ({train_size}) + validation_size ({validation_size}) exceeds total samples ({len(X)})")
                X_val = X[train_size:train_size + validation_size]
                y_val = y[train_size:train_size + validation_size]
            else:
                X_val = X[train_size:]
                y_val = y[train_size:]
        else:
            # Proportion-based split (for backward compatibility)
            from sklearn.model_selection import train_test_split
            if test_size is None:
                test_size = 0.2
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train SVM
        print(f"Training SVM classifier with C={C} (soft margin)...")
        self.model = SVC(kernel='rbf', probability=True, random_state=random_state, C=C)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_train_pred = self.model.predict(X_train_scaled)
        y_val_pred = self.model.predict(X_val_scaled)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        
        # Calculate precision, recall, and F1 scores
        train_precision = precision_score(y_train, y_train_pred, zero_division=0)
        train_recall = recall_score(y_train, y_train_pred, zero_division=0)
        train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
        
        val_precision = precision_score(y_val, y_val_pred, zero_division=0)
        val_recall = recall_score(y_val, y_val_pred, zero_division=0)
        val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
        
        metrics = {
            'train_accuracy': float(train_accuracy),
            'validation_accuracy': float(val_accuracy),
            'train_precision': float(train_precision),
            'train_recall': float(train_recall),
            'train_f1': float(train_f1),
            'validation_precision': float(val_precision),
            'validation_recall': float(val_recall),
            'validation_f1': float(val_f1),
            'train_samples': len(X_train),
            'validation_samples': len(X_val),
            'positive_samples_train': int(np.sum(y_train)),
            'negative_samples_train': int(len(y_train) - np.sum(y_train)),
            'positive_samples_validation': int(np.sum(y_val)),
            'negative_samples_validation': int(len(y_val) - np.sum(y_val))
        }
        
        print(f"\nTraining Results:")
        print(f"  Train Accuracy: {train_accuracy:.4f}")
        print(f"  Validation Accuracy: {val_accuracy:.4f}")
        print(f"  Train Precision: {train_precision:.4f}")
        print(f"  Train Recall: {train_recall:.4f}")
        print(f"  Train F1: {train_f1:.4f}")
        print(f"  Validation Precision: {val_precision:.4f}")
        print(f"  Validation Recall: {val_recall:.4f}")
        print(f"  Validation F1: {val_f1:.4f}")
        print(f"  Train Samples: {len(X_train)} (Positive: {metrics['positive_samples_train']}, Negative: {metrics['negative_samples_train']})")
        print(f"  Validation Samples: {len(X_val)} (Positive: {metrics['positive_samples_validation']}, Negative: {metrics['negative_samples_validation']})")
        
        print(f"\nValidation Set Classification Report:")
        print(classification_report(y_val, y_val_pred, target_names=['Non-GEO', 'GEO']))
        
        print(f"\nConfusion Matrix (Validation Set):")
        cm = confusion_matrix(y_val, y_val_pred)
        print(f"  True Negatives: {cm[0, 0]}, False Positives: {cm[0, 1]}")
        print(f"  False Negatives: {cm[1, 0]}, True Positives: {cm[1, 1]}")
        
        return metrics
    
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
        
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return int(prediction), float(probability[1])  # Return probability of GEO class
    
    def save_model(self, model_path: Path, scaler_path: Optional[Path] = None):
        """
        Save the trained model and scaler.
        
        Args:
            model_path: Path to save the model
            scaler_path: Path to save the scaler (if None, uses model_path with _scaler suffix)
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
        
        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
    
    def load_model(self, model_path: Path, scaler_path: Optional[Path] = None):
        """
        Load a saved model and scaler.
        
        Args:
            model_path: Path to the saved model
            scaler_path: Path to the saved scaler (if None, uses model_path with _scaler suffix)
        """
        model_path = Path(model_path)
        if scaler_path is None:
            scaler_path = model_path.parent / f"{model_path.stem}_scaler{model_path.suffix}"
        else:
            scaler_path = Path(scaler_path)
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        print(f"Model loaded from: {model_path}")
        print(f"Scaler loaded from: {scaler_path}")


def train_svm_classifier(
    optimization_dataset_path: str,
    output_dir: Optional[str] = None,
    limit: int = 1000,
    train_size: int = 300,
    validation_size: int = 700,
    C: float = 1.5,
    model_name: str = 'svm_geo_detector.pkl'
) -> SVMGEODetector:
    """
    Train an SVM classifier on optimization dataset using only semantic embeddings.
    
    Args:
        optimization_dataset_path: Path to optimization dataset JSON file
        output_dir: Directory to save the trained model (default: src/classification)
        limit: Number of entries to use (default: 1000)
        train_size: Number of samples to use for training (default: 300)
        validation_size: Number of samples to use for validation (default: 700)
        C: Regularization parameter for SVM (default: 1.5, higher = harder margin)
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
    detector = SVMGEODetector()
    print(f"Embedding model: {detector.embedding_model.get_sentence_embedding_dimension()} dimensions")
    
    # Prepare training data directly from optimization dataset
    print(f"\nPreparing training data from first {limit} entries...")
    X, y = detector.prepare_training_data(
        optimization_dataset,
        limit=limit
    )
    print(f"Extracted {len(X)} samples ({np.sum(y)} positive, {len(y) - np.sum(y)} negative)")
    print(f"Feature vector dimension: {X.shape[1]} (semantic embeddings only)")
    
    # Verify we have enough samples
    if len(X) < train_size + validation_size:
        print(f"Warning: Only {len(X)} samples available, but train_size={train_size} + validation_size={validation_size} = {train_size + validation_size}")
        print(f"Adjusting: using {train_size} for training, {len(X) - train_size} for validation")
        validation_size = len(X) - train_size
    
    print(f"\nSplitting data: {train_size} samples for training, {validation_size} samples for validation")
    print(f"SVM C parameter: {C} (soft margin, higher = more restrictive)")
    
    # Train
    metrics = detector.train(X, y, train_size=train_size, validation_size=validation_size, C=C)
    
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
    print(f"Training metrics saved to: {metrics_path}")
    
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
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for saved model')
    parser.add_argument('--model-name', type=str, default='svm_geo_detector.pkl',
                        help='Name of the model file')
    
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
        model_name=args.model_name
    )
    
    print("\n" + "="*80)
    print("SVM Classifier Training Complete!")
    print("="*80)

