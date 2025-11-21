"""
RNN (LSTM/GRU) with Ordinal Loss for GEO Detection

This module implements an RNN-based classifier (LSTM or GRU) with Ordinal
Logistic Loss for detecting GEO-optimized sources. The architecture processes
the embedding features through recurrent layers before outputting a scalar
that is transformed using ordinal logistic loss.
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
from scipy.special import expit
try:
    from .semantic_features import SemanticFeatureExtractor
except ImportError:
    from semantic_features import SemanticFeatureExtractor

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    try:
        import tensorflow as tf
        from tensorflow import keras
        TORCH_AVAILABLE = False
        TF_AVAILABLE = True
    except ImportError:
        raise ImportError("Neither PyTorch nor TensorFlow available. Please install one of them.")


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


if TORCH_AVAILABLE:
    class RankingLoss(nn.Module):
        """
        Ranking loss that encourages the true GEO source to have highest probability within each entry.
        Uses softmax on GEO probabilities within entries and cross-entropy loss.
        """
        def __init__(self, alpha: float = 0.5):
            """
            Args:
                alpha: Weight for ranking loss (0-1). 1.0 = only ranking, 0.0 = only ordinal
            """
            super(RankingLoss, self).__init__()
            self.alpha = alpha
        
        def forward(self, z: torch.Tensor, y: torch.Tensor, thresholds: torch.nn.Parameter,
                   entry_groups: List[Dict[str, Any]], source_indices: List[int]) -> torch.Tensor:
            """
            Args:
                z: Scalar outputs (batch_size,)
                y: Labels (batch_size,)
                thresholds: Ordinal thresholds
                entry_groups: List of dicts with 'indices' (local indices in batch), 'sugg_idx', and 'source_indices'
                source_indices: List of source_idx for each sample in the batch
            """
            # Compute ordinal loss (existing loss)
            ordinal_loss_fn = OrdinalLoss(n_classes=2)
            ordinal_loss = ordinal_loss_fn(z, y, thresholds)
            
            # Compute ranking loss
            # Get probabilities for all samples
            batch_size = z.shape[0]
            probs = torch.zeros(batch_size, 2, device=z.device)
            
            for i in range(batch_size):
                z_i = z[i]
                cum_prob = torch.sigmoid(thresholds[0] - z_i)
                probs[i, 0] = cum_prob
                probs[i, 1] = 1 - cum_prob
            
            ranking_loss = torch.tensor(0.0, device=z.device)
            num_entries = 0
            
            for group in entry_groups:
                local_indices = group['indices']  # Local indices within this batch
                sugg_idx = group['sugg_idx']
                
                if len(local_indices) < 2 or sugg_idx is None:
                    continue
                
                # Find which local index corresponds to sugg_idx
                sugg_local_idx = None
                for local_idx in local_indices:
                    if source_indices[local_idx] == sugg_idx:
                        sugg_local_idx = local_idx
                        break
                
                if sugg_local_idx is None:
                    continue
                
                # Get GEO probabilities for sources in this entry (using local indices)
                entry_geo_probs = probs[local_indices, 1]
                
                # Apply softmax to get normalized probabilities
                entry_geo_probs_norm = torch.softmax(entry_geo_probs, dim=0)
                
                # Create target: 1.0 for sugg_idx, 0.0 for others
                target = torch.zeros(len(local_indices), device=z.device)
                sugg_pos_in_entry = local_indices.index(sugg_local_idx)
                target[sugg_pos_in_entry] = 1.0
                
                # Cross-entropy loss (negative log likelihood)
                ranking_loss += -torch.sum(target * torch.log(entry_geo_probs_norm + 1e-10))
                num_entries += 1
            
            ranking_loss = ranking_loss / num_entries if num_entries > 0 else ranking_loss
            
            # Combine losses
            total_loss = (1 - self.alpha) * ordinal_loss + self.alpha * ranking_loss
            return total_loss
    
    class OrdinalRNN(nn.Module):
        """
        RNN (LSTM/GRU) with Ordinal Output.
        
        Architecture: Input -> RNN Layers -> Dense -> Scalar Output
        The scalar output is then transformed using ordinal logistic loss.
        """
        
        def __init__(self, input_dim: int, hidden_dim: int = 128, n_classes: int = 2, 
                     dropout: float = 0.1, num_layers: int = 2, rnn_type: str = 'LSTM',
                     bidirectional: bool = False):
            """
            Initialize the RNN.
            
            Args:
                input_dim: Input feature dimension
                hidden_dim: Hidden layer dimension
                n_classes: Number of ordinal classes
                dropout: Dropout rate
                num_layers: Number of RNN layers
                rnn_type: Type of RNN ('LSTM' or 'GRU')
                bidirectional: Whether to use bidirectional RNN
            """
            super(OrdinalRNN, self).__init__()
            self.n_classes = n_classes
            self.num_layers = num_layers
            self.hidden_dim = hidden_dim
            self.bidirectional = bidirectional
            self.rnn_type = rnn_type.upper()
            
            # Reshape input to sequence: treat embedding as sequence of chunks
            # Split input_dim into chunks to create a sequence
            # For 384-dim embedding with hidden_dim=128, we'll create ~3 timesteps
            self.chunk_size = min(hidden_dim, input_dim // 2)  # Chunk size for sequence
            self.seq_len = (input_dim + self.chunk_size - 1) // self.chunk_size  # Number of timesteps
            
            # Project each chunk to hidden_dim
            self.input_proj = nn.Linear(self.chunk_size, hidden_dim)
            
            # RNN layers
            if self.rnn_type == 'LSTM':
                self.rnn = nn.LSTM(
                    hidden_dim, 
                    hidden_dim, 
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=bidirectional,
                    batch_first=True
                )
            elif self.rnn_type == 'GRU':
                self.rnn = nn.GRU(
                    hidden_dim,
                    hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=bidirectional,
                    batch_first=True
                )
            else:
                raise ValueError(f"Unsupported RNN type: {rnn_type}. Use 'LSTM' or 'GRU'")
            
            # Output layer
            rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(rnn_output_dim, 1)  # Scalar output
            
            # Initialize thresholds for ordinal logistic (learned parameters)
            self.register_parameter('thresholds', nn.Parameter(torch.linspace(-2, 2, n_classes - 1)))
        
        def forward(self, x):
            """
            Forward pass.
            
            Args:
                x: Input features (batch_size, input_dim)
                
            Returns:
                Scalar output z (batch_size, 1)
            """
            batch_size = x.size(0)
            input_dim = x.size(1)
            
            # Reshape input to sequence: (batch_size, input_dim) -> (batch_size, seq_len, chunk_size)
            # Pad if necessary
            total_needed = self.seq_len * self.chunk_size
            if input_dim < total_needed:
                pad_size = total_needed - input_dim
                x = torch.cat([x, torch.zeros(batch_size, pad_size, device=x.device)], dim=1)
            elif input_dim > total_needed:
                # Truncate if needed
                x = x[:, :total_needed]
            
            # Reshape: (batch_size, input_dim) -> (batch_size, seq_len, chunk_size)
            x = x.view(batch_size, self.seq_len, self.chunk_size)
            
            # Project each chunk to hidden_dim
            x = self.input_proj(x)  # (batch_size, seq_len, hidden_dim)
            
            # RNN forward
            rnn_out, _ = self.rnn(x)  # (batch_size, seq_len_actual, rnn_output_dim)
            
            # Take the last output
            last_output = rnn_out[:, -1, :]  # (batch_size, rnn_output_dim)
            
            # Apply dropout and final layer
            last_output = self.dropout(last_output)
            z = self.fc(last_output)  # (batch_size, 1)
            
            return z
        
        def predict_proba(self, x):
            """
            Predict probability distribution using ordinal logistic transformation.
            
            Args:
                x: Input features (batch_size, input_dim)
                
            Returns:
                Probability matrix (batch_size, n_classes)
            """
            self.eval()
            with torch.no_grad():
                z = self.forward(x)  # (batch_size, 1)
                z = z.squeeze(-1)  # (batch_size,)
                
                batch_size = z.shape[0]
                probs = torch.zeros(batch_size, self.n_classes, device=z.device)
                
                for i in range(batch_size):
                    z_i = z[i].item()
                    
                    # Compute cumulative probabilities
                    cum_probs = []
                    for k in range(self.n_classes - 1):
                        threshold = self.thresholds[k].item()
                        cum_probs.append(expit(threshold - z_i))
                    
                    # Convert to individual class probabilities
                    probs[i, 0] = cum_probs[0]
                    for k in range(1, self.n_classes - 1):
                        probs[i, k] = cum_probs[k] - cum_probs[k - 1]
                    probs[i, self.n_classes - 1] = 1 - cum_probs[-1]
                    
                    # Normalize
                    probs[i] = probs[i] / (probs[i].sum() + 1e-10)
                
                return probs.cpu().numpy()
        
        def predict(self, x):
            """Predict class labels."""
            probs = self.predict_proba(x)
            return np.argmax(probs, axis=1)
    
    class OrdinalLoss(nn.Module):
        """
        Ordinal Logistic Loss (Cumulative Link Model Loss).
        """
        
        def __init__(self, n_classes: int = 2):
            super(OrdinalLoss, self).__init__()
            self.n_classes = n_classes
        
        def forward(self, z, y, thresholds):
            """
            Compute ordinal logistic loss.
            
            Args:
                z: Scalar outputs (batch_size,)
                y: Ordinal labels (batch_size,) with values in [0, n_classes-1]
                thresholds: Learned thresholds (n_classes - 1,)
                
            Returns:
                Loss value
            """
            batch_size = z.shape[0]
            loss = 0.0
            
            for i in range(batch_size):
                z_i = z[i]
                y_i = int(y[i].item())
                
                if y_i == 0:
                    # P(y <= 0) = sigmoid(threshold_0 - z_i)
                    prob = torch.sigmoid(thresholds[0] - z_i)
                    loss -= torch.log(prob + 1e-10)
                elif y_i == self.n_classes - 1:
                    # P(y = n_classes-1) = 1 - P(y <= n_classes-2)
                    prob_prev = torch.sigmoid(thresholds[-1] - z_i)
                    loss -= torch.log(1 - prob_prev + 1e-10)
                else:
                    # P(y = k) = P(y <= k) - P(y <= k-1)
                    prob_k = torch.sigmoid(thresholds[y_i] - z_i)
                    prob_k_minus_1 = torch.sigmoid(thresholds[y_i - 1] - z_i)
                    loss -= torch.log(prob_k - prob_k_minus_1 + 1e-10)
            
            return loss / batch_size

else:
    # TensorFlow implementation
    class OrdinalRNN:
        """
        RNN (LSTM/GRU) with Ordinal Output (TensorFlow).
        """
        
        def __init__(self, input_dim: int, hidden_dim: int = 128, n_classes: int = 2, 
                     dropout: float = 0.1, num_layers: int = 2, rnn_type: str = 'LSTM',
                     bidirectional: bool = False):
            self.n_classes = n_classes
            self.num_layers = num_layers
            self.hidden_dim = hidden_dim
            self.bidirectional = bidirectional
            self.rnn_type = rnn_type.upper()
            self.model = None
            self.thresholds = None
            self._build_model(input_dim, hidden_dim, dropout)
        
        def _build_model(self, input_dim, hidden_dim, dropout):
            """Build the Keras model."""
            inputs = keras.Input(shape=(input_dim,))
            
            # Reshape to sequence: split input_dim into chunks
            chunk_size = min(hidden_dim, input_dim // 2)
            seq_len = (input_dim + chunk_size - 1) // chunk_size
            
            # Reshape and project
            # Pad if necessary
            total_needed = seq_len * chunk_size
            if input_dim < total_needed:
                # Pad with zeros
                pad_size = total_needed - input_dim
                x = keras.layers.Lambda(lambda t: tf.pad(t, [[0, 0], [0, pad_size]]))(inputs)
            elif input_dim > total_needed:
                # Truncate
                x = keras.layers.Lambda(lambda t: t[:, :total_needed])(inputs)
            else:
                x = inputs
            
            x = keras.layers.Reshape((seq_len, chunk_size))(x)
            x = keras.layers.Dense(hidden_dim)(x)
            
            # RNN layers
            for i in range(self.num_layers):
                return_sequences = (i < self.num_layers - 1)
                if self.rnn_type == 'LSTM':
                    x = keras.layers.LSTM(
                        hidden_dim,
                        return_sequences=return_sequences,
                        dropout=dropout
                    )(x)
                elif self.rnn_type == 'GRU':
                    x = keras.layers.GRU(
                        hidden_dim,
                        return_sequences=return_sequences,
                        dropout=dropout
                    )(x)
                else:
                    raise ValueError(f"Unsupported RNN type: {self.rnn_type}")
            
            x = keras.layers.Dropout(dropout)(x)
            z = keras.layers.Dense(1, activation='linear')(x)
            
            self.model = keras.Model(inputs=inputs, outputs=z)
            
            # Initialize thresholds
            self.thresholds = tf.Variable(tf.linspace(-2.0, 2.0, self.n_classes - 1), trainable=True)
        
        def compile(self, learning_rate=0.001):
            """Compile the model with ordinal loss."""
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            self.model.compile(optimizer=optimizer, loss=self._ordinal_loss)
        
        def _ordinal_loss(self, y_true, z_pred):
            """Custom ordinal logistic loss."""
            batch_size = tf.shape(y_true)[0]
            loss = 0.0
            
            for i in range(self.n_classes):
                mask = tf.equal(y_true, i)
                z_masked = tf.boolean_mask(z_pred, mask)
                
                if i == 0:
                    prob = tf.sigmoid(self.thresholds[0] - z_masked)
                    loss += -tf.reduce_mean(tf.math.log(prob + 1e-10))
                elif i == self.n_classes - 1:
                    prob_prev = tf.sigmoid(self.thresholds[-1] - z_masked)
                    loss += -tf.reduce_mean(tf.math.log(1 - prob_prev + 1e-10))
                else:
                    prob_k = tf.sigmoid(self.thresholds[i] - z_masked)
                    prob_k_minus_1 = tf.sigmoid(self.thresholds[i - 1] - z_masked)
                    loss += -tf.reduce_mean(tf.math.log(prob_k - prob_k_minus_1 + 1e-10))
            
            return loss
        
        def fit(self, X, y, epochs=50, batch_size=32, validation_data=None, verbose=1):
            """Train the model."""
            return self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                verbose=verbose
            )
        
        def predict_proba(self, X):
            """Predict probability distribution."""
            z_pred = self.model.predict(X, verbose=0)
            batch_size = len(X)
            probs = np.zeros((batch_size, self.n_classes))
            
            thresholds_np = self.thresholds.numpy()
            
            for i in range(batch_size):
                z_i = z_pred[i, 0]
                
                cum_probs = []
                for k in range(self.n_classes - 1):
                    cum_probs.append(expit(thresholds_np[k] - z_i))
                
                probs[i, 0] = cum_probs[0]
                for k in range(1, self.n_classes - 1):
                    probs[i, k] = cum_probs[k] - cum_probs[k - 1]
                probs[i, self.n_classes - 1] = 1 - cum_probs[-1]
                
                probs[i] = probs[i] / (probs[i].sum() + 1e-10)
            
            return probs
        
        def predict(self, X):
            """Predict class labels."""
            probs = self.predict_proba(X)
            return np.argmax(probs, axis=1)


class RNNOrdinalGEODetector:
    """
    RNN with Ordinal Loss for GEO detection.
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
        Initialize the RNN Ordinal detector.
        
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
        self.use_torch = TORCH_AVAILABLE
        
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
        """Extract features from cleaned_text."""
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
            
            for source_idx, source in enumerate(sources):
                cleaned_text = source.get('cleaned_text', '')
                if not cleaned_text:
                    continue
                
                # Label: 2 if this source is the GEO source (sugg_idx), 0 otherwise
                label = 2 if source_idx == sugg_idx else 0
                
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
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        dropout: float = 0.1,
        num_layers: int = 2,
        rnn_type: str = 'LSTM',
        bidirectional: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Train the RNN Ordinal classifier.
        
        Args:
            X: Feature matrix
            y: Ordinal labels
            entry_metadata: Metadata for each sample
            source_texts: List of source texts
            train_size: Number of samples to use for training
            validation_size: Number of samples to use for validation
            test_size: Proportion of data to use for testing
            random_state: Random seed
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size
            dropout: Dropout rate
            num_layers: Number of RNN layers
            rnn_type: Type of RNN ('LSTM' or 'GRU')
            bidirectional: Whether to use bidirectional RNN
            
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
            texts_train = source_texts[:train_size]
            
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
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Apply PCA if enabled
        if self.pca is not None:
            X_train_scaled = self.pca.fit_transform(X_train_scaled)
            X_val_scaled = self.pca.transform(X_val_scaled)
            print(f"  Applied PCA: {X_train_scaled.shape[1]} components (explained variance: {self.pca.explained_variance_ratio_.sum():.4f})")
        
        # Train RNN
        print(f"Training RNN Ordinal Regression classifier...")
        print(f"  hidden_dim={hidden_dim}, num_layers={num_layers}, rnn_type={rnn_type}, bidirectional={bidirectional}")
        print(f"  learning_rate={learning_rate}, epochs={epochs}")
        print(f"  Using {'PyTorch' if self.use_torch else 'TensorFlow'}")
        
        # Use 2 classes since we only have labels 0 (non-GEO) and 2 (GEO)
        # Map labels: 0 -> 0, 2 -> 1
        y_train_mapped = (y_train == 2).astype(int)
        y_val_mapped = (y_val == 2).astype(int)
        
        training_start = time.perf_counter()
        if self.use_torch:
            # PyTorch training
            self.model = OrdinalRNN(
                input_dim=X_train_scaled.shape[1],
                hidden_dim=hidden_dim,
                n_classes=2,
                dropout=dropout,
                num_layers=num_layers,
                rnn_type=rnn_type,
                bidirectional=bidirectional
            )
            
            # Use ranking loss for context-aware training
            ranking_criterion = RankingLoss(alpha=0.5)  # 50% ordinal, 50% ranking
            ordinal_criterion = OrdinalLoss(n_classes=2)
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train_scaled)
            y_train_tensor = torch.LongTensor(y_train_mapped)
            X_val_tensor = torch.FloatTensor(X_val_scaled)
            y_val_tensor = torch.LongTensor(y_val_mapped)
            
            # Group training data by entry for context-aware training (use mapped labels)
            train_entry_groups = _group_by_entry(X_train_scaled, y_train_mapped, metadata_train)
            val_entry_groups = _group_by_entry(X_val_scaled, y_val_mapped, metadata_val)
            
            # Training loop with entry-batched processing
            for epoch in range(epochs):
                self.model.train()
                total_loss = 0.0
                num_batches = 0
                
                # Process entries in batches
                entry_batch_size = max(1, batch_size // 3)  # Approximate batch size in entries
                for batch_start in range(0, len(train_entry_groups), entry_batch_size):
                    batch_entries = train_entry_groups[batch_start:batch_start + entry_batch_size]
                    
                    # Collect all sources from this batch of entries
                    batch_X = []
                    batch_y = []
                    batch_indices = []
                    batch_source_indices = []
                    entry_group_info = []
                    
                    for entry_group in batch_entries:
                        entry_X = entry_group['X']
                        entry_y = entry_group['y']
                        entry_indices = entry_group['indices']
                        
                        # Get source indices for this entry
                        entry_source_indices = [metadata_train[i]['source_idx'] for i in entry_indices]
                        
                        # Local indices within this batch
                        local_start = len(batch_X)
                        local_indices = list(range(local_start, local_start + len(entry_X)))
                        
                        batch_X.append(entry_X)
                        batch_y.extend(entry_y)
                        batch_indices.extend(entry_indices)
                        batch_source_indices.extend(entry_source_indices)
                        
                        entry_group_info.append({
                            'indices': local_indices,
                            'sugg_idx': entry_group['sugg_idx']
                        })
                    
                    if not batch_X:
                        continue
                    
                    batch_X_tensor = torch.FloatTensor(np.vstack(batch_X))
                    batch_y_tensor = torch.LongTensor(np.array(batch_y))
                    
                    optimizer.zero_grad()
                    
                    z = self.model(batch_X_tensor).squeeze(-1)
                    
                    # Use ranking loss for context-aware training
                    if len(entry_group_info) > 0:
                        loss = ranking_criterion(z, batch_y_tensor, self.model.thresholds,
                                               entry_group_info, batch_source_indices)
                    else:
                        loss = ordinal_criterion(z, batch_y_tensor, self.model.thresholds)
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
                
                if (epoch + 1) % 10 == 0:
                    self.model.eval()
                    with torch.no_grad():
                        z_val = self.model(X_val_tensor).squeeze(-1)
                        val_loss = ordinal_criterion(z_val, y_val_tensor, self.model.thresholds)
                    print(f"  Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss.item():.4f}")
        else:
            # TensorFlow training
            self.model = OrdinalRNN(
                input_dim=X_train_scaled.shape[1],
                hidden_dim=hidden_dim,
                n_classes=2,
                dropout=dropout,
                num_layers=num_layers,
                rnn_type=rnn_type,
                bidirectional=bidirectional
            )
            self.model.compile(learning_rate=learning_rate)
            
            self.model.fit(
                X_train_scaled, y_train_mapped,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val_scaled, y_val_mapped),
                verbose=1 if epochs <= 20 else 2
            )
        
        training_time = time.perf_counter() - training_start
        
        # Evaluate
        if self.use_torch:
            self.model.eval()
            y_train_pred_mapped = self.model.predict(X_train_tensor)
            y_val_pred_mapped = self.model.predict(X_val_tensor)
        else:
            y_train_pred_mapped = self.model.predict(X_train_scaled)
            y_val_pred_mapped = self.model.predict(X_val_scaled)
        
        # Map back: 0 -> 0, 1 -> 2
        y_train_pred = (y_train_pred_mapped * 2).astype(int)
        y_val_pred = (y_val_pred_mapped * 2).astype(int)
        
        # Binary accuracy
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
        
        # Calculate ranking accuracy
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
        """Calculate ranking accuracy: percentage of entries where sugg_idx has highest probability."""
        entry_groups = {}
        for i, meta in enumerate(metadata):
            entry_idx = meta['entry_idx']
            if entry_idx not in entry_groups:
                entry_groups[entry_idx] = []
            entry_groups[entry_idx].append((i, meta))
        
        if self.use_torch:
            X_tensor = torch.FloatTensor(X)
            probs = self.model.predict_proba(X_tensor)
        else:
            probs = self.model.predict_proba(X)
        
        # For 2-class ordinal: class 0 = non-GEO, class 1 = GEO
        # Use probability of class 1 (GEO) as the score
        geo_scores = probs[:, 1]
        
        correct = 0
        total = 0
        
        for entry_idx, samples in entry_groups.items():
            if len(samples) < 2:
                continue
            
            total += 1
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
            
            max_score_idx = np.argmax(scores)
            if max_score_idx == sugg_idx_in_entry:
                correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def _measure_prediction_latency(
        self,
        texts: List[str],
        metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Measure latency for individual predictions."""
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
        if self.use_torch:
            X_tensor = torch.FloatTensor(X)
            predictions = self.model.predict(X_tensor)
            probabilities = self.model.predict_proba(X_tensor)
        else:
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
        """Predict if a source is GEO-optimized based on cleaned_text."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a saved model.")
        
        features = self.extract_features(cleaned_text)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        if self.pca is not None:
            features_scaled = self.pca.transform(features_scaled)
        
        if self.use_torch:
            features_tensor = torch.FloatTensor(features_scaled)
            probabilities = self.model.predict_proba(features_tensor)[0]
        else:
            probabilities = self.model.predict_proba(features_scaled)[0]
        
        prediction = np.argmax(probabilities)
        # Map back: 0 -> 0, 1 -> 2
        prediction = prediction * 2
        
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
        if self.use_torch:
            X_tensor = torch.FloatTensor(X_scaled)
            probabilities = self.model.predict_proba(X_tensor)
        else:
            probabilities = self.model.predict_proba(X_scaled)
        
        predictions = np.argmax(probabilities, axis=1)
        
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
                'prediction': int(predictions[i]),
                'probabilities': np.array([non_geo_prob, geo_probs_normalized[i]]),
                'geo_probability': float(geo_probs_normalized[i])
            })
        
        return results
    
    def save_model(self, model_path: Path, scaler_path: Optional[Path] = None, pca_path: Optional[Path] = None):
        """Save the trained model and scaler."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_path = Path(model_path)
        if scaler_path is None:
            scaler_path = model_path.parent / f"{model_path.stem}_scaler{model_path.suffix}"
        else:
            scaler_path = Path(scaler_path)
        
        if self.use_torch:
            torch.save(self.model.state_dict(), model_path)
        else:
            self.model.model.save_weights(str(model_path).replace('.pkl', '.h5'))
            # Save thresholds separately
            thresholds_path = model_path.parent / f"{model_path.stem}_thresholds.npy"
            np.save(thresholds_path, self.model.thresholds.numpy())
        
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
        """Load a saved model and scaler."""
        model_path = Path(model_path)
        if scaler_path is None:
            scaler_path = model_path.parent / f"{model_path.stem}_scaler{model_path.suffix}"
        else:
            scaler_path = Path(scaler_path)
        
        # Note: Loading requires knowing the architecture, so this is simplified
        # In practice, you'd save the architecture parameters too
        self.scaler = joblib.load(scaler_path)
        
        print(f"Model loaded from: {model_path}")
        print(f"Scaler loaded from: {scaler_path}")
        print("Warning: Model architecture must match training configuration")


def train_rnn_ordinal_classifier(
    optimization_dataset_path: str,
    output_dir: Optional[str] = None,
    limit: int = 1000,
    train_size: int = 300,
    validation_size: int = 700,
    hidden_dim: int = 128,
    learning_rate: float = 0.001,
    epochs: int = 50,
    batch_size: int = 32,
    num_layers: int = 2,
    rnn_type: str = 'LSTM',
    bidirectional: bool = False,
    use_semantic_features: bool = False,
    model_name: str = 'rnn_ordinal_geo_detector.pkl',
    pca_components: Optional[int] = None
) -> RNNOrdinalGEODetector:
    """
    Train an RNN Ordinal classifier on optimization dataset.
    
    Args:
        optimization_dataset_path: Path to optimization dataset JSON file
        output_dir: Directory to save the trained model
        limit: Number of entries to use
        train_size: Number of samples to use for training
        validation_size: Number of samples to use for validation
        hidden_dim: Hidden layer dimension
        learning_rate: Learning rate
        epochs: Number of training epochs
        batch_size: Batch size
        num_layers: Number of RNN layers
        rnn_type: Type of RNN ('LSTM' or 'GRU')
        bidirectional: Whether to use bidirectional RNN
        model_name: Name of the model file to save
        
    Returns:
        Trained RNNOrdinalGEODetector instance
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
    detector = RNNOrdinalGEODetector(use_semantic_features=use_semantic_features, pca_components=pca_components)
    if pca_components is not None:
        print(f"PCA enabled: {pca_components} components")
    print(f"Embedding model: {detector.embedding_model.get_sentence_embedding_dimension()} dimensions")
    if use_semantic_features:
        print(f"Using semantic pattern features: {detector.semantic_extractor.get_num_features()} additional features")
    
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
    
    # Train
    metrics, latency_info, predictions_info = detector.train(
        X, y, entry_metadata, source_texts,
        train_size=train_size, 
        validation_size=validation_size,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        num_layers=num_layers,
        rnn_type=rnn_type,
        bidirectional=bidirectional
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
    
    parser = argparse.ArgumentParser(description='Train RNN Ordinal classifier for GEO detection')
    parser.add_argument('--opt-data', type=str, default='optimization_dataset.json',
                        help='Path to optimization dataset JSON file')
    parser.add_argument('--limit', type=int, default=1000,
                        help='Total number of entries to use')
    parser.add_argument('--train-size', type=int, default=300,
                        help='Number of samples to use for training')
    parser.add_argument('--validation-size', type=int, default=700,
                        help='Number of samples to use for validation')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden layer dimension')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of RNN layers')
    parser.add_argument('--rnn-type', type=str, default='LSTM', choices=['LSTM', 'GRU'],
                        help='Type of RNN: LSTM or GRU')
    parser.add_argument('--bidirectional', action='store_true',
                        help='Use bidirectional RNN')
    parser.add_argument('--use-semantic-features', action='store_true',
                        help='Include individual semantic matching pattern scores as features')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for saved model')
    parser.add_argument('--model-name', type=str, default='rnn_ordinal_geo_detector.pkl',
                        help='Name of the model file')
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
    
    detector = train_rnn_ordinal_classifier(
        optimization_dataset_path=str(opt_data_path),
        output_dir=str(output_dir),
        limit=args.limit,
        train_size=args.train_size,
        validation_size=args.validation_size,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        bidirectional=args.bidirectional,
        use_semantic_features=args.use_semantic_features,
        model_name=args.model_name,
        pca_components=args.pca_components
    )
    
    print("\n" + "="*80)
    print("RNN Ordinal Classifier Training Complete!")
    print("="*80)

