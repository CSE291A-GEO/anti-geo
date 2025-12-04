"""
ListNet Ranking Model with Pairwise Loss for GEO Source Ranking

This module implements a ranking model that focuses on ranking sources within queries,
using a combination of ListNet loss (top-1 probability distribution) and pairwise ranking loss.
The model predicts relevance scores for each source and learns to rank them correctly.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
try:
    from .semantic_features import SemanticFeatureExtractor
except ImportError:
    from semantic_features import SemanticFeatureExtractor

# Import GEO detector for s_geo_max
import sys
import importlib.util
spec_geo = importlib.util.spec_from_file_location(
    "similarity_scores",
    Path(__file__).parent.parent / "pattern_recognition" / "similarity_scores.py"
)
similarity_scores_module = importlib.util.module_from_spec(spec_geo)
sys.modules["similarity_scores"] = similarity_scores_module
spec_geo.loader.exec_module(similarity_scores_module)
SemanticGEODetector = similarity_scores_module.SemanticGEODetector


class RankingDataset(Dataset):
    """Dataset for ranking model - groups sources by query."""
    
    def __init__(self, entries: List[Dict[str, Any]], feature_extractor, scaler=None):
        self.entries = entries
        self.feature_extractor = feature_extractor
        self.scaler = scaler
        
        # Prepare data: group by entry
        self.query_data = []
        for entry in entries:
            query_features = []
            query_ranks = []  # Actual ranks (ge_rank values)
            query_source_indices = []
            
            query_text = entry.get('query', '')
            for source in entry['sources']:
                cleaned_text = source.get('cleaned_text', '')
                category = source.get('category', None)
                s_geo_max = source.get('s_geo_max', None)
                
                # Extract features
                features = feature_extractor.extract_features(
                    cleaned_text, 
                    category=category,
                    query=query_text,
                    s_geo_max=s_geo_max
                )
                
                # Get actual rank (ge_rank, fallback to se_rank)
                ge_rank = source.get('ge_rank', -1)
                se_rank = source.get('se_rank', -1)
                
                # Use ge_rank if available, otherwise use se_rank
                rank = ge_rank if ge_rank >= 0 else se_rank
                
                if rank < 0:
                    continue  # Skip sources without any rank
                
                query_features.append(features)
                query_ranks.append(rank)  # Use rank (ge_rank or se_rank)
                query_source_indices.append(source.get('source_idx', len(query_features) - 1))
            
            if len(query_features) > 0:
                # Convert to numpy arrays
                query_features = np.array(query_features)
                
                # Scale features if scaler is provided
                if self.scaler is not None:
                    query_features = self.scaler.transform(query_features)
                
                self.query_data.append({
                    'features': torch.FloatTensor(query_features),
                    'ranks': torch.LongTensor(query_ranks),  # Actual ranks
                    'source_indices': query_source_indices,
                    'entry_idx': entry.get('entry_idx', len(self.query_data)),
                    'query': query_text  # Store query text for potential use
                })
    
    def __len__(self):
        return len(self.query_data)
    
    def __getitem__(self, idx):
        return self.query_data[idx]


class ListNetRankingModel(nn.Module):
    """
    Neural network for ranking sources within queries.
    Uses ListNet loss (top-1 probability distribution) and pairwise ranking loss.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64], dropout: float = 0.1):
        super(ListNetRankingModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output: single relevance score
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch_size, num_sources, feature_dim) or (num_sources, feature_dim)
        
        Returns:
            scores: (batch_size, num_sources) or (num_sources,)
        """
        if len(features.shape) == 2:
            # Single query: (num_sources, feature_dim)
            return self.network(features).squeeze(-1)
        else:
            # Batch of queries: (batch_size, num_sources, feature_dim)
            batch_size, num_sources, feature_dim = features.shape
            features_flat = features.view(-1, feature_dim)
            scores_flat = self.network(features_flat)
            return scores_flat.view(batch_size, num_sources)


class ListNetLoss(nn.Module):
    """
    ListNet loss: top-1 probability distribution.
    Compares the predicted top-1 probability distribution with the true ranking.
    """
    
    def forward(self, scores: torch.Tensor, ranks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: (num_sources,) predicted relevance scores
            ranks: (num_sources,) actual ranks (lower is better)
        
        Returns:
            loss: scalar
        """
        # Convert ranks to relevance (lower rank = higher relevance)
        # Use negative ranks so lower rank (better) gets higher relevance
        relevance = -ranks.float()
        
        # Normalize to get probability distribution
        # True distribution: softmax of negative ranks
        true_probs = torch.softmax(relevance, dim=0)
        
        # Predicted distribution: softmax of scores
        pred_probs = torch.softmax(scores, dim=0)
        
        # Cross-entropy loss
        loss = -torch.sum(true_probs * torch.log(pred_probs + 1e-10))
        
        return loss


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss: encourages correct pairwise ordering.
    For each pair (i, j) where rank_i < rank_j (i is better), 
    we want score_i > score_j.
    """
    
    def forward(self, scores: torch.Tensor, ranks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: (num_sources,) predicted relevance scores
            ranks: (num_sources,) actual ranks (lower is better)
        
        Returns:
            loss: scalar
        """
        num_sources = scores.shape[0]
        if num_sources < 2:
            return torch.tensor(0.0, device=scores.device)
        
        loss = torch.tensor(0.0, device=scores.device)
        num_pairs = 0
        
        # For each pair (i, j)
        for i in range(num_sources):
            for j in range(i + 1, num_sources):
                rank_i = ranks[i].item()
                rank_j = ranks[j].item()
                
                # If i has better rank (lower number) than j
                if rank_i < rank_j:
                    # We want score_i > score_j
                    # Use margin ranking loss: max(0, margin - (score_i - score_j))
                    margin = 1.0
                    diff = scores[i] - scores[j]
                    pair_loss = torch.clamp(margin - diff, min=0.0)
                    loss += pair_loss
                    num_pairs += 1
                elif rank_j < rank_i:
                    # j is better, we want score_j > score_i
                    margin = 1.0
                    diff = scores[j] - scores[i]
                    pair_loss = torch.clamp(margin - diff, min=0.0)
                    loss += pair_loss
                    num_pairs += 1
        
        if num_pairs > 0:
            loss = loss / num_pairs
        
        return loss


class CombinedRankingLoss(nn.Module):
    """
    Combined loss: ListNet + Pairwise ranking loss.
    """
    
    def __init__(self, listnet_weight: float = 0.5):
        super(CombinedRankingLoss, self).__init__()
        self.listnet_weight = listnet_weight
        self.listnet_loss = ListNetLoss()
        self.pairwise_loss = PairwiseRankingLoss()
    
    def forward(self, scores: torch.Tensor, ranks: torch.Tensor) -> torch.Tensor:
        listnet = self.listnet_loss(scores, ranks)
        pairwise = self.pairwise_loss(scores, ranks)
        return self.listnet_weight * listnet + (1 - self.listnet_weight) * pairwise


class ListNetRankingClassifier:
    """
    Ranking classifier using ListNet with pairwise loss.
    Focuses on ranking accuracy per query rather than binary classification.
    """
    
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2',
                 hidden_dims: List[int] = [128, 64],
                 dropout: float = 0.1,
                 listnet_weight: float = 0.5,
                 use_semantic_features: bool = True,
                 use_geo_score: bool = True,
                 use_query_aware: bool = True,
                 baseline_embeddings_path: Optional[str] = None):
        """
        Initialize the ranking classifier.
        
        Args:
            embedding_model_name: Name of the sentence transformer model
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            listnet_weight: Weight for ListNet loss (1.0 = only ListNet, 0.0 = only pairwise)
            use_semantic_features: Whether to use semantic pattern features
            baseline_embeddings_path: Path to baseline embeddings for demeaning
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.listnet_weight = listnet_weight
        self.use_semantic_features = use_semantic_features
        self.use_geo_score = use_geo_score
        self.use_query_aware = use_query_aware
        
        # Initialize GEO detector for s_geo_max
        self.geo_detector = None
        if use_geo_score:
            try:
                self.geo_detector = SemanticGEODetector()
                print("Initialized GEO detector for s_geo_max feature")
            except Exception as e:
                print(f"Warning: Could not initialize GEO detector: {e}")
                self.use_geo_score = False
        
        # Semantic feature extractor
        self.semantic_extractor = None
        if use_semantic_features:
            try:
                self.semantic_extractor = SemanticFeatureExtractor()
            except Exception as e:
                print(f"Warning: Could not initialize semantic feature extractor: {e}")
        
        # Baseline embeddings for demeaning
        self.baseline_embeddings = None
        if baseline_embeddings_path:
            try:
                with open(baseline_embeddings_path, 'r', encoding='utf-8') as f:
                    baseline_data = json.load(f)
                    self.baseline_embeddings = {}
                    for category, data in baseline_data.items():
                        self.baseline_embeddings[category] = np.array(data['embedding_mean'])
                print(f"Loaded baseline embeddings for {len(self.baseline_embeddings)} categories")
            except Exception as e:
                print(f"Warning: Failed to load baseline embeddings: {e}")
        
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def extract_features(self, cleaned_text: str, category: Optional[str] = None, 
                        query: Optional[str] = None, s_geo_max: Optional[float] = None) -> np.ndarray:
        """
        Extract features from cleaned_text.
        
        Args:
            cleaned_text: The cleaned text to embed
            category: Optional category name for demeaning embeddings
            query: Optional query text for query-aware features
            s_geo_max: Optional pre-computed GEO score
        
        Returns:
            numpy array of features
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
            category_mean = self.baseline_embeddings.get(
                category, 
                self.baseline_embeddings.get('Unknown', np.zeros(self.embedding_dim))
            )
            base_features = base_features - category_mean
        
        features = [base_features]
        
        # Add semantic pattern scores if enabled
        if self.use_semantic_features and self.semantic_extractor:
            semantic_scores = self.semantic_extractor.extract_pattern_scores(cleaned_text)
            features.append(semantic_scores)
        
        # Add s_geo_max if enabled
        if self.use_geo_score:
            if s_geo_max is None and self.geo_detector:
                try:
                    geo_score, _ = self.geo_detector.score(cleaned_text, top_k=3, parsed=False)
                    s_geo_max = float(geo_score)
                except Exception as e:
                    s_geo_max = 0.0
            elif s_geo_max is None:
                s_geo_max = 0.0
            features.append(np.array([s_geo_max]))
        
        # Add query-aware features if enabled (always add to maintain consistent dimensions)
        if self.use_query_aware:
            if query:
                try:
                    query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
                    # Compute cosine similarity between query and source
                    query_source_sim = np.dot(base_features, query_embedding) / (
                        np.linalg.norm(base_features) * np.linalg.norm(query_embedding) + 1e-10
                    )
                    features.append(np.array([query_source_sim]))
                except Exception as e:
                    features.append(np.array([0.0]))
            else:
                # Always add query-aware feature, even if query is empty, to maintain consistent dimensions
                features.append(np.array([0.0]))
        
        return np.concatenate(features)
    
    def prepare_training_data(self, entries: List[Dict[str, Any]], 
                            train_size: Optional[int] = None,
                            val_size: Optional[int] = None) -> Tuple[RankingDataset, RankingDataset]:
        """
        Prepare training and validation datasets.
        
        Args:
            entries: List of entries (queries with sources)
            train_size: Number of entries for training
            val_size: Number of entries for validation
        
        Returns:
            (train_dataset, val_dataset)
        """
        # Filter entries to only those with valid ranks
        valid_entries = []
        for entry in entries:
            has_ranks = any(
                source.get('ge_rank') is not None 
                for source in entry.get('sources', [])
            )
            if has_ranks:
                valid_entries.append(entry)
        
        print(f"Total entries: {len(entries)}, Valid entries with ranks: {len(valid_entries)}")
        
        if train_size is None:
            train_size = len(valid_entries)
        if val_size is None:
            val_size = 0
        
        # Split data
        train_entries = valid_entries[:train_size]
        val_entries = valid_entries[train_size:train_size + val_size] if val_size > 0 else []
        
        print(f"Training entries: {len(train_entries)}, Validation entries: {len(val_entries)}")
        
        # Extract features for all entries to fit scaler
        all_features = []
        for entry in train_entries:
            query_text = entry.get('query', '')
            for source in entry['sources']:
                cleaned_text = source.get('cleaned_text', '')
                category = source.get('category', None)
                s_geo_max = source.get('s_geo_max', None)
                features = self.extract_features(
                    cleaned_text, 
                    category=category,
                    query=query_text,
                    s_geo_max=s_geo_max
                )
                all_features.append(features)
        
        if len(all_features) > 0:
            all_features = np.array(all_features)
            self.scaler.fit(all_features)
            print(f"Fitted scaler on {len(all_features)} samples, feature dim: {all_features.shape[1]}")
        
        # Create datasets
        train_dataset = RankingDataset(train_entries, self, self.scaler)
        val_dataset = RankingDataset(val_entries, self, self.scaler) if val_entries else None
        
        return train_dataset, val_dataset
    
    def train(self, train_dataset: RankingDataset, val_dataset: Optional[RankingDataset] = None,
              epochs: int = 50, batch_size: int = 1, learning_rate: float = 0.001,
              patience: int = 10):
        """
        Train the ranking model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            epochs: Number of training epochs
            batch_size: Batch size (typically 1 for ranking per query)
            learning_rate: Learning rate
            patience: Early stopping patience
        """
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty")
        
        # Get feature dimension from first sample
        sample = train_dataset[0]
        feature_dim = sample['features'].shape[1]
        
        # Initialize model
        self.model = ListNetRankingModel(
            input_dim=feature_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        ).to(self.device)
        
        # Loss and optimizer
        criterion = CombinedRankingLoss(listnet_weight=self.listnet_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            num_queries = 0
            
            # Train on each query
            for query_data in train_dataset:
                features = query_data['features'].to(self.device)
                ranks = query_data['ranks'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                scores = self.model(features)
                
                # Compute loss
                loss = criterion(scores, ranks)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                num_queries += 1
            
            avg_train_loss = train_loss / num_queries if num_queries > 0 else 0.0
            
            # Validation
            if val_dataset is not None and len(val_dataset) > 0:
                self.model.eval()
                val_loss = 0.0
                num_val_queries = 0
                
                with torch.no_grad():
                    for query_data in val_dataset:
                        features = query_data['features'].to(self.device)
                        ranks = query_data['ranks'].to(self.device)
                        
                        scores = self.model(features)
                        loss = criterion(scores, ranks)
                        
                        val_loss += loss.item()
                        num_val_queries += 1
                
                avg_val_loss = val_loss / num_val_queries if num_val_queries > 0 else float('inf')
                
                print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}")
    
    def predict_ranking(self, query_data: Dict[str, Any]) -> Tuple[List[int], List[float]]:
        """
        Predict ranking for a single query.
        
        Args:
            query_data: Dictionary with 'features' (tensor) and 'source_indices' (list)
        
        Returns:
            (ranked_source_indices, scores)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        with torch.no_grad():
            features = query_data['features'].to(self.device)
            scores = self.model(features)
            scores_np = scores.cpu().numpy()
        
        # Get source indices
        source_indices = query_data['source_indices']
        
        # Sort by score (descending)
        sorted_indices = np.argsort(-scores_np)
        ranked_source_indices = [source_indices[i] for i in sorted_indices]
        ranked_scores = [float(scores_np[i]) for i in sorted_indices]
        
        return ranked_source_indices, ranked_scores
    
    def evaluate_ranking_accuracy(self, dataset: RankingDataset) -> Dict[str, float]:
        """
        Evaluate ranking accuracy per query and average rank deviation.
        
        Args:
            dataset: Dataset to evaluate
        
        Returns:
            Dictionary with metrics:
            - ranking_accuracy: Percentage of queries where top-1 is correct
            - mean_rank_deviation: Average absolute deviation between predicted and actual rank
            - mean_reciprocal_rank: Mean reciprocal rank
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        
        correct_top1 = 0
        total_queries = 0
        rank_deviations = []
        reciprocal_ranks = []
        
        with torch.no_grad():
            for query_data in dataset:
                features = query_data['features'].to(self.device)
                ranks = query_data['ranks'].cpu().numpy()
                source_indices = query_data['source_indices']
                
                # Predict scores
                scores = self.model(features)
                scores_np = scores.cpu().numpy()
                
                # Get actual best source (lowest rank)
                actual_best_idx = np.argmin(ranks)
                actual_best_source_idx = source_indices[actual_best_idx]
                
                # Get predicted best source (highest score)
                predicted_best_idx = np.argmax(scores_np)
                predicted_best_source_idx = source_indices[predicted_best_idx]
                
                # Check if top-1 is correct
                if actual_best_source_idx == predicted_best_source_idx:
                    correct_top1 += 1
                
                # Calculate rank deviations for all sources
                # Sort by predicted score (descending)
                sorted_by_pred = np.argsort(-scores_np)
                
                for pred_pos, actual_idx in enumerate(sorted_by_pred):
                    actual_rank = ranks[actual_idx]
                    predicted_rank = pred_pos + 1  # 1-indexed
                    deviation = abs(actual_rank - predicted_rank)
                    rank_deviations.append(deviation)
                
                # Calculate reciprocal rank (position of actual best in predicted ranking)
                actual_best_predicted_pos = None
                for pred_pos, actual_idx in enumerate(sorted_by_pred):
                    if source_indices[actual_idx] == actual_best_source_idx:
                        actual_best_predicted_pos = pred_pos + 1
                        break
                
                if actual_best_predicted_pos is not None:
                    reciprocal_ranks.append(1.0 / actual_best_predicted_pos)
                else:
                    reciprocal_ranks.append(0.0)
                
                total_queries += 1
        
        ranking_accuracy = (correct_top1 / total_queries * 100) if total_queries > 0 else 0.0
        mean_rank_deviation = np.mean(rank_deviations) if rank_deviations else 0.0
        mean_reciprocal_rank = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
        
        return {
            'ranking_accuracy': ranking_accuracy,
            'mean_rank_deviation': mean_rank_deviation,
            'mean_reciprocal_rank': mean_reciprocal_rank,
            'total_queries': total_queries
        }


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ListNet Ranking Model')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to JSON file with entries')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of entries')
    parser.add_argument('--train-size', type=int, default=40,
                       help='Number of entries for training')
    parser.add_argument('--validation-size', type=int, default=25,
                       help='Number of entries for validation')
    parser.add_argument('--output', type=str, default='src/classification/output',
                       help='Output directory')
    parser.add_argument('--model-name', type=str, default='listnet_ranking.pkl',
                       help='Model filename')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[128, 64],
                       help='Hidden layer dimensions')
    parser.add_argument('--listnet-weight', type=float, default=0.5,
                       help='Weight for ListNet loss (0.0-1.0)')
    parser.add_argument('--baseline-embeddings', type=str, default=None,
                       help='Path to baseline embeddings for demeaning')
    parser.add_argument('--use-geo-score', action='store_true', default=True,
                       help='Use s_geo_max as a feature')
    parser.add_argument('--no-geo-score', dest='use_geo_score', action='store_false',
                       help='Disable s_geo_max feature')
    parser.add_argument('--use-query-aware', action='store_true', default=True,
                       help='Use query-aware features (query-source similarity)')
    parser.add_argument('--no-query-aware', dest='use_query_aware', action='store_false',
                       help='Disable query-aware features')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    with open(args.data, 'r', encoding='utf-8') as f:
        entries = json.load(f)
    
    if args.limit:
        entries = entries[:args.limit]
        print(f"Limited to first {args.limit} entries")
    
    # Initialize classifier
    classifier = ListNetRankingClassifier(
        hidden_dims=args.hidden_dims,
        listnet_weight=args.listnet_weight,
        use_semantic_features=True,  # Enable by default for optimization
        use_geo_score=args.use_geo_score,
        use_query_aware=args.use_query_aware,
        baseline_embeddings_path=args.baseline_embeddings
    )
    
    # Prepare data
    train_dataset, val_dataset = classifier.prepare_training_data(
        entries,
        train_size=args.train_size,
        val_size=args.validation_size
    )
    
    # Train
    print("\nTraining ranking model...")
    start_time = time.time()
    classifier.train(
        train_dataset,
        val_dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    training_time = time.time() - start_time
    
    # Evaluate
    print("\nEvaluating model...")
    train_metrics = classifier.evaluate_ranking_accuracy(train_dataset)
    val_metrics = classifier.evaluate_ranking_accuracy(val_dataset) if val_dataset else {}
    
    # Print results
    print("\n" + "="*80)
    print("Training Results:")
    print("="*80)
    print(f"  Training Time: {training_time:.2f}s")
    print(f"  Train Ranking Accuracy: {train_metrics['ranking_accuracy']:.2f}%")
    print(f"  Train Mean Rank Deviation: {train_metrics['mean_rank_deviation']:.4f}")
    print(f"  Train Mean Reciprocal Rank: {train_metrics['mean_reciprocal_rank']:.4f}")
    
    if val_metrics:
        print(f"  Validation Ranking Accuracy: {val_metrics['ranking_accuracy']:.2f}%")
        print(f"  Validation Mean Rank Deviation: {val_metrics['mean_rank_deviation']:.4f}")
        print(f"  Validation Mean Reciprocal Rank: {val_metrics['mean_reciprocal_rank']:.4f}")
    
    # Save model
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / args.model_name
    scaler_path = output_dir / f"{args.model_name.replace('.pkl', '_scaler.pkl')}"
    metrics_path = output_dir / f"{args.model_name.replace('.pkl', '_metrics.json')}"
    
    # Save model state
    torch.save(classifier.model.state_dict(), model_path)
    joblib.dump(classifier.scaler, scaler_path)
    
    # Save metrics
    metrics = {
        'train_ranking_accuracy': train_metrics['ranking_accuracy'],
        'train_mean_rank_deviation': train_metrics['mean_rank_deviation'],
        'train_mean_reciprocal_rank': train_metrics['mean_reciprocal_rank'],
        'validation_ranking_accuracy': val_metrics.get('ranking_accuracy', 0.0),
        'validation_mean_rank_deviation': val_metrics.get('mean_rank_deviation', 0.0),
        'validation_mean_reciprocal_rank': val_metrics.get('mean_reciprocal_rank', 0.0),
        'training_time': training_time,
        'total_train_queries': train_metrics['total_queries'],
        'total_val_queries': val_metrics.get('total_queries', 0)
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Metrics saved to: {metrics_path}")
    print("\n" + "="*80)
    print("ListNet Ranking Classifier Training Complete!")
    print("="*80)


if __name__ == "__main__":
    main()

