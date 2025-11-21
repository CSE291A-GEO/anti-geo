#!/usr/bin/env python3
"""
Script to add PCA support to remaining classifiers (GBM, Neural, RNN)
"""
import re
from pathlib import Path

def update_save_load_methods(filepath, class_name):
    """Update save_model and load_model methods to handle PCA"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Update save_model signature and implementation
    save_pattern = r'def save_model\(self, model_path: Path, scaler_path: Optional\[Path\] = None\):'
    save_replacement = r'def save_model(self, model_path: Path, scaler_path: Optional[Path] = None, pca_path: Optional[Path] = None):'
    content = re.sub(save_pattern, save_replacement, content)
    
    # Update save_model body
    if 'if self.pca is not None:' not in content or 'joblib.dump(self.pca' not in content:
        save_body_pattern = r'(joblib\.dump\(self\.scaler, scaler_path\)\s+print\(f"Model saved to: \{model_path\}"\)\s+print\(f"Scaler saved to: \{scaler_path\}"\))'
        save_body_replacement = r'''joblib.dump(self.scaler, scaler_path)
        
        if self.pca is not None:
            if pca_path is None:
                pca_path = model_path.parent / f"{model_path.stem}_pca{model_path.suffix}"
            else:
                pca_path = Path(pca_path)
            joblib.dump(self.pca, pca_path)
            print(f"PCA saved to: {pca_path}")
        
        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")'''
        content = re.sub(save_body_pattern, save_body_replacement, content, flags=re.DOTALL)
    
    # Update load_model signature and implementation
    load_pattern = r'def load_model\(self, model_path: Path, scaler_path: Optional\[Path\] = None\):'
    load_replacement = r'def load_model(self, model_path: Path, scaler_path: Optional[Path] = None, pca_path: Optional[Path] = None):'
    content = re.sub(load_pattern, load_replacement, content)
    
    # Update load_model body
    if 'if pca_path.exists():' not in content:
        load_body_pattern = r'(self\.model = joblib\.load\(model_path\)\s+self\.scaler = joblib\.load\(scaler_path\)\s+print\(f"Model loaded from: \{model_path\}"\)\s+print\(f"Scaler loaded from: \{scaler_path\}"\))'
        load_body_replacement = r'''self.model = joblib.load(model_path)
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
        print(f"Scaler loaded from: {scaler_path}")'''
        content = re.sub(load_body_pattern, load_body_replacement, content, flags=re.DOTALL)
    
    with open(filepath, 'w') as f:
        f.write(content)
    print(f'Updated save/load methods in {Path(filepath).name}')

def update_train_function(filepath, class_name):
    """Update train function to accept pca_components"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Add pca_components parameter to train function
    train_pattern = r'(def train_\w+_classifier\([^)]+model_name: str = [^)]+\))'
    if 'pca_components' not in content or f'def train_{class_name.lower()}_classifier' in content:
        # Find the train function signature
        train_func_pattern = r'(def train_\w+_classifier\([^)]+model_name: str = [^,)]+[^)]*\))'
        match = re.search(train_func_pattern, content)
        if match:
            old_sig = match.group(1)
            if 'pca_components' not in old_sig:
                new_sig = old_sig.rstrip(')') + ',\n    pca_components: Optional[int] = None\n)'
                content = content.replace(old_sig, new_sig)
    
    # Update detector initialization
    detector_init_pattern = rf'({class_name}GEODetector\([^)]*use_semantic_features=use_semantic_features[^)]*\))'
    if 'pca_components=pca_components' not in content:
        content = re.sub(
            detector_init_pattern,
            lambda m: m.group(1).rstrip(')') + ', pca_components=pca_components)',
            content
        )
    
    # Add PCA print statement
    if 'PCA enabled:' not in content:
        init_pattern = r'(detector = \w+GEODetector\([^)]+\))'
        content = re.sub(
            init_pattern,
            lambda m: m.group(1) + '\n    if pca_components is not None:\n        print(f"PCA enabled: {pca_components} components")',
            content,
            count=1
        )
    
    # Add pca_components to CLI args
    if '--pca-components' not in content:
        parser_pattern = r'(parser\.add_argument\([^)]+--model-name[^)]+\))'
        content = re.sub(
            parser_pattern,
            lambda m: m.group(1) + '\n    parser.add_argument(\'--pca-components\', type=int, default=None,\n                        help=\'Number of PCA components to keep (None = no PCA)\')',
            content
        )
    
    # Add pca_components to function call
    train_call_pattern = r'(train_\w+_classifier\([^)]+model_name=args\.model_name\s*\))'
    if 'pca_components=args.pca_components' not in content:
        content = re.sub(
            train_call_pattern,
            lambda m: m.group(1).rstrip(')') + ',\n        pca_components=args.pca_components\n    )',
            content
        )
    
    with open(filepath, 'w') as f:
        f.write(content)
    print(f'Updated train function in {Path(filepath).name}')

# Update GBM
gbm_file = 'src/classification/gbm_ordinal_classifier.py'
update_save_load_methods(gbm_file, 'GBMOrdinal')
update_train_function(gbm_file, 'GBMOrdinal')

# Update Neural
neural_file = 'src/classification/neural_ordinal_classifier.py'
update_save_load_methods(neural_file, 'NeuralOrdinal')
update_train_function(neural_file, 'NeuralOrdinal')

# Update RNN
rnn_file = 'src/classification/rnn_ordinal_classifier.py'
update_save_load_methods(rnn_file, 'RNNOrdinal')
update_train_function(rnn_file, 'RNNOrdinal')

print('Done!')

