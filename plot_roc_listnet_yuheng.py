import numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from src.classification.semantic_features import SemanticFeatureExtractor

SEED=42
np.random.seed(SEED)
YUHENG_CSV='yuheng_data.csv'
MODEL_PATH='src/classification/output/listnet_yuheng_312_134.pkl'
SCALER_PATH='src/classification/output/listnet_yuheng_312_134_scaler.pkl'
EMB_MODEL='all-MiniLM-L6-v2'
EMB_DIM=384
PATTERN_DIM=5
INPUT_DIM=EMB_DIM+PATTERN_DIM

embedder = SentenceTransformer(EMB_MODEL)
sem = SemanticFeatureExtractor(model_name=EMB_MODEL)

def featurize(texts):
    embs = embedder.encode(list(texts), convert_to_numpy=True)
    patt = np.stack([sem.extract_pattern_scores(t) for t in texts], axis=0)
    return np.concatenate([embs.astype(np.float32), patt.astype(np.float32)], axis=1)

# load data
print('Loading yuheng...')
df = pd.read_csv(YUHENG_CSV)
X_text = list(df['original_article']) + list(df['best_edited_article'])
y = np.array([0]*len(df) + [1]*len(df))

X = featurize(X_text)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
scaler = joblib.load(SCALER_PATH)
X_train_s = scaler.transform(X_train)
X_val_s = scaler.transform(X_val)

model = torch.nn.Sequential(
    torch.nn.Linear(INPUT_DIM, 256), torch.nn.ReLU(), torch.nn.Dropout(0.1),
    torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(0.1),
    torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Dropout(0.1),
    torch.nn.Linear(64, 1)
)
sd = torch.load(MODEL_PATH, map_location='cpu')
sd = { (k.replace('network.','') if k.startswith('network.') else k): v for k,v in sd.items() }
model.load_state_dict(sd)
model.eval()
with torch.no_grad():
    scores = model(torch.from_numpy(X_val_s).float()).squeeze(-1).cpu().numpy()

fpr, tpr, _ = roc_curve(y_val, scores)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ListNet (AUC = {roc_auc:.3f})')
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0]); plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC - ListNet (yuheng per-source)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_listnet_yuheng.png', dpi=200)
print(f'AUC: {roc_auc:.4f}, saved roc_listnet_yuheng.png')
