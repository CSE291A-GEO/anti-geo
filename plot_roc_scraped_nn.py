import json
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

PRED_PATH = 'src/classification/output/neural_scraped_baseline_predictions.json'
OUT_PNG = 'roc_scraped_nn.png'

with open(PRED_PATH) as f:
    data = json.load(f)

labels = []
scores = []
for entry in data.get('entries', []):
    for src in entry.get('sources', []):
        lbl = src.get('true_label')
        prob = src.get('probabilities', {}).get('class_2')
        # fall back to class_1 if class_2 is missing
        if prob is None:
            prob = src.get('probabilities', {}).get('class_1')
        if prob is None:
            continue
        labels.append(1 if lbl == 2 else 0)
        scores.append(prob)

labels = np.array(labels)
scores = np.array(scores)

fpr, tpr, _ = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'Scraped NN (AUC = {roc_auc:.3f})')
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0]); plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC - Neural Classifier (Scraped Data)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
print(f'AUC: {roc_auc:.4f}, saved {OUT_PNG}, N={len(labels)}, positives={labels.sum()}')
