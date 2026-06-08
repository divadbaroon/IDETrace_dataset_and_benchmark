import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from data import safe_multiclass_auc


def evaluate_ensemble(xgb_probs, seq_probs, y_test, task_type='binary', weight_xgb=0.5):
    avg_probs = weight_xgb * xgb_probs + (1 - weight_xgb) * seq_probs
    preds = avg_probs.argmax(axis=1)

    results = {
        'accuracy': accuracy_score(y_test, preds),
        'macro_f1': f1_score(y_test, preds, average='macro', zero_division=0),
    }
    if task_type == 'binary':
        try:
            results['auc'] = roc_auc_score(y_test, avg_probs[:, 1])
        except:
            results['auc'] = 0.5
    else:
        results['auc'] = safe_multiclass_auc(y_test, avg_probs)

    return results