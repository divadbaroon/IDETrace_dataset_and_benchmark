import numpy as np
import pandas as pd
import warnings
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report

warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("  NOTE: xgboost not installed. XGBoost baseline will be skipped.")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data import safe_multiclass_auc


def get_baselines():
    baselines = {
        'Majority': DummyClassifier(strategy='most_frequent'),
        'LogReg': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1),
    }
    if HAS_XGBOOST:
        baselines['XGBoost'] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=42, use_label_encoder=False,
            eval_metric='logloss', verbosity=0,
        )
    return baselines


def evaluate(model, X_train, y_train, X_test, y_test, task_type='binary'):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'macro_f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
    }

    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(Xte)
        if task_type == 'binary':
            try:
                results['auc'] = roc_auc_score(y_test, y_prob[:, 1])
            except:
                results['auc'] = 0.5
        else:
            results['auc'] = safe_multiclass_auc(y_test, y_prob)
    else:
        results['auc'] = 0.5

    if task_type == 'multiclass':
        all_labels = sorted(set(y_test.unique()) | set(y_pred))
        results['per_class'] = classification_report(
            y_test, y_pred, labels=all_labels,
            output_dict=True, zero_division=0,
        )

    return results