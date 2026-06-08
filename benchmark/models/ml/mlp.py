import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from data import safe_multiclass_auc


if HAS_TORCH:
    class MLPModel(nn.Module):
        def __init__(self, input_dim, hidden=128, n_classes=2, dropout=0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, n_classes),
            )

        def forward(self, x):
            return self.net(x)

    class FlatDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.FloatTensor(X)
            self.y = torch.LongTensor(y)
        def __len__(self):
            return len(self.y)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    def train_mlp(X_train, y_train, X_test, y_test,
                   n_classes=2, epochs=50, hidden=128, return_probs=False):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train.values if hasattr(X_train, 'values') else X_train)
        Xte = scaler.transform(X_test.values if hasattr(X_test, 'values') else X_test)

        train_dl = DataLoader(FlatDataset(Xtr, y_train.values), batch_size=64, shuffle=True)
        test_dl = DataLoader(FlatDataset(Xte, y_test.values), batch_size=64, shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MLPModel(Xtr.shape[1], hidden=hidden, n_classes=n_classes).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        crit = nn.CrossEntropyLoss()

        model.train()
        for _ in range(epochs):
            for bx, by in train_dl:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                crit(model(bx), by).backward()
                opt.step()

        model.eval()
        probs, preds = [], []
        with torch.no_grad():
            for bx, _ in test_dl:
                out = model(bx.to(device))
                probs.append(torch.softmax(out, dim=1).cpu().numpy())
                preds.append(out.argmax(dim=1).cpu().numpy())

        probs = np.vstack(probs)
        preds = np.concatenate(preds)

        results = {
            'accuracy': accuracy_score(y_test, preds),
            'macro_f1': f1_score(y_test, preds, average='macro', zero_division=0),
        }
        if n_classes == 2:
            try:
                results['auc'] = roc_auc_score(y_test, probs[:, 1])
            except:
                results['auc'] = 0.5
        else:
            results['auc'] = safe_multiclass_auc(y_test, probs)

        if return_probs:
            return results, probs
        return results