import numpy as np
import pandas as pd
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
from data import STATE_NAMES, safe_multiclass_auc

STATE_TO_IDX = {s: i for i, s in enumerate(STATE_NAMES)}
SUBTYPE_TO_IDX = {
    '': 0, 'none': 0,
    'thinking-task': 1, 'thinking-llm': 2,
    'thinking-error': 3, 'thinking-code': 4,
}
SEG_SEQ_LEN = 30


if HAS_TORCH:
    class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden=64, layers=2, n_classes=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden, n_classes)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(self.drop(out[:, -1, :]))

    class GRUModel(nn.Module):
        def __init__(self, input_dim, hidden=64, layers=2, n_classes=2, dropout=0.3):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden, layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden, n_classes)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(self.drop(out[:, -1, :]))

    class TemporalCNNModel(nn.Module):
        def __init__(self, input_dim, hidden=64, n_classes=2, dropout=0.3):
            super().__init__()
            self.conv1 = nn.Conv1d(input_dim, hidden, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(hidden, n_classes)
            self.drop = nn.Dropout(dropout)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = x.transpose(1, 2)
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.pool(x).squeeze(-1)
            return self.fc(self.drop(x))

    class TransformerModel(nn.Module):
        def __init__(self, input_dim, hidden=64, n_heads=4, layers=2, n_classes=2, dropout=0.3):
            super().__init__()
            self.proj = nn.Linear(input_dim, hidden)
            self.pos_enc = nn.Parameter(torch.randn(1, SEG_SEQ_LEN, hidden) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4,
                dropout=dropout, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.fc = nn.Linear(hidden, n_classes)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            x = self.proj(x) + self.pos_enc[:, :x.size(1), :]
            x = self.encoder(x)
            x = x.mean(dim=1)
            return self.fc(self.drop(x))

    class SegSeqDataset(Dataset):
        def __init__(self, sequences, labels):
            self.sequences = sequences
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            return torch.FloatTensor(self.sequences[idx]), torch.LongTensor([self.labels[idx]])

    def build_segment_sequences(df, seg_df, time_col='window_end_s', mode='window'):
        feat_dim = 12
        sequences = []

        seg_grouped = {}
        for sid, group in seg_df.groupby('student_id'):
            seg_grouped[str(sid)] = group.sort_values('start_time_ms')

        col = 'window_end_s' if mode == 'window' else 'time_since_session_start_s'

        for _, row in df.iterrows():
            sid = str(row['student_id'])
            cutoff_ms = row[col] * 1000

            student_segs = seg_grouped.get(sid, pd.DataFrame())
            if len(student_segs) > 0:
                student_segs = student_segs[student_segs['start_time_ms'] < cutoff_ms]

            student_segs = student_segs.tail(SEG_SEQ_LEN)

            seq = np.zeros((SEG_SEQ_LEN, feat_dim))
            offset = SEG_SEQ_LEN - len(student_segs)

            for i, (_, seg) in enumerate(student_segs.iterrows()):
                pos = offset + i
                state_idx = STATE_TO_IDX.get(seg['behavioral_state'], 0)
                seq[pos, state_idx] = 1.0

                subtype = seg.get('thinking_subtype', '') or ''
                subtype_idx = SUBTYPE_TO_IDX.get(subtype, 0)
                seq[pos, 5 + subtype_idx] = 1.0

                dur = max(0.01, seg.get('duration_s', 0) or 0)
                seq[pos, 10] = min(dur / 60.0, 5.0)
                seq[pos, 11] = np.log1p(dur)

            sequences.append(seq)

        return sequences

    def train_seq_model(model_class, sequences_train, y_train, sequences_test, y_test,
                        n_classes=2, epochs=50, hidden=64, return_probs=False):
        train_dl = DataLoader(SegSeqDataset(sequences_train, y_train.values), batch_size=64, shuffle=True)
        test_dl = DataLoader(SegSeqDataset(sequences_test, y_test.values), batch_size=64, shuffle=False)

        feat_dim = sequences_train[0].shape[1]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model_class(feat_dim, hidden=hidden, n_classes=n_classes).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        crit = nn.CrossEntropyLoss()

        model.train()
        for _ in range(epochs):
            for bx, by in train_dl:
                bx, by = bx.to(device), by.squeeze().to(device)
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