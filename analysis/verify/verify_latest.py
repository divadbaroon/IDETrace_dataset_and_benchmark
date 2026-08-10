from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on path

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from main import load_all_deployments
from analysis.windows import compute_w1_windows, compute_w2_windows
from analysis.clustering import reached_all_pass
from analysis.session_patterns import (
    build_w3_session_rows,
    cluster_w3_session_patterns,
)

DEPS = ('deployment_1', 'deployment_2')
W1_FEATURES = ('time_in_editor_s', 'time_in_terminal_s', 'time_in_chat_s')
W2_FEATURES = ('time_in_editor_s', 'thinking_time_s', 'error_self_fix')


def fit_selected(rows, features, k, outcome):
    xraw = np.asarray([[r['features'][f] for f in features] for r in rows], dtype=float)
    x = StandardScaler().fit_transform(xraw)
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(x)
    sizes = [int((labels == c).sum()) for c in range(k)]
    rates = [
        float(np.mean([
            outcome[rows[i]['session_id']]
            for i in range(len(rows))
            if labels[i] == c
        ]))
        for c in range(k)
    ]
    return {
        'features': list(features),
        'k': k,
        'sizes': sizes,
        'silhouette': float(silhouette_score(x, labels)),
        'completion_rates': rates,
        'spread': float(max(rates) - min(rates)),
    }


def main():
    all_raw = load_all_deployments()
    raw = {sid: row for sid, row in all_raw.items() if row['deployment'] in DEPS}
    outcome = {sid: reached_all_pass(row['events']) for sid, row in raw.items()}

    w1 = compute_w1_windows(raw)
    w1_cold = [w for w in w1 if w['features']['code_edits'] == 0 and w['features']['terminal_runs'] == 0]
    w1_active = [w for w in w1 if w not in w1_cold]

    w2 = compute_w2_windows(raw)
    w2_passive = [w for w in w2 if w['features']['code_edits'] == 0 and w['features']['terminal_runs'] == 0]
    w2_active = [w for w in w2 if w not in w2_passive]

    w3_rows, w3_population = build_w3_session_rows(raw, w2, min_intervals=2)
    w3_result = cluster_w3_session_patterns(w3_rows)
    best = w3_result['per_k'][w3_result['best_k']]

    profile_summaries = sorted(
        best['cluster_summaries'],
        key=lambda summary: summary['profile'],
    )
    result = {
        'w1': {
            'total': len(w1),
            'cold_start': len(w1_cold),
            'active': len(w1_active),
            'selected': fit_selected(w1_active, W1_FEATURES, 2, outcome),
        },
        'w2': {
            'total': len(w2),
            'passive': len(w2_passive),
            'active': len(w2_active),
            'selected': fit_selected(w2_active, W2_FEATURES, 3, outcome),
        },
        'w3': {
            'population': w3_population,
            'eligible_learners': len(w3_rows),
            'best_k': w3_result['best_k'],
            'features': w3_result['feature_names'],
            'sizes': best['sizes'],
            'silhouette': best['silhouette'],
            'seed_ari_mean': best['seed_ari_mean'],
            'seed_ari_min': best['seed_ari_min'],
            'profiles': profile_summaries,
        },
    }
    out = Path('results/latest_verification.json')
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()