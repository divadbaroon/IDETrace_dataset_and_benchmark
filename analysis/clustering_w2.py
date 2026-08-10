import itertools
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


POST_RESPONSE_FEATURES = [
    'response_reading_latency_s', 'code_after_response',
    'response_to_code_or_next_query_s',
    'time_in_editor_s', 'time_in_terminal_s', 'time_in_chat_s',
    'time_in_tests_s', 'time_in_task_s', 'longest_idle',
]

NEW_EFFORT_FEATURES = [
    'thinking_time_s', 'terminal_runs', 'terminal_errors', 'code_edits',
    'code_deletions', 'code_edit_rate', 'chars_inserted', 'chars_deleted',
    'net_code_growth', 'delete_type_ratio', 'max_consecutive_errors',
    'mean_time_between_runs_s', 'error_reading_time_s', 'error_to_edit_s',
    'error_self_fix', 'seeking_help_time_s',
    'failed_test_to_edit_s', 'failed_test_self_fix',
]


def _is_defined(value):
    """True only for finite numeric values; None/NaN/inf are undefined."""
    if value is None:
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _eligible_features(features, rows, threshold=0.80):
    """Return common, finite, non-sparse features and exclusion diagnostics."""
    clean = []
    dropped_undefined = []
    dropped_sparse = []
    for f in features:
        vals = [r['features'].get(f) for r in rows]
        if any(not _is_defined(v) for v in vals):
            dropped_undefined.append(f)
            continue
        numeric = np.asarray([float(v) for v in vals], dtype=float)
        zero_frac = float((numeric == 0).sum()) / len(numeric)
        if zero_frac > threshold:
            dropped_sparse.append(f)
            continue
        clean.append(f)
    return clean, dropped_undefined, dropped_sparse


def _search(active_rows, feature_list, outcome_by_key, label, ks=(2, 3), min_cluster_size=15):
    X = np.array(
        [[float(r['features'][f]) for f in feature_list] for r in active_rows],
        dtype=float,
    )
    y = np.array([outcome_by_key[r['_key']] for r in active_rows], dtype=float)

    keep = [i for i in range(X.shape[1]) if X[:, i].std() > 1e-9]
    names = [feature_list[i] for i in keep]
    X = X[:, keep]

    results = []
    for combo in itertools.combinations(range(len(names)), 3):
        Xraw = X[:, combo]
        Xs = StandardScaler().fit_transform(Xraw)
        for K in ks:
            if K >= len(Xs):
                continue
            km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(Xs)
            labels = km.labels_
            if len(set(labels)) < K:
                continue
            sizes = [int((labels == c).sum()) for c in range(K)]
            if min(sizes) < min_cluster_size:
                continue
            try:
                sil = silhouette_score(Xs, labels)
            except Exception:
                continue
            pass_rates = [float(y[labels == c].mean()) for c in range(K)]
            spread = max(pass_rates) - min(pass_rates)
            cluster_means = [
                [float(Xraw[labels == c, j].mean()) for j in range(Xraw.shape[1])]
                for c in range(K)
            ]
            results.append({
                'search': label,
                'score': sil * spread,
                'silhouette': sil,
                'spread': spread,
                'pass_rates': [round(p * 100, 1) for p in pass_rates],
                'K': K,
                'sizes': sizes,
                'features': tuple(names[i] for i in combo),
                'cluster_means': cluster_means,
            })
    results.sort(key=lambda r: r['score'], reverse=True)
    return results


def search_w2(w2_windows, outcome_by_key, min_cluster_size=15, sparse_threshold=0.80):
    passive, active = [], []
    for w in w2_windows:
        f = w['features']
        if f['code_edits'] == 0 and f['terminal_runs'] == 0:
            passive.append(w)
        else:
            active.append(w)

    post_feats, post_undefined, post_sparse = _eligible_features(
        POST_RESPONSE_FEATURES, active, threshold=sparse_threshold
    )
    effort_feats, effort_undefined, effort_sparse = _eligible_features(
        NEW_EFFORT_FEATURES, active, threshold=sparse_threshold
    )
    combined_feats = list(dict.fromkeys(post_feats + effort_feats))

    results = _search(
        active, combined_feats, outcome_by_key, 'combined',
        min_cluster_size=min_cluster_size
    )

    return {
        'passive_n': len(passive),
        'active_n': len(active),
        'features': combined_feats,
        'source_pools': {
            'post_response': post_feats,
            'new_effort': effort_feats,
        },
        'dropped_undefined': {
            'post_response': post_undefined,
            'new_effort': effort_undefined,
        },
        'dropped_sparse': {
            'post_response': post_sparse,
            'new_effort': effort_sparse,
        },
        'results': results,
    }