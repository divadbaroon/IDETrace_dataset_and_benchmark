import itertools
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


DEFAULT_POOL_EXCLUDE = {
    # Interface instrumentation — sparse; produce outlier-isolation clusters.
    'tab_switches', 'tab_hidden_time_s',
    'copy_events', 'paste_events', 'undo_events', 'redo_events', 'select_events',
    'code_pastes',
    # Chat-activity metrics — undefined in a pre-first-query window by construction.
    'query_count', 'mean_query_length_chars', 'mean_query_interval_s',
    'response_reading_time_s', 'chat_to_code_latency_s',
    # Segment-derived help-seeking time — not part of the W1 behavioral pool.
    'seeking_help_time_s',
}


def reached_all_pass(events):
    """Outcome variable: did the student ever reach an all-pass TEST_CASE_RESULT."""
    for e in events:
        if e['type'] == 'TEST_CASE_RESULT':
            p = e.get('payload', {})
            if p.get('total_tests', 0) > 0 and p.get('passed_count', 0) == p.get('total_tests', 0):
                return 1
    return 0


def _is_defined(value):
    """True only for finite numeric values; None/NaN/inf are undefined."""
    if value is None:
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _remove_sparse(feat_names, X, threshold=0.80):
    """Drop features equal to zero in more than ``threshold`` of rows."""
    keep = []
    n_rows = X.shape[0]
    for i in range(len(feat_names)):
        zero_frac = float((X[:, i] == 0).sum()) / n_rows
        if zero_frac <= threshold:
            keep.append(i)
    return [feat_names[i] for i in keep], X[:, keep]


def search_w1(w1_windows, outcome_by_sid, pool_exclude=None, ks=(2, 3),
              min_cluster_size=15, sparse_threshold=0.80):
    """Carve Cold Start, then search eligible three-feature combinations."""
    if pool_exclude is None:
        pool_exclude = set(DEFAULT_POOL_EXCLUDE)
    else:
        pool_exclude = set(DEFAULT_POOL_EXCLUDE) | set(pool_exclude)

    cold_start, active = [], []
    for w in w1_windows:
        f = w['features']
        if f['code_edits'] == 0 and f['terminal_runs'] == 0:
            cold_start.append(w)
        else:
            active.append(w)

    if not active:
        return {
            'cold_start_n': len(cold_start), 'active_n': 0,
            'pool': [], 'dropped_undefined': [], 'results': []
        }

    candidate_names = [k for k in active[0]['features'] if k not in pool_exclude]

    # A general K-means search needs one common row population. Event-conditioned
    # metrics that are undefined for any active observation are therefore excluded,
    # rather than treating "not applicable" as an observed zero latency.
    dropped_undefined = [
        f for f in candidate_names
        if any(not _is_defined(w['features'].get(f)) for w in active)
    ]
    feat_names = [f for f in candidate_names if f not in dropped_undefined]

    X = np.array(
        [[float(w['features'][f]) for f in feat_names] for w in active],
        dtype=float,
    )
    y = np.array([outcome_by_sid[w['session_id']] for w in active], dtype=float)

    keep = [i for i in range(X.shape[1]) if X[:, i].std() > 1e-9]
    feat_names = [feat_names[i] for i in keep]
    X = X[:, keep]
    feat_names, X = _remove_sparse(feat_names, X, threshold=sparse_threshold)

    results = []
    for combo in itertools.combinations(range(len(feat_names)), 3):
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
                'score': sil * spread,
                'silhouette': sil,
                'spread': spread,
                'pass_rates': [round(p * 100, 1) for p in pass_rates],
                'K': K,
                'sizes': sizes,
                'features': tuple(feat_names[i] for i in combo),
                'cluster_means': cluster_means,
            })

    results.sort(key=lambda r: r['score'], reverse=True)
    return {
        'cold_start_n': len(cold_start),
        'active_n': len(active),
        'pool': feat_names,
        'dropped_undefined': dropped_undefined,
        'results': results,
    }