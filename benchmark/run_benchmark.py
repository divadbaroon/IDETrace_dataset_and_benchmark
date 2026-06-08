import os
import json
import numpy as np
import pandas as pd
import warnings
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

from data import (
    ROOT_DIR, STATE_NAMES,
    LAYER_1_FEATURES, LAYER_2_FEATURES, LAYER_3_FEATURES, Q_PRE_FEATURES,
    load_manifest, load_dataset, load_query_labels,
)
from models.ml.classical import get_baselines, evaluate, HAS_XGBOOST
from models.ml.ensemble import evaluate_ensemble

try:
    from models.ml.mlp import train_mlp, HAS_TORCH
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from models.ml.sequential import (
        LSTMModel, GRUModel, TemporalCNNModel, TransformerModel,
        build_segment_sequences, train_seq_model,
    )


# ══════════════════════════════════════════════════════════════
#  ABLATION RUNNER
# ══════════════════════════════════════════════════════════════

def run_ablation(df_train, df_test, task_name, target_col, feature_layers,
                 task_type='binary', seg_train=None, seg_test=None, time_col='window_end_s'):
    train = df_train.dropna(subset=[target_col]).copy()
    test = df_test.dropna(subset=[target_col]).copy()

    if task_type == 'multiclass':
        le = LabelEncoder()
        le.fit(pd.concat([train[target_col], test[target_col]]))
        y_train = pd.Series(le.transform(train[target_col]), index=train.index)
        y_test = pd.Series(le.transform(test[target_col]), index=test.index)
        n_classes = len(le.classes_)
    else:
        y_train = train[target_col].astype(int)
        y_test = test[target_col].astype(int)
        n_classes = 2

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        print(f"  {task_name}: SKIPPED (insufficient class diversity)")
        return None

    if task_type == 'binary':
        print(f"  Train: {len(y_train)} (pos: {y_train.mean()*100:.1f}%) | Test: {len(y_test)} (pos: {y_test.mean()*100:.1f}%)")
    else:
        print(f"  Train: {len(y_train)} | Test: {len(y_test)} | Classes: {n_classes}")

    print(f"\n  {'Model':<25s}", end='')
    for layer_name in feature_layers:
        print(f"  {layer_name:>22s}", end='')
    print()
    print(f"  {'-' * (25 + 24 * len(feature_layers))}")

    baselines = get_baselines()
    all_results = {}
    layer_names = list(feature_layers.keys())
    last_layer = layer_names[-1]

    xgb_probs = None
    for model_name, model in baselines.items():
        print(f"  {model_name:<25s}", end='')
        all_results[model_name] = {}
        for layer_name, cols in feature_layers.items():
            avail = [c for c in cols if c in train.columns]
            X_train = train[avail].fillna(0)
            X_test = test[avail].fillna(0)
            res = evaluate(clone(model), X_train, y_train, X_test, y_test, task_type)
            all_results[model_name][layer_name] = res
            print(f"  {res['auc']:>8.3f} / {res['macro_f1']:>.3f}", end='')

            if model_name == 'XGBoost' and layer_name == last_layer:
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(X_train)
                Xte = scaler.transform(X_test)
                m = clone(model)
                m.fit(Xtr, y_train)
                xgb_probs = m.predict_proba(Xte)
        print()

    if HAS_TORCH:
        print(f"  {'MLP':<25s}", end='')
        all_results['MLP'] = {}
        for layer_name, cols in feature_layers.items():
            avail = [c for c in cols if c in train.columns]
            X_train_layer = train[avail].fillna(0)
            X_test_layer = test[avail].fillna(0)
            res = train_mlp(X_train_layer, y_train, X_test_layer, y_test, n_classes=n_classes)
            all_results['MLP'][layer_name] = res
            print(f"  {res['auc']:>8.3f} / {res['macro_f1']:>.3f}", end='')
        print()

    best_seq_probs = None
    best_seq_auc = 0
    best_seq_name = None

    if HAS_TORCH and seg_train is not None and seg_test is not None:
        mode = 'window' if time_col == 'window_end_s' else 'query'
        seq_tr = build_segment_sequences(train, seg_train, time_col=time_col, mode=mode)
        seq_te = build_segment_sequences(test, seg_test, time_col=time_col, mode=mode)

        seq_models = {
            'Seq-LSTM':        LSTMModel,
            'Seq-GRU':         GRUModel,
            'Seq-CNN':         TemporalCNNModel,
            'Seq-Transformer': TransformerModel,
        }

        for model_name, model_class in seq_models.items():
            print(f"  {model_name:<25s}", end='')
            all_results[model_name] = {}

            for layer_name in layer_names:
                if layer_name == last_layer:
                    res, probs = train_seq_model(
                        model_class, seq_tr, y_train, seq_te, y_test,
                        n_classes=n_classes, return_probs=True
                    )
                    all_results[model_name][layer_name] = res
                    print(f"  {res['auc']:>8.3f} / {res['macro_f1']:>.3f}", end='')

                    if res.get('auc', 0) > best_seq_auc:
                        best_seq_auc = res['auc']
                        best_seq_probs = probs
                        best_seq_name = model_name
                else:
                    all_results[model_name][layer_name] = {'auc': 0, 'macro_f1': 0, 'accuracy': 0}
                    print(f"  {'—':>14s}", end='')
            print()

    if xgb_probs is not None and best_seq_probs is not None:
        ensemble_name = f'XGB + {best_seq_name}'
        print(f"  {ensemble_name:<25s}", end='')
        all_results[ensemble_name] = {}

        for layer_name in layer_names:
            if layer_name == last_layer:
                res = evaluate_ensemble(xgb_probs, best_seq_probs, y_test, task_type=task_type)
                all_results[ensemble_name][layer_name] = res
                print(f"  {res['auc']:>8.3f} / {res['macro_f1']:>.3f}", end='')
            else:
                all_results[ensemble_name][layer_name] = {'auc': 0, 'macro_f1': 0, 'accuracy': 0}
                print(f"  {'—':>14s}", end='')
        print()

    return all_results


# ══════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════

def print_importance(df, target_col, task_name, feature_cols, top_n=15):
    train = df.dropna(subset=[target_col])
    y = train[target_col]
    if isinstance(y.iloc[0], str):
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)
    else:
        y = y.astype(int)

    if y.nunique() < 2:
        return

    avail = [c for c in feature_cols if c in train.columns]
    X = train[avail].fillna(0)

    rf = RandomForestClassifier(300, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=avail).sort_values(ascending=False)

    print(f"\n  {task_name}:")
    print(f"  {'-' * 50}")
    for i, (feat, val) in enumerate(imp.head(top_n).items()):
        print(f"  {i+1:>2d}. {feat:<35s}  {val:.4f}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  IDETRACE BENCHMARK")
    print("=" * 60)

    manifest = load_manifest()
    tasks = manifest.get('tasks', {})
    deployments = manifest.get('deployments', {})

    train_names = [n for n, c in deployments.items() if c.get('split') == 'train' and c.get('enabled', True)]
    test_names = [n for n, c in deployments.items() if c.get('split') == 'test' and c.get('enabled', True)]

    if not train_names or not test_names:
        print("  ERROR: Need at least one train and one test deployment.")
        return

    print(f"\n  Loading data...")
    train_windows = pd.concat([load_dataset(n, 'windows') for n in train_names], ignore_index=True)
    test_windows = pd.concat([load_dataset(n, 'windows') for n in test_names], ignore_index=True)
    train_segments = pd.concat([load_dataset(n, 'segments') for n in train_names], ignore_index=True)
    test_segments = pd.concat([load_dataset(n, 'segments') for n in test_names], ignore_index=True)

    print(f"  Train: {len(train_windows):,} windows, {train_windows['student_id'].nunique()} students")
    print(f"  Test:  {len(test_windows):,} windows, {test_windows['student_id'].nunique()} students")

    window_layers = {
        'Raw telemetry':     [c for c in LAYER_1_FEATURES if c in train_windows.columns],
        '+Observable':       [c for c in LAYER_2_FEATURES if c in train_windows.columns],
        '+Behav. sequences': [c for c in LAYER_3_FEATURES if c in train_windows.columns],
    }

    results = {}

    # Task 1: Next behavioral state
    if tasks.get('next_behavioral_state'):
        print("\n" + "=" * 60)
        print("  TASK 1: NEXT BEHAVIORAL STATE (5-class)")
        print("=" * 60)
        res = run_ablation(train_windows, test_windows, 'Next behavioral state',
                          'label_next_state', window_layers, task_type='multiclass',
                          seg_train=train_segments, seg_test=test_segments)
        if res:
            results['next_behavioral_state'] = res

    # Task 2: Error imminence
    if tasks.get('error_imminence'):
        print("\n" + "=" * 60)
        print("  TASK 2: ERROR IMMINENCE")
        print("=" * 60)
        for horizon in [5, 10, 15, 30, 45, 60]:
            label_col = f'label_error_imminence_{horizon}s'
            if label_col not in train_windows.columns:
                continue
            print(f"\n  --- {horizon}s horizon ---")
            res = run_ablation(train_windows, test_windows, f'Error imminence ({horizon}s)',
                              label_col, window_layers, task_type='binary',
                              seg_train=train_segments, seg_test=test_segments)
            if res:
                results[f'error_imminence_{horizon}s'] = res

    # Task 3: Query imminence
    if tasks.get('query_imminence'):
        print("\n" + "=" * 60)
        print("  TASK 3: QUERY IMMINENCE")
        print("=" * 60)
        for horizon in [5, 10, 15, 30, 45, 60]:
            label_col = f'label_query_imminence_{horizon}s'
            if label_col not in train_windows.columns:
                continue
            print(f"\n  --- {horizon}s horizon ---")
            res = run_ablation(train_windows, test_windows, f'Query imminence ({horizon}s)',
                              label_col, window_layers, task_type='binary',
                              seg_train=train_segments, seg_test=test_segments)
            if res:
                results[f'query_imminence_{horizon}s'] = res

    # Task 4: Query type (guided vs dependent)
    if tasks.get('query_type'):
        print("\n" + "=" * 60)
        print("  TASK 4: QUERY TYPE (guided vs dependent)")
        print("=" * 60)

        engagement_map = {'guided': 0, 'dependent': 1}

        # 4a: Window-level
        if 'label_next_query_type' in train_windows.columns:
            train_qt = train_windows[train_windows['label_next_query_type'].isin(engagement_map.keys())].copy()
            test_qt = test_windows[test_windows['label_next_query_type'].isin(engagement_map.keys())].copy()
            train_qt['label_dependent'] = train_qt['label_next_query_type'].map(engagement_map)
            test_qt['label_dependent'] = test_qt['label_next_query_type'].map(engagement_map)

            if len(train_qt) > 0 and len(test_qt) > 0:
                print(f"\n  --- Window-level ---")
                res = run_ablation(train_qt, test_qt, 'Query type (window)',
                                  'label_dependent', window_layers, task_type='binary',
                                  seg_train=train_segments, seg_test=test_segments)
                if res:
                    results['query_type_window'] = res

        # 4b: Query-level
        train_queries = pd.concat([load_dataset(n, 'queries') for n in train_names], ignore_index=True)
        test_queries = pd.concat([load_dataset(n, 'queries') for n in test_names], ignore_index=True)
        train_labels = load_query_labels(train_names)
        test_labels = load_query_labels(test_names)

        if len(train_labels) > 0 and len(test_labels) > 0:
            train_queries = train_queries.merge(train_labels[['student_id', 'query_index', 'query_type']],
                                                on=['student_id', 'query_index'], how='left')
            test_queries = test_queries.merge(test_labels[['student_id', 'query_index', 'query_type']],
                                              on=['student_id', 'query_index'], how='left')
            train_queries = train_queries[train_queries['query_type'].isin(engagement_map.keys())].copy()
            test_queries = test_queries[test_queries['query_type'].isin(engagement_map.keys())].copy()
            train_queries['label_dependent'] = train_queries['query_type'].map(engagement_map)
            test_queries['label_dependent'] = test_queries['query_type'].map(engagement_map)

            if len(train_queries) > 0 and len(test_queries) > 0:
                print(f"\n  --- Query-level ---")
                q_feats = [c for c in Q_PRE_FEATURES if c in train_queries.columns]
                res = run_ablation(train_queries, test_queries, 'Query type (query-level)',
                                  'label_dependent', {'Pre-query features': q_feats},
                                  task_type='binary', seg_train=train_segments, seg_test=test_segments,
                                  time_col='time_since_session_start_s')
                if res:
                    results['query_type_query_level'] = res

    # Feature importance
    print("\n" + "=" * 60)
    print("  FEATURE IMPORTANCE (Top 15)")
    print("=" * 60)

    if tasks.get('next_behavioral_state'):
        print_importance(train_windows, 'label_next_state', 'Next behavioral state', LAYER_3_FEATURES)
    if tasks.get('error_imminence'):
        print_importance(train_windows, 'label_error_imminence_15s', 'Error imminence (15s)', LAYER_3_FEATURES)
    if tasks.get('query_imminence'):
        print_importance(train_windows, 'label_query_imminence_15s', 'Query imminence (15s)', LAYER_3_FEATURES)

    # Save results
    results_dir = os.path.join(ROOT_DIR, 'benchmark', 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'results.json')
    serialized = {}
    for task_name, task_res in results.items():
        serialized[task_name] = {}
        for model_name, model_res in task_res.items():
            serialized[task_name][model_name] = {}
            for layer_name, metrics in model_res.items():
                serialized[task_name][model_name][layer_name] = {
                    k: round(float(v), 4) if isinstance(v, (float, np.floating)) else v
                    for k, v in metrics.items() if k != 'per_class'
                }

    with open(out_path, 'w') as f:
        json.dump(serialized, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    main()