import os
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATASET_DIR = os.path.join(ROOT_DIR, 'dataset')
MANIFEST_PATH = os.path.join(ROOT_DIR, 'manifest.yaml')

STATE_NAMES = ['thinking', 'implementing', 'debugging', 'seekingHelp', 'testing']

# Window-level feature groups for ablation
LAYER_1_FEATURES = [
    'code_events', 'terminal_runs', 'terminal_errors', 'test_results',
    'query_count', 'event_density', 'longest_idle_s', 'thinking_time_s',
]

LAYER_2_FEATURES = LAYER_1_FEATURES + [
    'cum_code_rate', 'cum_query_rate', 'query_count_so_far',
    'time_since_session_start_s', 'net_code_growth', 'delete_ratio',
    'time_since_last_query_s', 'error_self_fix',
    'prior_no_effort_rate',
]

LAYER_3_FEATURES = LAYER_2_FEATURES + [
    'segments_in_window', 'pct_thinking', 'pct_implementing',
    'pct_debugging', 'pct_seekingHelp', 'pct_testing',
]

# Query-level features (pre-query behavioral features only, no query content)
Q_PRE_FEATURES = [
    'pre_code_edits', 'pre_terminal_runs', 'pre_terminal_errors',
    'pre_chars_inserted', 'pre_chars_deleted', 'pre_net_code_growth',
    'thinking_time_s', 'pre_duration_s', 'is_first_query',
    'time_since_session_start_s', 'query_index', 'total_queries',
    'pre_time_in_editor_s', 'pre_time_in_terminal_s', 'pre_time_in_chat_s',
    'pre_code_edit_rate', 'pre_code_deletes', 'pre_delete_type_ratio',
    'pre_max_consecutive_errors', 'pre_mean_time_between_runs_s',
    'pre_error_self_fix', 'pre_error_ai_fix',
    'pre_error_reading_time_s', 'pre_error_to_edit_s',
    'pre_failed_test_self_fix', 'pre_failed_test_ai_fix',
    'pre_failed_test_to_edit_s',
    'pre_longest_idle_s', 'pre_time_in_task_s', 'pre_time_in_tests_s',
    'pre_response_reading_time_s', 'pre_chat_to_code_latency_s',
    'pre_tab_switches', 'pre_tab_hidden_time_s',
    'thinking_task_s', 'thinking_llm_s', 'thinking_error_s', 'thinking_code_s',
    'time_since_last_query_s',
    'implementing_time_s', 'debugging_time_s', 'testing_time_s',
    'seeking_help_time_s',
]


def safe_multiclass_auc(y_true, y_prob):
    """Compute multiclass AUC, handling mismatched class counts between train and test."""
    try:
        test_classes = sorted(np.unique(y_true))
        n_prob_cols = y_prob.shape[1]

        if len(test_classes) < 2:
            return 0.5

        if len(test_classes) < n_prob_cols:
            y_prob_filtered = y_prob[:, test_classes]
            row_sums = y_prob_filtered.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            y_prob_filtered = y_prob_filtered / row_sums
            label_map = {c: i for i, c in enumerate(test_classes)}
            if isinstance(y_true, pd.Series):
                y_true_remapped = y_true.map(label_map).values
            else:
                y_true_remapped = np.array([label_map[c] for c in y_true])
            return roc_auc_score(y_true_remapped, y_prob_filtered, multi_class='ovr', average='macro')
        else:
            return roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except Exception:
        return 0.5


def load_manifest():
    with open(MANIFEST_PATH) as f:
        return yaml.safe_load(f)


def load_dataset(deployment_name, dataset_type):
    paths = {
        'segments': os.path.join(DATASET_DIR, 'behavioral_sequences', f'{deployment_name}_segments.csv'),
        'windows':  os.path.join(DATASET_DIR, 'observable_metrics', 'window_level', f'{deployment_name}_windows.csv'),
        'queries':  os.path.join(DATASET_DIR, 'observable_metrics', 'query_level', f'{deployment_name}_queries.csv'),
    }
    path = paths[dataset_type]
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found. Run prepare_data.py first.")
        return None
    df = pd.read_csv(path)
    df['student_id'] = df['student_id'].astype(str)
    return df


def load_query_labels(deployment_names):
    """Load and concatenate query type labels for given deployments."""
    dfs = []
    for name in deployment_names:
        label_path = os.path.join(DATASET_DIR, 'query_labels', f'{name}_labels.csv')
        if os.path.exists(label_path):
            ldf = pd.read_csv(label_path)
            ldf['student_id'] = ldf['student_id'].astype(str)
            dfs.append(ldf)
        else:
            print(f"  WARNING: {label_path} not found")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()