"""Shared prompts, parsers, data loading, and task runners for LLM baselines."""

import os
import re
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize


# ── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Respond with ONLY the requested value. No reasoning, no explanation, no extra text. "
    "Do NOT use <think> tags or any internal reasoning. Output only the final answer."
)

WINDOW_FEATURES_TEMPLATE = """Behavioral features for this window:
- Code edits: {code_events}
- Terminal runs: {terminal_runs}
- Terminal errors: {terminal_errors}
- Event density (events/sec): {event_density:.2f}
- Longest idle period: {longest_idle_s:.1f}s
- Thinking time: {thinking_time_s:.1f}s
- Net code growth (chars): {net_code_growth}
- Delete ratio: {delete_ratio:.2f}
- Time since last AI query: {time_since_last_query_s:.1f}s
- Time since session start: {time_since_session_start_s:.1f}s
- Cumulative code rate: {cum_code_rate:.3f}
- Cumulative query rate: {cum_query_rate:.4f}
- Current behavioral state: {current_state}
- Previous behavioral state: {prev_state}
- Segments in window: {segments_in_window}
- % time thinking: {pct_thinking:.1f}%
- % time implementing: {pct_implementing:.1f}%
- % time debugging: {pct_debugging:.1f}%
- % time seeking help: {pct_seekingHelp:.1f}%
- % time testing: {pct_testing:.1f}%"""


def format_window_features(row):
    return WINDOW_FEATURES_TEMPLATE.format(
        code_events=int(row.get('code_events', 0)),
        terminal_runs=int(row.get('terminal_runs', 0)),
        terminal_errors=int(row.get('terminal_errors', 0)),
        event_density=row.get('event_density', 0),
        longest_idle_s=row.get('longest_idle_s', 0),
        thinking_time_s=row.get('thinking_time_s', 0),
        net_code_growth=int(row.get('net_code_growth', 0)),
        delete_ratio=row.get('delete_ratio', 0),
        time_since_last_query_s=row.get('time_since_last_query_s', 0),
        time_since_session_start_s=row.get('time_since_session_start_s', 0),
        cum_code_rate=row.get('cum_code_rate', 0),
        cum_query_rate=row.get('cum_query_rate', 0),
        current_state=row.get('current_state', 'unknown'),
        prev_state=row.get('prev_state', 'unknown'),
        segments_in_window=int(row.get('segments_in_window', 0)),
        pct_thinking=row.get('pct_thinking', 0),
        pct_implementing=row.get('pct_implementing', 0),
        pct_debugging=row.get('pct_debugging', 0),
        pct_seekingHelp=row.get('pct_seekingHelp', 0),
        pct_testing=row.get('pct_testing', 0),
    )


def build_next_state_prompt(row):
    features = format_window_features(row)
    return f"""You are analyzing a programming student's IDE activity during a 30-second window. Based on the behavioral features below, predict what the student will do next.

The possible behavioral states are:
- thinking: pausing to read code, errors, or task description
- implementing: writing new code
- debugging: fixing errors in existing code
- seekingHelp: typing a query to an AI assistant
- testing: running code and reviewing output

{features}

Respond with ONLY one of these five words: thinking, implementing, debugging, seekingHelp, testing. No explanation."""


def build_error_imminence_prompt(row):
    features = format_window_features(row)
    return f"""You are analyzing a programming student's IDE activity during a 30-second window. Based on the behavioral features below, predict the probability that this student will encounter a terminal error within the next 15 seconds.

{features}

Respond with ONLY a number between 0.0 and 1.0 representing the probability of an error within 15 seconds. No explanation."""


def build_query_imminence_prompt(row):
    features = format_window_features(row)
    return f"""You are analyzing a programming student's IDE activity during a 30-second window. Based on the behavioral features below, predict the probability that this student will submit a query to an AI assistant within the next 15 seconds.

{features}

Respond with ONLY a number between 0.0 and 1.0 representing the probability the student will query within 15 seconds. No explanation."""


def build_query_engagement_prompt(row):
    features = format_window_features(row)
    return f"""You are analyzing a programming student's IDE activity during a 30-second window. The student is about to submit a query to an AI assistant within the next 15 seconds. Based on the behavioral features below, predict the probability that this query will be DEPENDENT rather than GUIDED.

DEPENDENT means the student is offloading cognitive work to the AI — pasting code with no question, vague requests like "help" or "idk", delegating with "ok do that", or asking the AI to just write the code. The student has NOT done cognitive work to identify what they need.

GUIDED means the student demonstrates independent thinking — asking a specific question, identifying a problem or confusion, describing what they tried and what went wrong. The student has done cognitive work to formulate what they need.

{features}

Respond with ONLY a number between 0.0 and 1.0 representing the probability the query will be DEPENDENT. No explanation."""


# ── Parsers ──────────────────────────────────────────────────────────────────

def strip_think_tags(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def parse_probability(response_text):
    if response_text is None:
        return None
    text = strip_think_tags(response_text)
    if not text:
        return None
    try:
        return max(0.0, min(1.0, float(text)))
    except ValueError:
        pass
    matches = re.findall(r'(?<!\d)([01]\.?\d*)', text)
    if matches:
        try:
            return max(0.0, min(1.0, float(matches[0])))
        except ValueError:
            pass
    return None


def parse_state(response_text):
    if response_text is None:
        return None
    text = strip_think_tags(response_text).lower()
    if not text:
        return None
    state_names = ['thinking', 'implementing', 'debugging', 'seekinghelp', 'testing']
    if text in state_names:
        return text
    for name in state_names:
        if name in text:
            return name
    return None


# ── Data loading ─────────────────────────────────────────────────────────────

def load_manifest(root_dir):
    with open(os.path.join(root_dir, 'manifest.yaml')) as f:
        manifest = yaml.safe_load(f)
    return [name for name, config in manifest['deployments'].items()
            if config.get('enabled', True) and config.get('split') == 'test']


def load_test_data(data_dir, test_deployments, data_type):
    subdir = 'window_level' if data_type == 'windows' else 'query_level'
    suffix = '_windows.csv' if data_type == 'windows' else '_queries.csv'
    dfs = []
    for dep in test_deployments:
        path = os.path.join(data_dir, 'observable_metrics', subdir, f'{dep}{suffix}')
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['deployment'] = dep
            dfs.append(df)
            print(f"  Loaded {len(df)} {data_type} from {dep}")
    if not dfs:
        raise FileNotFoundError(f"No test {data_type} files found")
    return pd.concat(dfs, ignore_index=True)


def stratified_subsample(df, label_col, n=1000, seed=42):
    df_clean = df.dropna(subset=[label_col])
    if len(df_clean) <= n:
        return df_clean
    groups = df_clean.groupby(label_col)
    samples = []
    for label, group in groups:
        k = max(1, min(int(n * len(group) / len(df_clean)), len(group)))
        samples.append(group.sample(n=k, random_state=seed))
    result = pd.concat(samples).sample(frac=1, random_state=seed)
    print(f"  Subsampled {len(result)} from {len(df_clean)} (target: {n})")
    return result


# ── Task runners ─────────────────────────────────────────────────────────────

def run_binary_task(call_fn, df, sample_size, task_name, label_col, prompt_builder):
    print(f"\n{'=' * 60}")
    print(f"  TASK: {task_name}")
    print(f"{'=' * 60}")

    if label_col not in df.columns:
        print(f"  ERROR: {label_col} not found")
        return None

    sample = stratified_subsample(df, label_col, n=sample_size)
    pos_rate = sample[label_col].mean()
    print(f"  Positive rate: {pos_rate:.1%}")

    predictions, labels, errors = [], [], 0
    for i, (_, row) in enumerate(sample.iterrows()):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i + 1}/{len(sample)}...")

        response = call_fn(prompt_builder(row))
        prob = parse_probability(response)
        if prob is None:
            errors += 1
            continue
        predictions.append(prob)
        labels.append(int(row[label_col]))

    if len(predictions) < 10:
        print(f"  ERROR: Only {len(predictions)} valid predictions")
        return None

    auc = roc_auc_score(labels, predictions)
    f1 = f1_score(labels, [1 if p > 0.5 else 0 for p in predictions], average='macro')
    print(f"  AUC: {auc:.3f} | F1: {f1:.3f} | N: {len(predictions)} | Errors: {errors}")

    return {'task': task_name, 'auc': round(auc, 3), 'f1': round(f1, 3),
            'n_samples': len(predictions), 'n_errors': errors}


def run_multiclass_task(call_fn, df, sample_size, task_name, label_col,
                        prompt_builder, parse_fn, class_names):
    print(f"\n{'=' * 60}")
    print(f"  TASK: {task_name}")
    print(f"{'=' * 60}")

    if label_col not in df.columns:
        print(f"  ERROR: {label_col} not found")
        return None

    sample = stratified_subsample(df, label_col, n=sample_size)
    predictions, labels, errors = [], [], 0

    for i, (_, row) in enumerate(sample.iterrows()):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i + 1}/{len(sample)}...")

        response = call_fn(prompt_builder(row))
        parsed = parse_fn(response)
        if parsed is None:
            errors += 1
            continue
        predictions.append(parsed)
        labels.append(row[label_col])

    if len(predictions) < 10:
        print(f"  ERROR: Only {len(predictions)} valid predictions")
        return None

    unique_labels = sorted(set(labels) | set(predictions))
    labels_bin = label_binarize(labels, classes=unique_labels)
    preds_bin = label_binarize(predictions, classes=unique_labels)

    try:
        auc = roc_auc_score(labels_bin, preds_bin, average='macro', multi_class='ovr')
    except ValueError:
        auc = 0.5
    f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    print(f"  AUC: {auc:.3f} | F1: {f1:.3f} | N: {len(predictions)} | Errors: {errors}")

    return {'task': task_name, 'auc': round(auc, 3), 'f1': round(f1, 3),
            'n_samples': len(predictions), 'n_errors': errors}


def run_all_tasks(call_fn, args):
    """Run all benchmark tasks with the given API caller."""
    tasks_to_run = [t.strip() for t in args.tasks.split(',')]
    test_deployments = load_manifest(args.root_dir)

    print(f"  Test: {test_deployments} | Sample: {args.sample_size}")

    df_windows = load_test_data(os.path.join(args.root_dir, 'dataset'), test_deployments, 'windows')
    results = []

    if 'next_state' in tasks_to_run:
        r = run_multiclass_task(call_fn, df_windows, args.sample_size,
            'next_behavioral_state', 'label_next_state',
            build_next_state_prompt, parse_state,
            ['thinking', 'implementing', 'debugging', 'seekinghelp', 'testing'])
        if r: results.append(r)

    if 'error_imminence' in tasks_to_run:
        r = run_binary_task(call_fn, df_windows, args.sample_size,
            'error_imminence_15s', 'label_error_imminence_15s',
            build_error_imminence_prompt)
        if r: results.append(r)

    if 'query_imminence' in tasks_to_run:
        r = run_binary_task(call_fn, df_windows, args.sample_size,
            'query_imminence_15s', 'label_query_imminence_15s',
            build_query_imminence_prompt)
        if r: results.append(r)

    if 'query_type' in tasks_to_run:
        df_qt = df_windows[df_windows['label_next_query_type'].isin(['guided', 'dependent'])].copy()
        df_qt['label_dependent'] = (df_qt['label_next_query_type'] == 'dependent').astype(int)
        if len(df_qt) > 0:
            r = run_binary_task(call_fn, df_qt, args.sample_size,
                'query_engagement', 'label_dependent',
                build_query_engagement_prompt)
            if r: results.append(r)

    return results, test_deployments


def save_and_print(results, test_deployments, provider, model, output_path):
    """Save results to JSON and print summary."""
    output = {'provider': provider, 'model': model,
              'test_deployments': test_deployments, 'results': results}
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY — {provider}: {model}")
    print(f"{'=' * 60}")
    print(f"  {'Task':<30s} {'AUC':>8s} {'F1':>8s}")
    print(f"  {'-' * 48}")
    for r in results:
        print(f"  {r['task']:<30s} {r['auc']:>8.3f} {r['f1']:>8.3f}")


import json  # needed for save_and_print