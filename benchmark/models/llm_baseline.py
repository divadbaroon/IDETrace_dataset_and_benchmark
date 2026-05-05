"""
LLM Baseline Evaluation for TutorTrace Benchmark (Closed-Source).

Evaluates closed-source LLMs via OpenAI API on four tasks:
  1. Next behavioral state (5-class) - subsampled 1000 windows
  2. Error imminence 15s (binary) - subsampled 1000 windows
  3. Query imminence 15s (binary) - subsampled 1000 windows
  4. Query engagement (binary: guided vs dependent) - window-level

Usage:
  cd tutortrace_dataset_and_benchmark
  python3 benchmark/models/llm_baseline.py --model gpt-4o-mini --tasks query_type
  python3 benchmark/models/llm_baseline.py --model gpt-4o --tasks query_type
  python3 benchmark/models/llm_baseline.py --model gpt-5.5 --tasks query_type
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import os
import re
import time
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import label_binarize
from openai import OpenAI


# ── Prompt builders ──────────────────────────────────────────────────────────

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


# ── API caller ───────────────────────────────────────────────────────────────

def call_llm(client, prompt, model="gpt-4o-mini", max_retries=3):
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "Respond with ONLY the requested value. No reasoning, no explanation, no extra text."},
                    {"role": "user", "content": prompt},
                ],
            }
            if model.startswith("gpt-5"):
                kwargs["max_completion_tokens"] = 500
            else:
                kwargs["temperature"] = 0.0
                kwargs["max_tokens"] = 20

            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"    Retry {attempt + 1}/{max_retries} after error: {e}")
                time.sleep(wait)
            else:
                print(f"    Failed after {max_retries} retries: {e}")
                return None


def parse_probability(response_text):
    if response_text is None:
        return None
    try:
        val = float(response_text.strip())
        return max(0.0, min(1.0, val))
    except ValueError:
        matches = re.findall(r'(?<!\d)([01]\.?\d*)', response_text)
        if matches:
            try:
                val = float(matches[0])
                return max(0.0, min(1.0, val))
            except ValueError:
                pass
    return None


def parse_state(response_text):
    if response_text is None:
        return None
    text = response_text.strip().lower()
    STATE_NAMES = ['thinking', 'implementing', 'debugging', 'seekinghelp', 'testing']
    if text in STATE_NAMES:
        return text
    for state_name in STATE_NAMES:
        if state_name in text:
            return state_name
    return None


# ── Manifest & data loading ──────────────────────────────────────────────────

def load_manifest(root_dir):
    manifest_path = os.path.join(root_dir, 'manifest.yaml')
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)
    test_deployments = []
    for name, config in manifest['deployments'].items():
        if config.get('enabled', True) and config.get('split') == 'test':
            test_deployments.append(name)
    return test_deployments


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
        else:
            print(f"  WARNING: {path} not found")
    if not dfs:
        raise FileNotFoundError(f"No test {data_type} files found")
    return pd.concat(dfs, ignore_index=True)


def stratified_subsample(df, label_col, n=1000, seed=42):
    df_clean = df.dropna(subset=[label_col])
    if len(df_clean) <= n:
        print(f"  Dataset size ({len(df_clean)}) <= sample size ({n}), using full set")
        return df_clean
    groups = df_clean.groupby(label_col)
    samples = []
    for label, group in groups:
        frac = len(group) / len(df_clean)
        k = max(1, int(n * frac))
        k = min(k, len(group))
        samples.append(group.sample(n=k, random_state=seed))
    result = pd.concat(samples).sample(frac=1, random_state=seed)
    print(f"  Subsampled {len(result)} from {len(df_clean)} (target: {n})")
    return result


# ── Task runners ─────────────────────────────────────────────────────────────

def run_binary_task(client, df, model, sample_size, task_name, label_col, prompt_builder):
    print(f"\n{'=' * 60}")
    print(f"  TASK: {task_name}")
    print(f"{'=' * 60}")

    if label_col not in df.columns:
        print(f"  ERROR: {label_col} not found in data")
        return None

    sample = stratified_subsample(df, label_col, n=sample_size)
    pos_rate = sample[label_col].mean()
    print(f"  Positive rate: {pos_rate:.1%}")

    predictions = []
    labels = []
    errors = 0

    for i, (idx, row) in enumerate(sample.iterrows()):
        if (i + 1) % 100 == 0:
            valid = len(predictions)
            err_pct = (errors / (i + 1)) * 100
            print(f"  Processing {i + 1}/{len(sample)}... ({valid} valid, {errors} errors [{err_pct:.0f}%])")

        prompt = prompt_builder(row)
        response = call_llm(client, prompt, model)
        prob = parse_probability(response)

        if prob is None:
            errors += 1
            if errors <= 3:
                print(f"    Could not parse: '{response}'")
            continue

        predictions.append(prob)
        labels.append(int(row[label_col]))

    if len(predictions) < 10:
        print(f"  ERROR: Only {len(predictions)} valid predictions, skipping")
        return None

    auc = roc_auc_score(labels, predictions)
    binary_preds = [1 if p > 0.5 else 0 for p in predictions]
    f1 = f1_score(labels, binary_preds, average='macro')

    print(f"\n  Results:")
    print(f"  Valid predictions: {len(predictions)}/{len(sample)} ({errors} errors)")
    print(f"  AUC:      {auc:.3f}")
    print(f"  Macro F1: {f1:.3f}")

    return {
        'task': task_name,
        'auc': round(auc, 3),
        'f1': round(f1, 3),
        'n_samples': len(predictions),
        'n_errors': errors,
        'pos_rate': round(pos_rate, 3),
    }


def run_multiclass_task(client, df, model, sample_size, task_name, label_col,
                        prompt_builder, parse_fn, class_names):
    print(f"\n{'=' * 60}")
    print(f"  TASK: {task_name}")
    print(f"{'=' * 60}")

    if label_col not in df.columns:
        print(f"  ERROR: {label_col} not found in data")
        return None

    sample = stratified_subsample(df, label_col, n=sample_size)

    print(f"\n  Distribution:")
    for val, count in sample[label_col].value_counts().items():
        print(f"    {val}: {count} ({count/len(sample)*100:.1f}%)")

    predictions = []
    labels = []
    errors = 0

    for i, (idx, row) in enumerate(sample.iterrows()):
        if (i + 1) % 100 == 0:
            valid = len(predictions)
            err_pct = (errors / (i + 1)) * 100
            print(f"  Processing {i + 1}/{len(sample)}... ({valid} valid, {errors} errors [{err_pct:.0f}%])")

        prompt = prompt_builder(row)
        response = call_llm(client, prompt, model)
        parsed = parse_fn(response)

        if parsed is None:
            errors += 1
            if errors <= 5:
                print(f"    Could not parse: '{response}'")
            continue

        predictions.append(parsed)
        labels.append(row[label_col])

    if len(predictions) < 10:
        print(f"  ERROR: Only {len(predictions)} valid predictions, skipping")
        return None

    unique_labels = sorted(set(labels) | set(predictions))
    if len(unique_labels) < 2:
        print(f"  ERROR: Only {len(unique_labels)} unique labels")
        return None

    labels_bin = label_binarize(labels, classes=unique_labels)
    preds_bin = label_binarize(predictions, classes=unique_labels)

    try:
        auc = roc_auc_score(labels_bin, preds_bin, average='macro', multi_class='ovr')
    except ValueError:
        auc = 0.5

    f1 = f1_score(labels, predictions, average='macro', zero_division=0)

    print(f"\n  Results:")
    print(f"  Valid predictions: {len(predictions)}/{len(sample)} ({errors} errors)")
    print(f"  AUC:      {auc:.3f}")
    print(f"  Macro F1: {f1:.3f}")

    report = classification_report(labels, predictions, labels=unique_labels, zero_division=0)
    print(f"\n{report}")

    return {
        'task': task_name,
        'auc': round(auc, 3),
        'f1': round(f1, 3),
        'n_samples': len(predictions),
        'n_errors': errors,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='LLM Baseline Evaluation (Closed-Source)')
    parser.add_argument('--root-dir', default='.', help='Path to project root')
    parser.add_argument('--sample-size', type=int, default=1000, help='Subsample size for window tasks')
    parser.add_argument('--model', default='gpt-4o-mini', help='OpenAI model name')
    parser.add_argument('--output', default='llm_results.json', help='Output JSON path')
    parser.add_argument('--tasks', default='next_state,error_imminence,query_imminence,query_type',
                        help='Comma-separated tasks to run')
    args = parser.parse_args()

    tasks_to_run = [t.strip() for t in args.tasks.split(',')]
    data_dir = os.path.join(args.root_dir, 'dataset')

    test_deployments = load_manifest(args.root_dir)
    if not test_deployments:
        print("  ERROR: No enabled test deployments found in manifest.yaml")
        return

    print("=" * 60)
    print("  TUTORTRACE LLM BASELINE (CLOSED-SOURCE)")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Test deployments: {test_deployments}")
    print(f"  Sample size: {args.sample_size}")
    print(f"  Tasks: {tasks_to_run}")
    print()

    client = OpenAI()
    results = []

    # Load window-level data (used by all tasks)
    df_windows = None
    if any(t in tasks_to_run for t in ['next_state', 'error_imminence', 'query_imminence', 'query_type']):
        print("  Loading test windows...")
        df_windows = load_test_data(data_dir, test_deployments, 'windows')
        print(f"  Total test windows: {len(df_windows)}")

    # Task 1: Next behavioral state
    if 'next_state' in tasks_to_run and df_windows is not None:
        result = run_multiclass_task(
            client, df_windows, args.model, args.sample_size,
            'NEXT BEHAVIORAL STATE (5-class)', 'label_next_state',
            build_next_state_prompt, parse_state,
            ['thinking', 'implementing', 'debugging', 'seekinghelp', 'testing'],
        )
        if result:
            result['task'] = 'next_behavioral_state'
            results.append(result)

    # Task 2: Error imminence 15s
    if 'error_imminence' in tasks_to_run and df_windows is not None:
        result = run_binary_task(
            client, df_windows, args.model, args.sample_size,
            'error_imminence_15s', 'label_error_imminence_15s',
            build_error_imminence_prompt,
        )
        if result:
            results.append(result)

    # Task 3: Query imminence 15s
    if 'query_imminence' in tasks_to_run and df_windows is not None:
        result = run_binary_task(
            client, df_windows, args.model, args.sample_size,
            'query_imminence_15s', 'label_query_imminence_15s',
            build_query_imminence_prompt,
        )
        if result:
            results.append(result)

    # Task 4: Query engagement (guided vs dependent)
    if 'query_type' in tasks_to_run and df_windows is not None:
        # Map to binary: dependent=1, guided=0
        df_qt = df_windows[df_windows['label_next_query_type'].isin(['guided', 'dependent'])].copy()
        df_qt['label_dependent'] = (df_qt['label_next_query_type'] == 'dependent').astype(int)

        if len(df_qt) > 0:
            result = run_binary_task(
                client, df_qt, args.model, args.sample_size,
                'query_engagement', 'label_dependent',
                build_query_engagement_prompt,
            )
            if result:
                result['task'] = 'query_engagement'
                results.append(result)
        else:
            print("  ERROR: No query engagement labels found in windows")

    # Save results
    output = {
        'model': args.model,
        'test_deployments': test_deployments,
        'sample_size': args.sample_size,
        'results': results,
    }

    output_path = os.path.join(args.root_dir, args.output)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY — {args.model}")
    print(f"{'=' * 60}")
    print(f"  {'Task':<30s} {'AUC':>8s} {'F1':>8s} {'N':>8s}")
    print(f"  {'-' * 56}")
    for r in results:
        print(f"  {r['task']:<30s} {r['auc']:>8.3f} {r['f1']:>8.3f} {r['n_samples']:>8d}")


if __name__ == '__main__':
    main()