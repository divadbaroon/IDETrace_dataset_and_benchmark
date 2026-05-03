"""
LLM Baseline Evaluation for TutorTrace Benchmark (Closed-Source).

Evaluates closed-source LLMs via OpenAI API on four tasks:
  1. Next behavioral state (5-class) - subsampled 1000 windows
  2. Error imminence 15s (binary) - subsampled 1000 windows
  3. Query imminence 15s (binary) - subsampled 1000 windows
  4. Post-query improvement (binary) - full query-level test set

Reads test deployments from manifest.yaml automatically.

Usage:
  cd tutortrace_dataset_and_benchmark
  python3 benchmark/models/llm_baseline.py --model gpt-4o-mini
  python3 benchmark/models/llm_baseline.py --model gpt-4o
  python3 benchmark/models/llm_baseline.py --model gpt-5.5

Run all three back to back:
  python3 benchmark/models/llm_baseline.py --model gpt-4o-mini --output llm_results_4o_mini.json && \
  python3 benchmark/models/llm_baseline.py --model gpt-4o --output llm_results_4o.json && \
  python3 benchmark/models/llm_baseline.py --model gpt-5.5 --output llm_results_5_5.json
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import os
import time
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, classification_report
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


def build_post_query_improvement_prompt(row):
    return f"""You are analyzing a programming student's behavior before an AI query. Based on the features below, predict the probability that the student's test pass count will increase after this AI interaction.

Pre-query behavior (what the student did before this query):
- Code edits before query: {int(row.get('pre_code_edits', 0))}
- Terminal runs before query: {int(row.get('pre_terminal_runs', 0))}
- Terminal errors before query: {int(row.get('pre_terminal_errors', 0))}
- Net code growth: {int(row.get('pre_net_code_growth', 0))}
- Thinking time before query: {row.get('thinking_time_s', 0):.1f}s
- Time spent on errors: {row.get('thinking_error_s', 0):.1f}s
- Time spent on code: {row.get('thinking_code_s', 0):.1f}s
- Debugging time: {row.get('debugging_time_s', 0):.1f}s
- Time in editor: {row.get('pre_time_in_editor_s', 0):.1f}s
- Time in terminal: {row.get('pre_time_in_terminal_s', 0):.1f}s
- Time in chat: {row.get('pre_time_in_chat_s', 0):.1f}s
- Longest idle before query: {row.get('pre_longest_idle_s', 0):.1f}s
- Duration before query: {row.get('pre_duration_s', 0):.1f}s
- Failed test self-fix attempts: {int(row.get('pre_failed_test_self_fix', 0))}
- Failed test to edit time: {row.get('pre_failed_test_to_edit_s', 0):.1f}s
- Error self-fix attempts: {int(row.get('pre_error_self_fix', 0))}

Session context:
- Query index: {int(row.get('query_index', 0))}
- Time since session start: {row.get('time_since_session_start_s', 0):.1f}s
- Time since last query: {row.get('time_since_last_query_s', 0):.1f}s
- Query length (chars): {int(row.get('query_length_chars', 0))}
- AI response length (chars): {int(row.get('ai_response_length_chars', 0))}

Test state at query:
- Tests passed: {int(row.get('test_passed_at_query', 0))}
- Tests total: {int(row.get('test_total_at_query', 0))}

Respond with ONLY a number between 0.0 and 1.0 representing the probability that tests will improve after this AI interaction. No explanation."""


# ── API caller ───────────────────────────────────────────────────────────────

def call_llm(client, prompt, model="gpt-4o-mini", max_retries=3):
    """Call OpenAI API with retries."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Respond with ONLY the requested value. No reasoning, no explanation, no extra text."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=20,
            )
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
    """Extract a probability value from LLM response."""
    if response_text is None:
        return None
    try:
        val = float(response_text.strip())
        return max(0.0, min(1.0, val))
    except ValueError:
        import re
        matches = re.findall(r'(?<!\d)([01]\.?\d*)', response_text)
        if matches:
            try:
                val = float(matches[0])
                return max(0.0, min(1.0, val))
            except ValueError:
                pass
    return None


def parse_state(response_text):
    """Extract a behavioral state from LLM response."""
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
    """Load and concatenate test deployment CSVs."""
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
    """Run a binary probability prediction task."""
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


def run_next_state_task(client, df, model, sample_size):
    """Run next behavioral state prediction (5-class)."""
    print(f"\n{'=' * 60}")
    print(f"  TASK: NEXT BEHAVIORAL STATE (5-class)")
    print(f"{'=' * 60}")

    label_col = 'label_next_state'
    if label_col not in df.columns:
        print(f"  ERROR: {label_col} not found in data")
        return None

    STATE_MAP = {'thinking': 0, 'implementing': 1, 'debugging': 2, 'seekinghelp': 3, 'testing': 4}
    STATE_NAMES = {v: k for k, v in STATE_MAP.items()}

    sample = stratified_subsample(df, label_col, n=sample_size)

    predictions = []
    labels = []
    errors = 0

    for i, (idx, row) in enumerate(sample.iterrows()):
        if (i + 1) % 100 == 0:
            valid = len(predictions)
            err_pct = (errors / (i + 1)) * 100
            print(f"  Processing {i + 1}/{len(sample)}... ({valid} valid, {errors} errors [{err_pct:.0f}%])")

        prompt = build_next_state_prompt(row)
        response = call_llm(client, prompt, model)
        matched_state = parse_state(response)

        if matched_state is None:
            errors += 1
            if errors <= 3:
                print(f"    Could not parse: '{response}'")
            continue

        predictions.append(STATE_MAP[matched_state])
        label_val = row[label_col]
        if isinstance(label_val, str):
            labels.append(STATE_MAP.get(label_val.lower(), -1))
        else:
            labels.append(int(label_val))

    if len(predictions) < 10:
        print(f"  ERROR: Only {len(predictions)} valid predictions, skipping")
        return None

    from sklearn.preprocessing import label_binarize
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

    report = classification_report(
        labels, predictions,
        target_names=[STATE_NAMES[i] for i in unique_labels],
        labels=unique_labels, zero_division=0,
    )
    print(f"\n{report}")

    return {
        'task': 'next_behavioral_state',
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
    parser.add_argument('--tasks', default='next_state,error_imminence,query_imminence,post_query',
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

    # Load window-level data
    df_windows = None
    if any(t in tasks_to_run for t in ['next_state', 'error_imminence', 'query_imminence']):
        print("  Loading test windows...")
        df_windows = load_test_data(data_dir, test_deployments, 'windows')
        print(f"  Total test windows: {len(df_windows)}")

    # Load query-level data
    df_queries = None
    if 'post_query' in tasks_to_run:
        print("\n  Loading test queries...")
        df_queries = load_test_data(data_dir, test_deployments, 'queries')
        print(f"  Total test queries: {len(df_queries)}")

    # Task 1: Next behavioral state
    if 'next_state' in tasks_to_run and df_windows is not None:
        result = run_next_state_task(client, df_windows, args.model, args.sample_size)
        if result:
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

    # Task 4: Post-query improvement
    if 'post_query' in tasks_to_run and df_queries is not None:
        result = run_binary_task(
            client, df_queries, args.model, len(df_queries),
            'post_query_improvement', 'label_post_query_improvement',
            build_post_query_improvement_prompt,
        )
        if result:
            results.append(result)

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

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY — {args.model}")
    print(f"{'=' * 60}")
    print(f"  {'Task':<30s} {'AUC':>8s} {'F1':>8s} {'N':>8s}")
    print(f"  {'-' * 56}")
    for r in results:
        print(f"  {r['task']:<30s} {r['auc']:>8.3f} {r['f1']:>8.3f} {r['n_samples']:>8d}")


if __name__ == '__main__':
    main()