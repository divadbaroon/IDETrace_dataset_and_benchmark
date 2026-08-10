from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from main import load_all_deployments
from analysis.windows import compute_w1_windows, compute_w2_windows
from analysis.clustering import search_w1, reached_all_pass
from analysis.clustering_w2 import search_w2
from analysis.session_patterns import (
    build_w3_session_rows,
    cluster_w3_session_patterns,
)

TAXONOMY_DEPLOYMENTS = ('deployment_1', 'deployment_2')

W1_EXCLUDE = {
    'session_duration_ms', 'session_duration_min',
    'seeking_help_time_s', 'response_reading_time_s', 'duration_s',
}
W1_TARGET = {'time_in_editor_s', 'time_in_terminal_s', 'time_in_chat_s'}
W2_TARGET = {'time_in_editor_s', 'thinking_time_s', 'error_self_fix'}


def _find_target(results: list[dict[str, Any]], target: set[str]) -> dict[str, Any] | None:
    for rank, result in enumerate(results, start=1):
        if set(result['features']) == target:
            return {'rank': rank, **result}
    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_w3_csvs(output_dir: Path, w3_result: dict[str, Any]) -> None:
    assignments = w3_result.get('assignments', [])
    if assignments:
        path = output_dir / 'w3_session_pattern_assignments.csv'
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(assignments[0].keys()))
            writer.writeheader()
            writer.writerows(assignments)

    best_k = w3_result.get('best_k')
    if best_k is not None:
        summaries = w3_result['per_k'][best_k]['cluster_summaries']
        summaries = sorted(summaries, key=lambda row: row.get('profile', ''))
        path = output_dir / 'w3_session_pattern_summary.csv'
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)


def main() -> None:
    raw_all = load_all_deployments()
    raw = {
        session_id: student
        for session_id, student in raw_all.items()
        if student['deployment'] in TAXONOMY_DEPLOYMENTS
    }
    outcome_by_sid = {
        session_id: reached_all_pass(student['events'])
        for session_id, student in raw.items()
    }

    # Window 1
    w1 = compute_w1_windows(raw)
    out1 = search_w1(
        w1,
        outcome_by_sid,
        pool_exclude=W1_EXCLUDE,
        min_cluster_size=15,
    )

    # Window 2
    w2 = compute_w2_windows(raw)
    outcome_by_key: dict[str, int] = {}
    for index, window in enumerate(w2):
        key = f"{window['session_id']}#{window.get('gap_index', index)}"
        window['_key'] = key
        outcome_by_key[key] = outcome_by_sid[window['session_id']]
    out2 = search_w2(w2, outcome_by_key, min_cluster_size=15)

    # Window 3
    w3_rows, w3_population = build_w3_session_rows(raw, w2, min_intervals=2)
    out3 = cluster_w3_session_patterns(
        w3_rows,
        ks=(2, 3, 4, 5, 6),
        n_init=100,
        min_cluster_size=15,
        stability_seeds=20,
    )

    result = {
        'method': {
            'deployments': list(TAXONOMY_DEPLOYMENTS),
            'monotonicity_filter': False,
            'minimum_cluster_size': 15,
            'w2_search': 'one combined exhaustive search',
            'w2_failed_response_intervals': 'excluded',
            'w2_duration_s': 'excluded from candidate pool',
            'ranking_w1_w2': 'silhouette * completion-rate spread',
            'w3_minimum_valid_intervals': 2,
            'w3_features': out3['feature_names'],
            'w3_selection': 'highest silhouette among eligible K; completion not used',
        },
        'w1': {
            'total': len(w1),
            'cold_start': out1['cold_start_n'],
            'active': out1['active_n'],
            'pool': out1['pool'],
            'dropped_undefined': out1['dropped_undefined'],
            'valid_solutions': len(out1['results']),
            'winner': out1['results'][0] if out1['results'] else None,
            'target': _find_target(out1['results'], W1_TARGET),
        },
        'w2': {
            'total': len(w2),
            'passive': out2['passive_n'],
            'active': out2['active_n'],
            'pool': out2['features'],
            'source_pools': out2['source_pools'],
            'dropped_undefined': out2['dropped_undefined'],
            'dropped_sparse': out2['dropped_sparse'],
            'valid_solutions': len(out2['results']),
            'winner': out2['results'][0] if out2['results'] else None,
            'target': _find_target(out2['results'], W2_TARGET),
        },
        'w3': {
            'population': w3_population,
            **out3,
        },
    }

    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'latest_taxonomy_results.json'
    output_path.write_text(json.dumps(_json_safe(result), indent=2))
    _write_w3_csvs(output_dir, out3)

    print(f"W1: total={len(w1)}, Cold Start={out1['cold_start_n']}, active={out1['active_n']}")
    print(f"W1 winner: {out1['results'][0] if out1['results'] else None}")
    print(f"W2: total={len(w2)}, Passive={out2['passive_n']}, active={out2['active_n']}")
    print(f"W2 winner: {out2['results'][0] if out2['results'] else None}")
    print(f"W3: eligible learners={len(w3_rows)}, best K={out3['best_k']}")
    if out3['best_k'] is not None:
        print(f"W3 best: {out3['per_k'][out3['best_k']]}")
    print(f"Saved {output_path}")


if __name__ == '__main__':
    main()