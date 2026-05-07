"""
Export data for paper figures.

Generates:
  1. figures/word_cloud_queries.csv — all query texts with guided/dependent labels
  2. figures/behavioral_distribution.csv — segments with session progress and query context

Usage:
  cd tutortrace_dataset_and_benchmark
  python3 figures/export_figure_data.py
"""

import os
import json
import yaml
import pandas as pd
import numpy as np

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATASET_DIR = os.path.join(ROOT_DIR, 'dataset')
MANIFEST_PATH = os.path.join(ROOT_DIR, 'manifest.yaml')
OUT_DIR = os.path.join(ROOT_DIR, 'figures')

QUERY_TYPES = {'CHAT_QUERY', 'CHAT_SEND'}
RESPONSE_TYPES = {'CHAT_RESPONSE', 'CHAT_RECEIVE'}


def load_manifest():
    with open(MANIFEST_PATH) as f:
        return yaml.safe_load(f)


def load_raw(path):
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    rows = []
    for sid, sdata in raw.items():
        for ev in sdata.get('events', []):
            payload = ev.get('payload', {}) or {}
            text = ''
            if isinstance(payload, dict):
                text = payload.get('text') or payload.get('content') or payload.get('message') or ''

            rows.append({
                'student_id': str(sid),
                'timestamp_ms': ev.get('timestamp'),
                'type': ev.get('type'),
                'text': text,
            })

    df = pd.DataFrame(rows)
    df['timestamp_ms'] = pd.to_numeric(df['timestamp_ms'], errors='coerce')
    df = df.dropna(subset=['timestamp_ms'])
    return df.sort_values(['student_id', 'timestamp_ms']).reset_index(drop=True)


def export_word_cloud(manifest):
    """Export all query texts with labels for word cloud figure."""
    print("Exporting word cloud data...")
    rows = []

    for name, config in manifest['deployments'].items():
        if not config.get('enabled', True):
            continue

        raw_path = os.path.join(DATASET_DIR, config['raw_telemetry'])
        if not os.path.exists(raw_path):
            continue

        df = load_raw(raw_path)

        # Load labels
        label_path = os.path.join(DATASET_DIR, 'query_labels', f'{name}_labels.csv')
        label_map = {}
        if os.path.exists(label_path):
            ldf = pd.read_csv(label_path)
            ldf['student_id'] = ldf['student_id'].astype(str)
            for _, r in ldf.iterrows():
                label_map[(str(r['student_id']), int(r['query_index']))] = r['query_type']

        # Extract queries
        for sid in df['student_id'].unique():
            s = df[df['student_id'] == sid]
            queries = s[s['type'].isin(QUERY_TYPES)].sort_values('timestamp_ms')

            for qi, (_, q) in enumerate(queries.iterrows()):
                text = q['text'].strip()
                if not text:
                    continue
                label = label_map.get((sid, qi + 1), '')

                rows.append({
                    'deployment': name,
                    'student_id': sid,
                    'query_index': qi + 1,
                    'query_text': text,
                    'query_type': label,
                })

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, 'word_cloud_queries.csv')
    out.to_csv(out_path, index=False)
    print(f"  {len(out)} queries → {out_path}")
    print(f"  Labeled: {(out['query_type'] != '').sum()}")
    print(f"  Distribution: {out['query_type'].value_counts().to_dict()}")


def export_behavioral_distribution(manifest):
    """Export segment data with session progress and query context."""
    print("\nExporting behavioral distribution data...")
    rows = []

    for name, config in manifest['deployments'].items():
        if not config.get('enabled', True):
            continue

        raw_path = os.path.join(DATASET_DIR, config['raw_telemetry'])
        seg_path = os.path.join(DATASET_DIR, 'behavioral_sequences', f'{name}_segments.csv')

        if not os.path.exists(raw_path) or not os.path.exists(seg_path):
            continue

        df = load_raw(raw_path)
        seg_df = pd.read_csv(seg_path)
        seg_df['student_id'] = seg_df['student_id'].astype(str)

        for sid in seg_df['student_id'].unique():
            s = df[df['student_id'] == sid]
            if len(s) == 0:
                continue

            student_segs = seg_df[seg_df['student_id'] == sid].sort_values('start_time_ms')
            if len(student_segs) == 0:
                continue

            session_start_ms = s['timestamp_ms'].min()
            session_end_ms = s['timestamp_ms'].max()
            session_duration_s = (session_end_ms - session_start_ms) / 1000

            if session_duration_s < 10:
                continue

            # Get query times relative to session start
            queries = s[s['type'].isin(QUERY_TYPES)].sort_values('timestamp_ms')
            query_times_ms = queries['timestamp_ms'].tolist()

            # Get response times
            responses = s[s['type'].isin(RESPONSE_TYPES)].sort_values('timestamp_ms')
            response_times_ms = responses['timestamp_ms'].tolist()

            for _, seg in student_segs.iterrows():
                seg_start_ms = seg['start_time_ms']
                seg_start_abs_ms = session_start_ms + seg_start_ms

                # Which query number does this segment fall between?
                # 0 = before first query, 1 = after first query/response, etc.
                query_number = 0
                for qt in query_times_ms:
                    if seg_start_abs_ms > qt:
                        query_number += 1

                # Is this segment in the 30s before a query?
                before_query = False
                for qt in query_times_ms:
                    if 0 < (qt - seg_start_abs_ms) <= 30000:
                        before_query = True
                        break

                # Is this segment in the 30s after a response?
                after_response = False
                for rt in response_times_ms:
                    if 0 < (seg_start_abs_ms - rt) <= 30000:
                        after_response = True
                        break

                start_s = seg_start_ms / 1000
                session_progress = min(1.0, start_s / max(1, session_duration_s))

                rows.append({
                    'deployment': name,
                    'student_id': sid,
                    'segment_index': seg['segment_index'],
                    'behavior': seg['behavioral_state'],
                    'thinking_subtype': seg.get('thinking_subtype', ''),
                    'start_time_s': round(start_s, 2),
                    'duration_s': seg['duration_s'],
                    'session_duration_s': round(session_duration_s, 2),
                    'session_progress': round(session_progress, 4),
                    'query_number': query_number,
                    'total_queries': len(query_times_ms),
                    'before_query_30s': int(before_query),
                    'after_response_30s': int(after_response),
                })

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, 'behavioral_distribution.csv')
    out.to_csv(out_path, index=False)
    print(f"  {len(out)} segments → {out_path}")
    print(f"  Deployments: {out['deployment'].nunique()}")
    print(f"  Students: {out['student_id'].nunique()}")
    print(f"  Behaviors: {out['behavior'].value_counts().to_dict()}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = load_manifest()

    export_word_cloud(manifest)
    export_behavioral_distribution(manifest)

    print("\nDone!")


if __name__ == '__main__':
    main()