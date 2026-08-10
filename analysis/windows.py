import yaml
from analysis.feature_extraction import compute_student_features

TELEMETRY_EVENTS_PATH = "constants/telemetry_events.yaml"


# ─── Event tracker helpers ──────────────────────────────────────────────

def get_event_tracker_template():
    with open(TELEMETRY_EVENTS_PATH) as f:
        telemetry_events = yaml.safe_load(f)
    template = {}
    for group in telemetry_events.values():
        for event in group:
            template[event] = 0
    return template


def build_event_tracker(events, template):
    """Count events / regions over a (possibly sliced) event list."""
    tracker = dict(template)
    for e in events:
        if e['type'] in tracker:
            tracker[e['type']] += 1
        if 'payload' in e and 'region' in e['payload']:
            region = e['payload']['region']
            if region in tracker:            # guard: unknown region won't crash
                tracker[region] += 1
    return tracker


# ─── Population filters (match the original pipeline) ────────────────────

def truncate_at_first_all_pass(events):
    """Cut events at the first all-pass TEST_CASE_RESULT (inclusive). Everything after
       the student first solves the task is discarded before any feature computation.
       Matches the original: filteredEvents = sorted.slice(0, allPassedIndex + 1)."""
    s = sorted(events, key=lambda e: e['timestamp'])
    all_pass_idx = None
    for i, e in enumerate(s):
        if e['type'] == 'TEST_CASE_RESULT':
            p = e.get('payload', {})
            if p.get('total_tests', 0) > 0 and p.get('passed_count', 0) == p.get('total_tests', 0):
                all_pass_idx = i
                break
    return s[:all_pass_idx + 1] if all_pass_idx is not None else s


def tab_hidden_seconds(events):
    """Total seconds the tab was hidden over `events` (TAB_STATE visible:false -> true).
       Matches calculateStats tabVisibility.hiddenTime."""
    if not events:
        return 0.0
    s = sorted(events, key=lambda e: e['timestamp'])
    period_end = s[-1]['timestamp']
    hidden_ms = 0
    last_hidden = None
    for e in s:
        if e['type'] == 'TAB_STATE':
            vis = e.get('payload', {}).get('visible')
            if vis is False:
                last_hidden = e['timestamp']
            elif vis is True and last_hidden is not None:
                hidden_ms += e['timestamp'] - last_hidden
                last_hidden = None
    if last_hidden is not None:
        hidden_ms += period_end - last_hidden
    return hidden_ms / 1000


# ─── W1: pre-query window ────────────────────────────────────────────────

def get_w1_slice(events):
    """Pre-query window: events from session start up to (not including) the first
       CHAT_QUERY. Returns None if the student never queried (not in taxonomy population).
       Assumes `events` is already sorted (call after truncate_at_first_all_pass)."""
    for i, e in enumerate(events):
        if e['type'] == 'CHAT_QUERY':
            return events[:i]
    return None


def compute_w1_windows(deployment):
    """One W1 feature row per taxonomy-eligible student.

    Eligible = made >=1 query AND <=30s tab-hidden in the pre-query window.
    Features are computed over the truncated pre-query slice.
    """
    template = get_event_tracker_template()
    windows = []
    for session_id, student in deployment.items():
        # 1. Truncate the whole session at first all-pass BEFORE anything else
        events = truncate_at_first_all_pass(student['events'])

        # 2. Pre-query slice (None => never queried => excluded from taxonomy)
        w1_events = get_w1_slice(events)
        if w1_events is None or len(w1_events) < 2:
            continue

        # 3. Off-platform filter: drop if tab hidden > 30s in the pre-query window
        if tab_hidden_seconds(w1_events) > 30:
            continue

        # 4. Compute features over the pre-query slice
        tracker = build_event_tracker(w1_events, template)
        features = compute_student_features(w1_events, tracker)
        windows.append({
            'session_id': session_id,
            'student_name': student['student_name'],
            'deployment': student.get('deployment', ''),
            'window': 'W1',
            'features': features,
        })
    return windows


# ─── W2: between-query windows ───────────────────────────────────────────

def get_w2_slices(events):
    """Return gaps between consecutive ``CHAT_QUERY`` events.

    Each item contains the events from the earlier query up to, but not
    including, the next query, plus the next-query timestamp. The tail after
    the final query is not a Window 2 observation.
    """
    query_indices = [i for i, e in enumerate(events) if e['type'] == 'CHAT_QUERY']
    slices = []
    for k in range(1, len(query_indices)):
        start = query_indices[k - 1]
        end = query_indices[k]
        slices.append({
            'events': events[start:end],
            'next_query_timestamp': events[end]['timestamp'],
        })
    return slices


def _post_response_features(events, next_query_timestamp):
    """Compute fully observed post-response measures for a valid W2 interval.

    ``code_after_response`` distinguishes whether code activity occurred before
    the next query. ``response_to_code_or_next_query_s`` records time to the
    first code event, or to the next query when no code event occurred. This
    avoids encoding an absent event as a zero-second latency.
    """
    response = next((e for e in events if e['type'] == 'CHAT_RESPONSE'), None)
    if response is None:
        return None

    first_code = next(
        (
            e for e in events
            if e['timestamp'] > response['timestamp']
            and e['type'].startswith('CODE_')
        ),
        None,
    )
    code_after_response = 1 if first_code is not None else 0
    endpoint = first_code['timestamp'] if first_code is not None else next_query_timestamp
    latency_s = max(0.0, (endpoint - response['timestamp']) / 1000)

    return {
        'code_after_response': code_after_response,
        'response_to_code_or_next_query_s': round(latency_s),
    }


def compute_w2_windows(deployment):
    """One W2 feature row per valid interval between consecutive queries.

    Intervals are excluded when the preceding query did not receive an AI
    response (for example, ``CHAT_QUERY`` followed by ``CHAT_ERROR``), because
    W2 characterizes behavior following a returned AI response.
    """
    template = get_event_tracker_template()
    windows = []
    for session_id, student in deployment.items():
        events = truncate_at_first_all_pass(student['events'])
        for idx, item in enumerate(get_w2_slices(events)):
            w2_events = item['events']
            next_query_timestamp = item['next_query_timestamp']
            if len(w2_events) < 2:
                continue
            if tab_hidden_seconds(w2_events) > 30:
                continue

            post_response = _post_response_features(w2_events, next_query_timestamp)
            if post_response is None:
                continue

            tracker = build_event_tracker(w2_events, template)
            features = compute_student_features(w2_events, tracker)
            features.update(post_response)
            windows.append({
                'session_id': session_id,
                'student_name': student['student_name'],
                'deployment': student.get('deployment', ''),
                'window': 'W2',
                'gap_index': idx,
                'features': features,
            })
    return windows