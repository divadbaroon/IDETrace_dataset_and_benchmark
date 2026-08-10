import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "behavioral_classifier"))
from auto_segmenter import auto_segment_events  # repo's segmenter, unmodified


def compute_char_counts(events):
    chars_inserted = 0
    chars_deleted = 0
    for e in events:
        changes = e.get('payload', {}).get('changes')
        if not changes:
            continue
        for c in changes:
            chars_inserted += len(c.get('text', ''))
            chars_deleted += c.get('to', 0) - c.get('from', 0)
    return chars_inserted, chars_deleted


def compute_student_features(events, event_tracker):
    features = {}

    # Auto-segment session
    start_time = events[0]['timestamp']
    duration = events[-1]['timestamp'] - events[0]['timestamp']
    segments = auto_segment_events(events, start_time, duration)

    # Session duration
    features['session_duration_ms'] = events[-1]['timestamp'] - events[0]['timestamp']
    features['session_duration_s'] = features['session_duration_ms'] / 1000
    features['session_duration_min'] = features['session_duration_ms'] / 60000

    # Code deletions (event count)
    features['code_deletions'] = event_tracker['CODE_DELETE']

    # Terminal runs / errors
    features['terminal_runs'] = event_tracker['TERMINAL_RUN']
    features['terminal_errors'] = event_tracker['TERMINAL_ERROR']

    # Character insertions / deletions
    chars_inserted, chars_deleted = compute_char_counts(events)
    features['chars_inserted'] = chars_inserted
    features['chars_deleted'] = chars_deleted

    # code_edits = TYPE + INDENT (Fix 2, matches calculateStats)
    features['code_edits'] = event_tracker['CODE_TYPE'] + event_tracker['CODE_INDENT']

    # Net code growth
    features['net_code_growth'] = chars_inserted - chars_deleted

    # delete_type_ratio = (DELETE + DELETE_SELECTION + CUT) / (TYPE + INDENT)  (Fix 3)
    delete_events = (event_tracker['CODE_DELETE'] + event_tracker['CODE_DELETE_SELECTION']
                     + event_tracker['CODE_CUT'])
    type_events = event_tracker['CODE_TYPE'] + event_tracker['CODE_INDENT']
    if type_events > 0:
        features['delete_type_ratio'] = delete_events / type_events
    else:
        features['delete_type_ratio'] = None

    # Code edit rate (per second)
    if features['session_duration_s'] > 0:
        features['code_edit_rate'] = features['code_edits'] / features['session_duration_s']
    else:
        features['code_edit_rate'] = 0

    # Longest idle gap (seconds)
    longest_idle = 0
    for i in range(1, len(events)):
        idle_period = (events[i]['timestamp'] - events[i - 1]['timestamp']) / 1000
        if idle_period > longest_idle:
            longest_idle = idle_period
    features['longest_idle'] = longest_idle

    # Mean time between terminal runs
    run_timestamps = []
    for e in events:
        if e['type'] == 'TERMINAL_RUN':
            run_timestamps.append(e['timestamp'])
    if len(run_timestamps) < 2:
        features['mean_time_between_runs_s'] = None
    else:
        total_gap_s = 0
        gap_count = 0
        for i in range(1, len(run_timestamps)):
            total_gap_s += (run_timestamps[i] - run_timestamps[i - 1]) / 1000
            gap_count += 1
        features['mean_time_between_runs_s'] = total_gap_s / gap_count

    # Max consecutive errors (Fix 4: reset on test-progress OR net-growth>5)
    max_consecutive = 0
    current_streak = 0
    last_test_passed = 0
    for e in events:
        if e['type'] == 'TERMINAL_ERROR':
            current_streak += 1
            if current_streak > max_consecutive:
                max_consecutive = current_streak
        elif e['type'] == 'TEST_CASE_RESULT':
            passed = e.get('payload', {}).get('passed_count', 0)
            if passed > last_test_passed:
                current_streak = 0
                last_test_passed = passed
        elif e['type'].startswith('CODE_') and e.get('payload', {}).get('changes'):
            net_change = 0
            for c in e['payload']['changes']:
                net_change += len(c.get('text', '')) - (c.get('to', 0) - c.get('from', 0))
            if net_change > 5:
                current_streak = 0
    features['max_consecutive_errors'] = max_consecutive

    edit_events = ('CODE_TYPE', 'CODE_DELETE', 'CODE_PASTE')

    # error_to_edit
    error_to_edit_sum_s = 0
    error_to_edit_count = 0
    for i in range(len(events)):
        if events[i]['type'] != 'TERMINAL_ERROR':
            continue
        for j in range(i + 1, len(events)):
            if events[j]['type'] in edit_events:
                error_to_edit_sum_s += (events[j]['timestamp'] - events[i]['timestamp']) / 1000
                error_to_edit_count += 1
                break
    features['error_to_edit_s'] = error_to_edit_sum_s / error_to_edit_count if error_to_edit_count > 0 else None

    # failed_test_to_edit
    failed_test_to_edit_sum_s = 0
    failed_test_to_edit_count = 0
    for i in range(len(events)):
        if events[i]['type'] != 'TEST_CASE_RESULT':
            continue
        payload = events[i].get('payload', {})
        if payload.get('total_tests', 0) - payload.get('passed_count', 0) < 1:
            continue
        for j in range(i + 1, len(events)):
            if events[j]['type'] in edit_events:
                failed_test_to_edit_sum_s += (events[j]['timestamp'] - events[i]['timestamp']) / 1000
                failed_test_to_edit_count += 1
                break
    features['failed_test_to_edit_s'] = failed_test_to_edit_sum_s / failed_test_to_edit_count if failed_test_to_edit_count > 0 else None

    # error_reading
    mouse_events = ('MOUSE_MOVE', 'MOUSE_CLICK')
    error_reading_sum_s = 0
    error_reading_count = 0
    for i in range(len(events)):
        if events[i]['type'] != 'TERMINAL_ERROR':
            continue
        for j in range(i + 1, len(events)):
            if events[j]['type'] not in mouse_events:
                error_reading_sum_s += (events[j]['timestamp'] - events[i]['timestamp']) / 1000
                error_reading_count += 1
                break
    features['error_reading_time_s'] = error_reading_sum_s / error_reading_count if error_reading_count > 0 else None

    # error_self_fix (Fix 5: COUNT of errors whose next code-or-query event is code)
    error_self_fix_count = 0
    for i in range(len(events)):
        if events[i]['type'] != 'TERMINAL_ERROR':
            continue
        for j in range(i + 1, len(events)):
            if events[j]['type'].startswith('CODE_') or events[j]['type'] == 'CHAT_QUERY':
                if events[j]['type'] != 'CHAT_QUERY':
                    error_self_fix_count += 1
                break
    features['error_self_fix'] = error_self_fix_count

    # failed_test_self_fix (Fix 5: COUNT version)
    failed_test_self_fix_count = 0
    for i in range(len(events)):
        if events[i]['type'] != 'TEST_CASE_RESULT':
            continue
        payload = events[i].get('payload', {})
        if payload.get('total_tests', 0) - payload.get('passed_count', 0) < 1:
            continue
        for j in range(i + 1, len(events)):
            if events[j]['type'].startswith('CODE_') or events[j]['type'] == 'CHAT_QUERY':
                if events[j]['type'] != 'CHAT_QUERY':
                    failed_test_self_fix_count += 1
                break
    features['failed_test_self_fix'] = failed_test_self_fix_count

    # Region times — credit only positive gaps shorter than 30 seconds.
    # Longer gaps are treated as idle/off-task rather than attributed to a region.
    region_time = {}
    region_events = [e for e in events if 'region' in e.get('payload', {})]
    for k in range(1, len(region_events)):
        prev = region_events[k - 1]
        gap_ms = region_events[k]['timestamp'] - prev['timestamp']
        if gap_ms <= 0 or gap_ms >= 30000:
            continue
        r = prev['payload']['region']
        region_time[r] = region_time.get(r, 0) + gap_ms / 1000
    features['time_in_editor_s'] = region_time.get('CODE_EDITOR', 0)
    features['time_in_terminal_s'] = region_time.get('TERMINAL', 0)
    features['time_in_chat_s'] = region_time.get('CHAT_WINDOW', 0)
    features['time_in_task_s'] = region_time.get('TASK_DESCRIPTION', 0)
    features['time_in_tests_s'] = region_time.get('TEST_CASES', 0)

    # chat_to_code_latency_s: mean time from a CHAT_RESPONSE to the next code edit
    # (matches calculateStats chatMetrics.chatToCodeLatency). Used by W2 post-response search.
    ctc_sum_s = 0
    ctc_count = 0
    for i in range(len(events)):
        if events[i]['type'] != 'CHAT_RESPONSE':
            continue
        for j in range(i + 1, len(events)):
            if events[j]['type'].startswith('CODE_'):
                ctc_sum_s += (events[j]['timestamp'] - events[i]['timestamp']) / 1000
                ctc_count += 1
                break
    features['chat_to_code_latency_s'] = ctc_sum_s / ctc_count if ctc_count > 0 else None

    # response_reading_latency_s: mean time from a CHAT_RESPONSE to the next non-mouse,
    # non-tab event (matches calculateStats chatMetrics.responseReadingTime). This is the
    # latency-based version W2 uses (distinct from the segment-based response_reading_time_s).
    rr_sum_s = 0
    rr_count = 0
    for i in range(len(events)):
        if events[i]['type'] != 'CHAT_RESPONSE':
            continue
        for j in range(i + 1, len(events)):
            if events[j]['type'] not in ('MOUSE_MOVE', 'TAB_STATE'):
                rr_sum_s += (events[j]['timestamp'] - events[i]['timestamp']) / 1000
                rr_count += 1
                break
    features['response_reading_latency_s'] = rr_sum_s / rr_count if rr_count > 0 else None

    # Segment-derived times
    thinking_time_ms = seeking_help_ms = response_reading_ms = 0
    for seg in segments:
        bid = (seg.get('suggestedBehavior') or {}).get('id')
        if bid == 'thinking':
            thinking_time_ms += seg['endTime'] - seg['startTime']
        if bid == 'seekingHelp':
            seeking_help_ms += seg['endTime'] - seg['startTime']
        if seg.get('suggestedThinkingSubcategory') == 'thinking-llm':
            response_reading_ms += seg['endTime'] - seg['startTime']
    features['thinking_time_s'] = thinking_time_ms / 1000
    features['seeking_help_time_s'] = seeking_help_ms / 1000
    features['response_reading_time_s'] = response_reading_ms / 1000

    if segments:
        last = segments[-1]
        features['duration_s'] = (last['endTime'] - last['startTime']) / 1000
    else:
        features['duration_s'] = None

    # ─── Integer rounding to match CSV storage (Math.round(v/1000)) ───
    # The original dataset stores all *_s features as whole integer seconds.
    # Round only the second-valued (_s) features; counts/ratios/rates stay as-is.
    ROUND_TO_INT = [
        'session_duration_s', 'longest_idle', 'mean_time_between_runs_s',
        'error_to_edit_s', 'failed_test_to_edit_s', 'error_reading_time_s',
        'time_in_editor_s', 'time_in_terminal_s', 'time_in_chat_s',
        'time_in_task_s', 'time_in_tests_s',
        'thinking_time_s', 'seeking_help_time_s', 'response_reading_time_s', 'duration_s',
        'chat_to_code_latency_s', 'response_reading_latency_s',
    ]
    for key in ROUND_TO_INT:
        if features.get(key) is not None:
            features[key] = round(features[key])

    # delete_type_ratio is stored as .toFixed(2) in the CSV — round to 2 decimals
    if features.get('delete_type_ratio') is not None:
        features['delete_type_ratio'] = round(features['delete_type_ratio'], 2)

    return features