from __future__ import annotations

import bisect
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "behavioral_classifier"))
from auto_segmenter import auto_segment_events  # repo's segmenter, unmodified


CHAT_COMPOSITION_EVENTS = {
    "CHAT_TYPE",
    "CHAT_PASTE",
    "CHAT_DELETE",
    "CHAT_QUERY",
}

CODE_EDIT_EVENTS = {
    "CODE_TYPE",
    "CODE_DELETE",
    "CODE_DELETE_SELECTION",
    "CODE_PASTE",
    "CODE_CUT",
    "CODE_UNDO",
    "CODE_REDO",
    "CODE_INDENT",
    "CODE_UNKNOWN",
}

DELETE_EVENTS = {
    "CODE_DELETE",
    "CODE_DELETE_SELECTION",
    "CODE_CUT",
}

INSERT_EVENTS = {
    "CODE_TYPE",
    "CODE_PASTE",
    "CODE_INDENT",
}

BEHAVIORS = ("thinking", "implementing", "debugging", "testing", "seekingHelp")


@dataclass(frozen=True)
class StudentSession:
    student_id: str
    events: tuple[dict[str, Any], ...]
    timestamps: tuple[int, ...]
    safe_events: tuple[dict[str, Any], ...]
    safe_timestamps: tuple[int, ...]
    query_timestamps: tuple[int, ...]
    start_time: int
    observed_end_time: int
    completion_time: int | None
    segments: tuple[dict[str, Any], ...]

    @property
    def analysis_end_time(self) -> int:
        if self.completion_time is None:
            return self.observed_end_time
        return min(self.completion_time, self.observed_end_time)


@dataclass
class PredictionDataset:
    task: str
    feature_dicts: dict[str, list[dict[str, float]]]
    labels: np.ndarray
    metadata: list[dict[str, Any]]


@dataclass
class ModelResult:
    feature_layer: str
    n_train: int
    n_test: int
    train_positive_rate: float
    test_positive_rate: float
    auroc: float
    average_precision: float
    macro_f1_at_05: float
    accuracy_at_05: float
    confusion_matrix_at_05: list[list[int]]
    n_features: int
    top_features: list[dict[str, float]]


def load_deployment(path: str | Path) -> dict[str, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if isinstance(raw, dict) and "students" in raw:
        return {
            str(item["student_id"]): {
                "student_name": item.get("name", f"Student {item['student_id']}"),
                "event_count": item.get("event_count", len(item.get("events", []))),
                "events": item.get("events", []),
            }
            for item in raw["students"]
        }

    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported deployment structure in {path}")
    return {str(k): v for k, v in raw.items()}


def first_completion_time(events: Sequence[Mapping[str, Any]]) -> int | None:
    for event in events:
        if event.get("type") != "TEST_CASE_RESULT":
            continue
        payload = event.get("payload", {})
        total = int(payload.get("total_tests", 0) or 0)
        passed = int(payload.get("passed_count", 0) or 0)
        if total > 0 and passed == total:
            return int(event["timestamp"])
    return None


def build_sessions(deployment: Mapping[str, Mapping[str, Any]]) -> dict[str, StudentSession]:
    sessions: dict[str, StudentSession] = {}
    for student_id, record in deployment.items():
        events = tuple(sorted(record.get("events", []), key=lambda e: int(e.get("timestamp", 0))))
        if not events:
            continue
        start = int(events[0]["timestamp"])
        end = int(events[-1]["timestamp"])
        safe_events = tuple(e for e in events if e.get("type") not in CHAT_COMPOSITION_EVENTS)
        safe_timestamps = tuple(int(e["timestamp"]) for e in safe_events)
        query_times = tuple(int(e["timestamp"]) for e in events if e.get("type") == "CHAT_QUERY")
        duration = max(0, end - start)
        segments = tuple(auto_segment_events(list(safe_events), start, duration)) if safe_events else tuple()
        sessions[str(student_id)] = StudentSession(
            student_id=str(student_id),
            events=events,
            timestamps=tuple(int(e["timestamp"]) for e in events),
            safe_events=safe_events,
            safe_timestamps=safe_timestamps,
            query_timestamps=query_times,
            start_time=start,
            observed_end_time=end,
            completion_time=first_completion_time(events),
            segments=segments,
        )
    return sessions


def load_query_labels(path: str | Path) -> dict[tuple[str, int], str]:
    labels: dict[tuple[str, int], str] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            student_id = str(row.get("student_id", "")).strip()
            query_index_raw = str(row.get("query_index", "")).strip()
            label = str(row.get("query_type", "")).strip().lower()
            if not student_id or not query_index_raw:
                continue
            if label not in {"guided", "dependent"}:
                continue
            labels[(student_id, int(query_index_raw))] = label
    return labels


def _slice_by_time(
    events: Sequence[dict[str, Any]],
    timestamps: Sequence[int],
    start_ms: int,
    end_ms: int,
) -> list[dict[str, Any]]:
    left = bisect.bisect_left(timestamps, start_ms)
    right = bisect.bisect_left(timestamps, end_ms)
    return list(events[left:right])


def _raw_features(events: Sequence[Mapping[str, Any]], window_seconds: float) -> dict[str, float]:
    counts = Counter(str(e.get("type", "UNKNOWN")) for e in events)
    result = {f"raw__{event_type}": float(count) for event_type, count in counts.items()}
    result["raw__event_count"] = float(len(events))
    result["raw__event_density_per_s"] = float(len(events)) / max(1e-6, window_seconds)
    return result


def _char_counts(events: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    inserted = 0
    deleted = 0
    for event in events:
        for change in event.get("payload", {}).get("changes", []) or []:
            text = str(change.get("text", ""))
            inserted += len(text)
            try:
                deleted += max(0, int(change.get("to", 0)) - int(change.get("from", 0)))
            except (TypeError, ValueError):
                continue
    return inserted, deleted


def _region_times(events: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    region_events = [e for e in events if e.get("payload", {}).get("region")]
    totals: Counter[str] = Counter()
    for prev, current in zip(region_events, region_events[1:]):
        gap_ms = int(current["timestamp"]) - int(prev["timestamp"])
        if 0 < gap_ms < 30_000:
            totals[str(prev.get("payload", {}).get("region"))] += gap_ms / 1000.0
    return {
        "editor": float(totals.get("CODE_EDITOR", 0.0)),
        "terminal": float(totals.get("TERMINAL", 0.0)),
        "chat": float(totals.get("CHAT_WINDOW", 0.0)),
        "task": float(totals.get("TASK_DESCRIPTION", 0.0)),
        "tests": float(totals.get("TEST_CASES", 0.0)),
    }


def _segments_in_window(
    session: StudentSession,
    start_ms: int,
    end_ms: int,
) -> list[tuple[dict[str, Any], int, int]]:
    rel_start = start_ms - session.start_time
    rel_end = end_ms - session.start_time
    output: list[tuple[dict[str, Any], int, int]] = []
    for segment in session.segments:
        seg_start = int(segment.get("startTime", 0))
        seg_end = int(segment.get("endTime", seg_start))
        overlap_start = max(rel_start, seg_start)
        overlap_end = min(rel_end, seg_end)
        if overlap_end > overlap_start:
            output.append((segment, overlap_start, overlap_end))
    return output


def _sequence_features(session: StudentSession, start_ms: int, end_ms: int) -> dict[str, float]:
    overlapping = _segments_in_window(session, start_ms, end_ms)
    window_seconds = max(1e-6, (end_ms - start_ms) / 1000.0)
    features: dict[str, float] = {}
    durations: Counter[str] = Counter()
    labels: list[str] = []

    for segment, overlap_start, overlap_end in overlapping:
        behavior = str((segment.get("suggestedBehavior") or {}).get("id", "unknown"))
        durations[behavior] += (overlap_end - overlap_start) / 1000.0
        if not labels or labels[-1] != behavior:
            labels.append(behavior)

    for behavior in BEHAVIORS:
        duration = float(durations.get(behavior, 0.0))
        features[f"sequence__duration_{behavior}_s"] = duration
        features[f"sequence__proportion_{behavior}"] = duration / window_seconds
        features[f"sequence__present_{behavior}"] = float(duration > 0)

    features["sequence__segment_count"] = float(len(labels))
    features["sequence__transition_count"] = float(max(0, len(labels) - 1))

    if labels:
        features[f"sequence__current_{labels[-1]}"] = 1.0
    else:
        features["sequence__current_unknown"] = 1.0
    if len(labels) >= 2:
        features[f"sequence__previous_{labels[-2]}"] = 1.0
    else:
        features["sequence__previous_none"] = 1.0

    for left, right in zip(labels, labels[1:]):
        key = f"sequence__transition_{left}_to_{right}"
        features[key] = features.get(key, 0.0) + 1.0

    # Preserve a compact suffix of the ordered sequence for tabular models.
    # Position 0 is the most recent behavior, position 1 the preceding one, etc.
    labeled_overlaps = []
    for segment, overlap_start, overlap_end in overlapping:
        behavior = str((segment.get("suggestedBehavior") or {}).get("id", "unknown"))
        duration_s = (overlap_end - overlap_start) / 1000.0
        if labeled_overlaps and labeled_overlaps[-1][0] == behavior:
            labeled_overlaps[-1] = (behavior, labeled_overlaps[-1][1] + duration_s)
        else:
            labeled_overlaps.append((behavior, duration_s))
    for position, (behavior, duration_s) in enumerate(reversed(labeled_overlaps[-8:])):
        features[f"sequence__recent_{position}_{behavior}"] = 1.0
        features[f"sequence__recent_{position}_duration_s"] = float(duration_s)

    # Thinking subtypes are useful ordered context but remain telemetry-derived.
    subtype_durations: Counter[str] = Counter()
    for segment, overlap_start, overlap_end in overlapping:
        subtype = segment.get("suggestedThinkingSubcategory")
        if subtype:
            subtype_durations[str(subtype)] += (overlap_end - overlap_start) / 1000.0
    for subtype, duration in subtype_durations.items():
        features[f"sequence__thinking_subtype_{subtype}_s"] = float(duration)

    return features


def _next_event_latency(
    events: Sequence[Mapping[str, Any]],
    origin_index: int,
    accepted_types: set[str],
) -> float | None:
    origin = int(events[origin_index]["timestamp"])
    for later in events[origin_index + 1 :]:
        if str(later.get("type")) in accepted_types:
            return max(0.0, (int(later["timestamp"]) - origin) / 1000.0)
    return None


def _observable_features(
    session: StudentSession,
    window_events: Sequence[dict[str, Any]],
    start_ms: int,
    end_ms: int,
    include_session_context: bool = False,
) -> dict[str, float]:
    window_seconds = max(1e-6, (end_ms - start_ms) / 1000.0)
    counts = Counter(str(e.get("type", "UNKNOWN")) for e in window_events)
    inserted, deleted = _char_counts(window_events)
    code_edits = sum(counts[t] for t in CODE_EDIT_EVENTS)
    deletes = sum(counts[t] for t in DELETE_EVENTS)
    insert_event_count = sum(counts[t] for t in INSERT_EVENTS)
    run_times = [int(e["timestamp"]) for e in window_events if e.get("type") == "TERMINAL_RUN"]

    timestamps = [int(e["timestamp"]) for e in window_events]
    longest_idle = 30.0 if not timestamps else max(
        [max(0.0, (timestamps[0] - start_ms) / 1000.0)]
        + [max(0.0, (b - a) / 1000.0) for a, b in zip(timestamps, timestamps[1:])]
        + [max(0.0, (end_ms - timestamps[-1]) / 1000.0)]
    )

    if len(run_times) >= 2:
        mean_between_runs = float(np.mean(np.diff(run_times))) / 1000.0
    else:
        mean_between_runs = 30.0

    max_consecutive_errors = 0
    current_errors = 0
    for event in window_events:
        event_type = str(event.get("type"))
        if event_type == "TERMINAL_ERROR":
            current_errors += 1
            max_consecutive_errors = max(max_consecutive_errors, current_errors)
        elif event_type == "TEST_CASE_RESULT":
            payload = event.get("payload", {})
            if int(payload.get("passed_count", 0) or 0) > 0:
                current_errors = 0
        elif event_type in CODE_EDIT_EVENTS:
            current_errors = 0

    error_to_edits: list[float] = []
    failed_to_edits: list[float] = []
    error_reading: list[float] = []
    error_self_fix = 0
    failed_self_fix = 0

    for idx, event in enumerate(window_events):
        event_type = str(event.get("type"))
        is_failed_test = False
        if event_type == "TEST_CASE_RESULT":
            payload = event.get("payload", {})
            total = int(payload.get("total_tests", 0) or 0)
            passed = int(payload.get("passed_count", 0) or 0)
            is_failed_test = total > 0 and passed < total

        if event_type == "TERMINAL_ERROR":
            latency = _next_event_latency(window_events, idx, CODE_EDIT_EVENTS)
            if latency is not None:
                error_to_edits.append(latency)
                error_self_fix += 1
            origin = int(event["timestamp"])
            for later in window_events[idx + 1 :]:
                if str(later.get("type")) not in {"MOUSE_MOVE", "MOUSE_CLICK", "TAB_STATE"}:
                    error_reading.append((int(later["timestamp"]) - origin) / 1000.0)
                    break

        if is_failed_test:
            latency = _next_event_latency(window_events, idx, CODE_EDIT_EVENTS)
            if latency is not None:
                failed_to_edits.append(latency)
                failed_self_fix += 1

    region = _region_times(window_events)
    sequence = _sequence_features(session, start_ms, end_ms)

    result = {
        "metric__code_edits": float(code_edits),
        "metric__code_edit_rate": float(code_edits) / window_seconds,
        "metric__chars_inserted": float(inserted),
        "metric__chars_deleted": float(deleted),
        "metric__code_deletes": float(deletes),
        "metric__net_code_growth": float(inserted - deleted),
        "metric__delete_type_ratio": float(deletes) / max(1.0, float(insert_event_count)),
        "metric__terminal_runs": float(counts["TERMINAL_RUN"]),
        "metric__terminal_errors": float(counts["TERMINAL_ERROR"]),
        "metric__max_consecutive_errors": float(max_consecutive_errors),
        "metric__mean_time_between_runs_s": float(mean_between_runs),
        "metric__error_self_fix": float(error_self_fix),
        "metric__failed_test_self_fix": float(failed_self_fix),
        "metric__error_to_edit_s": float(np.mean(error_to_edits)) if error_to_edits else 30.0,
        "metric__failed_test_to_edit_s": float(np.mean(failed_to_edits)) if failed_to_edits else 30.0,
        "metric__error_reading_time_s": float(np.mean(error_reading)) if error_reading else 0.0,
        "metric__time_in_editor_s": region["editor"],
        "metric__time_in_terminal_s": region["terminal"],
        "metric__time_in_chat_s": region["chat"],
        "metric__time_in_task_s": region["task"],
        "metric__time_in_tests_s": region["tests"],
        "metric__longest_idle_s": float(longest_idle),
        "metric__thinking_time_s": float(sequence.get("sequence__duration_thinking_s", 0.0)),
        "metric__seeking_help_time_s": float(sequence.get("sequence__duration_seekingHelp_s", 0.0)),
        "metric__response_reading_time_s": float(sequence.get("sequence__thinking_subtype_thinking-llm_s", 0.0)),
    }

    if include_session_context:
        # Session-history extension (NOT part of the paper's Table 7 configuration,
        # which computes features strictly within the observation window). Enable
        # via --include-session-context for the documented out-of-spec variant.
        prior_query_index = bisect.bisect_left(session.query_timestamps, end_ms) - 1
        if prior_query_index >= 0:
            time_since_query = (end_ms - session.query_timestamps[prior_query_index]) / 1000.0
        else:
            time_since_query = (end_ms - session.start_time) / 1000.0
        history_safe = _slice_by_time(
            session.safe_events,
            session.safe_timestamps,
            session.start_time,
            end_ms,
        )
        history_code_edits = sum(1 for e in history_safe if str(e.get("type")) in CODE_EDIT_EVENTS)
        elapsed_s = max(1e-6, (end_ms - session.start_time) / 1000.0)
        prior_queries = bisect.bisect_left(session.query_timestamps, end_ms)
        result.update({
            "metric__cum_code_rate": float(history_code_edits) / elapsed_s,
            "metric__cum_query_rate": float(prior_queries) / elapsed_s,
            "metric__time_since_query_s": float(time_since_query),
            "metric__time_since_start_s": float(elapsed_s),
        })
    return result


def build_feature_layers(
    session: StudentSession,
    start_ms: int,
    end_ms: int,
    include_session_context: bool = False,
) -> dict[str, dict[str, float]]:
    window_events = _slice_by_time(
        session.safe_events,
        session.safe_timestamps,
        start_ms,
        end_ms,
    )
    window_seconds = max(1e-6, (end_ms - start_ms) / 1000.0)
    raw = _raw_features(window_events, window_seconds)
    metrics = _observable_features(
        session, window_events, start_ms, end_ms,
        include_session_context=include_session_context,
    )
    sequences = _sequence_features(session, start_ms, end_ms)

    raw_layer = dict(raw)
    metric_layer = {**raw_layer, **metrics}
    sequence_layer = {**metric_layer, **sequences}
    return {
        "raw": raw_layer,
        "observable": metric_layer,
        "behavioral": sequence_layer,
    }


def build_query_imminence_dataset(
    sessions: Mapping[str, StudentSession],
    observation_seconds: int = 30,
    step_seconds: int = 5,
    horizon_seconds: int = 60,
    truncate_at_completion: bool = True,
    include_session_context: bool = False,
) -> PredictionDataset:
    layers = {"raw": [], "observable": [], "behavioral": []}
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []

    observation_ms = observation_seconds * 1000
    step_ms = step_seconds * 1000
    horizon_ms = horizon_seconds * 1000

    for student_id in sorted(sessions):
        session = sessions[student_id]
        analysis_end = session.analysis_end_time if truncate_at_completion else session.observed_end_time
        first_end = session.start_time + observation_ms
        last_end = analysis_end - horizon_ms
        if last_end < first_end:
            continue

        end_ms = first_end
        while end_ms <= last_end:
            next_query_idx = bisect.bisect_right(session.query_timestamps, end_ms)
            positive = 0
            next_query_time: int | None = None
            if next_query_idx < len(session.query_timestamps):
                candidate = session.query_timestamps[next_query_idx]
                if candidate <= end_ms + horizon_ms and candidate <= analysis_end:
                    positive = 1
                    next_query_time = candidate

            feature_layers = build_feature_layers(
                session, end_ms - observation_ms, end_ms,
                include_session_context=include_session_context,
            )
            for name in layers:
                layers[name].append(feature_layers[name])
            labels.append(positive)
            metadata.append(
                {
                    "student_id": student_id,
                    "window_start": end_ms - observation_ms,
                    "window_end": end_ms,
                    "label": positive,
                    "next_query_time": next_query_time,
                }
            )
            end_ms += step_ms

    return PredictionDataset(
        task="query_imminence",
        feature_dicts=layers,
        labels=np.asarray(labels, dtype=np.int8),
        metadata=metadata,
    )


def build_query_type_dataset(
    sessions: Mapping[str, StudentSession],
    labels_by_query: Mapping[tuple[str, int], str],
    observation_seconds: int = 15,
    truncate_at_completion: bool = True,
    include_session_context: bool = False,
) -> PredictionDataset:
    layers = {"raw": [], "observable": [], "behavioral": []}
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    observation_ms = observation_seconds * 1000

    for student_id in sorted(sessions):
        session = sessions[student_id]
        analysis_end = session.analysis_end_time if truncate_at_completion else session.observed_end_time
        for query_index, query_time in enumerate(session.query_timestamps, start=1):
            label_name = labels_by_query.get((student_id, query_index))
            if label_name not in {"guided", "dependent"}:
                continue
            if query_time > analysis_end:
                continue
            if query_time - session.start_time < observation_ms:
                continue

            feature_layers = build_feature_layers(
                session, query_time - observation_ms, query_time,
                include_session_context=include_session_context,
            )
            for name in layers:
                layers[name].append(feature_layers[name])
            numeric_label = 1 if label_name == "guided" else 0
            labels.append(numeric_label)
            metadata.append(
                {
                    "student_id": student_id,
                    "query_index": query_index,
                    "query_time": query_time,
                    "label": numeric_label,
                    "label_name": label_name,
                }
            )

    return PredictionDataset(
        task="query_type",
        feature_dicts=layers,
        labels=np.asarray(labels, dtype=np.int8),
        metadata=metadata,
    )


def fit_and_evaluate(
    train: PredictionDataset,
    test: PredictionDataset,
    feature_layer: str,
    random_state: int = 42,
) -> ModelResult:
    vectorizer = DictVectorizer(sparse=True, sort=True)
    x_train = vectorizer.fit_transform(train.feature_dicts[feature_layer])
    x_test = vectorizer.transform(test.feature_dicts[feature_layer])
    y_train = train.labels
    y_test = test.labels

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError(f"Both classes are required for {train.task}/{feature_layer}")

    model = RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int8)

    names = vectorizer.get_feature_names_out()
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:20]
    top_features = [
        {"feature": str(names[index]), "importance": float(importances[index])}
        for index in top_indices
        if importances[index] > 0
    ]

    return ModelResult(
        feature_layer=feature_layer,
        n_train=int(len(y_train)),
        n_test=int(len(y_test)),
        train_positive_rate=float(np.mean(y_train)),
        test_positive_rate=float(np.mean(y_test)),
        auroc=float(roc_auc_score(y_test, probabilities)),
        average_precision=float(average_precision_score(y_test, probabilities)),
        macro_f1_at_05=float(f1_score(y_test, predictions, average="macro")),
        accuracy_at_05=float(accuracy_score(y_test, predictions)),
        confusion_matrix_at_05=confusion_matrix(y_test, predictions, labels=[0, 1]).astype(int).tolist(),
        n_features=int(x_train.shape[1]),
        top_features=top_features,
    )


def evaluate_all_layers(
    train: PredictionDataset,
    test: PredictionDataset,
    random_state: int = 42,
) -> list[ModelResult]:
    return [
        fit_and_evaluate(train, test, layer, random_state=random_state)
        for layer in ("raw", "observable", "behavioral")
    ]


def validate_label_alignment(
    sessions: Mapping[str, StudentSession],
    labels: Mapping[tuple[str, int], str],
) -> dict[str, int]:
    raw_query_keys: set[tuple[str, int]] = set()
    for student_id, session in sessions.items():
        raw_query_keys.update((student_id, i) for i in range(1, len(session.query_timestamps) + 1))
    valid_label_keys = {key for key, value in labels.items() if value in {"guided", "dependent"}}
    return {
        "raw_queries": len(raw_query_keys),
        "valid_labels": len(valid_label_keys),
        "queries_without_valid_label": len(raw_query_keys - valid_label_keys),
        "labels_without_query": len(valid_label_keys - raw_query_keys),
    }


def as_serializable(result: ModelResult) -> dict[str, Any]:
    return {
        "feature_layer": result.feature_layer,
        "n_train": result.n_train,
        "n_test": result.n_test,
        "train_positive_rate": result.train_positive_rate,
        "test_positive_rate": result.test_positive_rate,
        "auroc": result.auroc,
        "average_precision": result.average_precision,
        "macro_f1_at_05": result.macro_f1_at_05,
        "accuracy_at_05": result.accuracy_at_05,
        "confusion_matrix_at_05": result.confusion_matrix_at_05,
        "n_features": result.n_features,
        "top_features": result.top_features,
    }