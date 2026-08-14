from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from analysis.feature_extraction import compute_student_features
from analysis.clustering import reached_all_pass
from analysis.windows import (
    _post_response_features,
    build_event_tracker,
    compute_w2_windows,
    get_event_tracker_template,
    get_w2_slices,
    tab_hidden_seconds,
    truncate_at_first_all_pass,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "dataset" / "raw_telemetry"
INTERVENTION_EXPORT = (
    ROOT / "dataset" / "raw_telemetry" / "deployment_4_intervention.json"
)
if not INTERVENTION_EXPORT.exists():
    raise SystemExit(
        "dataset/raw_telemetry/deployment_4_intervention.json (the intervention "
        "source export, wrapped format) was not found. It ships with this repository; "
        "if you have a stripped copy, obtain the export from the authors and "
        "place it at that path."
    )
OUT_JSON = ROOT / "results" / "preliminary_evaluation_results.json"
OUT_MD = ROOT / "results" / "preliminary_evaluation_report.md"

TRAIN_DEPLOYMENTS = ("deployment_1", "deployment_2")
BASELINE_DEPLOYMENT = "deployment_3_baseline"
SELECTED_FEATURES = ("time_in_editor_s", "thinking_time_s", "error_self_fix")
PROFILE_ORDER = ("Passive", "Iterating", "Debugging", "Spinning")
CLUSTER_TO_PROFILE = {0: "Debugging", 1: "Iterating", 2: "Spinning"}


def load_standard_deployment(path: Path, deployment_name: str) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text())
    return {
        str(student_id): {**student, "deployment": deployment_name}
        for student_id, student in raw.items()
    }


def load_wrapped_export(path: Path, deployment_name: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    raw = json.loads(path.read_text())
    metadata = {key: value for key, value in raw.items() if key != "students"}
    students = {
        str(student["student_id"]): {
            "student_name": student.get("name", f"Student {student['student_id']}"),
            "event_count": student.get("event_count", len(student.get("events", []))),
            "events": student.get("events", []),
            "deployment": deployment_name,
        }
        for student in raw["students"]
    }
    return students, metadata


def fit_fixed_w2_model() -> tuple[StandardScaler, KMeans]:
    training: dict[str, dict[str, Any]] = {}
    for deployment_name in TRAIN_DEPLOYMENTS:
        deployment = load_standard_deployment(
            RAW_DIR / f"{deployment_name}.json", deployment_name
        )
        for student_id, student in deployment.items():
            training[f"{deployment_name}:{student_id}"] = student

    windows = compute_w2_windows(training)
    active = [
        window
        for window in windows
        if not (
            window["features"]["code_edits"] == 0
            and window["features"]["terminal_runs"] == 0
        )
    ]
    matrix = np.asarray(
        [[window["features"][feature] for feature in SELECTED_FEATURES] for window in active],
        dtype=float,
    )
    scaler = StandardScaler().fit(matrix)
    model = KMeans(n_clusters=3, n_init=10, random_state=42).fit(scaler.transform(matrix))
    sizes = Counter(int(label) for label in model.labels_)
    expected = {0: 56, 1: 290, 2: 22}
    if dict(sizes) != expected:
        raise RuntimeError(f"Unexpected training cluster sizes: {dict(sizes)} != {expected}")
    return scaler, model


def classify_windows(
    windows: list[dict[str, Any]], scaler: StandardScaler, model: KMeans
) -> list[dict[str, Any]]:
    classified = []
    for window in windows:
        features = window["features"]
        if features["code_edits"] == 0 and features["terminal_runs"] == 0:
            profile = "Passive"
        else:
            vector = [[features[feature] for feature in SELECTED_FEATURES]]
            cluster = int(model.predict(scaler.transform(vector))[0])
            profile = CLUSTER_TO_PROFILE[cluster]
        classified.append({**window, "profile": profile})
    return classified


def build_audit_windows(
    deployment: dict[str, dict[str, Any]],
    *,
    truncate_at_completion: bool,
    require_response: bool,
    max_hidden_seconds: float | None,
) -> list[dict[str, Any]]:
    """Build W2 intervals under explicit legacy/current audit switches."""
    tracker_template = get_event_tracker_template()
    windows: list[dict[str, Any]] = []
    for session_id, student in deployment.items():
        events = sorted(student["events"], key=lambda event: event["timestamp"])
        if truncate_at_completion:
            events = truncate_at_first_all_pass(events)
        for gap_index, item in enumerate(get_w2_slices(events)):
            interval = item["events"]
            next_query_timestamp = item["next_query_timestamp"]
            if len(interval) < 2:
                continue
            hidden_seconds = tab_hidden_seconds(interval)
            if max_hidden_seconds is not None and hidden_seconds > max_hidden_seconds:
                continue
            post_response = _post_response_features(interval, next_query_timestamp)
            if require_response and post_response is None:
                continue
            tracker = build_event_tracker(interval, tracker_template)
            features = compute_student_features(interval, tracker)
            if post_response is not None:
                features.update(post_response)
            features["query_to_query_s"] = (
                next_query_timestamp - interval[0]["timestamp"]
            ) / 1000.0
            features["tab_hidden_s"] = hidden_seconds
            windows.append(
                {
                    "session_id": session_id,
                    "student_name": student["student_name"],
                    "deployment": student["deployment"],
                    "window": "W2",
                    "gap_index": gap_index,
                    "features": features,
                }
            )
    return windows


def ai_user_summary(deployment: dict[str, dict[str, Any]]) -> dict[str, Any]:
    users = []
    for session_id, student in deployment.items():
        events = sorted(student["events"], key=lambda event: event["timestamp"])
        truncated = truncate_at_first_all_pass(events)
        query_count = sum(event["type"] == "CHAT_QUERY" for event in truncated)
        if query_count == 0:
            continue
        users.append(
            {
                "session_id": session_id,
                "query_count": query_count,
                "completed": bool(reached_all_pass(events)),
            }
        )
    completed = sum(user["completed"] for user in users)
    return {
        "n": len(users),
        "completed": completed,
        "completion_rate": completed / len(users) if users else math.nan,
        "queries_before_completion": sum(user["query_count"] for user in users),
    }


def profile_summary(classified: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(window["profile"] for window in classified)
    total = len(classified)
    return {
        profile: {
            "n": counts[profile],
            "proportion": counts[profile] / total if total else math.nan,
        }
        for profile in PROFILE_ORDER
    }


def metric_summary(windows: list[dict[str, Any]]) -> dict[str, float]:
    def mean(feature: str) -> float:
        return float(np.mean([window["features"][feature] for window in windows]))

    return {
        "code_edits": mean("code_edits"),
        "terminal_runs": mean("terminal_runs"),
        "working_time_s": mean("session_duration_s"),
        "query_to_query_s": mean("query_to_query_s")
        if windows and "query_to_query_s" in windows[0]["features"]
        else float("nan"),
    }


def learner_level_rows(classified: list[dict[str, Any]]) -> list[dict[str, float]]:
    by_learner: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for window in classified:
        by_learner[window["session_id"]].append(window)

    rows = []
    for learner, windows in by_learner.items():
        rows.append(
            {
                "learner": learner,
                "n_windows": len(windows),
                "passive_rate": float(np.mean([w["profile"] == "Passive" for w in windows])),
                "mean_code_edits": float(np.mean([w["features"]["code_edits"] for w in windows])),
                "mean_terminal_runs": float(np.mean([w["features"]["terminal_runs"] for w in windows])),
                "mean_working_time_s": float(np.mean([w["features"]["session_duration_s"] for w in windows])),
            }
        )
    return rows


def mann_whitney(baseline: list[dict[str, float]], intervention: list[dict[str, float]], key: str) -> dict[str, float]:
    left = np.asarray([row[key] for row in baseline], dtype=float)
    right = np.asarray([row[key] for row in intervention], dtype=float)
    statistic, p_value = mannwhitneyu(left, right, alternative="two-sided")
    return {
        "baseline_mean": float(left.mean()),
        "intervention_mean": float(right.mean()),
        "baseline_median": float(np.median(left)),
        "intervention_median": float(np.median(right)),
        "U": float(statistic),
        "p": float(p_value),
    }


def current_comparison(
    baseline: dict[str, dict[str, Any]],
    intervention: dict[str, dict[str, Any]],
    scaler: StandardScaler,
    model: KMeans,
) -> dict[str, Any]:
    baseline_windows = compute_w2_windows(baseline)
    intervention_windows = compute_w2_windows(intervention)
    baseline_classified = classify_windows(baseline_windows, scaler, model)
    intervention_classified = classify_windows(intervention_windows, scaler, model)

    baseline_counts = Counter(window["profile"] for window in baseline_classified)
    intervention_counts = Counter(window["profile"] for window in intervention_classified)
    contingency = np.asarray(
        [
            [baseline_counts[profile] for profile in PROFILE_ORDER],
            [intervention_counts[profile] for profile in PROFILE_ORDER],
        ],
        dtype=int,
    )
    chi2, p_value, degrees_freedom, _ = chi2_contingency(contingency)
    cramers_v = math.sqrt(chi2 / (contingency.sum() * min(contingency.shape[0] - 1, contingency.shape[1] - 1)))

    baseline_learners = learner_level_rows(baseline_classified)
    intervention_learners = learner_level_rows(intervention_classified)

    baseline_ai = ai_user_summary(baseline)
    intervention_ai = ai_user_summary(intervention)
    completion_odds, completion_p = fisher_exact(
        [
            [baseline_ai["completed"], baseline_ai["n"] - baseline_ai["completed"]],
            [intervention_ai["completed"], intervention_ai["n"] - intervention_ai["completed"]],
        ]
    )

    return {
        "method": {
            "truncate_at_first_all_pass": True,
            "require_preceding_chat_response": True,
            "maximum_tab_hidden_seconds": 30,
            "active_profile_model": "fixed W2 scaler and K-means fit on deployments 1-2",
            "selected_features": list(SELECTED_FEATURES),
        },
        "baseline": {
            "raw_student_records": len(baseline),
            "ai_users": baseline_ai,
            "valid_w2_learners": len(baseline_learners),
            "valid_w2_windows": len(baseline_classified),
            "profiles": profile_summary(baseline_classified),
            "window_means": metric_summary(baseline_windows),
        },
        "intervention": {
            "raw_student_records": len(intervention),
            "ai_users": intervention_ai,
            "valid_w2_learners": len(intervention_learners),
            "valid_w2_windows": len(intervention_classified),
            "profiles": profile_summary(intervention_classified),
            "window_means": metric_summary(intervention_windows),
        },
        "profile_distribution_test": {
            "contingency_rows": ["baseline", "intervention"],
            "contingency_columns": list(PROFILE_ORDER),
            "counts": contingency.tolist(),
            "chi2": float(chi2),
            "df": int(degrees_freedom),
            "p": float(p_value),
            "cramers_v": float(cramers_v),
            "warning": "Window-level chi-square treats repeated windows from the same learner as independent.",
        },
        "learner_level_tests": {
            "baseline_learners": len(baseline_learners),
            "intervention_learners": len(intervention_learners),
            "passive_rate": mann_whitney(baseline_learners, intervention_learners, "passive_rate"),
            "mean_code_edits": mann_whitney(baseline_learners, intervention_learners, "mean_code_edits"),
            "mean_terminal_runs": mann_whitney(baseline_learners, intervention_learners, "mean_terminal_runs"),
            "mean_working_time_s": mann_whitney(baseline_learners, intervention_learners, "mean_working_time_s"),
        },
        "completion_test": {
            "odds_ratio": float(completion_odds),
            "fisher_exact_p": float(completion_p),
            "warning": "Completion is confounded with deployment/session and is not a causal learning-outcome estimate.",
        },
    }


def audit_variant(
    baseline: dict[str, dict[str, Any]],
    intervention: dict[str, dict[str, Any]],
    scaler: StandardScaler,
    model: KMeans,
    *,
    truncate_at_completion: bool,
    require_response: bool,
) -> dict[str, Any]:
    baseline_windows = build_audit_windows(
        baseline,
        truncate_at_completion=truncate_at_completion,
        require_response=require_response,
        max_hidden_seconds=None,
    )
    intervention_windows = build_audit_windows(
        intervention,
        truncate_at_completion=truncate_at_completion,
        require_response=require_response,
        max_hidden_seconds=None,
    )
    baseline_classified = classify_windows(baseline_windows, scaler, model)
    intervention_classified = classify_windows(intervention_windows, scaler, model)
    return {
        "method": {
            "truncate_at_first_all_pass": truncate_at_completion,
            "require_preceding_chat_response": require_response,
            "maximum_tab_hidden_seconds": None,
        },
        "baseline": {
            "windows": len(baseline_windows),
            "profiles": profile_summary(baseline_classified),
            "window_means": metric_summary(baseline_windows),
        },
        "intervention": {
            "windows": len(intervention_windows),
            "profiles": profile_summary(intervention_classified),
            "window_means": metric_summary(intervention_windows),
        },
    }


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def write_markdown(result: dict[str, Any]) -> None:
    current = result["current_taxonomy_comparison"]
    baseline = current["baseline"]
    intervention = current["intervention"]
    b_profiles = baseline["profiles"]
    i_profiles = intervention["profiles"]
    b_means = baseline["window_means"]
    i_means = intervention["window_means"]
    test = current["profile_distribution_test"]
    learner_tests = current["learner_level_tests"]

    lines = [
        "# Preliminary Evaluation Re-analysis",
        "",
        "## Current taxonomy rules",
        "",
        f"- Baseline: {baseline['ai_users']['n']} pre-completion AI users; "
        f"{baseline['valid_w2_windows']} valid W2 windows from {baseline['valid_w2_learners']} learners.",
        f"- Intervention: {intervention['ai_users']['n']} pre-completion AI users; "
        f"{intervention['valid_w2_windows']} valid W2 windows from {intervention['valid_w2_learners']} learners.",
        f"- Passive: {pct(b_profiles['Passive']['proportion'])} -> {pct(i_profiles['Passive']['proportion'])}.",
        f"- Iterating: {pct(b_profiles['Iterating']['proportion'])} -> {pct(i_profiles['Iterating']['proportion'])}.",
        f"- Debugging: {pct(b_profiles['Debugging']['proportion'])} -> {pct(i_profiles['Debugging']['proportion'])}.",
        f"- Spinning: {pct(b_profiles['Spinning']['proportion'])} -> {pct(i_profiles['Spinning']['proportion'])}.",
        f"- Window-level chi-square: chi2({test['df']})={test['chi2']:.2f}, p={test['p']:.3g}, Cramer's V={test['cramers_v']:.3f}.",
        f"- Mean edits/window: {b_means['code_edits']:.1f} -> {i_means['code_edits']:.1f}.",
        f"- Mean runs/window: {b_means['terminal_runs']:.1f} -> {i_means['terminal_runs']:.1f}.",
        f"- Mean working time/window: {b_means['working_time_s']:.1f}s -> {i_means['working_time_s']:.1f}s.",
        f"- Completion among pre-completion AI users: {pct(baseline['ai_users']['completion_rate'])} "
        f"({baseline['ai_users']['completed']}/{baseline['ai_users']['n']}) -> "
        f"{pct(intervention['ai_users']['completion_rate'])} "
        f"({intervention['ai_users']['completed']}/{intervention['ai_users']['n']}); "
        f"Fisher p={current['completion_test']['fisher_exact_p']:.3f}.",
        "",
        "The chi-square calculation is a descriptive window-level replication and assumes independence between windows. "
        "A learner-level Mann-Whitney comparison of each learner's Passive-window rate also differs between deployments "
        f"(U={learner_tests['passive_rate']['U']:.1f}, p={learner_tests['passive_rate']['p']:.3g}).",
        "",
        "## Sample",
        "",
        f"The intervention export contains {result['source_metadata']['raw_student_records']} student records and "
        f"{intervention['ai_users']['n']} learners who queried before first completion.",
        "",
        "## Interpretation",
        "",
        "Under the current taxonomy rules, the intervention deployment is associated with fewer Passive windows and "
        "more observable work between queries. Because condition is fully confounded with class session and learners "
        "contribute repeated windows, this is preliminary behavioral evidence rather than a causal treatment estimate.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    baseline = load_standard_deployment(
        RAW_DIR / f"{BASELINE_DEPLOYMENT}.json", BASELINE_DEPLOYMENT
    )
    intervention, source_metadata = load_wrapped_export(
        INTERVENTION_EXPORT, "deployment_4_intervention"
    )
    scaler, model = fit_fixed_w2_model()

    result = {
        "source_metadata": {
            **source_metadata,
            "raw_student_records": len(intervention),
            "note": "Wrapped {'students': [...]} export; read directly rather than through the taxonomy loader.",
        },
        "current_taxonomy_comparison": current_comparison(
            baseline, intervention, scaler, model
        ),
        "legacy_audit": {
            "truncate_at_completion_no_tab_or_response_filter": audit_variant(
                baseline,
                intervention,
                scaler,
                model,
                truncate_at_completion=True,
                require_response=False,
            ),
            "full_session_no_tab_or_response_filter": audit_variant(
                baseline,
                intervention,
                scaler,
                model,
                truncate_at_completion=False,
                require_response=False,
            ),
        },
    }

    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2))
    write_markdown(result)
    print(OUT_MD.read_text())
    print(f"Saved {OUT_JSON}")
    print(f"Saved {OUT_MD}")


if __name__ == "__main__":
    main()