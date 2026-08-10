from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from analysis.clustering import reached_all_pass
from analysis.windows import truncate_at_first_all_pass


PROFILE_ORDER = (
    "Passive Re-querying",
    "Active Testing",
    "Untested Editing",
)


def classify_interval(features: dict[str, Any]) -> str:
    """Assign a valid W2 interval to one mutually exclusive behavior category."""
    edits = float(features.get("code_edits") or 0)
    runs = float(features.get("terminal_runs") or 0)
    if runs > 0:
        return "tested"
    if edits > 0:
        return "active_untested"
    return "passive"


def longest_streak(sequence: list[str], target: str) -> int:
    best = current = 0
    for value in sequence:
        if value == target:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def build_w3_session_rows(
    raw: dict[str, dict[str, Any]],
    w2_windows: list[dict[str, Any]],
    min_intervals: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aggregate valid W2 intervals into one Window 3 row per eligible learner."""
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for window in w2_windows:
        by_session[window["session_id"]].append(window)
    for windows in by_session.values():
        windows.sort(key=lambda w: w.get("gap_index", 0))

    rows: list[dict[str, Any]] = []
    excluded_too_few = 0
    for session_id, student in raw.items():
        intervals = by_session.get(session_id, [])
        if len(intervals) < min_intervals:
            excluded_too_few += 1
            continue

        categories = [classify_interval(w["features"]) for w in intervals]
        n_intervals = len(categories)
        passive_n = categories.count("passive")
        tested_n = categories.count("tested")
        active_untested_n = categories.count("active_untested")
        passive_streak = longest_streak(categories, "passive")

        truncated_events = truncate_at_first_all_pass(student["events"])
        rows.append({
            "session_id": session_id,
            "deployment": student.get("deployment", ""),
            "n_intervals": n_intervals,
            "query_count": sum(e["type"] == "CHAT_QUERY" for e in truncated_events),
            "passive_rate": passive_n / n_intervals,
            "tested_rate": tested_n / n_intervals,
            "active_untested_rate": active_untested_n / n_intervals,
            "longest_passive_streak": passive_streak,
            "longest_passive_streak_rate": passive_streak / n_intervals,
            "q2_passive": int(categories[0] == "passive"),
            "completed": int(reached_all_pass(student["events"])),
            "terminal_runs_total": sum(e["type"] == "TERMINAL_RUN" for e in truncated_events),
            "terminal_errors_total": sum(e["type"] == "TERMINAL_ERROR" for e in truncated_events),
            "categories": categories,
        })

    diagnostics = {
        "raw_sessions": len(raw),
        "sessions_with_any_valid_w2": sum(bool(by_session.get(sid)) for sid in raw),
        "eligible_sessions": len(rows),
        "excluded_below_minimum_intervals": excluded_too_few,
        "minimum_valid_intervals": min_intervals,
    }
    if rows:
        counts = np.asarray([r["n_intervals"] for r in rows], dtype=float)
        diagnostics["interval_count_distribution"] = {
            "min": int(counts.min()),
            "median": float(np.median(counts)),
            "mean": float(counts.mean()),
            "max": int(counts.max()),
        }
    return rows, diagnostics


def _profile_name_map(cluster_summaries: list[dict[str, Any]]) -> dict[int, str]:
    """Name clusters from their dominant interval-rate characteristic."""
    remaining = {summary["cluster"] for summary in cluster_summaries}
    mapping: dict[int, str] = {}

    passive_cluster = max(cluster_summaries, key=lambda s: s["passive_rate"])["cluster"]
    mapping[passive_cluster] = "Passive Re-querying"
    remaining.remove(passive_cluster)

    tested_cluster = max(
        (s for s in cluster_summaries if s["cluster"] in remaining),
        key=lambda s: s["tested_rate"],
    )["cluster"]
    mapping[tested_cluster] = "Active Testing"
    remaining.remove(tested_cluster)

    for cluster in remaining:
        mapping[cluster] = "Untested Editing"
    return mapping


def cluster_w3_session_patterns(
    rows: list[dict[str, Any]],
    ks: tuple[int, ...] = (2, 3, 4, 5, 6),
    n_init: int = 100,
    min_cluster_size: int = 15,
    stability_seeds: int = 20,
) -> dict[str, Any]:
    """Select K by silhouette among behavior-only, minimum-size solutions."""
    feature_names = [
        "passive_rate",
        "tested_rate",
        "active_untested_rate",
        "longest_passive_streak_rate",
    ]
    X_raw = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows],
        dtype=float,
    )
    if len(X_raw) < 2:
        return {
            "feature_names": feature_names,
            "n": len(rows),
            "per_k": {},
            "best_k": None,
        }

    X = StandardScaler().fit_transform(X_raw)
    per_k: dict[int, dict[str, Any]] = {}

    for K in ks:
        if K >= len(X):
            continue
        model = KMeans(n_clusters=K, n_init=n_init, random_state=42).fit(X)
        labels = model.labels_
        if len(set(labels)) < K:
            continue
        sizes = [int((labels == c).sum()) for c in range(K)]
        if min(sizes) < min_cluster_size:
            continue

        seed_aris = []
        for seed in range(stability_seeds):
            seed_labels = KMeans(
                n_clusters=K,
                n_init=20,
                random_state=seed,
            ).fit_predict(X)
            seed_aris.append(adjusted_rand_score(labels, seed_labels))

        summaries = []
        for c in range(K):
            indices = np.where(labels == c)[0]
            members = [rows[i] for i in indices]
            summaries.append({
                "cluster": c,
                "n": len(members),
                "passive_rate": float(np.mean([r["passive_rate"] for r in members])),
                "tested_rate": float(np.mean([r["tested_rate"] for r in members])),
                "active_untested_rate": float(np.mean([r["active_untested_rate"] for r in members])),
                "longest_passive_streak_rate": float(np.mean([r["longest_passive_streak_rate"] for r in members])),
                "q2_passive_rate": float(np.mean([r["q2_passive"] for r in members])),
                "mean_valid_intervals": float(np.mean([r["n_intervals"] for r in members])),
                "completion_rate": float(np.mean([r["completed"] for r in members])),
                "mean_queries": float(np.mean([r["query_count"] for r in members])),
                "mean_runs": float(np.mean([r["terminal_runs_total"] for r in members])),
                "mean_errors": float(np.mean([r["terminal_errors_total"] for r in members])),
            })

        per_k[K] = {
            "silhouette": float(silhouette_score(X, labels)),
            "sizes": sizes,
            "labels": labels.tolist(),
            "centroids_standardized": model.cluster_centers_.tolist(),
            "seed_ari_mean": float(np.mean(seed_aris)),
            "seed_ari_min": float(np.min(seed_aris)),
            "cluster_summaries": summaries,
        }

    best_k = max(per_k, key=lambda k: per_k[k]["silhouette"]) if per_k else None
    result: dict[str, Any] = {
        "feature_names": feature_names,
        "n": len(rows),
        "per_k": per_k,
        "best_k": best_k,
        "selection": "highest silhouette among K with every cluster >= minimum size",
        "completion_used_for_clustering": False,
    }

    if best_k is not None:
        best = per_k[best_k]
        name_map = _profile_name_map(best["cluster_summaries"])
        for summary in best["cluster_summaries"]:
            summary["profile"] = name_map[summary["cluster"]]
        assignments = []
        for row, cluster in zip(rows, best["labels"]):
            assignments.append({
                "session_id": row["session_id"],
                "deployment": row["deployment"],
                "cluster": cluster,
                "profile": name_map[cluster],
                "n_intervals": row["n_intervals"],
                "passive_rate": row["passive_rate"],
                "tested_rate": row["tested_rate"],
                "active_untested_rate": row["active_untested_rate"],
                "longest_passive_streak_rate": row["longest_passive_streak_rate"],
                "q2_passive": row["q2_passive"],
                "completed": row["completed"],
            })
        result["profile_name_map"] = name_map
        result["assignments"] = assignments
    return result