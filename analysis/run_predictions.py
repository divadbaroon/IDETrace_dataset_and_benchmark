from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from analysis.pipeline import (
    as_serializable,
    build_query_imminence_dataset,
    build_query_type_dataset,
    build_sessions,
    evaluate_all_layers,
    load_deployment,
    load_query_labels,
    validate_label_alignment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--include-post-completion", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--include-session-context",
        action="store_true",
        help="Out-of-spec extension: add cumulative-rate and time-since-query features "
             "(NOT the paper's Table 7 configuration).",
    )
    return parser.parse_args()


def write_metadata_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.root
    raw_dir = root / "dataset" / "raw_telemetry"
    label_dir = root / "dataset" / "query_labels"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    d1_sessions = build_sessions(load_deployment(raw_dir / "deployment_1.json"))
    d2_sessions = build_sessions(load_deployment(raw_dir / "deployment_2.json"))
    d1_labels = load_query_labels(label_dir / "deployment_1_labels.csv")
    d2_labels = load_query_labels(label_dir / "deployment_2_labels.csv")

    truncate = not args.include_post_completion

    imminence_train = build_query_imminence_dataset(
        d1_sessions,
        observation_seconds=30,
        step_seconds=5,
        horizon_seconds=60,
        truncate_at_completion=truncate,
        include_session_context=args.include_session_context,
    )
    imminence_test = build_query_imminence_dataset(
        d2_sessions,
        observation_seconds=30,
        step_seconds=5,
        horizon_seconds=60,
        truncate_at_completion=truncate,
        include_session_context=args.include_session_context,
    )

    query_type_train = build_query_type_dataset(
        d1_sessions,
        d1_labels,
        observation_seconds=15,
        truncate_at_completion=truncate,
        include_session_context=args.include_session_context,
    )
    query_type_test = build_query_type_dataset(
        d2_sessions,
        d2_labels,
        observation_seconds=15,
        truncate_at_completion=truncate,
        include_session_context=args.include_session_context,
    )

    imminence_results = evaluate_all_layers(imminence_train, imminence_test, args.random_state)
    query_type_results = evaluate_all_layers(query_type_train, query_type_test, args.random_state)

    payload = {
        "configuration": {
            "train_deployment": 1,
            "test_deployment": 2,
            "observation_window_seconds": 30,
            "query_imminence_horizon_seconds": 60,
            "sliding_step_seconds": 5,
            "query_type_pre_query_window_seconds": 15,
            "include_session_context": args.include_session_context,
            "truncate_at_first_completion": truncate,
            "excluded_predictor_event_types": [
                "CHAT_TYPE",
                "CHAT_PASTE",
                "CHAT_DELETE",
                "CHAT_QUERY",
            ],
            "model": "RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight='balanced_subsample')",
            "primary_metric": "AUROC",
        },
        "label_alignment": {
            "deployment_1": validate_label_alignment(d1_sessions, d1_labels),
            "deployment_2": validate_label_alignment(d2_sessions, d2_labels),
        },
        "datasets": {
            "query_imminence": {
                "train_instances": len(imminence_train.labels),
                "test_instances": len(imminence_test.labels),
                "train_positive": int(imminence_train.labels.sum()),
                "test_positive": int(imminence_test.labels.sum()),
                "train_students": len({m["student_id"] for m in imminence_train.metadata}),
                "test_students": len({m["student_id"] for m in imminence_test.metadata}),
            },
            "query_type": {
                "train_instances": len(query_type_train.labels),
                "test_instances": len(query_type_test.labels),
                "train_guided": int(query_type_train.labels.sum()),
                "test_guided": int(query_type_test.labels.sum()),
                "train_dependent": int(len(query_type_train.labels) - query_type_train.labels.sum()),
                "test_dependent": int(len(query_type_test.labels) - query_type_test.labels.sum()),
                "train_students": len({m["student_id"] for m in query_type_train.metadata}),
                "test_students": len({m["student_id"] for m in query_type_test.metadata}),
            },
        },
        "results": {
            "query_imminence": [as_serializable(r) for r in imminence_results],
            "query_type": [as_serializable(r) for r in query_type_results],
        },
    }

    result_path = results_dir / "prediction_results.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_metadata_csv(results_dir / "query_imminence_train_instances.csv", imminence_train.metadata)
    write_metadata_csv(results_dir / "query_imminence_test_instances.csv", imminence_test.metadata)
    write_metadata_csv(results_dir / "query_type_train_instances.csv", query_type_train.metadata)
    write_metadata_csv(results_dir / "query_type_test_instances.csv", query_type_test.metadata)

    lines = [
        "# TutorTrace prediction rerun",
        "",
        "## Configuration",
        "",
        "- Train: Deployment 1; held-out test: Deployment 2.",
        "- Query imminence: 30-second windows, 5-second steps, 60-second horizon.",
        "- Query type: 15 seconds immediately before each labeled query.",
        "- Session-history features included."
        if args.include_session_context
        else "- All features computed strictly within the observation window.",
        "- Sessions are truncated at first all-pass completion." if truncate else "- Post-completion activity is retained.",
        "- CHAT_TYPE, CHAT_PASTE, CHAT_DELETE, and CHAT_QUERY are excluded from predictors.",
        "- Model: Random Forest; primary metric: AUROC.",
        "",
        "## Dataset sizes",
        "",
        f"- Query imminence: {len(imminence_train.labels):,} train windows; {len(imminence_test.labels):,} test windows.",
        f"- Query type: {len(query_type_train.labels):,} train queries; {len(query_type_test.labels):,} test queries.",
        "",
        "## Held-out results",
        "",
        "| Task | Feature layer | AUROC | Average precision | Macro F1 @ .5 |",
        "|---|---|---:|---:|---:|",
    ]
    for task, results in (("Query imminence", imminence_results), ("Query type", query_type_results)):
        for result in results:
            lines.append(
                f"| {task} | {result.feature_layer} | {result.auroc:.3f} | "
                f"{result.average_precision:.3f} | {result.macro_f1_at_05:.3f} |"
            )
    lines.extend([
        "",
        "The 0.5-threshold metrics are included only as diagnostics. AUROC is the primary reported measure.",
    ])
    (results_dir / "prediction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload["datasets"], indent=2))
    print("\nHeld-out AUROC")
    for task, results in (("query_imminence", imminence_results), ("query_type", query_type_results)):
        print(task)
        for result in results:
            print(f"  {result.feature_layer:10s} {result.auroc:.3f}")
    print(f"\nSaved: {result_path}")


if __name__ == "__main__":
    main()