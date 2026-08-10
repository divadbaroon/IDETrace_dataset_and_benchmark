from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path

import argparse
import bisect
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from analysis.pipeline import (
    build_query_imminence_dataset,
    build_query_type_dataset,
    build_sessions,
    load_deployment,
    load_query_labels,
)
from analysis.windows import compute_w1_windows, compute_w2_windows

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
SWEEP_DIR = RESULTS_DIR / "_sweep"

WINDOWS = (15, 30, 45, 60)
REPORT_WINDOWS = {"imminence": 30, "type": 15}  # Table 7 configurations featured in the appendix
LAYERS = ("raw", "observable", "behavioral")
TASKS = ("imminence", "type")
W1_FEATURES = ("time_in_editor_s", "time_in_terminal_s", "time_in_chat_s")
W2_FEATURES = ("time_in_editor_s", "thinking_time_s", "error_self_fix")
TABLE7 = {  # paper anchors: (task, window) -> {layer: auroc}
    ("imminence", 30): {"raw": 0.689, "observable": 0.726, "behavioral": 0.719},
    ("type", 15): {"raw": 0.690, "observable": 0.717, "behavioral": 0.705},
}


# ─── Cached units ────────────────────────────────────────────────────────

def dataset_path(task: str, window: int, split: str) -> Path:
    return SWEEP_DIR / f"{task}_{window}s_{split}.pkl"


def result_path(task: str, window: int, layer: str) -> Path:
    return SWEEP_DIR / f"{task}_{window}s_{layer}.json"


def get_dataset(task, window, split, sessions, labels):
    path = dataset_path(task, window, split)
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)
    if task == "imminence":
        dataset = build_query_imminence_dataset(sessions, observation_seconds=window)
    else:
        dataset = build_query_type_dataset(sessions, labels, observation_seconds=window)
    with path.open("wb") as handle:
        pickle.dump(dataset, handle)
    print(f"built {path.name}: n={len(dataset.labels)}", flush=True)
    return dataset


def fit_unit(task, window, layer, train, test, random_state):
    path = result_path(task, window, layer)
    if path.exists():
        return json.loads(path.read_text())
    vectorizer = DictVectorizer(sparse=True, sort=True)
    x_train = vectorizer.fit_transform(train.feature_dicts[layer])
    x_test = vectorizer.transform(test.feature_dicts[layer])
    model = RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, train.labels)
    probabilities = model.predict_proba(x_test)[:, 1]
    names = vectorizer.get_feature_names_out()
    order = np.argsort(model.feature_importances_)[::-1]
    contributions = [
        {"feature": str(names[i]), "importance": float(model.feature_importances_[i])}
        for i in order
        if model.feature_importances_[i] > 0
    ]
    payload = {
        "task": task,
        "window_seconds": window,
        "layer": layer,
        "n_train": int(len(train.labels)),
        "n_test": int(len(test.labels)),
        "train_positive_rate": float(np.mean(train.labels)),
        "test_positive_rate": float(np.mean(test.labels)),
        "auroc": float(roc_auc_score(test.labels, probabilities)),
        "n_features": int(x_train.shape[1]),
        "contributions": contributions,
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"fit {path.stem}: AUROC={payload['auroc']:.3f}", flush=True)
    return payload


# ─── Taxonomy profiles (deployments 1-2, published models) ───────────────

def taxonomy_assignments(d1_records, d2_records):
    raw = {}
    for dep_name, records in (("deployment_1", d1_records), ("deployment_2", d2_records)):
        for sid, record in records.items():
            raw[f"{dep_name}:{sid}"] = {**record, "deployment": dep_name}

    def fit(rows, features, k):
        x = np.asarray([[r["features"][f] for f in features] for r in rows], dtype=float)
        return KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(
            StandardScaler().fit_transform(x))

    w1 = compute_w1_windows(raw)
    cold = [w for w in w1 if w["features"]["code_edits"] == 0 and w["features"]["terminal_runs"] == 0]
    active = [w for w in w1 if w not in cold]
    labels1 = fit(active, W1_FEATURES, 2)
    sizes1 = [int((labels1 == c).sum()) for c in range(2)]
    if sorted(sizes1) != [21, 100]:
        raise RuntimeError(f"W1 cluster sizes {sizes1} != published [100, 21]")
    name1 = {sizes1.index(max(sizes1)): "Oriented", sizes1.index(min(sizes1)): "Struggling"}
    w1_profile = {w["session_id"]: "Cold Start" for w in cold}
    for i, w in enumerate(active):
        w1_profile[w["session_id"]] = name1[int(labels1[i])]

    w2 = compute_w2_windows(raw)
    passive = [w for w in w2 if w["features"]["code_edits"] == 0 and w["features"]["terminal_runs"] == 0]
    act2 = [w for w in w2 if w not in passive]
    labels2 = fit(act2, W2_FEATURES, 3)
    sizes2 = [int((labels2 == c).sum()) for c in range(3)]
    if sorted(sizes2) != [22, 56, 290]:
        raise RuntimeError(f"W2 cluster sizes {sizes2} != published [290, 56, 22]")
    order = np.argsort(sizes2)
    name2 = {int(order[2]): "Iterating", int(order[1]): "Debugging", int(order[0]): "Spinning"}
    w2_profile = {(w["session_id"], w["gap_index"]): "Passive" for w in passive}
    for i, w in enumerate(act2):
        w2_profile[(w["session_id"], w["gap_index"])] = name2[int(labels2[i])]
    return w1_profile, w2_profile


def imminence_distribution(test_dataset, sessions, w1_profile, w2_profile):
    counts: Counter[str] = Counter()
    positives: Counter[str] = Counter()
    for m in test_dataset.metadata:
        sid_full = f"deployment_2:{m['student_id']}"
        qts = sessions[m["student_id"]].query_timestamps
        we = m["window_end"]
        if not qts:
            context = "No query this session"
        else:
            idx = bisect.bisect_right(qts, we)
            if idx == 0:
                context = (f"W1: {w1_profile[sid_full]}" if sid_full in w1_profile
                           else "W1 context (excluded from population)")
            elif idx >= len(qts):
                context = "After last query"
            else:
                key = (sid_full, idx - 1)
                context = (f"W2: {w2_profile[key]}" if key in w2_profile
                           else "W2 context (invalid interval)")
        counts[context] += 1
        positives[context] += int(m["label"])
    total = sum(counts.values())
    return [
        {"context": context, "windows": counts[context],
         "share": counts[context] / total,
         "positive_rate": positives[context] / counts[context]}
        for context in sorted(counts, key=lambda c: -counts[c])
    ]


def type_distribution(test_dataset, w1_profile, w2_profile):
    counts: Counter[str] = Counter()
    guided: Counter[str] = Counter()
    for m in test_dataset.metadata:
        sid_full = f"deployment_2:{m['student_id']}"
        if m["query_index"] == 1:
            profile = (f"W1: {w1_profile[sid_full]}" if sid_full in w1_profile
                       else "W1 (excluded from population)")
        else:
            key = (sid_full, m["query_index"] - 2)
            profile = (f"W2: {w2_profile[key]}" if key in w2_profile
                       else "W2 (invalid interval)")
        counts[profile] += 1
        guided[profile] += int(m["label"])
    total = sum(counts.values())
    return [
        {"profile": profile, "queries": counts[profile],
         "share": counts[profile] / total,
         "guided_rate": guided[profile] / counts[profile]}
        for profile in sorted(counts, key=lambda c: -counts[c])
    ]


# ─── Reporting ───────────────────────────────────────────────────────────

def latex_escape_metric(name: str) -> str:
    return "\\metric{" + name + "}"


def emit_latex(sweep, reported, contributions, distributions):
    lines = ["% Auto-generated by run_breakdowns.py -- do not edit by hand.", ""]
    # Window sweep table.
    lines += [
        "\\begin{table}[H]", "\\centering",
        "\\caption{Held-out AUROC across observation-window sizes for both",
        "prediction tasks and all three feature layers. The 30-second",
        "query-imminence row and 15-second help-seeking-type row correspond to",
        "Table~\\ref{tab:ablation}; bold marks the configuration reported",
        "in the subsequent tables.}",
        "\\label{tab:window-sweep}", "\\small",
        "\\setlength{\\tabcolsep}{6pt}", "\\renewcommand{\\arraystretch}{1.12}",
        "\\begin{tabular}{llrrccc}", "\\toprule",
        "\\textbf{Task} & \\textbf{Window} & \\textbf{Test $n$} & \\textbf{Positive}"
        " & \\textbf{Raw} & \\textbf{+Obs.} & \\textbf{+Seq.} \\\\", "\\midrule",
    ]
    for task, task_name in (("imminence", "Query imminence"), ("type", "Help-seeking type")):
        best_window, best_layer, best_auc = reported[task]
        for wi, window in enumerate(WINDOWS):
            row = sweep[task][window]
            cells = []
            for layer in LAYERS:
                value = f"{row[layer]['auroc']:.3f}"
                if window == best_window and layer == best_layer:
                    value = "\\textbf{" + value + "}"
                cells.append(value)
            first = task_name if wi == 0 else ""
            lines.append(
                f"{first} & {window}\\,s & {row['raw']['n_test']:,} & "
                f"{100 * row['raw']['test_positive_rate']:.1f}\\% & "
                + " & ".join(cells) + " \\\\")
        if task == "imminence":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    # Contribution tables (paired columns to halve length).
    for task, task_name in (("imminence", "query imminence"), ("type", "help-seeking type")):
        best_window, best_layer, _ = reported[task]
        contribution = contributions[task]
        half = (len(contribution) + 1) // 2
        left, right = contribution[:half], contribution[half:]
        lines += [
            "\\begin{table}[H]", "\\centering",
            f"\\caption{{Complete feature-contribution list (Random-Forest importances,"
            f" all features with importance $>0$) for {task_name} at its"
            f" reported configuration ({best_window}\\,s window,"
            f" {best_layer} layer).}}",
            f"\\label{{tab:contrib-{task}}}", "\\scriptsize",
            "\\setlength{\\tabcolsep}{4pt}", "\\renewcommand{\\arraystretch}{1.05}",
            "\\begin{tabular}{lr@{\\hspace{18pt}}lr}", "\\toprule",
            "\\textbf{Feature} & \\textbf{Imp.} & \\textbf{Feature} & \\textbf{Imp.} \\\\",
            "\\midrule",
        ]
        for i in range(half):
            l = left[i]
            lcell = f"{latex_escape_metric(l['feature'])} & {l['importance']:.4f}"
            if i < len(right):
                r = right[i]
                rcell = f"{latex_escape_metric(r['feature'])} & {r['importance']:.4f}"
            else:
                rcell = " & "
            lines.append(f"{lcell} & {rcell} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    # Distribution tables at the best window.
    imminence_rows = distributions["imminence"]
    best_window = reported["imminence"][0]
    lines += [
        "\\begin{table}[H]", "\\centering",
        f"\\caption{{Distribution of behavioral contexts among held-out"
        f" query-imminence observation windows at the reported window"
        f" ({best_window}\\,s), with each context's positive rate. Profiles are"
        f" evaluation strata from the deployment-1--2 taxonomy models.}}",
        "\\label{tab:dist-imminence}", "\\small",
        "\\setlength{\\tabcolsep}{6pt}", "\\renewcommand{\\arraystretch}{1.12}",
        "\\begin{tabular}{lrrr}", "\\toprule",
        "\\textbf{Behavioral context} & \\textbf{Windows} & \\textbf{Share}"
        " & \\textbf{Positive rate} \\\\", "\\midrule",
    ]
    for row in imminence_rows:
        lines.append(f"{row['context']} & {row['windows']:,} & "
                     f"{100 * row['share']:.1f}\\% & {100 * row['positive_rate']:.1f}\\% \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    type_rows = distributions["type"]
    best_window = reported["type"][0]
    lines += [
        "\\begin{table}[H]", "\\centering",
        f"\\caption{{Distribution of behavioral profiles among held-out"
        f" help-seeking-type queries at the reported window"
        f" ({best_window}\\,s), with each profile's guided rate.}}",
        "\\label{tab:dist-type}", "\\small",
        "\\setlength{\\tabcolsep}{6pt}", "\\renewcommand{\\arraystretch}{1.12}",
        "\\begin{tabular}{lrrr}", "\\toprule",
        "\\textbf{Preceding profile} & \\textbf{Queries} & \\textbf{Share}"
        " & \\textbf{Guided} \\\\", "\\midrule",
    ]
    for row in type_rows:
        lines.append(f"{row['profile']} & {row['queries']} & "
                     f"{100 * row['share']:.1f}\\% & {100 * row['guided_rate']:.1f}\\% \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (RESULTS_DIR / "appendix_window_sweep_tables.tex").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    if args.rebuild:
        for path in SWEEP_DIR.glob("*"):
            path.unlink()

    d1_records = load_deployment(ROOT / "dataset/raw_telemetry/deployment_1.json")
    d2_records = load_deployment(ROOT / "dataset/raw_telemetry/deployment_2.json")
    d1_sessions = build_sessions(d1_records)
    d2_sessions = build_sessions(d2_records)
    d1_labels = load_query_labels(ROOT / "dataset/query_labels/deployment_1_labels.csv")
    d2_labels = load_query_labels(ROOT / "dataset/query_labels/deployment_2_labels.csv")

    sweep = {task: {} for task in TASKS}
    for task in TASKS:
        for window in WINDOWS:
            done = all(result_path(task, window, layer).exists() for layer in LAYERS)
            if done:
                sweep[task][window] = {
                    layer: json.loads(result_path(task, window, layer).read_text())
                    for layer in LAYERS
                }
                continue
            train = get_dataset(task, window, "train", d1_sessions, d1_labels)
            test = get_dataset(task, window, "test", d2_sessions, d2_labels)
            sweep[task][window] = {
                layer: fit_unit(task, window, layer, train, test, args.random_state)
                for layer in LAYERS
            }

    # Anchor check against the paper's Table 7.
    for (task, window), expected_layers in TABLE7.items():
        for layer, expected in expected_layers.items():
            actual = sweep[task][window][layer]["auroc"]
            if abs(actual - expected) >= 0.002:
                raise RuntimeError(
                    f"Table 7 anchor mismatch: {task}/{window}s/{layer} "
                    f"{actual:.3f} vs {expected:.3f}")
    print("Table 7 anchors confirmed (imminence@30s, type@15s).", flush=True)

    # Sweep maximum (for reference) and the reported configuration.
    sweep_max = {}
    reported = {}
    for task in TASKS:
        sweep_max[task] = max(
            ((window, layer, sweep[task][window][layer]["auroc"])
             for window in WINDOWS for layer in LAYERS),
            key=lambda item: item[2])
        window = REPORT_WINDOWS[task]
        layer = max(LAYERS, key=lambda l: sweep[task][window][l]["auroc"])
        reported[task] = (window, layer, sweep[task][window][layer]["auroc"])

    w1_profile, w2_profile = taxonomy_assignments(d1_records, d2_records)
    imm_best_test = get_dataset("imminence", reported["imminence"][0], "test",
                                d2_sessions, d2_labels)
    type_best_test = get_dataset("type", reported["type"][0], "test",
                                 d2_sessions, d2_labels)
    distributions = {
        "imminence": imminence_distribution(imm_best_test, d2_sessions,
                                            w1_profile, w2_profile),
        "type": type_distribution(type_best_test, w1_profile, w2_profile),
    }
    contributions = {
        task: sweep[task][reported[task][0]][reported[task][1]]["contributions"]
        for task in TASKS
    }

    payload = {
        "configuration": {
            "note": "Paper-faithful configuration; imminence horizon fixed at 60 s; "
                    "profiles are evaluation strata, never model inputs.",
            "windows_seconds": list(WINDOWS),
            "random_state": args.random_state,
        },
        "window_sweep": {
            task: {str(window): {layer: {k: v for k, v in sweep[task][window][layer].items()
                                         if k != "contributions"}
                                 for layer in LAYERS}
                   for window in WINDOWS}
            for task in TASKS
        },
        "reported": {task: {"window_seconds": reported[task][0], "layer": reported[task][1],
                            "auroc": reported[task][2]} for task in TASKS},
        "sweep_max": {task: {"window_seconds": sweep_max[task][0], "layer": sweep_max[task][1],
                             "auroc": sweep_max[task][2]} for task in TASKS},
        "feature_contributions_at_best": contributions,
        "profile_distribution_at_best": distributions,
    }
    (RESULTS_DIR / "prediction_breakdowns.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# Prediction breakdowns: window sweep, contributions, profiles", ""]
    for task, task_name in (("imminence", "Query imminence"), ("type", "Help-seeking type")):
        lines += [f"## {task_name}: held-out AUROC by observation window", "",
                  "| Window | Test n | Positive | Raw | +Obs. | +Seq. |",
                  "|---|---|---|---|---|---|"]
        for window in WINDOWS:
            row = sweep[task][window]
            lines.append(
                f"| {window}s | {row['raw']['n_test']:,} | "
                f"{100 * row['raw']['test_positive_rate']:.1f}% | "
                + " | ".join(f"{row[layer]['auroc']:.3f}" for layer in LAYERS) + " |")
        window, layer, auc = reported[task]
        mw, ml, ma = sweep_max[task]
        lines += ["", f"Reported: {window}s / {layer} (AUROC={auc:.3f}); "
                      f"sweep peak: {mw}s / {ml} (AUROC={ma:.3f}).", "",
                  f"### Complete feature contributions ({window}s, {layer}; "
                  f"{len(contributions[task])} features)", "",
                  "| Feature | Importance |", "|---|---|"]
        lines += [f"| {row['feature']} | {row['importance']:.4f} |"
                  for row in contributions[task]]
        lines += [""]
    lines += ["## Profile distribution at the reported windows", ""]
    lines += ["### Query imminence", "", "| Context | Windows | Share | Positive rate |",
              "|---|---|---|---|"]
    for row in distributions["imminence"]:
        lines.append(f"| {row['context']} | {row['windows']:,} | "
                     f"{100 * row['share']:.1f}% | {100 * row['positive_rate']:.1f}% |")
    lines += ["", "### Help-seeking type", "", "| Preceding profile | Queries | Share | Guided |",
              "|---|---|---|---|"]
    for row in distributions["type"]:
        lines.append(f"| {row['profile']} | {row['queries']} | "
                     f"{100 * row['share']:.1f}% | {100 * row['guided_rate']:.1f}% |")
    (RESULTS_DIR / "prediction_breakdowns.md").write_text("\n".join(lines) + "\n")

    emit_latex(sweep, reported, contributions, distributions)
    print((RESULTS_DIR / "prediction_breakdowns.md").read_text())
    print("Saved prediction_breakdowns.{json,md} and appendix_window_sweep_tables.tex")


if __name__ == "__main__":
    main()