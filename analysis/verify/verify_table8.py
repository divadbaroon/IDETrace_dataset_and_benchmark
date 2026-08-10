import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on path
import csv
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from analysis.windows import compute_w1_windows, compute_w2_windows

W1_FEATURES = ("time_in_editor_s", "time_in_terminal_s", "time_in_chat_s")
W2_FEATURES = ("time_in_editor_s", "thinking_time_s", "error_self_fix")


def load():
    raw = {}
    for dep in ("deployment_1", "deployment_2"):
        d = json.load(open(Path(__file__).resolve().parents[2] / f"dataset/raw_telemetry/{dep}.json"))
        for sid, s in d.items():
            raw[f"{dep}:{sid}"] = {**s, "deployment": dep}
    for s in raw.values():
        s["events"].sort(key=lambda e: e["timestamp"])
    labels = {}
    for dep in ("deployment_1", "deployment_2"):
        with open(Path(__file__).resolve().parents[2] / f"dataset/query_labels/{dep}_labels.csv", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                labels[(row["student_id"].strip(), int(row["query_index"]))] = row["query_type"].strip().lower()
    return raw, labels


def fit(rows, features, k):
    x = np.asarray([[r["features"][f] for f in features] for r in rows], dtype=float)
    xs = StandardScaler().fit_transform(x)
    return KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(xs)


def tab(name, pairs):
    g = sum(1 for p in pairs if p == "guided")
    d = sum(1 for p in pairs if p == "dependent")
    n = g + d
    print(f"  {name:12s} labeled n={n:3d}  guided={100*g/n:5.1f}%  dependent={100*d/n:5.1f}%")


def main():
    raw, labels = load()

    def lab(session_id, query_index):
        sid = session_id.split(":", 1)[1]
        return labels.get((sid, query_index))

    w1 = compute_w1_windows(raw)
    cold = [w for w in w1 if w["features"]["code_edits"] == 0 and w["features"]["terminal_runs"] == 0]
    active = [w for w in w1 if w not in cold]
    l1 = fit(active, W1_FEATURES, 2)
    sizes = [int((l1 == c).sum()) for c in range(2)]
    name1 = {sizes.index(max(sizes)): "Oriented", sizes.index(min(sizes)): "Struggling"}
    print("Window 1 (first query = query_index 1):")
    tab("Cold Start", [lab(w["session_id"], 1) for w in cold if lab(w["session_id"], 1)])
    for c in range(2):
        members = [active[i] for i in range(len(active)) if l1[i] == c]
        tab(name1[c], [lab(w["session_id"], 1) for w in members if lab(w["session_id"], 1)])

    w2 = compute_w2_windows(raw)
    passive = [w for w in w2 if w["features"]["code_edits"] == 0 and w["features"]["terminal_runs"] == 0]
    act2 = [w for w in w2 if w not in passive]
    l2 = fit(act2, W2_FEATURES, 3)
    sizes2 = [int((l2 == c).sum()) for c in range(3)]
    order = np.argsort(sizes2)
    prof = {int(order[2]): "Iterating", int(order[1]): "Debugging", int(order[0]): "Spinning"}
    print("Window 2 (following query = gap_index + 2):")
    tab("Passive", [lab(w["session_id"], w["gap_index"] + 2) for w in passive
                    if lab(w["session_id"], w["gap_index"] + 2)])
    for c in (int(order[2]), int(order[1]), int(order[0])):
        members = [act2[i] for i in range(len(act2)) if l2[i] == c]
        tab(prof[c], [lab(w["session_id"], w["gap_index"] + 2) for w in members
                      if lab(w["session_id"], w["gap_index"] + 2)])


if __name__ == "__main__":
    main()