import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from main import load_all_deployments
from analysis.windows import compute_w1_windows, compute_w2_windows
from analysis.clustering import reached_all_pass

DEPS = ("deployment_1", "deployment_2")
W1_FEATURES = ("time_in_editor_s", "time_in_terminal_s", "time_in_chat_s")
W2_FEATURES = ("time_in_editor_s", "thinking_time_s", "error_self_fix")


def fit(rows, features, k):
    x = np.asarray([[r["features"][f] for f in features] for r in rows], dtype=float)
    xs = StandardScaler().fit_transform(x)
    return KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(xs)


def describe(rows, outcome, label):
    n = len(rows)
    think = np.mean([r["features"]["thinking_time_s"] for r in rows])
    runs = np.mean([r["features"]["terminal_runs"] for r in rows])
    errs = np.mean([r["features"]["terminal_errors"] for r in rows])
    comp = np.mean([outcome[r["session_id"]] for r in rows])
    print(f"  {label:<18} N={n:<4} Think={think:6.1f}s  Runs={runs:4.1f}  Err={errs:4.1f}  Comp={100*comp:5.1f}%")
    return n


def main():
    all_raw = load_all_deployments()
    raw = {sid: s for sid, s in all_raw.items() if s["deployment"] in DEPS}
    outcome = {sid: reached_all_pass(s["events"]) for sid, s in raw.items()}

    # ---- Window 1 ----
    w1 = compute_w1_windows(raw)
    cold = [w for w in w1 if w["features"]["code_edits"] == 0 and w["features"]["terminal_runs"] == 0]
    active = [w for w in w1 if w not in cold]
    labels = fit(active, W1_FEATURES, 2)
    sizes = [int((labels == c).sum()) for c in range(2)]
    name = {sizes.index(max(sizes)): "Oriented", sizes.index(min(sizes)): "Struggling"}
    print(f"W1 (total {len(w1)}):")
    describe(cold, outcome, "Cold Start")
    for c in range(2):
        describe([active[i] for i in range(len(active)) if labels[i] == c], outcome, name[c])
    for grp, lab in ((cold, "Cold Start"), (active, "active")):
        print(f"    {lab} share: {100*len(grp)/len(w1):.0f}%")

    # ---- Window 2 ----
    w2 = compute_w2_windows(raw)
    passive = [w for w in w2 if w["features"]["code_edits"] == 0 and w["features"]["terminal_runs"] == 0]
    act2 = [w for w in w2 if w not in passive]
    labels2 = fit(act2, W2_FEATURES, 3)
    sizes2 = [int((labels2 == c).sum()) for c in range(3)]
    prof = {}
    order = np.argsort(sizes2)  # ascending: 22,56,290
    prof[int(order[2])] = "Iterating"   # 290
    prof[int(order[1])] = "Debugging"   # 56
    prof[int(order[0])] = "Spinning"    # 22
    print(f"\nW2 (total {len(w2)}):")
    describe(passive, outcome, "Passive")
    for c in (int(order[2]), int(order[1]), int(order[0])):
        describe([act2[i] for i in range(len(act2)) if labels2[i] == c], outcome, prof[c])
    print(f"    Passive share {100*len(passive)/len(w2):.0f}% | Iterating {100*sizes2[int(order[2])]/len(w2):.0f}% | Debugging {100*sizes2[int(order[1])]/len(w2):.0f}% | Spinning {100*sizes2[int(order[0])]/len(w2):.0f}%")


if __name__ == "__main__":
    main()