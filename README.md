# TutorTrace

A behavioral telemetry dataset and taxonomy for classifying learner behavioral states in AI-assisted coding environments. The dataset captures fine-grained IDE interactions from 480 students across 4 classroom deployments in two introductory Python courses with an integrated AI tutor.

## Dataset

Approximately 180K telemetry events from 480 students (317 using the AI tutor), organized into three layers:

- **Raw telemetry** — timestamped IDE events (code edits, terminal runs, errors, AI queries/responses, test results), 37 event types across 6 IDE regions
- **Observable metrics** — 27 continuously computed window-level metrics, retained from 35 candidates after pruning
- **Behavioral sequences** — 13,633 auto-segmented states (implementing, debugging, testing, seeking help, thinking)

Additionally, queries are labeled as *guided* or *dependent* help-seeking (GPT-4o labels validated against human annotators, κ = .897 human–human, κ = .709/.690 GPT–human).

## Taxonomy

Ten behavioral profiles across three temporal windows of the help-seeking cycle:

| Window | Unit | Profiles |
|--------|------|----------|
| W1 — Before first query | Learner | Cold Start, Oriented, Struggling |
| W2 — Between queries | Inter-query interval | Passive, Iterating, Debugging, Spinning |
| W3 — Session-wide patterns | Learner | Passive Re-querying, Active Testing, Untested Editing |

Results (deployments D1 + D2):

| Window | n | K | Silhouette | Cluster sizes |
|--------|:-:|:-:|:----------:|---------------|
| W1 | 152 | 2 | .420 | 31 (rule-defined) / 100 / 21 |
| W2 | 686 | 3 | .547 | 318 (rule-defined) / 290 / 56 / 22 |
| W3 | 112 | 3 | .399 | 47 / 31 / 34 |

W1 and W2 rank candidate three-metric combinations by `silhouette × completion-rate spread`; behavioral metrics alone determine cluster membership. W3 uses task completion neither for clustering nor for selecting K, and its assignments are identical across 20 random seeds (ARI = 1.000).

Two prediction tasks demonstrate downstream utility (train on D1, test on D2):

| Task | Granularity | Type | AUROC |
|------|------------|------|:-----:|
| Query Imminence (60s) | Window | Binary | .726 |
| Help-Seeking Type | Query | Binary | .717 |

## Deployments

| ID | Role | Task | Students | AI Users | AI Interactions |
|----|------|------|----------|----------|-----------------|
| D1 | Taxonomy; prediction train | Playlist | 190 | 94 | 428 |
| D2 | Taxonomy; prediction test | Playlist | 113 | 90 | 540 |
| D3 | Preliminary eval — baseline | Grade Book | 70 | 48 | 190 |
| D4 | Preliminary eval — intervention | Grade Book | 107 | 85 | 228 |

Deployment files are not self-describing; identify each by its measured student / AI-user / query counts before use. The preliminary evaluation (§7.3) additionally requires the consent-exclusion list applied to D4. See `DATA_NOTES.md`.

## Quick Start

```bash
pip install -r requirements.txt
```

Reproduce the taxonomy numbers reported in the paper:

```bash
python verify_latest.py
```

Re-run the full exhaustive searches (969 combinations for W1, 1,140 for W2):

```bash
python run_taxonomy.py
```

## Repository Structure

```
├── main.py                          # Telemetry loading and feature extraction
├── run_taxonomy.py                  # Full exhaustive taxonomy searches (W1, W2, W3)
├── verify_latest.py                 # Reproduces the paper's reported numbers
├── constants/
│   └── telemetry_events.yaml        # Telemetry event schema (37 types)
├── data/
│   └── raw_telemetry/               # Raw event streams (JSON)
├── analysis/
│   ├── feature_extraction.py        # Observable metric computation
│   └── taxonomy/
│       ├── windows.py               # W1 / W2 window construction and validity rules
│       ├── clustering.py            # W1 exhaustive search
│       ├── clustering_w2.py         # W2 exhaustive search
│       └── session_patterns.py      # W3 session-pattern clustering
├── results/
│   ├── latest_verification.json     # Verification output
│   ├── verified_w1_w2_exhaustive.json
│   ├── verified_w3_session_patterns.json
│   ├── w3_session_pattern_assignments.csv
│   └── w3_session_pattern_report.md
└── behavioral_classifier/
    ├── codes.py                     # Behavioral codes and event sets
    └── auto_segmenter.py            # Behavioral state classifier
```

## Data Format

Raw telemetry is stored as JSON per deployment, keyed by session ID:

```json
{
  "session_id": {
    "events": [
      {"timestamp": 1234567890, "type": "CODE_TYPE", "payload": {...}},
      {"timestamp": 1234567891, "type": "TERMINAL_RUN", "payload": {...}},
      {"timestamp": 1234567892, "type": "CHAT_QUERY", "payload": {"text": "..."}}
    ]
  }
}
```

Loading merges every `deployment_*.json` under a namespaced key (`deployment_N:session_id`) so session IDs cannot collide across files, and tags each record with its source deployment.

Sessions are truncated at the first all-pass test result, so no behavior after task completion enters any analysis. Windows containing more than 30 seconds of tab-hidden time are excluded. Event-conditioned metrics that do not apply within a window are treated as undefined rather than imputed as zero.

Reported event counts exclude continuous pointer sampling (`MOUSE_MOVE`, emitted every 50 ms), which accounts for roughly three quarters of raw event volume. See `DATA_NOTES.md` before comparing raw file totals against the paper.

## Ethics

All data is de-identified with randomized IDs. No personally identifiable information is included. Raw telemetry retains learner-authored code and query text as submitted. Study approved under university IRB.