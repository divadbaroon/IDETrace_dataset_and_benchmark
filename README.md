# TutorTrace

A dataset, behavioral classifier, and taxonomy for making learners' behavioral context visible and computable in real time during AI-assisted programming education.

TutorTrace captures fine-grained IDE telemetry from 480 students across four classroom deployments in two introductory Python courses, and transforms it into behavioral sequences, observable metrics, and behavioral profiles that an AI tutor can act on while a student is working.

## Dataset

Approximately 180K raw telemetry events from 480 students, organized into four abstraction layers:

| Layer | Representation | Scale |
|-------|----------------|-------|
| **A: Raw Telemetry** | Timestamped IDE events, 37 event types across 6 source regions | ~180K |
| **B: Observable Metrics** | Window-level measures of activity amount, frequency, and distribution | 27 features |
| **C: Behavioral Sequences** | Auto-classified behavioral segments | 13,633 |
| **D: Behavioral Profiles** | Clustered profiles across three temporal windows | 10 profiles |

## Behavioral Classifier

An automated pipeline segments raw telemetry into labeled behavioral states in real time. The segmentation and classification rules were derived from expert-labeler consensus, not authored heuristics: four domain-expert annotators independently coded session replays across 10 pilot sessions, reconciling segment boundaries and codebook definitions in weekly meetings.

Behavioral codes: `Implementing`, `Debugging`, `Testing`, `Thinking` (subtypes: Task, Code, Error, Pre-query, Response), `Seeking Help`, `Idle`, `Off-Topic`, `Unknown`.

**Validation.** A user study with 13 participants who self-labeled their own sessions, with six sessions independently labeled by domain experts, allows agreement to be triangulated across three sources:

| Comparison | Cohen's κ | Raw agreement |
|-----------|:---------:|:-------------:|
| Classifier vs. expert annotations | .83 | 87% |
| Classifier vs. learner self-reports | .73 | 78% |
| Expert annotations vs. learner self-reports | .79 | 83% |

Agreement is strongest for behaviors with clear telemetry signatures (`Implementing` .95, `Seeking Help` 1.00) and weakest where semantic understanding is required to disambiguate — `Thinking: Code` vs. `Thinking: Error` in particular (.34, .42 against expert labels).

## The TutorTrace Taxonomy

Ten behavioral profiles across three temporal windows, each capturing a distinct stage of the help-seeking cycle.

**Window 1 — Before the first AI query** (one observation per learner)

| Profile | n (%) | Think | Runs | Errors | Completion |
|---------|:-----:|:-----:|:----:|:------:|:----------:|
| Cold Start | 31 (20%) | 72s | 0.0 | 0.0 | 97% |
| Oriented | 100 (66%) | 98s | 1.2 | 0.7 | 92% |
| Struggling | 21 (14%) | 165s | 4.2 | 2.5 | 67% |

**Window 2 — Between consecutive queries** (one observation per inter-query interval)

| Profile | n (%) | Think | Runs | Errors | Completion |
|---------|:-----:|:-----:|:----:|:------:|:----------:|
| Passive | 318 (46%) | 18s | 0.0 | 0.0 | 81% |
| Iterating | 290 (42%) | 31s | 1.0 | 0.5 | 84% |
| Debugging | 56 (8%) | 57s | 3.7 | 2.8 | 79% |
| Spinning | 22 (3%) | 149s | 1.8 | 0.6 | 50% |

**Window 3 — Session-wide re-querying patterns** (learners with ≥2 valid inter-query intervals)

| Profile | n (%) | Passive | Tested | Completion |
|---------|:-----:|:-------:|:------:|:----------:|
| Passive Re-querying | 47 (42%) | 70% | 22% | 79% |
| Active Testing | 31 (28%) | 20% | 77% | 81% |
| Untested Editing | 34 (30%) | 35% | 23% | 88% |

Windows 1 and 2 use an outcome-guided clustering procedure: clustering runs on behavioral metrics only, with task completion used to arbitrate among candidate solutions that are already behaviorally distinct. Their completion rates are therefore **descriptive characteristics, not independent validation**. Window 3 excludes completion from both clustering and model selection.

Profile assignments are stable across random seeds (Window 3 ARI = 1.000 over 20 seeds).

## Prediction Tasks

A downstream demonstration that the abstraction layers carry signal beyond raw event counts. Models are Random Forests trained on Deployment 1 (*n* = 190) and evaluated on held-out Deployment 2 (*n* = 113) — a cross-cohort evaluation under the same instructor and task.

| Task | Window | Raw telemetry | +Observable metrics | +Behavioral sequences |
|------|:------:|:-------------:|:-------------------:|:---------------------:|
| **Query imminence** — will the learner query within 60s? | 30s | .689 | **.726** | .719 |
| **Help-seeking type** — will the query be guided or dependent? | 15s | .690 | **.717** | .705 |

Held-out AUROC. Feature representations are nested: each column adds to the previous.

Both tasks exclude query-composition events (`CHAT_TYPE`, `CHAT_DELETE`, `CHAT_PASTE`, `CHAT_QUERY`), query text, source-code content, chat history, and the subsequent AI response, so models rely only on preceding behavioral telemetry.

These are modest, honest numbers. They demonstrate that window-level summaries of learner activity carry usable signal about the timing and form of help-seeking — not that the task is solved.

## System

TutorTrace is a task-based IDE platform with LLM support, structurally similar to LeetCode or HackerRank: fixed layout with task description, test cases, code editor, terminal, and AI chat window. The constrained interaction space is what makes fine-grained behavioral observation feasible and reproducible.

Events are captured client-side and batched to the backend every five seconds with no impact on the student's workflow. Each event carries a millisecond timestamp, source region, and payload. The tutor is GPT-4o, prompted with a three-level scaffolding framework (Socratic questioning → conceptual hint → concrete scaffolding) and prohibited from emitting runnable code.

## Quick Start

```bash
pip install -r requirements.txt
```

Reproduce the taxonomy (all three windows):

```bash
python -m taxonomy.run_all
```

Run a single window:

```bash
python -m taxonomy.window1
python -m taxonomy.window2
python -m taxonomy.window3
```

Reproduce the prediction results (train D1, test D2):

```bash
python -m prediction.run_benchmark
```

Window-size sensitivity sweep (15s / 30s / 45s / 60s):

```bash
python -m prediction.run_window_sweep
```

## Repository Structure

```
├── dataset/
│   ├── raw_telemetry/               # Raw event streams per deployment (JSON)
│   ├── behavioral_sequences/        # Auto-classified segments (CSV)
│   ├── observable_metrics/
│   │   ├── window_level/            # Sliding-window features
│   │   └── query_level/             # Per-query behavioral context
│   └── query_labels/                # Guided / dependent help-seeking labels
├── behavioral_classifier/
│   ├── codes.py                     # Behavioral codes and event sets
│   └── auto_segmenter.py            # Seven-step segmentation pipeline
├── taxonomy/
│   ├── window1.py                   # Before first query
│   ├── window2.py                   # Between consecutive queries
│   ├── window3.py                   # Session-wide re-querying patterns
│   └── run_all.py
└── prediction/
    ├── run_benchmark.py             # D1 → D2 held-out evaluation
    ├── run_window_sweep.py          # Observation-window sensitivity
    └── features.py                  # Nested feature representations
```

## Ethics

Study approved under university IRB. Student identifiers are replaced with pseudonymous tokens and no directly identifying information is included.

Released data files retain verbatim chat text and code snapshots, since the behavioral context these represent is the substance of the dataset. Users of the dataset should treat this content as student-authored classroom work and handle it accordingly.

## License

Released under CC-BY 4.0, matching the paper.
