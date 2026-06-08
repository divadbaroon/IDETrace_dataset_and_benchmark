# IDETrace

A behavioral telemetry dataset for predicting novice programmer behavior in AI-assisted coding environments. The dataset captures fine-grained IDE interactions from 664 students across 8 classroom deployments of an introductory Python course with an integrated AI tutor.

## Dataset

882,367 telemetry events from 664 students (386 using the AI tutor), organized into three layers:

- **Raw telemetry** — timestamped IDE events (code edits, terminal runs, errors, AI queries/responses, test results)
- **Observable metrics** — 87,439 window-level observations (30s window, 5s step) and 1,576 query-level behavioral features
- **Behavioral sequences** — 15,991 auto-segmented states (implementing, debugging, testing, seeking help, thinking)

Additionally, 1,692 queries are labeled as *guided* or *dependent* help-seeking (GPT-4o labels validated against human annotators, κ = .897 human–human, κ = .709/.690 GPT–human).

## Benchmark

Four prediction tasks on behavioral telemetry:

| Task | Granularity | Type | Description |
|------|------------|------|-------------|
| Next Behavioral State | Window | 5-class | Predict the next behavioral state |
| Error Imminence | Window | Binary | Will a terminal error occur within *h* seconds? |
| Query Imminence | Window | Binary | Will the student query the AI within *h* seconds? |
| Query Type | Query | Binary | Will the query be *guided* or *dependent*? |

Results (train on D1, test on D2):

| Task | Best Model | AUC |
|------|-----------|:---:|
| Next Behavioral State | XGB+Seq-CNN | .828 |
| Error Imminence (15s) | XGB+Seq-Trans | .868 |
| Query Imminence (15s) | XGB+Seq-Trans | .843 |
| Query Type | Seq-GRU | .771 |

All LLMs tested (GPT-4o, Llama 3.1, Qwen 3.5, DeepSeek-R1) perform near chance, indicating behavioral prediction requires task-specific training.

## Deployments

| ID | Split | Task | Students | AI Users | Queries |
|----|-------|------|----------|----------|---------|
| D1 | Train | GradeBook | 190 | 94 | 428 |
| D2 | Test | GradeBook | 113 | 90 | 536 |
| D3 | Test | Playlist | 49 | 48 | 190 |
| D4 | Test | Rectangle | 14 | 13 | 47 |
| D5 | Test | Rectangle | 37 | 36 | 99 |
| D6 | Test | GradeScore (C) | 25 | 24 | 120 |
| D7 | Test | Rectangle | 43 | 42 | 202 |
| D8 | Test | Rectangle | 15 | 14 | 59 |

## Quick Start

```bash
pip install pandas numpy scikit-learn xgboost torch pyyaml
```

Prepare derived datasets from raw telemetry:

```bash
python prepare_data.py --force
```

Run the benchmark (D1 → D2):

```bash
cd benchmark
python run_benchmark.py
```

Cross-deployment evaluation:

```bash
python benchmark/run_all_benchmark.py
```

LLM baselines (requires API key or Ollama):

```bash
python benchmark/models/llm/closed_baseline.py --model gpt-4o-mini
python benchmark/models/llm/open_baseline.py --model llama3.1:8b
```

## Repository Structure

```
├── manifest.yaml                    # Dataset configuration
├── prepare_data.py                  # Data preparation pipeline
├── dataset/
│   ├── raw_telemetry/               # Raw event streams (JSON)
│   ├── behavioral_sequences/        # Auto-segmented states (CSV)
│   ├── observable_metrics/
│   │   ├── window_level/            # 30s sliding window features
│   │   └── query_level/             # Per-query behavioral features
│   ├── query_labels/                # Guided/dependent labels
│   ├── demographics/                # De-identified demographics
│   └── post_surveys/                # Self-reported understanding
├── benchmark/
│   ├── run_benchmark.py             # Main benchmark runner
│   ├── run_all_benchmark.py         # Cross-deployment evaluation
│   ├── run_window_ablation.py       # Window size ablation
│   ├── data.py                      # Constants and data loading
│   ├── results/                     # All output
│   └── models/
│       ├── ml/
│       │   ├── classical.py         # Majority, LogReg, RF, XGBoost
│       │   ├── mlp.py              # 3-layer MLP
│       │   ├── sequential.py       # LSTM, GRU, CNN, Transformer
│       │   └── ensemble.py         # XGB + best sequential
│       └── llm/
│           ├── prompts.py           # Shared prompts and parsers
│           ├── closed_baseline.py   # OpenAI (GPT-4o, GPT-5.5)
│           └── open_baseline.py     # Ollama (Llama, Qwen, DeepSeek)
└── behavioral_classifier/
    ├── codes.py                     # Behavioral codes and event sets
    └── auto_segmenter.py            # Behavioral state classifier
```

## Data Format

Raw telemetry is stored as JSON per deployment, keyed by student ID:

```json
{
  "student_id": {
    "events": [
      {"timestamp": 1234567890, "type": "CODE_TYPE", "payload": {...}},
      {"timestamp": 1234567891, "type": "TERMINAL_RUN", "payload": {...}},
      {"timestamp": 1234567892, "type": "CHAT_QUERY", "payload": {"text": "..."}}
    ]
  }
}
```

Window-level features (CSV) include three abstraction layers: raw event counts, derived observable metrics (edit rates, query rates, error self-fix patterns), and behavioral sequence features (state proportions, current/previous state). Each row also contains imminence labels at multiple horizons (5s, 10s, 15s, 30s, 45s, 60s).

Query-level features (CSV) capture pre-query behavioral context computed from the window between the last AI response and 15 seconds before the next query.

## Ethics

All data is de-identified with randomized IDs. No personally identifiable information is included. Query text content is not included in released features. Study approved under university IRB.
