# Data Notes

## Identifying deployments

Deployment files are not self-describing; identify each by its measured
counts. The UIST '26 paper uses Deployments 1-4:

| File | Paper role | Students | AI users | AI queries |
|---|---|---|---|---|
| `dataset/raw_telemetry/deployment_1.json` | D1 - taxonomy; prediction train | 190 | 94 | 428 |
| `dataset/raw_telemetry/deployment_2.json` | D2 - taxonomy; prediction test | 113 | 90 | 540 |
| `dataset/raw_telemetry/deployment_3.json` | D3 - preliminary-evaluation baseline | 70 | 48 | 190 |
| `dataset/raw_telemetry/deployment_4.json` | D4 - intervention session (raw export) | 122 (raw) | 85 | 256 (raw) |

Deployments 5-9 and the `benchmark/` and `figures/` directories are extended
material beyond the paper and are preserved as-is.

## Deployment 4 (intervention)

`deployment_4.json` is the intervention session's **raw source export** in
wrapped `{"students": [...]}` format (55,037,483 bytes; 122 records). The
paper's Table 6 reports n = 107 students and 228 interactions for D4 - the
post-consent-exclusion population; this raw export precedes that filtering.
`analysis/run_preliminary_evaluation.py` reads the file directly and
reproduces the paper's §7.3 exactly: 48 vs. 85 eligible AI users, Passive
50.0% -> 20.7%, chi-square(3) = 27.55, U = 1533, completion 33.3% -> 43.5%
(Fisher p = .273). Because the format is wrapped, the taxonomy loader
(`main.py`) skips this file automatically.

**Caveat:** `deployment_4_segments.csv`, `deployment_4_labels.csv`, and the
deployment_4 metric CSVs were generated from a *previous* 49-record file
under the same name and do not describe this export.

## Query labels

GPT-4o guided/dependent labels are human-validated for Deployments 1-2 only
(kappa = .897 between raters on 97 queries; kappa = .709/.690 model vs.
raters), as reported in the paper. Labels for other deployments are
model-generated and unvalidated.

## Event counts vs. the paper

Continuous pointer sampling (`MOUSE_MOVE`, every 50 ms) accounts for roughly
three quarters of raw event volume. The paper's "approximately 180K
telemetry events" excludes it; do not compare raw file totals directly.

## Analysis conventions (paper reproduction)

Sessions are truncated at the first all-pass test result; windows with more
than 30 seconds of tab-hidden time are excluded; event-conditioned metrics
that do not apply within a window are treated as undefined rather than
imputed as zero. See the paper's §7.1 and Appendix F.

## Known caveats in preserved material

- `prepare_data.py` fails at import as committed (its package-style
  segmenter import predates this update); it is left untouched because the
  paper reproduction does not use it.
- The `benchmark/` directory's feature layers, task horizons, and labels are
  extensions that differ from the paper's (its query-task feature list
  includes `total_queries`, which is future information at query time);
  treat its outputs as separate from the paper's results.