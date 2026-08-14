# Datasheet: TutorTrace Dataset

Following Gebru et al., "Datasheets for Datasets." Items marked
**[TO CONFIRM]** require author verification.

## Motivation
Created to make learners' behavioral context during AI-assisted programming
visible and computable: existing AI tutoring systems see only the query text,
while TutorTrace captures the fine-grained activity preceding each
help-seeking moment. Released with the UIST '26 paper "TutorTrace: A Dataset
and Taxonomy for Classifying Learner Behavioral States during AI-Assisted
Programming Education." **[TO CONFIRM: funding acknowledgments.]**

## Composition
Student sessions on a task-based IDE with LLM support: timestamped telemetry
(37 event types, six interface regions), auto-classified behavioral
segments, window- and query-level observable metrics, and GPT-4o
guided/dependent query labels. The repository contains the four raw
deployment files used by the UIST '26 paper (see `datanotes.md` for the
mapping and counts). Chat payloads include students'
free-text queries and AI responses; code payloads include program text at
character level. **[TO CONFIRM: a PII/self-disclosure scrub over chat and
code payloads was completed before release.]** Student identifiers are
pseudonymous; no names, demographics, or grades are included.

## Collection process
Passive client-side instrumentation of the TutorTrace web IDE during
in-class exercises (10-15 minute limits), batched every five seconds with no
impact on the student workflow. Conducted under institutional review with
informed consent; only consenting students' records are intended for
release. **[TO CONFIRM: IRB protocol number and consent-language summary.]**

## Preprocessing and labeling
Behavioral segments come from a rule-based auto-segmenter derived from an
expert codebook (overall Cohen's kappa 0.73-0.83 against expert labels and
learner self-reports). Query labels are GPT-4o generated with the paper's
published prompt; human validation covers Deployments 1-2 (kappa = .897
between raters; .709/.690 model vs. raters). Labels for other deployments
are unvalidated. The derived layers cover Deployments 1-3; Deployment 4
ships as raw telemetry only (see `datanotes.md`).

## Uses
Research on learner behavior modeling, help-seeking, and behavior-aware AI
tutoring; `analysis/` reproduces every number in the UIST '26 paper (see the README's Quick Start).
Out of scope: evaluating or profiling individual students,
re-identification, or high-stakes decisions. Data reflect short introductory
tasks at a single institution.

## Distribution and maintenance
Distributed via this repository under **[TO CONFIRM: license]**, with
Croissant metadata in `croissant.json`. Maintained by the TutorTrace team;
issues via the repository tracker. **[TO CONFIRM: contact email and erratum
policy.]**