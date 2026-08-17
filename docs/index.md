# Content Refresh Opportunity Scoring: A Model-Driven Approach to Prioritizing Page Reviews

## Abstract

Content teams need a repeatable way to decide which pages to refresh first, but manual review doesn't scale past a few hundred URLs. This project builds a machine-learning model that scores existing content for refresh priority, trained on 30,000 anonymized pages from FlyRank's search performance warehouse. The task is framed as predicting a declining-trend label from engagement, visibility, and content-age signals, using a client-holdout validation split to prevent leakage across clients. A random forest classifier reaches a Precision@50 of 0.740, compared to 0.240 for a transparent hand-written baseline rule — roughly a 3x improvement at identifying the highest-priority pages. The resulting ranked queue assigns each page a confidence tier and a specific suggested action (monitor, refresh, refresh-and-review-CTR, refresh-and-review-engagement, or expand-and-refresh), turning a scoring exercise into a decision-support tool a content team could act on directly.

## Introduction / Problem Statement

**Decision this supports:** Given a large catalog of existing content, which pages should a content team review or refresh first, and why?

Search visibility decays for reasons that aren't always obvious from a single metric — a page can lose impressions gradually as it ages, as competition increases, or as user intent shifts, and by the time a decline is visually obvious in a dashboard, meaningful traffic has often already been lost. A scoring system that flags declining or at-risk pages earlier, and explains *why* each page was flagged, lets a team act before the drop compounds.

This project treats the problem as a binary classification task: predict whether a page's performance trend is declining (`trend_direction == "down"`), using signals available before that trend is fully realized — impression history, click history, position, content age, and content structure — then rank pages by model confidence to build a prioritized action queue.

## Data

- **Source:** `data/raw/content_refresh_anonymized.csv`, the anonymized starter release from the FlyRank ML Internship warehouse — 30,000 rows, 44 columns.
- **Scope:** the dataset ships with no titles, URLs, client names, or domains — only anonymized `content_id` and `client_id` identifiers and behavioral/structural features.
- **Label:** `is_declining_label = (trend_direction == "down")`. Declining-label rate across the dataset: 0.542 (16,262 of 30,000 rows).
- **Validation split:** client_holdout — the evaluation split holds out entire clients rather than random rows, so the model isn't scored on clients it has already seen. This is a meaningfully stricter test than a random split, since it checks whether patterns generalize across accounts, not just across pages.
- **Exclusions:** no raw private exports, no client-identifying fields, and no keyword-level or query-level data were used, per the internship's data-use terms.

## Methodology

**Pipeline stages** (from `scripts/run_all.py`):
1. `01_prepare_features.py` — cleans the raw 44-column release and engineers a 52-column feature vector, defining the label.
2. `02_baseline_score.py` — builds a transparent, hand-written rule-based score as a comparison point ("fix this first" logic with human-readable reason codes).
3. `03_train_model.py` — trains three models (logistic regression, decision tree, random forest) on the client-holdout split.
4. `04_evaluate_and_export.py` — blends model output with the baseline into a final ranked queue, with charts and a written report.
5. `05_build_pdf_report.py` — exports a shareable PDF summary.

**Models compared:**

| Model | ROC AUC | Avg Precision | Precision@50 | Recall | F1 |
|---|---|---|---|---|---|
| decision_tree | 0.742 | 0.575 | 0.580 | 0.716 | 0.634 |
| logistic_regression | 0.700 | 0.522 | 0.400 | 0.567 | 0.566 |
| random_forest | 0.750 | 0.618 | 0.740 | 0.744 | 0.640 |
| baseline_rules | 0.627 | 0.468 | 0.240 | — | — |

**Model selection:** random_forest was selected as the best model, chosen by Precision@50 — the metric that matters most here, since a content team will only act on the top of the queue, not the full ranked list.

**Top features by importance:**

| Feature | Importance |
|---|---|
| days_with_impressions | 0.1581 |
| log_impressions_90d | 0.1286 |
| avg_position | 0.1092 |
| content_age_days | 0.0952 |
| char_count | 0.0426 |
| word_count | 0.0396 |
| log_clicks_90d | 0.0345 |
| ctr | 0.0333 |
| scroll_rate | 0.0312 |
| days_with_sessions | 0.0280 |

Visibility consistency (`days_with_impressions`) and recent impression volume (`log_impressions_90d`) dominate the model — ahead of raw content length or click-through rate — suggesting that *how consistently* a page shows up in search matters more to the model than any single content-quality signal.

**Leakage safeguard:** the client_holdout split specifically guards against the model learning client-specific patterns that wouldn't transfer to unseen clients, which is the most realistic failure mode for a tool meant to generalize across a portfolio.

## Results

The random forest model outperformed both the simpler ML baselines and the hand-written rule across every metric reported, with the sharpest gap in Precision@50 — the number of true declining pages found in the top 50 ranked results. The hand-written rule achieved 0.240 (roughly 1 in 4 correct in its top 50); the random forest achieved 0.740 (roughly 3 in 4 correct), a ~3x lift.

The final ranked queue split 30,000 pages into three confidence tiers:
- High-confidence: 3,602 pages
- Medium-confidence: 11,398 pages
- Low-confidence: 15,000 pages

And into five suggested actions:
- monitor: 13,083
- refresh: 8,188
- refresh_and_review_ctr: 6,654
- refresh_and_review_engagement: 1,993
- expand_and_refresh: 82

The top of the queue was dominated by pages flagged with overlapping reason codes — `declining_with_demand`, `low_ctr_visible_page`, `model_decline_risk`, and `visible_model_opportunity` — meaning the highest-priority pages are ones that still have search demand and visibility, but are underperforming relative to that visibility. This is a distinct (and arguably more actionable) signal than simply flagging pages that have lost demand entirely, since there's still traffic to recover.

*[Charts: action_mix.svg, confidence_mix.svg, trend_distribution.svg, top_reason_codes.svg, top_feature_importance.svg — embed alongside this section when deploying]*

## Limitations & Honest Framing

- This is an **observed, directional, decision-support** tool — it flags pages statistically similar to ones that declined in the past. It does not, and cannot, predict or explain Google's ranking algorithm, and no claim here should be read as causal.
- The model was trained and evaluated on a single anonymized 30,000-row snapshot; performance on the full ~79M-row warehouse, or on a different data distribution, is untested here.
- Precision@50 measures the top of the queue well but says little about ranking quality further down; the medium- and low-confidence tiers (26,398 of 30,000 pages) haven't been separately validated.
- The client_holdout split is a stronger test than random splitting, but with a bounded set of anonymized clients, results may not generalize identically to clients outside this dataset.
- Reason codes are model-derived heuristics for explainability, not independently verified causes of decline.

## Ranked Recommendations

1. **Act on the high-confidence tier first** (3,602 pages) — this is where model precision is strongest and the action queue is most reliable.
2. **Prioritize `refresh_and_review_ctr` pages** — these pages retain visibility but are underperforming on clicks, meaning the fastest, lowest-risk win is often a metadata/snippet review rather than a full content rewrite.
3. **Treat `expand_and_refresh` (82 pages) as a small, high-touch list** — worth manual review given how few pages qualify.
4. **Re-evaluate the medium- and low-confidence tiers on a slower cadence**, since they weren't validated with the same rigor as the top of the queue.
5. **Re-run the pipeline periodically** rather than treating this as a one-time score, since `days_with_impressions` and `log_impressions_90d` — the top two features — are inherently time-sensitive.

## Reproducibility

- Full pipeline: [`scripts/run_all.py`](../scripts/run_all.py), stages `01`–`05`
- Capstone notebook: [`work/notebooks/capstone.ipynb`](../work/notebooks/capstone.ipynb)
- Raw data: `data/raw/content_refresh_anonymized.csv` (anonymized, ships with repo)
- Full outputs: `outputs/refresh_queue.csv`, `outputs/model_report.md`, `outputs/model_results.json`, `outputs/charts/`
- Repo: https://github.com/Xcelrator0/Intership-Tasks

## Acknowledgments & Data Credit

Built on the FlyRank ML Internship dataset — [https://flyrank.ai](https://flyrank.ai)
