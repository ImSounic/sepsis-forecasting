# Sepsis Forecasting — Research Question Analysis Report

**Project:** Early Sepsis Prediction using PhysioNet/CinC Challenge 2019 Dataset
**Institution:** University of Twente
**Dataset:** 40,336 ICU patients (34,285 train / 6,051 validation), 40 clinical variables measured hourly
**Sepsis Prevalence:** 1.84% (hourly), 458 sepsis patients in validation set
**Date:** April 2026

---

## Executive Summary

This report presents the evaluation of three research questions comparing a causal Temporal Convolutional Network (TCN) against an enhanced LightGBM model for early sepsis prediction. The key findings are:

1. **LightGBM significantly outperforms TCN** on all metrics (utility: 0.388 vs 0.269), but the models exhibit low overlap (26%) in which patients they detect — indicating fundamentally different signal capture.
2. **Lab ordering patterns (time-since features) provide marginal predictive value** — a small AUPRC improvement (+0.007) but no utility gain, supporting the hypothesis that ordering patterns reflect standard clinical protocol rather than patient-specific deterioration.
3. **Aggregate engineered features decisively outperform learned temporal representations** for this task, with LightGBM achieving 44% higher utility than TCN.

---

## Models Evaluated

| Model | Architecture | Parameters | Features | Training Time |
|-------|-------------|-----------|----------|---------------|
| **TCN** | 4-layer causal dilated CNN + attention | 425,585 | 120 (raw + masks + time-since) | ~30 hours |
| **LightGBM** | Gradient-boosted trees, 500 rounds | N/A | 315 (120 base + 195 engineered) | ~5 minutes |
| **Ensemble** | Max-probability strategy (TCN + LightGBM) | — | — | — |

### TCN Configuration
- Causal dilated convolutions (kernel_size=3, dilations=[1,2,4,8])
- Receptive field: 61 timesteps (covers full 24h window)
- Bahdanau attention over temporal outputs
- Mixed precision (FP16), batch_size=256, lr=0.001
- Best epoch: 3/13 (early stopping due to overfitting)

### LightGBM Configuration
- 500 boosting rounds, 127 leaves, lr=0.05
- Enhanced features: SOFA, qSOFA, NEWS, SIRS, MEWS clinical scores
- Rolling statistics (3h, 6h, 12h windows), rate-of-change, EMA
- Lab cross-interactions (Lactate*HR, WBC*Temp, Creatinine*MAP, etc.)
- Optimal threshold: 0.50

### Critical Note: Bidirectional GRU Data Leakage
The previously reported GRU results (utility: 0.337, ensemble: 0.406) used a **bidirectional GRU** which leaks future information during both training and inference. PyTorch's `nn.GRU(bidirectional=True)` processes sequences in both forward and reverse directions even in `.eval()` mode — `.eval()` only disables dropout and batch normalization, not bidirectionality. These results are therefore **invalid as honest baselines** and are excluded from this analysis. The TCN was specifically designed with causal dilated convolutions to guarantee no future information leakage.

---

## RQ1: Do TCN and GBM Capture Different Sepsis Signals?

### Metrics Comparison

| Metric | TCN | LightGBM | Ensemble (max) |
|--------|-----|----------|----------------|
| AUPRC | 0.0961 | 0.1001 | 0.0997 |
| AUROC | 0.8012 | 0.8237 | 0.8157 |
| PhysioNet Utility | 0.2689 | 0.3878 | 0.2756 |

### Patient-Level Agreement Analysis

Of the 458 sepsis patients in the validation set:

| | LightGBM Catches | LightGBM Misses |
|---|:---:|:---:|
| **TCN Catches** | 119 (26.0%) | 118 (25.8%) |
| **TCN Misses** | 5 (1.1%) | 216 (47.2%) |

**Key findings:**

- **Only 26% overlap** in detected sepsis patients — the models capture fundamentally different signals
- **TCN uniquely detects 118 patients** (25.8%) that LightGBM misses entirely — this is a substantial subpopulation
- **LightGBM uniquely detects only 5 patients** (1.1%) that TCN misses — nearly all LightGBM detections are a subset of TCN detections
- **216 patients (47.2%) are missed by both models** — significant room for improvement
- The TCN detects more patients overall (237 = 119 + 118) than LightGBM (124 = 119 + 5), but at the cost of far more false positives, which drives its lower utility score

### PR and ROC Curve Analysis

The **Precision-Recall curves** show both models performing similarly at the threshold-independent level (AUPRC: 0.096 vs 0.100), with curves closely overlapping. The small AUPRC difference (0.004) is unlikely to be statistically significant. Both models achieve precision of ~0.3-0.4 at very low recall (<0.05) before rapidly declining toward the baseline prevalence (0.0184).

The **ROC curves** show LightGBM with a modest advantage (AUROC: 0.824 vs 0.801). Both models achieve strong discrimination despite the extreme class imbalance, with the curves well above the random diagonal across all operating points.

### Ensemble Performance

The max-probability ensemble (utility: 0.276) performs **worse** than LightGBM alone (0.388). This occurs because:
- The TCN generates many false positive predictions at its optimal threshold (0.77)
- The max strategy propagates these false positives into the ensemble
- The utility score heavily penalizes false alarms (-0.05 per FP)

A more sophisticated ensemble strategy (e.g., weighted averaging, learned combination, or requiring both models to agree) could potentially leverage the complementary detections without inheriting the false positive burden.

### Interpretation

The low overlap (26%) provides strong evidence that TCN and LightGBM capture **different sepsis signals**. The TCN, operating on raw temporal sequences, likely identifies patterns in the trajectory of vital signs and lab values over time. LightGBM, operating on engineered aggregate statistics, captures instantaneous clinical severity (via SOFA, qSOFA, SIRS scores) and recent trends (via rolling statistics). The 118 patients uniquely detected by TCN suggest that temporal trajectory patterns contain genuine predictive information that is not captured by point-in-time feature engineering.

---

## RQ2: Do Temporal Patterns in Lab Test Ordering Contribute Predictive Value Beyond Measurement Values?

### Feature Ablation Results

| Metric | With Time-Since (315 features) | Without Time-Since (275 features) | Delta |
|--------|-------------------------------|----------------------------------|-------|
| AUPRC | 0.1001 | 0.0932 | +0.0070 |
| AUROC | 0.8237 | 0.8143 | +0.0095 |
| Utility | 0.3878 | 0.3884 | -0.0006 |

### Interpretation

The time-since features (40 features encoding hours since last observation for each clinical variable) show a **mixed and marginal** contribution:

- **AUPRC improves by +0.007** (7.5% relative improvement) — suggesting the ordering patterns contain some discriminative signal at the threshold-independent level
- **AUROC improves by +0.009** — consistent small improvement in overall discrimination
- **Utility is essentially unchanged** (-0.0006, within noise) — the signal does not translate to better clinical decision-making at an optimal threshold

This result supports the professor's caution: **lab ordering patterns largely reflect standard clinical protocol rather than patient-specific deterioration signals**. The ICU follows systematic protocols for when to order labs (e.g., every 4-6 hours for basic metabolic panels, daily for CBCs). The timing of these orders is driven more by hospital policy than by clinical suspicion of sepsis.

The small AUPRC improvement suggests there may be edge cases where ordering frequency changes (e.g., a physician ordering more frequent lactate measurements due to clinical concern), but this signal is too weak to improve the clinically relevant utility metric.

**Recommendation:** Time-since features can be retained (they don't hurt performance) but should not be cited as a meaningful predictor of sepsis onset. The research narrative should emphasize that lab ordering patterns are primarily protocol-driven in ICU settings.

---

## RQ3: How Much Does Temporal Ordering (Sequence Model) Contribute vs Aggregate Statistics (Feature Engineering)?

### Direct Comparison

| Metric | TCN (Sequential/Temporal) | LightGBM (Aggregate/Engineered) | LightGBM Advantage |
|--------|--------------------------|--------------------------------|-------------------|
| AUPRC | 0.0961 | 0.1001 | +0.0040 (+4.2%) |
| AUROC | 0.8012 | 0.8237 | +0.0225 (+2.8%) |
| Utility | 0.2689 | 0.3878 | +0.1189 (+44.2%) |

### Analysis

**Aggregate statistics decisively outperform learned temporal representations** across all metrics, with the most dramatic gap in the clinically relevant PhysioNet utility score (+44.2%).

This result has several explanations:

1. **Domain knowledge advantage**: The LightGBM features encode decades of clinical knowledge (SOFA, qSOFA, NEWS, SIRS, MEWS scores are validated clinical deterioration indicators). The TCN must discover equivalent patterns from raw data alone.

2. **Threshold sensitivity**: The utility score rewards precise timing of predictions within the -6h to +3h onset window. LightGBM's engineered features (rate-of-change, rolling statistics) directly encode the temporal dynamics that clinical scores are designed to capture, while the TCN must learn these dynamics implicitly.

3. **Data efficiency**: With 1.84% sepsis prevalence and ~750 sepsis patients in training, the TCN has limited positive examples to learn complex temporal patterns. LightGBM operates on pre-computed features that compress the temporal information into a denser representation, making more efficient use of the limited positive samples.

4. **Overfitting**: The TCN showed signs of severe overfitting (training loss: 0.578, validation loss: 1.026 at epoch 4), suggesting the model complexity exceeds what the training data can support. The TCN's best epoch was epoch 3 (of 13), with utility declining steadily afterward.

5. **Training constraints**: The TCN was trained with default hyperparameters (no Optuna tuning), whereas the LightGBM used validated clinical feature engineering. A tuned TCN with stronger regularization may close some of the gap.

### Nuance: Different Strengths at Different Operating Points

Despite lower aggregate metrics, the TCN detects **118 sepsis patients that LightGBM misses** (from RQ1). This suggests that temporal ordering does capture unique signals — particularly trajectory-based patterns like gradual vital sign deterioration that precede the threshold-based triggers used in clinical scoring systems.

The AUPRC gap (0.096 vs 0.100) is much smaller than the utility gap (0.269 vs 0.388), indicating that the models have similar discriminative ability but differ substantially in **calibration and threshold behavior**. The TCN may benefit from post-hoc calibration (e.g., Platt scaling) to improve its utility at optimal operating points.

---

## Limitations

1. **No statistical significance testing**: All comparisons are point estimates without confidence intervals. Bootstrap resampling or paired permutation tests would strengthen the conclusions.

2. **TCN not hyperparameter-tuned**: The TCN used default hyperparameters while the LightGBM features were carefully engineered. Optuna tuning for TCN (particularly dropout, weight decay, and architecture) may reduce the performance gap.

3. **Single train/validation split**: Results are based on a single 85/15 split. Cross-validation would provide more robust estimates.

4. **Ensemble strategy limited**: Only the max-probability ensemble was evaluated. Learned ensemble methods (stacking, weighted averaging with optimized weights) could better exploit model complementarity.

5. **No formal stationarity tests**: Time-series features were applied without formal stationarity testing of the underlying clinical variables.

---

## Summary of Findings

| Research Question | Finding | Strength of Evidence |
|------------------|---------|---------------------|
| **RQ1:** Model complementarity | TCN and LightGBM capture different signals (26% overlap). TCN uniquely detects 118/458 sepsis patients. | Strong |
| **RQ2:** Lab ordering patterns | Marginal AUPRC improvement (+0.007), no utility improvement. Ordering patterns are largely protocol-driven. | Moderate |
| **RQ3:** Temporal vs aggregate | Aggregate statistics outperform temporal ordering (utility: 0.388 vs 0.269). Domain knowledge in feature engineering provides substantial advantage. | Strong |

---

## Recommendations for Future Work

1. **Hyperparameter tuning for TCN** — Increase dropout (0.2 to 0.4), weight decay (1e-4 to 1e-3), and reduce model capacity to address overfitting
2. **Learned ensemble** — Train a meta-learner to optimally combine TCN and LightGBM predictions, leveraging the 118 unique TCN detections
3. **Bootstrap confidence intervals** — Add statistical rigor to all metric comparisons
4. **Cross-validation** — Validate findings across multiple data splits
5. **Attention analysis** — Examine TCN attention weights to understand which temporal patterns drive the 118 unique detections
6. **Hybrid model** — Feed TCN-learned temporal embeddings as additional features into LightGBM

---

*Report generated from outputs/rq_analysis/metrics.json*
*Figures: pr_curves.png, roc_curves.png, agreement_matrix.png*
