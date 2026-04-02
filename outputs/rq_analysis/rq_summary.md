# Research Question Evaluation Results

## RQ1: Model Complementarity

| Metric | TCN | LightGBM | Ensemble |
|--------|-----|----------|----------|
| AUPRC | 0.0961 | 0.1001 | 0.0997 |
| AUROC | 0.8012 | 0.8237 | 0.8157 |
| Utility | 0.2689 | 0.3878 | 0.2756 |

Patient-level agreement (at optimal threshold):
- Both models catch: 119 patients
- TCN only: 118 patients
- LightGBM only: 5 patients
- Neither catches: 216 patients
- Overlap: 26.0%

## RQ2: Lab Ordering Patterns (Feature Ablation)

| Metric | With time-since | Without time-since | Delta |
|--------|----------------|-------------------|-------|
| AUPRC | 0.1001 | 0.0932 | +0.0070 |
| AUROC | 0.8237 | 0.8143 | +0.0095 |
| Utility | 0.3878 | 0.3884 | -0.0006 |
| Features | 315 | 275 | - |

## RQ3: Temporal Ordering vs Aggregate Statistics

| Metric | TCN (sequential) | LightGBM (aggregate) |
|--------|-----------------|---------------------|
| AUPRC | 0.0961 | 0.1001 |
| AUROC | 0.8012 | 0.8237 |
| Utility | 0.2689 | 0.3878 |
