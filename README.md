# Sepsis Forecasting from ICU Data

Early prediction of sepsis using deep learning on the [PhysioNet/Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/) dataset.

## Results

| Model | Utility Score | AUPRC | AUROC | Threshold |
|-------|--------------|-------|-------|-----------|
| **Ensemble (GRU + LightGBM, max)** | **0.4060** | — | — | **0.50** |
| Improved LightGBM (244 features) | 0.3976 | — | — | 0.50 |
| Enhanced LightGBM (315 features) | 0.3878 | 0.1001 | 0.8237 | 0.50 |
| GRU + Attention (optimized) | 0.3374 | — | — | 0.40 |
| Causal TCN + Attention | 0.2689 | 0.0961 | 0.8012 | 0.77 |
| Ensemble (TCN + LightGBM, max) | 0.2756 | 0.0997 | 0.8157 | — |
| LightGBM Baseline (160 features) | 0.1370 | — | — | 0.15 |

*PhysioNet 2019 Challenge winners achieved 0.36-0.43 utility on the hidden test set.*

## Research Question Analysis

### RQ1: Do TCN and GBM capture different sepsis signals?

Of 458 validation sepsis patients:

| | LightGBM Catches | LightGBM Misses |
|---|:---:|:---:|
| **TCN Catches** | 119 (26.0%) | 118 (25.8%) |
| **TCN Misses** | 5 (1.1%) | 216 (47.2%) |

26% overlap — the models capture fundamentally different signals. TCN uniquely detects 118 patients that LightGBM misses.

### RQ2: Do lab ordering patterns contribute predictive value?

| Metric | With time-since | Without time-since | Delta |
|--------|----------------|-------------------|-------|
| AUPRC | 0.1001 | 0.0932 | +0.0070 |
| AUROC | 0.8237 | 0.8143 | +0.0095 |
| Utility | 0.3878 | 0.3884 | -0.0006 |
| Features | 315 | 275 | — |

Marginal. AUPRC improves slightly but utility is unchanged. Lab ordering patterns largely reflect standard ICU protocol, not patient-specific deterioration.

### RQ3: Temporal ordering (TCN) vs aggregate statistics (LightGBM)?

| Metric | TCN (sequential) | LightGBM (aggregate) |
|--------|-----------------|---------------------|
| AUPRC | 0.0961 | 0.1001 |
| AUROC | 0.8012 | 0.8237 |
| Utility | 0.2689 | 0.3878 |

Aggregate statistics win decisively. Domain knowledge encoded in clinical scoring features (SOFA, qSOFA, NEWS, SIRS, MEWS) outperforms learned temporal representations by 44% on utility.

*Full analysis: [outputs/rq_analysis/RQ_ANALYSIS_REPORT.md](outputs/rq_analysis/RQ_ANALYSIS_REPORT.md)*

## Models

### Causal TCN + Temporal Attention

```
Input (batch, 24, 120)
  -> 4x Residual Blocks:
     -> Causal Dilated Conv1d (dilations: 1, 2, 4, 8)
     -> BatchNorm -> ReLU -> Dropout
     -> Causal Dilated Conv1d
     -> BatchNorm -> ReLU -> Dropout
     -> Residual connection (1x1 conv if needed)
  -> Bahdanau Attention (dim=64)
  -> Dense (128 -> 1)
```

- **Utility**: 0.2689, **Parameters**: 425,585
- **Config**: hidden=128, layers=4, kernel_size=3, dropout=0.2, lr=0.001, batch=256
- **Training**: 13 epochs (early stopping, best epoch 3), ~30 hours on NVIDIA A16
- **Receptive field**: 61 timesteps (covers full 24h window)

### Enhanced LightGBM

315 features built from the 120 base features:

| Feature Group | Count | Description |
|---------------|-------|-------------|
| Base features | 120 | 40 raw + 40 masks + 40 time_since |
| 6h rolling stats | 40 | mean/min/max/std + slope for 8 vitals |
| 12h rolling stats | 32 | mean/min/max/std for 8 vitals |
| 24h rolling stats | 32 | mean/min/max/std for 8 vitals |
| Rate of change (6h) | 8 | current - 6h ago for vitals |
| Rate of change (3h, 12h) | 16 | Multi-window rate of change |
| Time-weighted avg (3h) | 8 | Exponential decay (half-life=3h) |
| Time-weighted avg (6h) | 8 | Exponential decay (half-life=6h) |
| Clinical scores | 10 | SIRS (5 components) + MEWS (5 components) |
| Lab cross-interactions | 5 | Lactate*HR, WBC*Temp, Creatinine*MAP, etc. |
| Lab deviation from norm | 8 | Absolute z-scores for 8 key labs |
| SOFA, qSOFA, NEWS, Shock Index | 28 | Existing clinical severity scores |

- **Utility**: 0.3878, **Trees**: 500 (no early stopping)
- **Params**: num_leaves=127, lr=0.05, min_data_in_leaf=100

### GRU + Temporal Attention

```
Input (batch, 24, 120)
  -> LayerNorm
  -> Bidirectional GRU (hidden=256, 1 layer)
  -> Bahdanau Attention (dim=64)
  -> Dense (512 -> 64 -> 1)
```

- **Utility**: 0.3374, **Parameters**: 646,641
- **Config**: hidden=256, layers=1, dropout=0.1, lr=0.00095, batch=32, bidirectional=True
- **Training**: 47 epochs (early stopping at epoch 37), ~70 minutes on RTX 1000

### Improved LightGBM

244 features built from the 120 base features:

| Feature Group | Count | Description |
|---------------|-------|-------------|
| Base features | 120 | 40 raw + 40 masks + 40 time_since |
| 6h rolling stats | 40 | mean/min/max/std + slope for 8 vitals |
| 12h rolling stats | 32 | mean/min/max/std for 8 vitals |
| 24h rolling stats | 32 | mean/min/max/std for 8 vitals |
| Rate of change | 8 | current - 6h ago for vitals |
| Time-weighted avg | 8 | Exponential decay (half-life=3h) |
| Clinical interactions | 4 | HR*SBP, MAP trend, Resp*HR, temp deviation |

- **Utility**: 0.3976, **Trees**: 500 (no early stopping)
- **Params**: num_leaves=127, lr=0.05, min_data_in_leaf=100

### Ensemble (Max Strategy)

- **Utility**: 0.4060, **Threshold**: 0.50
- Strategy: `prediction = max(gru_prob, lgb_prob)`
- Only 27.7% overlap in true positive detections between models
- 60.1% of TPs detected in optimal -6h to 0h window

### Patient-Level Performance

| Category | Patients | Samples | % of Patients |
|----------|----------|---------|---------------|
| True Positive | 268 | 1,588 | 4.4% |
| True Negative | 4,303 | 220,734 | 71.1% |
| False Positive | 1,290 | 11,555 | 21.3% |
| False Negative | 190 | 2,777 | 3.1% |
| **Total** | **6,051** | **236,654** | **100%** |

## Hyperparameter Optimization (Optuna)

12 trials completed (2 pruned). Best trial: #8 (utility: 0.329).

| Trial | Utility | Hidden | Layers | Dropout | LR | Batch | Bidir | PW Mult |
|-------|---------|--------|--------|---------|------|-------|-------|---------|
| 0 | 0.300 | 128 | 1 | 0.3 | 0.0026 | 64 | Yes | 2.0 |
| 1 | 0.265 | 256 | 3 | 0.3 | 0.0011 | 128 | Yes | 1.0 |
| 2 | 0.221 | 256 | 3 | 0.3 | 0.0021 | 128 | No | 1.0 |
| 3 | 0.310 | 128 | 3 | 0.5 | 0.00052 | 64 | Yes | 0.5 |
| 4 | 0.188 | 128 | 2 | 0.5 | 0.0018 | 32 | No | 1.0 |
| 5 | 0.327 | 256 | 2 | 0.1 | 0.00016 | 64 | No | 2.0 |
| 6 | 0.324 | 256 | 2 | 0.3 | 0.00024 | 32 | Yes | 2.0 |
| 7 | 0.289 | 128 | 1 | 0.5 | 0.00044 | 64 | Yes | 1.0 |
| **8** | **0.329** | **256** | **1** | **0.1** | **0.00095** | **32** | **Yes** | **0.5** |
| 9 | pruned | - | - | - | - | - | - | - |
| 10 | pruned | - | - | - | - | - | - | - |
| 11 | 0.308 | 256 | 2 | 0.1 | 0.00012 | 64 | No | 0.5 |

## Dataset

| Property | Value |
|----------|-------|
| Total patients | 40,336 |
| Training patients | 34,285 |
| Validation patients | 6,051 |
| Training samples | 1,315,556 |
| Validation samples | 236,654 |
| Sepsis prevalence (hourly) | 1.84% |
| Positive class weight | 54.86 |
| Features (TCN / GRU) | 120 (40 raw + 40 masks + 40 time_since) |
| Features (Enhanced LightGBM) | 315 (120 base + 195 engineered) |
| Features (Improved LightGBM) | 244 (120 base + 124 engineered) |

**Source**: PhysioNet/CinC Challenge 2019 - 40,336 ICU patients across two hospital systems. 40 clinical variables (8 vitals, 26 labs, 6 demographics) measured hourly. Lab values exhibit 80-95% missingness.

## Setup

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support (optional but recommended)

### Installation
```bash
git clone https://github.com/ImSounic/sepsis-forecasting.git
cd sepsis-forecasting

# Option 1: Use setup script
./setup.sh

# Option 2: Manual setup
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Usage

```bash
source .venv/bin/activate

# Train TCN model
python -m src.training.train_tcn --config configs/tcn.yaml

# Resume TCN training from checkpoint
python -m src.training.train_tcn --config configs/tcn.yaml --resume

# Train GRU model
python -m src.training.train --config configs/optimized.yaml

# Train enhanced LightGBM (315 features)
python -m src.training.train_baseline --config configs/default.yaml --improved

# Train LightGBM baseline
python -m src.training.train_baseline --config configs/default.yaml

# Run RQ evaluation (PR curves, agreement, feature ablation)
python -m src.training.evaluate_rq --config configs/tcn.yaml

# Evaluate ensemble strategies
python -m src.training.evaluate_ensemble --config configs/optimized.yaml --full-analysis

# Run explainability analysis (attention + SHAP)
python -m src.explainability.run_analysis --config configs/optimized.yaml

# Hyperparameter optimization
python -m src.training.hyperopt --config configs/default.yaml --n_trials 50
python -m src.training.hyperopt_tcn --config configs/tcn.yaml --n_trials 12

# Run tests
pytest tests/
```

## Project Structure

```
sepsis-forecasting/
├── src/
│   ├── data/                # Data loading, preprocessing, caching
│   ├── models/              # TCN, GRU, LightGBM (baseline + improved), ensemble
│   ├── training/            # Training loops, hyperopt, ensemble evaluation
│   └── explainability/      # SHAP analysis, attention visualization
├── configs/
│   ├── default.yaml         # Base configuration
│   ├── optimized.yaml       # Optuna Trial 8 best hyperparameters
│   ├── best.yaml            # Copy of optimized
│   └── tcn.yaml             # TCN configuration
├── outputs/
│   ├── models/
│   │   ├── gru/             # GRU checkpoints
│   │   ├── tcn/             # TCN checkpoints (best.pt, last.pt)
│   │   ├── lightgbm/        # LightGBM baseline + improved models
│   │   └── ensemble/        # Ensemble predictions
│   ├── rq_analysis/         # RQ evaluation outputs (PR/ROC curves, agreement matrix)
│   ├── figures/             # Attention, SHAP, training visualizations
│   └── docs/                # Project report
├── tests/                   # Unit tests
└── data/                    # Raw and processed data (gitignored)
```

## Preprocessing Pipeline

Raw PSV files are transformed through: forward-fill imputation, missingness masks, time-since features, and z-score normalization. Each patient's 40 clinical variables become 120 features per timestep. Results are cached with hash-based validation for reproducibility.

## Evaluation

The PhysioNet 2019 utility score rewards early sepsis detection (optimal at 6h before onset) and penalizes late/missed predictions and false alarms. This is used instead of standard metrics like accuracy or AUROC because it directly encodes clinical value.

## Hardware

- **University JupyterLab**: 8x NVIDIA A16 (15GB VRAM each), 144 CPU cores (2x Xeon Platinum 8352V), 251GB RAM
- **Local**: NVIDIA RTX PRO 1000 (8GB VRAM), 32GB RAM, Intel Ultra 7
- TCN training: ~30 hours (13 epochs, NVIDIA A16)
- GRU training: ~70 minutes per run (RTX 1000)
- LightGBM training: ~5 minutes
- RQ evaluation: ~10-15 minutes
- SHAP analysis (1000 bg, 1000 test): ~15 minutes
