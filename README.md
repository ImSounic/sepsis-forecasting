# Sepsis Forecasting from ICU Data

Early prediction of sepsis using deep learning on the [PhysioNet/Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/) dataset.

## Results

| Model | Utility Score | Threshold |
|-------|--------------|-----------|
| **Ensemble (max)** | **0.4060** | **0.50** |
| Improved LightGBM (244 features) | 0.3976 | 0.50 |
| GRU + Attention (optimized) | 0.3374 | 0.40 |
| LightGBM Baseline (160 features) | 0.1370 | 0.15 |

*PhysioNet 2019 Challenge winners achieved 0.36-0.43 utility on the hidden test set.*

The ensemble combines the GRU and improved LightGBM using a max strategy: if either model predicts sepsis, we predict sepsis. This works because the models are highly complementary, with only 27.7% overlap in true positive detections.

### Key Features
- Bidirectional GRU with Bahdanau temporal attention (646,641 parameters)
- Improved LightGBM with 244 engineered features (rolling stats, trends, clinical interactions)
- Max ensemble achieving 0.4060 utility (within PhysioNet winner range)
- 120 input features per timestep (40 raw + 40 missingness masks + 40 time-since)
- PhysioNet utility score optimization with threshold search
- SHAP + attention-based explainability analysis
- Optuna hyperparameter optimization (12 trials)

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
| Features (GRU) | 120 (40 raw + 40 masks + 40 time_since) |
| Features (Improved LightGBM) | 244 (120 base + 124 engineered) |

**Source**: PhysioNet/CinC Challenge 2019 - 40,336 ICU patients across two hospital systems. 40 clinical variables (8 vitals, 26 labs, 6 demographics) measured hourly. Lab values exhibit 80-95% missingness.

## Models

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

# Train GRU model
python -m src.training.train --config configs/optimized.yaml

# Train LightGBM baseline
python -m src.training.train_baseline --config configs/default.yaml

# Train improved LightGBM (244 features)
python -m src.training.train_baseline --config configs/default.yaml --improved

# Evaluate ensemble strategies
python -m src.training.evaluate_ensemble --config configs/optimized.yaml
python -m src.training.evaluate_ensemble --config configs/optimized.yaml --full-analysis

# Run explainability analysis (attention + SHAP)
python -m src.explainability.run_analysis --config configs/optimized.yaml
python -m src.explainability.run_analysis --config configs/optimized.yaml --skip-shap

# Run hyperparameter optimization
python -m src.training.hyperopt --config configs/default.yaml --n_trials 50

# Run tests
pytest tests/
```

## Project Structure

```
sepsis-forecasting/
├── src/
│   ├── data/                # Data loading, preprocessing, caching
│   ├── models/              # GRU, LightGBM (baseline + improved), ensemble
│   ├── training/            # Training loops, hyperopt, ensemble evaluation
│   └── explainability/      # SHAP analysis, attention visualization
├── configs/
│   ├── default.yaml         # Base configuration
│   ├── optimized.yaml       # Optuna Trial 8 best hyperparameters
│   └── best.yaml            # Copy of optimized
├── outputs/
│   ├── models/
│   │   ├── gru/             # GRU checkpoints
│   │   ├── lightgbm/        # LightGBM baseline + improved models
│   │   └── ensemble/        # Ensemble predictions
│   ├── figures/
│   │   ├── gru/             # Attention distributions and patient plots
│   │   │   ├── attention/   # Aggregate attention weight analysis
│   │   │   └── patients/    # Per-patient plots (tp/fp/tn/fn/highest_pred)
│   │   ├── shap/            # SHAP feature importance + model comparison
│   │   ├── lightgbm/        # LightGBM feature importance
│   │   ├── ensemble/        # Model agreement visualizations
│   │   └── training/        # Training curves
│   └── docs/                # Project report
├── tests/                   # Unit tests
└── data/                    # Raw and processed data (gitignored)
```

## Preprocessing Pipeline

Raw PSV files are transformed through: forward-fill imputation, missingness masks, time-since features, and z-score normalization. Each patient's 40 clinical variables become 120 features per timestep. Results are cached with hash-based validation for reproducibility.

## Evaluation

The PhysioNet 2019 utility score rewards early sepsis detection (optimal at 6h before onset) and penalizes late/missed predictions and false alarms. This is used instead of standard metrics like accuracy or AUROC because it directly encodes clinical value.

## Hardware

- Local: NVIDIA RTX PRO 1000 (8GB VRAM), 32GB RAM, Intel Ultra 7
- GRU training: ~70 minutes per run
- LightGBM training: ~5 minutes
- SHAP analysis (1000 bg, 1000 test): ~15 minutes
