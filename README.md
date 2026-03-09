# Sepsis Forecasting from ICU Data

Early prediction of sepsis using deep learning on the [PhysioNet/Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/) dataset.

## Results

| Model | Utility Score | Threshold | Parameters |
|-------|--------------|-----------|------------|
| GRU + Attention (Optuna best) | **0.329** | TBD | ~350K |
| GRU + Attention (optimized.yaml) | 0.322 | 0.65 | 225K |
| GRU + Attention (default.yaml) | 0.176 | 0.90 | 212K |
| LightGBM Baseline | 0.137 | 0.15 | N/A |

*PhysioNet 2019 Challenge winners achieved 0.36-0.43 utility.*

### Key Features
- Bidirectional GRU with Bahdanau temporal attention
- 120 input features per timestep (40 raw + 40 missingness masks + 40 time-since)
- PhysioNet utility score optimization with threshold search
- SHAP + attention-based explainability analysis
- Optuna hyperparameter optimization

## Dataset Statistics

| Property | Value |
|----------|-------|
| Total patients | 40,336 |
| Training patients | 34,285 |
| Validation patients | 6,051 |
| Training samples | 1,315,556 |
| Validation samples | 236,654 |
| Sepsis prevalence (hourly) | ~1.79% |
| Positive class weight | 54.86 |
| Features | 120 (40 raw + 40 masks + 40 time_since) |

## LightGBM Baseline

- **Utility**: 0.137 (threshold: 0.15)
- **Training loss**: 0.097 (train), 0.101 (val)
- **Features**: 120 base + 40 rolling/trend = 160 total

Top 10 features by importance (gain):
1. ICULOS (4,089,459)
2. PaCO2_time_since (1,189,753)
3. Temp_rolling_max (645,124)
4. HospAdmTime (596,571)
5. HR_rolling_max (252,669)
6. MAP_rolling_min (234,893)
7. Temp (199,848)
8. WBC (184,819)
9. BUN (176,441)
10. AST (154,451)

## GRU + Attention (Optimized)

- **Utility**: 0.322 (threshold: 0.65)
- **Parameters**: 225,265 (all trainable)
- **Training**: 36 epochs (early stopping at epoch 26), ~70 minutes
- **Validation loss**: 1.3134
- **Positive class weight**: 54.86

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
# Clone the repository
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

# Train GRU model (default config)
python -m src.training.train --config configs/default.yaml

# Train with optimized hyperparameters
python -m src.training.train --config configs/optimized.yaml

# Train with Optuna best config
python -m src.training.train --config configs/best.yaml

# Run hyperparameter optimization
python -m src.training.hyperopt --config configs/default.yaml --n_trials 50

# Train LightGBM baseline
python -m src.training.train_baseline --config configs/default.yaml

# Run explainability analysis
python -m src.explainability.run_analysis --config configs/optimized.yaml

# Run tests
pytest tests/
```

## Project Structure
```
sepsis-forecasting/
├── src/
│   ├── data/             # Data loading, preprocessing, caching
│   ├── models/           # GRU + Attention, LightGBM baseline
│   ├── training/         # Training loop, hyperopt, utility scoring
│   └── explainability/   # SHAP analysis, attention visualization
├── configs/
│   ├── default.yaml      # Base configuration
│   ├── optimized.yaml    # Optimized hyperparameters (Trial 0)
│   └── best.yaml         # Best hyperparameters from Optuna (Trial 8)
├── tests/                # Unit tests
└── outputs/              # Models, figures, logs (gitignored)
```

## Architecture

The primary model is a bidirectional GRU with Bahdanau temporal attention:

```
Input (batch, 24, 120)
  → LayerNorm
  → Bidirectional GRU (hidden=256, 1 layer)
  → Temporal Attention (Bahdanau, dim=64)
  → FC (512 → 64 → 1)
```

Evaluation uses the official PhysioNet 2019 normalized utility score, which rewards early sepsis detection (optimal at 6 hours before onset) and penalizes late/missed predictions and false alarms.

## Dataset

- **Source**: PhysioNet/CinC Challenge 2019
- **Size**: 40,336 ICU patients (20,336 setA + 20,000 setB)
- **Features**: 40 clinical variables (8 vitals, 26 labs, 6 demographics) measured hourly
- **Missingness**: 80-95% in lab values
- **Sepsis prevalence**: ~1.79% (hourly), ~7% (patient-level)
- **Preprocessing**: Forward-fill imputation, missingness masks, time-since-measurement features
