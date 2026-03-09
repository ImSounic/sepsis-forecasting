# Sepsis Forecasting from ICU Data

Early prediction of sepsis using deep learning on the [PhysioNet/Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/) dataset.

## Results

| Model | Utility Score | Threshold | Parameters |
|-------|--------------|-----------|------------|
| GRU + Temporal Attention | **0.3082** | 0.65 | 225K |
| LightGBM Baseline | 0.1754 | 0.15 | - |

*PhysioNet 2019 Challenge winners achieved 0.36-0.43 utility.*

### Key Features
- Bidirectional GRU with Bahdanau temporal attention
- 120 input features per timestep (40 raw + 40 missingness masks + 40 time-since)
- PhysioNet utility score optimization with threshold search
- SHAP + attention-based explainability analysis
- Optuna hyperparameter optimization

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

### Data
Download the PhysioNet 2019 Challenge data (requires free PhysioNet account):
1. Register at https://physionet.org
2. Complete required training
3. Download training_setA and training_setB
4. Place contents in `data/raw/`

## Usage

```bash
source .venv/bin/activate

# Train GRU model (default config)
python -m src.training.train --config configs/default.yaml

# Train with optimized hyperparameters
python -m src.training.train --config configs/optimized.yaml

# Run hyperparameter optimization
python -m src.training.hyperopt --config configs/default.yaml --n_trials 50

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
│   └── optimized.yaml    # Best hyperparameters from Optuna
├── tests/                # Unit tests
└── outputs/              # Models, figures, logs (gitignored)
```

## Architecture

The primary model is a bidirectional GRU with Bahdanau temporal attention:

```
Input (batch, 24, 120)
  → LayerNorm
  → Bidirectional GRU (hidden=128)
  → Temporal Attention (Bahdanau, dim=64)
  → FC (256 → 64 → 1)
```

Evaluation uses the official PhysioNet 2019 normalized utility score, which rewards early sepsis detection (optimal at 6 hours before onset) and penalizes late/missed predictions and false alarms.

## Dataset

- **Source**: PhysioNet/CinC Challenge 2019
- **Size**: ~40,000 ICU patients (20,336 setA + 20,000 setB)
- **Features**: 40 clinical variables (vitals, labs, demographics) measured hourly
- **Missingness**: 80-95% in lab values
- **Sepsis prevalence**: ~1.8%
- **Preprocessing**: Forward-fill imputation, missingness masks, time-since-measurement features
