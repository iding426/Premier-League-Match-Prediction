# Premier League Match Prediction

Deep learning models for predicting Premier League match outcomes and goal differences using historical performance data, expected goals (xG) metrics, and contextual features.

## Overview

This project implements two neural network architectures to predict:
- **Match result**: Home win / Draw / Away win (3-class classification)
- **Goal difference**: Predicted score differential (regression)

The models are trained on 10 years of Premier League data (2015-2024) from Understat, incorporating team performance history, advanced statistics, venue information, and weather conditions.

## Features

### Input Features (651 dimensions for MLP)

**Team History (320 features per team)**
- 20 most recent matches per team
- 14 features per match:
  - Home/away indicator
  - Goals scored/conceded
  - Expected goals (xG) for/against
  - Shots and shots on target (for/against)
  - Deep/progressive passes (for/against)
  - PPDA - Passes Per Defensive Action (for/against)

**Match Context (11 features)**
- Venue type (open-air, retractable roof, dome)
- Weather conditions (6 normalized features):
  - Temperature, precipitation, wind speed
  - Cloud cover, humidity, dome indicator
- Home team record (wins, draws, losses from historical matches)

### Models

**MLP (Multi-Layer Perceptron)**
- Architecture: 651 → 256 → 128 → 64 → 4
- Dropout regularization (0.4, 0.4, 0.3)
- Flattened team history (no temporal modeling)
- Fast training, good baseline performance

**Transformer**
- Sequence-based architecture with positional encoding
- Treats each team's 20 matches as a temporal sequence
- 41 tokens total: [20 home matches][20 away matches][1 match context]
- Better at capturing temporal patterns

## Dataset

**Source**: Understat (https://understat.com)
- **Period**: 2015-2024 seasons
- **Matches**: 3,420 Premier League games
- **Features**: 16 per-match statistics including xG, shots, possession metrics

**Data Split** (temporal to prevent leakage):
- Training: 70% (oldest matches)
- Validation: 15%
- Test: 15% (most recent matches)

## Installation

### Using Conda (Recommended)

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate soccerpred
```

### Manual Installation

```bash
conda create -n soccerpred python=3.10
conda activate soccerpred
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pandas numpy scikit-learn matplotlib seaborn tqdm jupyter -c conda-forge
pip install requests python-dotenv
```

## Usage

### Training

Train the MLP model:

```bash
bash train.sh
# Or directly:
python scripts/train.py
```

**Key hyperparameters** (in `train.py`):
- Learning rate: 5e-5
- Batch size: 64
- Epochs: 100
- Weight decay: 1e-3
- Label smoothing: 0.1
- Dropout: 0.4 / 0.4 / 0.3

Training saves checkpoints to `weights/`:
- `mlp_epoch_10.pt`, `mlp_epoch_20.pt`, ... (periodic saves)
- `mlp_best.pt` (best validation accuracy)

### Evaluation

Evaluate model on 2022-2024 test matches:

```bash
bash eval.sh
# Or directly:
python scripts/eval.py \
    --model mlp \
    --checkpoint weights/mlp_best.pt \
    --data-path data/understat_match_1524.csv \
    --output-dir eval_results
```

**Outputs**:
- Overall accuracy and per-class accuracy (win/draw/loss)
- Average goal difference error
- Sample predictions CSV with probabilities
- Confusion matrix (if plots enabled)

### Sample Predictions

Generate predictions for recent matches:

```bash
bash sample.sh
# Or directly:
python scripts/sample.py \
    --model mlp \
    --checkpoint weights/mlp_best.pt \
    --data-path data/understat_match_1524.csv \
    --output sample_predictions.csv \
    --num-samples 50 \
    --year 2024
```

**Output CSV columns**:
- Date, home team, away team
- Actual result vs predicted result
- Win/draw/loss probabilities
- Actual score vs predicted goal difference

## Project Structure

```
SoccerPred/
├── data/
│   ├── dataset.py              # PyTorch Dataset classes
│   ├── understat_match_1524.csv # Match data (2015-2024)
│   ├── venue_info.py           # Stadium locations & types
│   ├── weather_fetcher.py      # Weather API integration
│   └── weather_cache.csv       # Cached weather data
├── models/
│   ├── linear.py               # MLP architecture
│   └── transformer.py          # Transformer architecture
├── scripts/
│   ├── train.py                # Training script
│   ├── eval.py                 # Evaluation script
│   ├── sample.py               # Sample prediction generator
│   └── plots.py                # Visualization utilities
├── weights/                    # Model checkpoints
├── environment.yml             # Conda environment spec
├── train.sh                    # Training launcher
├── eval.sh                     # Evaluation launcher
├── sample.sh                   # Sampling launcher
└── README.md
```

## Key Implementation Details

### Data Leakage Prevention

- **Team records** (W-D-L) computed only from matches *before* current match
- Team history uses `_get_team_history_block(team, current_idx)` which filters `i < current_idx`
- Temporal split ensures test set is chronologically after training set

### Regularization Strategy

To combat overfitting:
- High dropout rates (0.4 → 0.4 → 0.3)
- Weight decay (L2 regularization): 1e-3
- Label smoothing: 0.1
- Reduced model capacity (651 → 256 → 128 → 64 → 4)

### Weather Integration

- Stadium locations and roof types hardcoded in `venue_info.py`
- Historical weather cached in `weather_cache.csv` to avoid repeated API calls
- Weather only fetched for open-air stadiums
- 6 normalized weather features (temperature, precipitation, wind, clouds, humidity, dome flag)

## Performance

Typical results on 2022-2024 test set:
- **Overall accuracy**: ~50-55%
- **Home win accuracy**: ~60-65%
- **Draw accuracy**: ~20-30% (hardest to predict)
- **Away win accuracy**: ~45-55%
- **Avg goal difference error**: ~1.2-1.5 goals

**Note**: Football match prediction is inherently difficult due to high randomness. Professional bookmakers achieve ~52-54% accuracy.

## Troubleshooting

### Dimension Mismatch Error

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x1291 and 651x256)
```

**Cause**: Checkpoint was trained with old input dimensions (1291) before feature optimization.

**Solution**: Retrain model with current code, or temporarily load with `MLP(input_space=1291)` to match checkpoint.

### Missing Weather Cache

If weather fetching is slow:
1. Set `fetch_weather=False` in dataset initialization
2. Or pre-generate cache using weather fetcher notebook
3. Cache path: `data/weather_cache.csv`

### Import Errors on Cluster

Add project root to path (already in scripts):
```python
import sys
import pathlib
repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
```

## Future Improvements

- [ ] Add player-level features (lineups, injuries, form)
- [ ] Incorporate betting odds as features
- [ ] Implement ensemble methods (combine MLP + Transformer)
- [ ] Add recurrent architectures (LSTM/GRU) for better temporal modeling
- [ ] Feature importance analysis
- [ ] Transfer learning from other leagues
- [ ] Real-time prediction API

## Citation

Dataset source:
```
@misc{understat2024,
  title = {Understat - Football Analytics},
  author = {Understat},
  year = {2024},
  url = {https://understat.com}
}
```

## License

MIT License - See LICENSE file for details

## Authors

- Ian Ding (@iding426)
- Mofolaoluwarera Oladipo

## Acknowledgments

- Understat for providing detailed match statistics
- Premier League for the beautiful game
- PyTorch team for the deep learning framework
