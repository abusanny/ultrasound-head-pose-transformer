# Project Structure

This document describes the directory organization and development workflow for the **Ultrasound Head Pose Estimation with Tiny Transformer (6DoF)** project.

## Directory Tree

```
ultrasound-head-pose-transformer/
├── src/                          # Source code for training, evaluation, and inference
│   ├── __init__.py              # Package initialization
│   ├── data_loader.py           # Dataset loading, preprocessing, and augmentation
│   ├── preprocessing.py         # Image normalization, windowing, standardization
│   ├── model.py                 # USTransformer6DoFSimple architecture definition
│   ├── train.py                 # Training script with callbacks and logging
│   ├── evaluate.py              # Validation and test set evaluation
│   ├── predict.py               # Inference pipeline for new scans
│   └── utils.py                 # Utility functions (standardization, conversions, metrics)
│
├── notebooks/                    # Jupyter notebooks for experimentation
│   └── Transformation_matrix_Tiny_Transformer_from_Scratch.ipynb
│       # Full development notebook: data exploration, model design, training, results
│
├── data/                         # Data directory (git-ignored)
│   ├── raw/                     # Raw ultrasound scan files (.nii, .dcm, or custom format)
│   │   ├── scan_001/
│   │   ├── scan_002/
│   │   └── ...
│   ├── processed/               # Preprocessed sequences (sliding windows)
│   │   └── train_sequences.pkl
│   └── splits/                  # Train/val/test split definitions
│       └── split_config.json
│
├── models/                       # Trained model checkpoints (git-ignored)
│   ├── best_model.h5            # Best model based on validation loss
│   ├── latest_checkpoint.h5     # Latest checkpoint during training
│   └── model_summary.txt        # Model architecture summary
│
├── results/                      # Evaluation results and predictions (git-ignored)
│   ├── metrics/                 # Training/validation/test metrics
│   │   ├── train_metrics.json
│   │   ├── val_metrics.json
│   │   └── test_metrics.json
│   ├── predictions/             # Per-scan prediction CSVs
│   │   ├── scan_001_predictions.csv
│   │   ├── scan_002_predictions.csv
│   │   └── ...
│   └── visualizations/          # Plots and analysis figures
│       ├── training_loss.png
│       ├── pose_trajectories.png
│       └── attention_weights.png
│
├── logs/                         # Training and inference logs (git-ignored)
│   ├── training_log_YYYY-MM-DD_HH-MM-SS.txt
│   └── inference_log.txt
│
├── config/                       # Configuration files
│   ├── __init__.py
│   ├── default_config.py        # Default hyperparameters and paths
│   └── experiment_config.yaml   # Experiment-specific settings (optional)
│
├── requirements.txt              # Python dependencies (pip installable)
├── README.md                     # Project overview and usage guide
├── PROJECT_STRUCTURE.md          # This file
├── .gitignore                    # Git ignore rules (excludes data/, models/, results/, logs/)
└── LICENSE                       # (Optional) License file
```

## File Descriptions

### Source Code (`src/`)

**data_loader.py**
- Functions for discovering ultrasound scans from directory structure
- Loading 6DoF labels from CSV files
- Creating sliding-window sequences (WINDOW=3, STRIDE=1)
- Building tf.data.Dataset pipelines with batching and prefetching
- Handles train/val/test splits

**preprocessing.py**
- Image loading and resizing (224×224)
- Grayscale normalization (0-1 range or z-score)
- Label standardization (Z-score normalization for training)
- Label unstandardization (reverse transform for inference)
- Angle conversion: radians ↔ degrees

**model.py**
- Defines `USTransformer6DoFSimple` Keras/TensorFlow model
- CNN frame encoder with temporal pooling
- MultiHeadAttention with configurable heads and dimensions
- Positional embeddings for sequence context
- Final Dense layers for 6DoF regression output
- Model summary and parameter count

**train.py**
- Main training loop
- Learning rate scheduling and early stopping
- Model checkpointing (best validation loss)
- Training history logging
- Command-line argument parsing for hyperparameters
- Example usage: `python src/train.py --epochs 30 --batch_size 8 --learning_rate 1e-4`

**evaluate.py**
- Validation and test evaluation on tf.data.Dataset
- Computes loss and MAE metrics
- Generates confusion matrices or metric summaries (for classification)
- Saves results to JSON in `results/metrics/`
- Example usage: `python src/evaluate.py --model_path models/best_model.h5 --data_split val`

**predict.py**
- Loads trained model and performs inference
- Handles single-scan or batch prediction
- Merges overlapping sliding-window predictions
- Unstandardizes predictions and converts units
- Outputs per-frame [rx, ry, rz, tx, ty, tz] to CSV
- Example usage: `python src/predict.py --scan_path data/raw/scan_001 --model_path models/best_model.h5 --output_csv results/predictions/scan_001.csv`

**utils.py**
- Standardization/unstandardization helper functions
- Angle conversion utilities
- Metric calculation functions (MAE, MSE, RMSE)
- Visualization helpers (plot training curves, pose trajectories)
- File I/O utilities

### Data (`data/`)

**raw/**: Original ultrasound scans in native format (.nii, .dcm, or custom)
- Each scan is in its own subdirectory with accompanying CSV labels
- CSV format: frame_id, rx, ry, rz, tx, ty, tz

**processed/**: Preprocessed sliding-window sequences (pickle format)
- Accelerates data loading during training
- Optional; can be regenerated on-the-fly from raw/

**splits/**: Train/val/test split configuration
- Ensures reproducibility across experiments

### Models (`models/`)

- `best_model.h5`: Best model weights (lowest validation loss)
- Typically generated during training with EarlyStopping callback
- Can be loaded with `tf.keras.models.load_model()`

### Results (`results/`)

**metrics/**: JSON files with evaluation metrics
- Useful for tracking experiment performance over time

**predictions/**: CSV files with per-frame pose predictions
- Used for downstream analysis, visualization, and comparison with ground truth

**visualizations/**: PNG/PDF plots
- Training curves, loss evolution, MAE per DOF, attention heatmaps, etc.

### Logs (`logs/`)

- Text logs from training and inference runs
- Timestamped for easy organization
- Contains stdout redirects, error messages, and debug info

### Config (`config/`)

**default_config.py**: Central location for hyperparameters
```python
DATAROOT = 'data/raw'
BATCH_SIZE = 8
EPOCHS = 30
BASE_LR = 1e-4
WINDOW = 3
STRIDE = 1
DMODEL = 128
NUMHEADS = 4
MLPDIM = 256
DROPOUT = 0.1
STANDARDIZE_Y = True
ANGLES_IN_DEGREES = True
```

## Workflow

### 1. Setup
```bash
git clone https://github.com/abusanny/ultrasound-head-pose-transformer.git
cd ultrasound-head-pose-transformer
pip install -r requirements.txt
```

### 2. Prepare Data
- Place raw ultrasound scans in `data/raw/` with subdirectories per scan
- Include CSV label files with 6DoF annotations per frame
- (Optional) Run preprocessing: `python src/preprocessing.py --input data/raw/ --output data/processed/`

### 3. Train Model
```bash
python src/train.py \
    --epochs 30 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --model_checkpoint models/best_model.h5 \
    --log_dir logs/
```

### 4. Evaluate
```bash
python src/evaluate.py \
    --model_path models/best_model.h5 \
    --data_split val \
    --output_json results/metrics/val_metrics.json
```

### 5. Inference
```bash
python src/predict.py \
    --scan_dir data/raw/ \
    --model_path models/best_model.h5 \
    --output_dir results/predictions/
```

### 6. Visualize & Analyze
```bash
# Optional: Generate plots
python src/utils.py --plot_type training_curves --input logs/
python src/utils.py --plot_type pose_trajectories --input results/predictions/ --output results/visualizations/
```

## Development Guidelines

1. **Git Workflow**:
   - Keep `.gitignore` entries (`data/`, `models/`, `results/`, `logs/`) to avoid bloating the repository
   - Commit code changes and config files only
   - Document major changes in commit messages

2. **Testing**:
   - Test data loading and preprocessing on small subsets first
   - Validate model architecture with `model.summary()`
   - Run evaluation on validation set after training

3. **Logging**:
   - Enable TensorFlow logging: `tf.get_logger().setLevel('ERROR')`
   - Use Python `logging` module for custom application logs
   - Include timestamps and run metadata in log filenames

4. **Configuration**:
   - Always parametrize hyperparameters (avoid hardcoding)
   - Use `config/default_config.py` as a single source of truth
   - Document non-obvious configuration choices in README

5. **Documentation**:
   - Update README when adding new features
   - Include docstrings in all functions
   - Link to relevant papers or references in comments

## Troubleshooting

- **Out of memory during training**: Reduce `BATCH_SIZE`, use gradient checkpointing, or enable mixed precision
- **Nan loss**: Check label standardization, learning rate, and data normalization
- **Slow data loading**: Preprocess data into `data/processed/` or enable multi-worker DataLoader
- **Poor predictions**: Verify model checkpoint path, ensure label standardization is reversed at inference

## Future Extensions

- Add multi-GPU training support via `tf.distribute.Strategy`
- Implement custom training loops for gradient accumulation
- Add dataset augmentation (rotation, translation) to `preprocessing.py`
- Support for different backbone architectures (ResNet, EfficientNet)
- Export model to ONNX or TFLite for deployment

---

**Last Updated:** 2024
