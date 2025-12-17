# Ultrasound Head Pose Estimation with Tiny Transformer (6DoF)

## Overview

This project implements a lightweight Transformer-based architecture for 6 Degree-of-Freedom (6DoF) head pose estimation from ultrasound image sequences. The model predicts head orientation angles (roll, pitch, yaw: rx, ry, rz) and 3D position coordinates (tx, ty, tz) from consecutive ultrasound frames using multi-head attention mechanisms.

**Key Features:**
- Lightweight CNN-based frame encoder with temporal attention
- Single Transformer encoder block with positional embeddings
- Efficient sliding-window preprocessing for sequence modeling
- Mean-squared-error (MSE) loss with early stopping and model checkpointing
- Supports both radians and degrees for angle representation

## Problem Statement

Head pose estimation is crucial for surgical guidance, fetal monitoring, and ultrasound probe calibration. Traditional frame-by-frame methods fail to capture temporal coherence. This project leverages sequence-level context through attention, enabling robust 6DoF pose predictions from limited ultrasound data (12 scans, ~100 frames per scan).

## Dataset

**Composition:**
- 12 ultrasound scans (grayscale, single channel)
- Approximately 1,200 total frames (~100 per scan)
- Sliding-window sequences extracted with `WINDOW=3` frames and `STRIDE=1`
- 6DoF labels per frame: [rx, ry, rz, tx, ty, tz]

**Data Pipeline:**
1. Scan discovery from directory structure (raw data root: `DATAROOT`)
2. CSV label loading with 6DoF poses per frame
3. Label standardization (Z-score normalization using mean/std from training set)
4. Sliding-window frame extraction (3-frame sequences)
5. Per-frame image loading, resizing to 224×224, and normalization
6. tf.data.Dataset creation with batch size 8 and prefetching

**Preprocessing Configuration:**
- Image size: 224×224
- Channels: 1 (grayscale)
- Angle representation: configurable radians (default) or degrees via `ANGLESAREDEGREES`
- Label standardization: `STANDARDIZEY=True` (applied during training, reversed during inference)

## Model Architecture

**USTransformer6DoFSimple** (~251k parameters)

### Components:

1. **Frame Encoder** (CNN):
   - Small convolutional network to embed 224×224 grayscale images into fixed-size representations
   - Outputs: `(batch, num_frames, embedding_dim)` tensors

2. **Temporal Processing** (TimeDistributed):
   - Applies frame encoder independently to each frame in the sequence

3. **Positional Embeddings**:
   - Learnable positional encodings added to frame embeddings to preserve temporal order

4. **Transformer Encoder Block** (1 layer):
   ```
   Input: (batch, seq_len=3, embedding_dim=128)
   ├─ LayerNorm
   ├─ MultiHeadAttention(num_heads=4, embedding_dim=128)
   │  └─ 4 attention heads, 32-dim each
   ├─ Add & Dropout(0.1)
   ├─ LayerNorm
   ├─ MLP (Dense-256 → GELU → Dropout(0.1) → Dense-128)
   ├─ Add & Dropout(0.1)
   └─ Output: (batch, seq_len=3, embedding_dim=128)
   ```

5. **Pose Head**:
   - LayerNorm → Dense(64, activation='gelu') → Dense(6)
   - Outputs 6DoF predictions per frame

### Hyperparameters:
- **DMODEL**: 128 (embedding dimension)
- **NUMHEADS**: 4 (attention heads)
- **MLPDIM**: 256 (MLP inner dimension)
- **DROPOUT**: 0.1
- **WINDOW**: 3 (sequence length)
- **STRIDE**: 1 (sliding-window stride)

## Training Configuration

**Hyperparameters:**
- **Batch size**: 8
- **Epochs**: 30
- **Base learning rate**: 1e-4 (Adam optimizer)
- **Loss function**: Mean Squared Error (MSE)
- **Metrics**: MAE (unstandardized, in original units)

**Training Setup:**
```python
optimizer = Adam(learning_rate=BASELR)
loss = 'mse'
metrics = ['mae']
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True,
    verbose=1
)

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)
```

**Data Split:**
- Training: 70% of sequences
- Validation: 30% of sequences
- Test: held-out scan(s) or k-fold cross-validation

## Evaluation Metrics

**On Validation Set:**
- **Loss (MSE)**: mean squared error across all 6 DOF
- **MAE (unstandardized)**: mean absolute error in original coordinate units
  - Angles (rx, ry, rz): reported in degrees (if `ANGLESAREDEGREES=True`) or radians
  - Positions (tx, ty, tz): reported in original measurement units (e.g., mm)

**Example Metrics:**
```
val_loss: 0.0342
val_mae (unstandardized): [0.18°, 0.21°, 0.15°, 2.3mm, 1.9mm, 2.1mm]
```

## Usage

### Installation

1. Clone the repository:
```bash
git clone https://github.com/abusanny/ultrasound-head-pose-transformer.git
cd ultrasound-head-pose-transformer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training

```python
python src/train.py --epochs 30 --batch_size 8 --learning_rate 1e-4
```

Configurable arguments:
- `--dataroot`: Path to ultrasound scans
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size (default: 8)
- `--learning_rate`: Base learning rate (default: 1e-4)
- `--window`: Sequence window length (default: 3)
- `--stride`: Sliding-window stride (default: 1)
- `--model_checkpoint`: Path to save best model
- `--standardize_labels`: Enable label standardization (default: True)
- `--angles_in_degrees`: Output angles in degrees (default: True)

### Evaluation

```python
python src/evaluate.py --model_path models/best_model.h5 --data_split val
```

Output:
```
Validation Loss: 0.0342
Validation MAE (unstandardized): [0.18°, 0.21°, 0.15°, 2.3mm, 1.9mm, 2.1mm]
```

### Inference (Single Scan)

```python
python src/predict.py --scan_path data/raw/scan_001.nii --model_path models/best_model.h5 --output_csv results/predictions.csv
```

Output CSV (results/predictions.csv):
```
frame_id,rx,ry,rz,tx,ty,tz
0,12.3,15.7,8.2,45.2,-12.3,89.5
1,12.5,15.8,8.1,45.3,-12.2,89.4
...
```

### Batch Prediction (Full Dataset)

```python
python src/predict.py --scan_dir data/raw/ --model_path models/best_model.h5 --output_dir results/
```

Generates per-scan prediction CSV files in `results/`.

## Project Structure

Detailed directory organization and development workflow are documented in [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

```
ultrasound-head-pose-transformer/
├── src/                          # Source code
│   ├── data_loader.py           # Dataset loading and preprocessing
│   ├── preprocessing.py         # Image normalization, windowing
│   ├── model.py                 # USTransformer6DoFSimple architecture
│   ├── train.py                 # Training script with early stopping
│   ├── evaluate.py              # Validation/test evaluation
│   ├── predict.py               # Inference on new scans
│   └── utils.py                 # Utilities (standardization, conversions)
├── notebooks/                    # Jupyter notebooks
│   └── Transformation_matrix_Tiny_Transformer_from_Scratch.ipynb
├── data/                         # Data directory (gitignored)
│   ├── raw/                     # Raw ultrasound scans
│   └── processed/               # Preprocessed sequences
├── models/                       # Trained models (gitignored)
│   └── best_model.h5
├── results/                      # Evaluation results and predictions (gitignored)
│   ├── metrics.json
│   └── predictions/
├── logs/                         # Training logs (gitignored)
│   └── training_log.txt
├── config/                       # Configuration files
│   └── default_config.py
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── PROJECT_STRUCTURE.md          # Detailed project layout
└── .gitignore                    # Git ignore rules
```

## Results

**Training Dynamics:**
- Early stopping typically triggers after 18–24 epochs
- Validation loss stabilizes around 0.03–0.05
- MAE (unstandardized): rotation angles ±0.15–0.25°, positions ±2–3mm

**Qualitative Observations:**
- Model captures smooth pose trajectories across sequences
- Attention weights show concentration on middle frame (context aggregation)
- Sliding-window overlap ensures temporal continuity in predictions

## Inference Pipeline

1. **Load scan**: Read all frames from raw ultrasound data
2. **Create sliding windows**: Generate 3-frame sequences with stride 1
3. **Predict**: Forward pass through model for each window
4. **Merge predictions**: Average overlapping window predictions per frame
5. **Standardization reversal**: Un-standardize predictions using training mean/std
6. **Unit conversion**: Convert angles to degrees if `ANGLESAREDEGREES=True`
7. **Output CSV**: Save per-frame [rx, ry, rz, tx, ty, tz] to results/

## Future Work

- **Multi-modal fusion**: Incorporate RGB probe views alongside ultrasound
- **Uncertainty quantification**: Add Bayesian layers or ensemble methods
- **Real-time streaming**: Optimize for online inference on resource-constrained devices
- **Generalization**: Test on independent test set from different ultrasound machines/probes
- **Physics-based regularization**: Add soft constraints (e.g., smoothness priors) on pose sequences
- **Extended temporal context**: Experiment with longer sequences (WINDOW > 3) and dilated convolutions

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Chollet, F. (2016). "Xception: Deep Learning with Depthwise Separable Convolutions." CVPR.
3. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.
4. Ronneberger, O., et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{abusanny2024ultrasound,
  author = {Abu Sanny},
  title = {Ultrasound Head Pose Estimation with Tiny Transformer (6DoF)},
  year = {2024},
  url = {https://github.com/abusanny/ultrasound-head-pose-transformer}
}
```

## License

This project is provided as-is for research and educational purposes.

## Contact

For questions or collaboration inquiries, please contact: [abusanny40@gmail.com](mailto:your.email@example.com) or open an issue on GitHub.

---

**Last Updated:** 2024
**Status:** Research Prototype
