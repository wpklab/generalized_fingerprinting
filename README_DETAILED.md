# Generalized Source Identification of Additively Manufactured Parts

Code for the paper: *Learned Region Selection and Deep Learning for Generalized Source Identification of Additively Manufactured Parts*

<img width="1001" height="388" alt="image" src="https://github.com/user-attachments/assets/91fb9945-3828-49ee-8efb-f3f1bd8a3bea" />

## Model Architecture

<img width="1436" height="525" alt="image" src="https://github.com/user-attachments/assets/5f7b865f-5834-4b4d-afcd-c536cb50e567" />

## Video Inference

<img width="1452" height="485" alt="image" src="https://github.com/user-attachments/assets/487d79f9-31db-4d4e-b2c9-bce7ff412f7d" />

---

## Overview

This repository implements a deep learning system for fingerprinting 3D-printed parts to identify their source device. The approach combines learned region selection (Differentiable Patch Selection - DPS) with transformer-based classifiers and advanced data augmentation techniques.

### Key Features

- **Differentiable Patch Selection (DPS):** Automatically selects discriminative regions from high-resolution images
- **Multi-Architecture Support:** Compatible with EVA02, ConvNeXt, DINOv3, EfficientNet, ViT, and more
- **Distributed Training:** Full support for multi-GPU training with PyTorch DDP
- **Advanced Augmentation:** CLAHE, exposure simulation, noise injection, and geometric transformations
- **Experiment Tracking:** Integrated with Weights & Biases (wandb) for hyperparameter sweeps
- **Flexible Dataset Handling:** In-memory GPU caching and design-based data splits

---

## File Structure & Documentation

### Core Model Components

#### `fingerprint_proposal.py` (396 lines)
Implements the core machine learning models for differentiable patch selection:

**Key Classes:**
- **`PerturbedTopK`** - Differentiable top-k selection layer that enables gradient computation through discrete selection operations
- **`PerturbedTopKFunction`** - Custom autograd function with forward/backward passes for perturbed top-k
- **`Scorer`** - ResNet18-based scoring network that assigns importance scores to image regions
- **`DPS` (Differentiable Patch Selection)** - Main framework combining patch selection with classification
- **`Position_proposal`** - Generates candidate patch positions and extracts patches
- **`TransformerClassifier`** - Transformer-based classification head for aggregating patch predictions
- **`FlexibleMLP`** - Flexible multi-layer perceptron for classification
- **`adjust_sigma`** - Utility for dynamic sigma adjustment during training

---

### Training Utilities

#### `region_fingerprint_dpp.py` (~1000 lines)
Core training infrastructure and utilities for model training and evaluation:

**Key Functions:**
- **`initialize_model(model_name, num_classes, ...)`** 
  - Creates and configures pre-trained models with custom classifiers
  - Supports 20+ architectures: ResNet, ConvNeXt, EfficientNet, ViT, PoolFormer, etc.
  - Configurable classifier layers (simple to complex)
  - Optional backbone freezing for fine-tuning
  
- **`train_model(model, dataloaders, ...)`**
  - Main training loop with distributed training support
  - Mixed precision training (torch.amp for efficiency)
  - Early stopping callback
  - Learning rate scheduling
  - Comprehensive metric tracking and logging
  
- **`test_model(model, dataloaders, ...)`**
  - Validation/testing with per-patch and per-image predictions
  - Metrics aggregation across all GPU processes
  - Visualization support for attention maps
  
- **`gather_tensors()/gather_indices()`** 
  - Distributed communication utilities for collecting results from all GPUs
  - Handles tensors of varying sizes across processes
  
- **`EarlyStopping`** 
  - Early stopping callback with configurable patience and delta
  - Prevents overfitting

**Key Classes:**
- **`EarlyStopping`** - Monitors validation metrics and halts training when improvement plateaus

#### `ml_models.py`
Duplicate of `region_fingerprint_dpp.py` - **Note:** Should be consolidated (see RESTRUCTURING_PROPOSAL.md)

---

### Training Scripts

#### `AIMS_fingerprint_design_sweep.py` (~800 lines)
Hyperparameter sweep with design-based dataset splitting:

**Purpose:** Tests model generalization across different design types

**Key Features:**
- Separates training data by design type (Connector, Switch, King, Target, Oring, Clip, Fishbait, Plug, Minion)
- Validates on different design types not seen during training
- Useful for testing cross-design generalization
- Integrates with wandb for automated hyperparameter sweeps
- Design-based data filtering prevents data leakage

**Usage:**
```bash
wandb sweep sweep_config.yaml
python AIMS_fingerprint_design_sweep.py
```

**Output:** 
- Model checkpoint saved as `data/Models/aim_all_designs_all_views_design_separation_20_train_10_val_[run_id]_dpp.pth`
- Results logged to wandb with per-design metrics

---

#### `AIMS_fingerprint_dpp.py` (~900 lines)
Standard DPS-based training with random dataset splits:

**Purpose:** Primary training script for device fingerprinting

**Key Features:**
- Uses standard random train/val splits
- Focuses on device identification across all designs
- Generates cleaner checkpoint names for deployment
- Suitable for production training
- Integrated in-memory GPU dataset loading

**Usage:**
```bash
python AIMS_fingerprint_dpp.py
```

**Output:**
- Model checkpoint: `data/Models/[config_name]_[run_id]_dpp.pth`
- Results file: `data/Results/[config_name].txt`
- wandb metrics dashboard

---

#### `AIMS_fingerprint_original_model.py` (~800 lines)
Baseline training without design-based splits:

**Purpose:** Simple starting point for new datasets or quick experiments

**Key Features:**
- Simpler data loading pipeline
- Uses random crop augmentation
- No design-based filtering
- Good for prototyping and testing

**Usage:**
```bash
python AIMS_fingerprint_original_model.py
```

**Output:**
- Model checkpoint: `data/Models/aim_all_designs_all_views_[run_id]_dpp.pth`
- Results tracking via wandb

---

## Data Processing Components

All training scripts include the following data transformation classes (located inline):

### Augmentation Classes

| Class | Purpose | Parameters |
|-------|---------|-----------|
| **`PadToSize`** | Pads images to target dimensions | size, fill_value |
| **`RandomTranslateWithPadding`** | Random translation (up to 25% of image) | max_translate |
| **`CLAHE`** | Contrast Limited Adaptive Histogram Equalization | clip_limit, tile_grid_size |
| **`RandomDownscaleUpscaleTensor`** | Downscale then upscale (simulates compression artifacts) | min_scale, max_scale |
| **`RandomExposureSimulator`** | Adjusts brightness, contrast, gamma (exposure simulation) | brightness, contrast, gamma ranges |
| **`RandomNoise`** | Adds Gaussian or salt-pepper noise | noise_types, max_intensity |
| **`PerImageNormalize`** | Per-image z-score normalization | eps (epsilon for stability) |
| **`RandomApply`** | Applies transforms probabilistically | transform, probability |

### Dataset Classes

| Class | Purpose | Details |
|-------|---------|---------|
| **`SubsetInMemoryDataset`** | Loads dataset partition into GPU memory | Faster training, requires sufficient VRAM |
| **`DINOv3FeatureExtractor`** | Wrapper for DINOv3 backbone | Gradient control, model freezing options |

---

## Quick Start

### Prerequisites

```bash
pip install torch torchvision timm wandb opencv-python torch-optimizer
```

### Basic Training

```bash
wandb login

python AIMS_fingerprint_dpp.py
```

### Key Hyperparameters (via wandb config)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | float | 0.0005 | Learning rate |
| `model_name` | str | 'eva02' | Model architecture |
| `batch_size` | int | varies | Batch size per GPU |
| `num_epochs` | int | 50 | Training epochs |
| `test_samples` | int | 8 | Number of patches per image |
| `sigma` | float | 0.01 | Perturbation noise for top-k selection |
| `weight_decay` | float | 0.01 | L2 regularization |
| `contrast_mod` | float | 0.0 | Probability of exposure augmentation |
| `scale_mod` | float | 0.1 | Probability of scale augmentation |
| `sensor_noise` | float | 0.0 | Maximum noise intensity |
| `clahe_grid` | int | 64 | CLAHE grid size |
| `attn_dropout` | float | 0.1 | Dropout in transformer attention |
| `dropout` | float | 0.1 | General dropout rate |

### Distributed Training (Multi-GPU)

The scripts automatically detect and use all available GPUs:

```bash
srun -N 1 -n 8 python AIMS_fingerprint_dpp.py
```

**Environment Setup (automatic via SLURM):**
- `MASTER_ADDR`, `MASTER_PORT` - Set by SLURM
- `RANK`, `WORLD_SIZE` - Set by SLURM  
- `LOCAL_RANK` - Computed from SLURM variables
- `NCCL_DEBUG` - Set to INFO for debugging
- `NCCL_ALGO` - Set to Ring for NVLink topologies

---

## Model Architecture Details

### Inference Pipeline

```
High-Resolution Image (5100×5100)
         ↓
    [DPS Module]
    ├─ Scorer: Generate importance map
    ├─ PerturbedTopK: Select top-k patches
    └─ Extract & resize patches (448×448)
         ↓
[Feature Extraction] (EVA02, ConvNeXt, DINOv3, etc.)
         ↓
[Transformer Classifier] - Aggregate patch features
         ↓
[Output] - Device/source predictions
```

### Key Components

| Component | Architecture | Input | Output |
|-----------|-------------|-------|--------|
| Scorer | ResNet18 (pretrained) | High-res image | Importance map |
| PerturbedTopK | Custom autograd | Scores | Differentiable mask |
| Feature Extractor | Pre-trained backbone | Patches | Feature vectors |
| Transformer | Multi-head attention | Features | Aggregated vector |
| Classifier | Linear layer(s) | Aggregated features | Class predictions |

### Transformer Configuration

```python
TransformerClassifier(
    input_dim=1024,          # Feature dimension
    num_classes=6,           # Number of devices
    n_layer=2,              # Number of transformer layers
    n_token=1,              # Number of tokens per layer
    n_head=16,              # Number of attention heads
    d_k=64, d_v=64,         # Key/value dimensions
    d_model=1024,           # Model dimension
    d_inner=4096,           # Inner FFN dimension
    attn_dropout=0.1,       # Attention dropout
    dropout=0.1             # General dropout
)
```

---

## Dataset Structure

Expected directory layout:

```
data/aim_all_designs_all_views/
├── train/
│   ├── Stratasys450mc-1/
│   │   ├── Connector/
│   │   │   ├── Device_6_Print_001_001.jpg
│   │   │   ├── Device_6_Print_001_002.jpg
│   │   │   └── ...
│   │   ├── Switch/
│   │   └── King/
│   ├── Stratasys450mc-2/
│   │   └── (same structure)
│   └── ...
└── val/
    └── (identical structure)
```

### Class Hierarchy

- **6 Devices:** Stratasys450mc-1 through Stratasys450mc-6
- **9 Designs:** Connector, Switch, King, Target, Oring, Clip, Fishbait, Plug, Minion
- **Multiple Views:** Different imaging angles and orientations per print
- **Filename Format:** `Device_X_Print_YYY_ZZZ.jpg` where X=device ID, YYY=print ID, ZZZ=view ID

---

## Training Configuration & Optimization

### Data Augmentation Pipeline

```python
train_transforms = Compose([
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    CLAHE(clip_limit=3.0, tile_grid_size=(64, 64)),
    PerImageNormalize(),
    RandomTranslateWithPadding(max_translate=0.25),
    RandomApply(RandomRotation(degrees=180), p=0.5),
    RandomApply(RandomExposureSimulator(brightness=(0.75, 1.25), contrast=(0.75, 1.25)), p=0.3),
    RandomApply(RandomDownscaleUpscaleTensor(min_scale=0.75, max_scale=1.25), p=0.1),
    RandomApply(RandomNoise(max_intensity=0.0), p=0.5),
])
```

### Optimization Configuration

| Setting | Value | Purpose |
|---------|-------|---------|
| **Optimizer** | AdamW or Lamb | Adaptive learning with weight decay |
| **LR Scheduler** | OneCycleLR | Cosine annealing with warm-up |
| **Loss Function** | CrossEntropyLoss | Multi-class classification |
| **Mixed Precision** | torch.amp | Reduces memory, speeds training |
| **Gradient Scaling** | GradScaler | Prevents gradient underflow in FP16 |

### Learning Rate Scheduling

OneCycleLR configuration:
- **Warm-up Phase:** 30% of total steps with increasing LR
- **Decay Phase:** 70% of total steps with cosine annealing
- **Initial LR:** `max_lr / 25`
- **Final LR:** `max_lr / 1000`

### Distributed Training Setup

- **Backend:** NCCL (NVIDIA Collective Communications Library)
- **Strategy:** DistributedDataParallel (DDP)
- **Synchronization:** All-reduce for aggregating metrics
- **Data Distribution:** DistributedSampler with shuffle per epoch
- **Metrics Collection:** Custom gather functions for cross-GPU aggregation

---

## Output Files

After training completes:

```
data/Models/
└── [dataset_name]_[run_id]_dpp.pth    # Best model state dict

data/Results/
└── [dataset_name].txt                 # Metrics summary
                                        # Line 1: loss values
                                        # Line 2: accuracy values
                                        # Line 3: best accuracy
```

### Weights & Biases Logging

Automatically logged metrics:
- Training loss and accuracy per epoch
- Validation loss and accuracy
- Learning rate schedule
- Per-patch attention weight statistics
- Best validation accuracy
- Training time per epoch
- Memory usage statistics

---

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| **Out of Memory (OOM)** | Batch too large or test_samples too high | Reduce `batch_size` or `test_samples` |
| **Slow Training** | Insufficient parallelization | Increase `batch_size` or add more GPUs |
| **NaN Loss** | Unstable gradients | Reduce `lr` or increase gradient clipping |
| **Poor Generalization** | Insufficient augmentation | Increase augmentation probabilities |
| **NCCL Errors** | Communication issues | Check network, reduce batch size |
| **High Variance** | Small batch size | Increase batch size for stability |

---

## Performance Benchmarks

Typical validation accuracy on held-out test set:

| Model | Devices | Accuracy | Notes |
|-------|---------|----------|-------|
| EVA02 Enormous | 6 | ~95% | Largest model, best accuracy |
| ConvNeXt XXLarge | 6 | ~93% | Good balance of speed/accuracy |
| DINOv3 Giant | 6 | ~91% | Vision Transformer based |
| EfficientNet V2 M | 6 | ~88% | Faster inference |

*Accuracy varies based on dataset splits, augmentation settings, and hyperparameters*

---

## Recommended Workflow

### Phase 1: Quick Baseline
```bash
python AIMS_fingerprint_original_model.py
```
- Gets system running quickly
- Tests basic pipeline

### Phase 2: Production Training
```bash
python AIMS_fingerprint_dpp.py
```
- Full training with all augmentations
- Best for deployment

### Phase 3: Generalization Testing
```bash
python AIMS_fingerprint_design_sweep.py
```
- Tests cross-design generalization
- Refines augmentation strategy

### Phase 4: Hyperparameter Optimization
```bash
wandb sweep sweep_config.yaml
wandb agent [sweep_id]
```
- Find optimal hyperparameters
- Use Bayesian optimization

### Phase 5: Deploy Best Model
- Load checkpoint from Phase 2 or 3
- Use for inference on new parts

---

## Citation

If using this code, please cite:

```bibtex
@article{generalized_fingerprinting,
  title={Learned Region Selection and Deep Learning for Generalized Source Identification of Additively Manufactured Parts},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

---

## Additional Resources

- **Restructuring Proposal:** See `RESTRUCTURING_PROPOSAL.md` for planned code improvements
- **Paper:** [Link to be added]
- **Weights & Biases:** https://wandb.ai/wpklab/AIMS

---

## Disclaimer

This repository is **not plug-and-play** and requires the following modifications:

1. **Update Paths:** Modify `data_dir` and checkpoint paths in training scripts
2. **Dataset Structure:** Organize your dataset to match expected directory layout
3. **Class Names:** Update `class_names` list to match your devices
4. **Augmentation Parameters:** Tune for your specific imaging setup
5. **wandb Configuration:** Set your wandb project name and entity
6. **Batch Size:** Adjust based on your GPU memory

For questions, refer to inline script documentation or create an issue.

**Dataset:** Available on [Kaggle - link to be added]

