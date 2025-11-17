# Codebase Restructuring Proposal

## Current Issues

### 1. **Code Duplication**
- `ml_models.py` and `region_fingerprint_dpp.py` contain nearly identical code
- Three training scripts (`AIMS_fingerprint_*.py`) have massive code duplication in data loading, model initialization, and training loops

### 2. **Poor Separation of Concerns**
- Data transformations scattered across multiple files
- Model initialization mixed with training logic
- Training utilities not modular (hard to reuse)

### 3. **Lack of Structure**
- No clear package organization
- No configuration management
- Hardcoded paths and hyperparameters throughout

### 4. **Difficult to Maintain**
- Changes must be made in 3+ places
- Difficult to test individual components
- Hard to extend functionality

---

## Proposed Restructure

### Directory Structure
```
generalized_fingerprinting/
├── models/
│   ├── __init__.py
│   ├── core.py              # PerturbedTopK, Scorer, DPS
│   ├── classifiers.py       # TransformerClassifier, FlexibleMLP
│   └── feature_extractors.py # DINOv3FeatureExtractor
│
├── data/
│   ├── __init__.py
│   ├── transforms.py        # All augmentation classes
│   ├── loaders.py           # DataLoader utilities, SubsetInMemoryDataset
│   └── datasets.py          # Custom dataset classes
│
├── training/
│   ├── __init__.py
│   ├── trainer.py           # Main training loop (single implementation)
│   ├── validators.py        # Validation/testing logic
│   ├── callbacks.py         # EarlyStopping, logging utilities
│   └── distributed.py       # DDP utilities, gathering functions
│
├── configs/
│   ├── __init__.py
│   ├── base.py              # Base configuration class
│   ├── models.py            # Model configurations
│   └── training.py          # Training configurations
│
├── scripts/
│   ├── train_baseline.py    # Simplified baseline training
│   ├── train_design_sweep.py # Design-based split training
│   └── train_dpp.py         # DPP-based training
│
├── utils/
│   ├── __init__.py
│   ├── seeds.py             # Random seed utilities
│   ├── paths.py             # Path management
│   └── logging.py           # Logging setup
│
├── README.md
├── requirements.txt
└── setup.py
```

---

## Benefits of This Structure

### **1. Modularity**
- Each module has a single responsibility
- Easy to import and use individual components
- Easy to test each component independently

### 2. Reusability**
- Write training once, use everywhere
- Data transforms are standardized
- Model initialization is centralized

### 3. Maintainability**
- Changes in one place affect all scripts
- Clear dependencies between modules
- Easy to add new models/transforms

### 4. Scalability**
- Add new training scripts without code duplication
- Easy to add new models or data sources
- Simple to extend for new use cases

### 5. Professionalism**
- Industry-standard layout
- Better for collaboration
- Easier to package and distribute
- Simpler CI/CD integration

---

## Key Implementation Steps

### Step 1: Create Core Modules
Extract:
- `models/core.py` from `fingerprint_proposal.py`
- `models/classifiers.py` from `fingerprint_proposal.py`
- `data/transforms.py` from training scripts

### Step 2: Create Configuration System
```python
# configs/base.py
class Config:
    model_name: str
    learning_rate: float
    batch_size: int
    num_epochs: int
    ...
```

### Step 3: Create Unified Trainer
```python
# training/trainer.py
class DistributedTrainer:
    def train(self, model, dataloaders, config):
        # Single implementation used by all scripts
```

### Step 4: Create Training Scripts
Each script becomes thin wrapper:
```python
# scripts/train_design_sweep.py
config = DesignSweepConfig(...)
trainer = DistributedTrainer(config)
trainer.train(model, dataloaders, config)
```

### Step 5: Clean Up Utilities
Move all utility functions to `utils/`

---

## Migration Path

**Phase 1:** Create new package structure (keep old files)
**Phase 2:** Migrate one training script, verify it works
**Phase 3:** Migrate other scripts
**Phase 4:** Remove old files
**Phase 5:** Add tests and documentation

---

## Code Quality Improvements

### Immediate (No restructuring needed)
- ✅ Remove AI-generated comments (DONE)
- [ ] Add type hints throughout
- [ ] Add docstrings to public functions
- [ ] Extract magic numbers to constants

### After Restructuring
- [ ] Add unit tests for each module
- [ ] Add integration tests for training pipeline
- [ ] Add example usage scripts
- [ ] Create API documentation
- [ ] Add configuration validation

---

## Example: Before vs After

### Before (Duplicated)
```python
# AIMS_fingerprint_dpp.py (900+ lines)
class CLAHE(torch.nn.Module): ...
class RandomNoise(torch.nn.Module): ...
def run_model(...): ... # 500+ lines

# AIMS_fingerprint_design_sweep.py (900+ lines)
class CLAHE(torch.nn.Module): ...  # DUPLICATE
class RandomNoise(torch.nn.Module): ...  # DUPLICATE
def run_model(...): ... # 500+ lines, 90% similar
```

### After (Single Source of Truth)
```python
# data/transforms.py
class CLAHE(torch.nn.Module): ...
class RandomNoise(torch.nn.Module): ...

# training/trainer.py
class DistributedTrainer:
    def train(...): ...

# scripts/train_dpp.py
from models import DPS
from data import get_dataloaders
from training import DistributedTrainer

trainer = DistributedTrainer(config)
trainer.train(...)

# scripts/train_design_sweep.py (reuses same trainer)
trainer = DistributedTrainer(config)
trainer.train(...)
```

---

## Implementation Priority

1. **High Impact, Low Effort:** Extract transforms to `data/transforms.py`
2. **High Impact, Medium Effort:** Create unified `training/trainer.py`
3. **Medium Impact, Medium Effort:** Create `configs/` system
4. **Low Effort:** Create directory structure and move files

