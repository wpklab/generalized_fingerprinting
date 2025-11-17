# Restructuring Implementation Examples

This document provides concrete code examples for implementing the proposed restructuring.

---

## Module 1: Data Transforms (`data/transforms.py`)

Extract all augmentation classes into a single module:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import random
from typing import Tuple, Optional

class PadToSize:
    def __init__(self, size: Tuple[int, int], fill: int = 0):
        self.size = size
        self.fill = fill

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        padding = [
            (self.size[1] - img.size()[2]) // 2,
            (self.size[0] - img.size()[1]) // 2,
            (self.size[1] - img.size()[2] + 1) // 2,
            (self.size[0] - img.size()[1] + 1) // 2
        ]
        return torch.nn.functional.pad(img, padding, value=self.fill)


class RandomTranslateWithPadding:
    def __init__(self, max_translate: float):
        self.max_translate = max_translate

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape[-2:]
        max_dx = int(self.max_translate * w)
        max_dy = int(self.max_translate * h)
        dx = torch.randint(-max_dx, max_dx + 1, (1,)).item()
        dy = torch.randint(-max_dy, max_dy + 1, (1,)).item()

        translated = torch.zeros_like(img)
        x1, x2 = max(0, dx), min(w, w + dx)
        y1, y2 = max(0, dy), min(h, h + dy)
        ox1, ox2 = max(0, -dx), min(w, w - dx)
        oy1, oy2 = max(0, -dy), min(h, h - dy)

        translated[..., y1:y2, x1:x2] = img[..., oy1:oy2, ox1:ox2]
        return translated


class CLAHE(nn.Module):
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        super().__init__()
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        img_np = img.cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * 255).astype(np.uint8)
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        
        l, a, b = cv2.split(img_lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        cl = clahe.apply(l)
        
        img_clahe = cv2.merge((cl, a, b))
        img_result = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2RGB)
        return torch.from_numpy(img_result.transpose(2, 0, 1).astype(np.float32) / 255.0)


class RandomDownscaleUpscaleTensor:
    def __init__(self, min_scale: float = 0.5, max_scale: float = 1.5):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        c, h, w = img.shape
        scale = random.uniform(self.min_scale, self.max_scale)
        img_batch = img.unsqueeze(0)

        if scale <= 1.0:
            new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
            img_down = F.interpolate(img_batch, size=(new_h, new_w), mode='bilinear', align_corners=False)
            img_result = F.interpolate(img_down, size=(h, w), mode='bilinear', align_corners=False)
        else:
            new_h, new_w = int(h * scale), int(w * scale)
            img_up = F.interpolate(img_batch, size=(new_h, new_w), mode='bilinear', align_corners=False)
            h_start, w_start = (new_h - h) // 2, (new_w - w) // 2
            img_result = img_up[:, :, h_start:h_start+h, w_start:w_start+w]

        return img_result.squeeze(0)


class RandomExposureSimulator(nn.Module):
    def __init__(self, brightness: Tuple[float, float] = (0.5, 1.5),
                 contrast: Tuple[float, float] = (0.5, 1.5),
                 gamma: Tuple[float, float] = (0.5, 2.0)):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.gamma = gamma

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        import torchvision.transforms.functional as TF
        
        brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
        img = TF.adjust_brightness(img, brightness_factor)
        
        contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
        img = TF.adjust_contrast(img, contrast_factor)
        
        gamma_factor = random.uniform(self.gamma[0], self.gamma[1])
        img = TF.adjust_gamma(img, gamma_factor)
        
        return img


class RandomNoise(nn.Module):
    def __init__(self, noise_types: list = None, max_intensity: float = 0.1):
        super().__init__()
        self.noise_types = noise_types or ['gaussian', 'salt_pepper']
        self.max_intensity = max_intensity

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        noise_type = random.choice(self.noise_types)
        
        if noise_type == 'gaussian':
            noise = torch.randn_like(img) * random.uniform(0, self.max_intensity)
            return torch.clamp(img + noise, 0, 1)
        elif noise_type == 'salt_pepper':
            img = img.clone()
            noise = torch.rand_like(img)
            img[noise > 1 - self.max_intensity/2] = 1
            img[noise < self.max_intensity/2] = 0
            return img
        return img


class PerImageNormalize(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(img, dim=(1, 2), keepdim=True)
        std = torch.std(img, dim=(1, 2), keepdim=True) + self.eps
        return (img - mean) / std


class RandomApply(nn.Module):
    def __init__(self, transform: nn.Module, p: float = 0.5):
        super().__init__()
        self.transform = transform
        self.p = p

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return self.transform(img)
        return img


def build_train_transforms(config) -> torch.nn.Sequential:
    import torchvision.transforms as transforms
    
    return torch.nn.Sequential(
        RandomApply(transforms.RandomHorizontalFlip(), p=0.5),
        RandomApply(transforms.RandomVerticalFlip(), p=0.5),
        CLAHE(clip_limit=3.0, tile_grid_size=(config.clahe_grid, config.clahe_grid)),
        PerImageNormalize(),
        RandomApply(RandomTranslateWithPadding(max_translate=0.25), p=0.5),
        RandomApply(transforms.RandomRotation(degrees=180), p=0.5),
        RandomApply(RandomExposureSimulator(brightness=(0.75, 1.25), 
                                           contrast=(0.75, 1.25)), 
                   p=config.contrast_mod),
        RandomApply(RandomDownscaleUpscaleTensor(min_scale=0.75, max_scale=1.25), 
                   p=config.scale_mod),
        RandomApply(RandomNoise(max_intensity=config.sensor_noise), p=0.5),
    )


def build_val_transforms() -> torch.nn.Sequential:
    import torchvision.transforms as transforms
    
    return torch.nn.Sequential(
        transforms.Resize((2550, 2550)),
        CLAHE(clip_limit=3.0),
        PerImageNormalize(),
    )
```

**Usage:**
```python
from data.transforms import build_train_transforms, build_val_transforms

config = Config(clahe_grid=64, contrast_mod=0.3, scale_mod=0.1, sensor_noise=0.05)
train_aug = build_train_transforms(config)
val_aug = build_val_transforms()
```

---

## Module 2: Configuration System (`configs/base.py`)

Centralized configuration:

```python
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class DataConfig:
    data_dir: Path = Path('data')
    batch_size: int = 32
    num_workers: int = 4
    test_samples: int = 8
    image_size: int = 448
    high_res_size: int = 5100
    
    clahe_grid: int = 64
    contrast_mod: float = 0.3
    scale_mod: float = 0.1
    sensor_noise: float = 0.05


@dataclass
class ModelConfig:
    name: str = 'eva02'
    num_classes: int = 6
    freeze_backbone: bool = False
    
    n_layer: int = 2
    n_token: int = 1
    n_head: int = 16
    d_k: int = 64
    d_v: int = 64
    d_model: int = 1024
    d_inner: int = 4096
    attn_dropout: float = 0.1
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    learning_rate: float = 0.0005
    weight_decay: float = 0.01
    num_epochs: int = 50
    
    optimizer: str = 'adamw'
    scheduler: str = 'onecyclelr'
    
    log_interval: int = 5
    early_stopping_patience: int = 50
    early_stopping_delta: float = 0.0001
    
    mixed_precision: bool = True
    gradient_clip_value: Optional[float] = None
    
    sigma: float = 0.01
    lr_gamma: float = 0.99


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    project_name: str = 'generalized_fingerprinting'
    wandb_entity: str = 'wpklab'
    
    def to_dict(self):
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
        }


@dataclass
class DesignSweepConfig(Config):
    train_design: str = 'Connector'
    num_train_designs: int = 1
    training_parts_num: int = 1
```

**Usage:**
```python
from configs import Config, ModelConfig, TrainingConfig, DataConfig

config = Config(
    model=ModelConfig(name='eva02', num_classes=6),
    training=TrainingConfig(learning_rate=0.0005, num_epochs=50),
    data=DataConfig(batch_size=32, clahe_grid=64)
)
```

---

## Module 3: Unified Trainer (`training/trainer.py`)

Single implementation used by all scripts:

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import logging
from typing import Tuple, Dict
import numpy as np


class DistributedTrainer:
    def __init__(self, config, rank: int = 0, world_size: int = 1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}')
        
        self.logger = self._setup_logger()
        self.scaler = torch.amp.GradScaler()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        if self.rank == 0:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(handler)
        return logger

    def _create_optimizer(self, model: nn.Module):
        if self.config.training.optimizer == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == 'lamb':
            import torch_optimizer as optim_custom
            optimizer = optim_custom.Lamb(
                model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
        
        return optimizer

    def _create_scheduler(self, optimizer: optim.Optimizer, num_steps: int):
        if self.config.training.scheduler == 'onecyclelr':
            return OneCycleLR(
                optimizer,
                max_lr=self.config.training.learning_rate,
                total_steps=num_steps,
                pct_start=0.3,
                anneal_strategy='cos',
                div_factor=25,
                final_div_factor=1000
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.training.scheduler}")

    def train_epoch(self, 
                   model: nn.Module,
                   train_loader: DataLoader,
                   optimizer: optim.Optimizer,
                   criterion: nn.Module) -> Tuple[float, float]:
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad()

            if self.config.training.mixed_precision:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                if self.config.training.gradient_clip_value:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        self.config.training.gradient_clip_value
                    )
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                if self.config.training.gradient_clip_value:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config.training.gradient_clip_value
                    )
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += inputs.size(0)

        epoch_loss = total_loss / total_samples
        epoch_acc = total_correct / total_samples
        
        return epoch_loss, epoch_acc

    def validate(self,
                model: nn.Module,
                val_loader: DataLoader,
                criterion: nn.Module) -> Tuple[float, float]:
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if self.config.training.mixed_precision:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                total_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                total_correct += (preds == labels).sum().item()
                total_samples += inputs.size(0)

        epoch_loss = total_loss / total_samples
        epoch_acc = total_correct / total_samples
        
        return epoch_loss, epoch_acc

    def fit(self,
           model: nn.Module,
           train_loader: DataLoader,
           val_loader: DataLoader,
           num_epochs: int = None):
        
        num_epochs = num_epochs or self.config.training.num_epochs
        criterion = nn.CrossEntropyLoss()
        optimizer = self._create_optimizer(model)
        scheduler = self._create_scheduler(optimizer, len(train_loader) * num_epochs)
        
        model = model.to(self.device)
        best_val_acc = 0.0
        best_model_wts = None

        if self.rank == 0:
            self.logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc = self.validate(model, val_loader, criterion)
            scheduler.step()

            if self.rank == 0:
                self.logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_wts = model.state_dict().copy()

        if best_model_wts is not None and self.rank == 0:
            self.logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
            model.load_state_dict(best_model_wts)

        return model, best_val_acc
```

**Usage:**
```python
from training.trainer import DistributedTrainer

trainer = DistributedTrainer(config, rank=0, world_size=1)
model, best_acc = trainer.fit(model, train_loader, val_loader)
```

---

## Module 4: Simplified Training Script (`scripts/train_dpp.py`)

Before: 900 lines of duplicated code
After: Simple and clear

```python
import torch
import torch.distributed as dist
from pathlib import Path
import wandb
import os

from configs import Config, ModelConfig, TrainingConfig, DataConfig
from data.transforms import build_train_transforms, build_val_transforms
from data.loaders import get_dataloaders
from models import create_model
from training.trainer import DistributedTrainer
from fingerprint_proposal import DPS
from utils.seeds import set_seed


def main():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
    
    set_seed(42 + rank)
    
    config = Config(
        model=ModelConfig(name='eva02', num_classes=6),
        training=TrainingConfig(learning_rate=0.0005, num_epochs=50),
        data=DataConfig(batch_size=32, clahe_grid=64)
    )
    
    if rank == 0:
        wandb.init(project=config.project_name, config=config.to_dict())
    
    train_aug = build_train_transforms(config)
    val_aug = build_val_transforms()
    
    train_loader, val_loader = get_dataloaders(
        config,
        train_transforms=train_aug,
        val_transforms=val_aug
    )
    
    model = create_model(config)
    
    trainer = DistributedTrainer(config, rank=rank, world_size=world_size)
    model, best_acc = trainer.fit(model, train_loader, val_loader)
    
    if rank == 0:
        torch.save(model.state_dict(), f'best_model_{wandb.run.id}.pth')
        wandb.log({'best_accuracy': best_acc})
        wandb.finish()
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
```

---

## Benefits Demonstrated

### Code Reduction
- **Before:** 900 lines (+ 800 in other scripts + 1000 in utilities)
- **After:** 50 lines (+ shared 1500 lines in modules)
- **Reduction:** 70% for training scripts

### Duplication Elimination
- **Transforms:** 1 copy (was 3) → 2 fewer updates needed
- **Training logic:** 1 implementation (was 3) → 2 fewer bugs to fix
- **Utilities:** 1 location (was 2) → No more sync issues

### Maintainability
- Bug fixes apply everywhere automatically
- New features only need to be added once
- Easy to add new augmentations
- Simple to add new model architectures

---

## Migration Strategy

1. Create new modules alongside existing code
2. Migrate one script to use new modules
3. Verify results match original
4. Migrate remaining scripts
5. Remove old duplicated code
6. Update documentation

This approach ensures zero downtime and easy rollback if needed.

