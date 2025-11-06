import torch.utils
import torch.utils.data
import torch_optimizer as optim

import os
import pdb
import time
import numpy as np
import gc
import pandas as pd
import typing as t
import random
import sys
import subprocess
import logging
import traceback
import cv2
import timm

import matplotlib.pyplot as plt
import torch
#import torch.optim as optim
import torchvision
from torch import nn
from torchvision import datasets, transforms

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import Dataset
from socket import gethostname

# from S3N import MultiSmoothLoss

from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

from PIL import Image

## Testing out Weights and Biases
import wandb
from joblib.externals.loky.backend.context import get_context

import torch
import torchvision.transforms as transforms
import PIL.Image as Image
from matplotlib import pyplot as plt
import region_fingerprint_faster as ml_models
from fingerprint_proposal import DPS, TransformerClassifier, FlexibleMLP

from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress bar

import torch.nn.functional as F
import torchvision.transforms as T
import random

os.environ['WANDB_HTTP_TIMEOUT'] = '60'

class PadToSize:
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        padding = [
            (self.size[1] - img.size()[2]) // 2,
            (self.size[0] - img.size()[1]) // 2,
            (self.size[1] - img.size()[2] + 1) // 2,
            (self.size[0] - img.size()[1] + 1) // 2
        ]
        return transforms.functional.pad(img, padding, fill=self.fill)
    
class RandomTranslateWithPadding:
    def __init__(self, max_translate):
        self.max_translate = max_translate

    def __call__(self, img):
        h, w = img.shape[-2:]

        max_dx = int(self.max_translate * w)
        max_dy = int(self.max_translate * h)
        dx = torch.randint(-max_dx, max_dx + 1, (1,)).item()
        dy = torch.randint(-max_dy, max_dy + 1, (1,)).item()

        translated_img = torch.zeros_like(img)

        x1 = max(0, dx)
        x2 = min(w, w + dx)
        y1 = max(0, dy)
        y2 = min(h, h + dy)

        orig_x1 = max(0, -dx)
        orig_x2 = min(w, w - dx)
        orig_y1 = max(0, -dy)
        orig_y2 = min(h, h - dy)

        translated_img[..., y1:y2, x1:x2] = img[..., orig_y1:orig_y2, orig_x1:orig_x2]

        return translated_img
    
class CLAHE(torch.nn.Module):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        super().__init__()
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        
    def forward(self, img):
        img_np = img.cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * 255).astype(np.uint8)
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        
        l, a, b = cv2.split(img_lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        cl = clahe.apply(l)
        
        img_clahe = cv2.merge((cl, a, b))
        img_result = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2RGB)
        img_result = torch.from_numpy(img_result.transpose(2, 0, 1).astype(np.float32) / 255.0)
        
        return img_result
    
class DINOv3FeatureExtractor(nn.Module):
    def __init__(self, base_model, freeze_backbone=False):
        super().__init__()
        self.base_model = base_model
        
        if freeze_backbone:
            for param in self.base_model.parameters():
                param.requires_grad = False
            self.base_model.eval()
        else:
            for param in self.base_model.parameters():
                param.requires_grad = True
            self.base_model.train()
        
    def forward(self, x, **kwargs):
        outputs = self.base_model(x, output_hidden_states=True, **kwargs)
        features = outputs.pooler_output
        return features

class RandomDownscaleUpscaleTensor:
    def __init__(self, min_scale=0.5, max_scale=1.5, mode='bilinear', align_corners=False):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.mode = mode
        self.align_corners = align_corners

    def __call__(self, img):
        C, H, W = img.shape
        scale = random.uniform(self.min_scale, self.max_scale)
        
        img_batch = img.unsqueeze(0)
        
        if scale <= 1.0:
            new_H = max(1, int(H * scale))
            new_W = max(1, int(W * scale))
            img_down = F.interpolate(img_batch, size=(new_H, new_W), mode=self.mode, align_corners=self.align_corners)
            img_result = F.interpolate(img_down, size=(H, W), mode=self.mode, align_corners=self.align_corners)
        else:
            new_H = int(H * scale)
            new_W = int(W * scale)
            img_up = F.interpolate(img_batch, size=(new_H, new_W), mode=self.mode, align_corners=self.align_corners)
            
            h_start = (new_H - H) // 2
            w_start = (new_W - W) // 2
            img_result = img_up[:, :, h_start:h_start+H, w_start:w_start+W]
        
        return img_result.squeeze(0)
    
class RandomExposureSimulator(torch.nn.Module):
    def __init__(self, brightness=(0.5, 1.5), contrast=(0.5, 1.5), gamma=(0.5, 2.0)):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.gamma = gamma
        
    def forward(self, img):
        brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
        img = T.functional.adjust_brightness(img, brightness_factor)
        
        contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
        img = T.functional.adjust_contrast(img, contrast_factor)
        
        gamma_factor = random.uniform(self.gamma[0], self.gamma[1])
        img = T.functional.adjust_gamma(img, gamma_factor)
        
        return img
    

class RandomNoise(torch.nn.Module):
    def __init__(self, noise_types=['gaussian', 'salt_pepper'], max_intensity=0.1):
        super().__init__()
        self.noise_types = noise_types
        self.max_intensity = max_intensity
        
    def forward(self, img):
        noise_type = random.choice(self.noise_types)
        if noise_type == 'gaussian':
            noise = torch.randn_like(img) * random.uniform(0, self.max_intensity)
            img = img + noise
            img = torch.clamp(img, 0, 1)
        elif noise_type == 'salt_pepper':
            noise = torch.rand_like(img)
            salt = (noise > 1 - self.max_intensity/2)
            pepper = (noise < self.max_intensity/2)
            img = img.clone()
            img[salt] = 1
            img[pepper] = 0
        return img
    
class PerImageNormalize(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        
    def forward(self, img):
        mean = torch.mean(img, dim=(1, 2), keepdim=True)
        std = torch.std(img, dim=(1, 2), keepdim=True) + self.eps
        return (img - mean) / std
    
class SubsetInMemoryDataset(Dataset):
    def __init__(self, image_folder_dataset, indices, pre_transform, device, transform=None, rank=None):
        self.data = []
        self.targets = []
        self.pre_transform = pre_transform
        self.transform = transform
        
        start_time = time.time()
        total_memory_before = torch.cuda.memory_allocated(device) / (1024 ** 3)
        print(f"[Rank {rank}] Starting to load {len(indices)} images into memory. Initial GPU memory: {total_memory_before:.2f} GB")

        for i, idx in enumerate(tqdm(indices, desc=f"[Rank {rank}] Loading data", disable=rank is not None and rank != 0)):
            img, label = image_folder_dataset[idx]
            if not isinstance(img, torch.Tensor):
                img = self.pre_transform(img)
            img = img.to(device).half()
            self.data.append(img)
            self.targets.append(torch.tensor(label, device=device))
            
            if (i+1) % 100 == 0 or i == len(indices)-1:
                current_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
                print(f"[Rank {rank}] Loaded {i+1}/{len(indices)} images. GPU memory: {current_memory:.2f} MB")
        
        end_time = time.time()
        total_memory_after = torch.cuda.memory_allocated(device) / (1024 ** 2)
        memory_used = total_memory_after - total_memory_before
        print(f"[Rank {rank}] Finished loading {len(indices)} images in {end_time - start_time:.2f}s. Memory used: {memory_used:.2f} MB")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.transform:
            return (self.transform(self.data[idx])), self.targets[idx]
        else:
            return self.data[idx], self.targets[idx]
        
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class RandomApply(torch.nn.Module):
    def __init__(self, transform, p=0.5):
        super().__init__()
        self.transform = transform
        self.p = p
        
    def forward(self, img):
        if random.random() < self.p:
            return self.transform(img)
        return img

def run_model(local_rank, world_size, rank):

    gc.collect()
    torch.cuda.empty_cache()

    try:
        lr = 0.0005
        model_name = ''
        weight_decay = 0.01
        lr_gamma = 0.99
        test_samples = 8
        sigma = 0.01
        
        patch_size = 448
        
        #Transformer
        n_layer = 2
        n_token = 1
        n_head = 16
        d_k = 64
        d_v = 64
        d_model = 1024
        d_inner = 4096
        attn_dropout = 0.1
        dropout = 0.1
        
        sensor_noise = 0.0
        contrast_mod = 0.0
        scale_mod = 0.1
        clahe_grid = 64
        num_epochs = 50

        if rank == 0:
            wandb.init()
            lr = wandb.config.lr
            model_name  = wandb.config.model_name
            weight_decay = wandb.config.weight_decay
            test_samples = wandb.config.test_samples
            lr_gamma = wandb.config.lr_gamma
            sigma = wandb.config.sigma
            n_layer = wandb.config.n_layer
            n_token = wandb.config.n_token
            n_head = wandb.config.n_head
            d_k = wandb.config.d_k
            d_v = wandb.config.d_v
            d_model = wandb.config.d_model
            d_inner = wandb.config.d_inner
            attn_dropout = wandb.config.attn_dropout
            dropout = wandb.config.dropout
            patch_size = wandb.config.patch_size
            
            sensor_noise = wandb.config.sensor_noise
            contrast_mod = wandb.config.contrast_mod
            scale_mod = wandb.config.scale_mod
            clahe_grid = wandb.config.clahe_grid
            num_epochs = wandb.config.num_epochs

        max_length = 50
        model_name = model_name.ljust(max_length)

        lr = torch.tensor(lr).cuda()
        model_name = torch.tensor([ord(c) for c in model_name]).cuda()
        weight_decay = torch.tensor(weight_decay).cuda()
        lr_gamma = torch.tensor(lr_gamma).cuda()
        test_samples = torch.tensor(test_samples).cuda()
        sigma = torch.tensor(sigma).cuda()
        n_layer = torch.tensor(n_layer).cuda()
        n_token = torch.tensor(n_token).cuda()
        n_head = torch.tensor(n_head).cuda()
        d_k = torch.tensor(d_k).cuda()
        d_v = torch.tensor(d_v).cuda()
        d_model = torch.tensor(d_model).cuda()
        d_inner = torch.tensor(d_inner).cuda()
        attn_dropout = torch.tensor(attn_dropout).cuda()
        dropout = torch.tensor(dropout).cuda()
        patch_size = torch.tensor(patch_size).cuda()
        sensor_noise = torch.tensor(sensor_noise).cuda()
        contrast_mod = torch.tensor(contrast_mod).cuda()
        scale_mod = torch.tensor(scale_mod).cuda()
        clahe_grid = torch.tensor(clahe_grid).cuda()
        num_epochs = torch.tensor(num_epochs).cuda()

        dist.broadcast(lr, src=0)
        dist.broadcast(model_name, src=0)
        dist.broadcast(weight_decay, src=0)
        dist.broadcast(lr_gamma, src=0)
        dist.broadcast(test_samples, src=0)
        dist.broadcast(sigma, src=0)
        dist.broadcast(n_layer, src=0)
        dist.broadcast(n_token, src=0)  
        dist.broadcast(n_head, src=0)
        dist.broadcast(d_k, src=0)
        dist.broadcast(d_v, src=0)  
        dist.broadcast(d_model, src=0)
        dist.broadcast(d_inner, src=0)
        dist.broadcast(attn_dropout, src=0)
        dist.broadcast(dropout, src=0)
        dist.broadcast(patch_size, src=0)
        dist.broadcast(sensor_noise, src=0)
        dist.broadcast(contrast_mod, src=0)
        dist.broadcast(scale_mod, src=0)
        dist.broadcast(clahe_grid, src=0)
        dist.broadcast(num_epochs, src=0)
        
        dist.barrier()
        
        logging.info(f"Rank {rank} received the broadcasted configuration values.")

        model_name = ''.join([chr(int(c)) for c in model_name.cpu().numpy()]).strip()

        lr = lr.item()
        weight_decay = weight_decay.item()
        lr_gamma = lr_gamma.item()
        test_samples = test_samples.item()
        sigma = sigma.item()
        n_layer = n_layer.item()
        n_token = n_token.item()
        n_head = n_head.item()
        d_k = d_k.item()
        d_v = d_v.item()
        d_model = d_model.item()
        d_inner = d_inner.item()
        attn_dropout = attn_dropout.item()
        dropout = dropout.item()
        patch_size = patch_size.item()
        
        sensor_noise = sensor_noise.item()
        contrast_mod = contrast_mod.item()
        scale_mod = scale_mod.item()
        clahe_grid = clahe_grid.item()
        num_epochs = num_epochs.item()

        main_dir = 'data'
        mean = [0.2606, 0.2654, 0.2964]
        std =[0.1024, 0.1033, 0.1126]

        freeze_layers = False
        if model_name == 'eva02':
            batch_size = 6
        elif model_name == 'convnext_xxlarge':
            batch_size = 4
        else:
            batch_size = 24

        jigsaw = False
        num_patches = test_samples
        feature_extract = True
        num_workers = 0

        if rank == 0:
            print((os.environ["SLURM_JOBID"]))
            print("Initializing Datasets and Dataloaders...")
            wandb.log({'batch_size': batch_size})

        # transform used for preprocessing/augmentation
        # create high resolution image
        data_transforms = {
            "train": transforms.Compose([
                transforms.ToTensor(),
                # PadToSize([5200, 5200]),
                # transforms.CenterCrop((5100, 5100)),
                # transforms.Resize((int(5100/2), int(5100/2))),
                # transforms.Normalize([mean[0], mean[1], mean[2]], [std[0], std[1], std[2]]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(5),  # Reduced rotation
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.5, 1.5)),  # More subtle
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),  # Less aggressive
                RandomApply(RandomNoise(
                    max_intensity=sensor_noise), p=0.25),
                CLAHE(clip_limit=3.0, tile_grid_size=(clahe_grid, clahe_grid)),
                PerImageNormalize(),
                RandomTranslateWithPadding(max_translate=0.1),
            ]),
            "val": transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((int(5100/2), int(5100/2))),
                CLAHE(clip_limit=3.0, tile_grid_size=(clahe_grid, clahe_grid)),
                # PadToSize([5200, 5200]),
                # transforms.CenterCrop((5100, 5100)),
                # transforms.Resize((int(5100/2), int(5100/2))),
                # transforms.Normalize([mean[0], mean[1], mean[2]], [std[0], std[1], std[2]]),
                PerImageNormalize(),
            ]),
        }

        # initialize datasets

        # c = 'printer_identification_29_150_0'
        # c = 'printer_22_efficiency_100'
        # c = 'printer_DLS_11_150_0'
        c = 'aim_all_designs_all_views'

        data_dir = 'data/'+ str(c)
        if rank == 0:
            model_path = 'data/Models/'+ str(c) + '_' + wandb.run.id + '_dpp.pth'

        ssd_dir = 'data/'+ str(c)


        class_names = ['Stratasys450mc-1','Stratasys450mc-2', 'Stratasys450mc-3', 'Stratasys450mc-4', 'Stratasys450mc-5', 'Stratasys450mc-6']
        
        num_classes = len(class_names)
        if rank == 0:
            print("Class names", class_names)
            print("Dataloaders initialized")
            
        image_datasets = {x: datasets.ImageFolder(os.path.join(ssd_dir, x), data_transforms[x]) for x in ['train', 'val']}

        train_sampler = DistributedSampler(image_datasets['train'], shuffle=True, num_replicas=world_size, rank=rank, drop_last=True)
        val_sampler = DistributedSampler(
            image_datasets['val'], 
            num_replicas=world_size,# * 8,
            shuffle=True,
            rank=rank, 
            drop_last=True,
        )
        

        train_sampler.set_epoch(0)
        val_sampler.set_epoch(0)
        train_indices = list(train_sampler)
        val_indices = list(val_sampler)
        
        n_train_samples = len(train_sampler)
        n_val_samples = len(val_sampler)


        
        data_loading_device = torch.device(f'cuda:{local_rank}')

        dataloaders_dict = {
            'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True),
            'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=True)
        }
        
        

        if rank == 0:
            print("Initializing model and optimizer...")
            
            
        if model_name == 'eva02':
            model = timm.create_model('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=True)
        elif model_name == 'convnext_xxlarge':
            model = timm.create_model('convnext_xxlarge.clip_laion2b_soup_ft_in12k', pretrained=True)
        elif model_name == 'dinov3_vits16':
            from transformers import AutoImageProcessor, AutoModel
            from transformers.image_utils import load_image

            pretrained_model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
            model = AutoModel.from_pretrained(
                pretrained_model_name
            ).base_model
            model = DINOv3FeatureExtractor(model)

        if model_name == 'inception_next_base':
            model.head.fc2 = nn.Identity()
        elif model_name == 'efficientnetv2_m':
            model.classifier = nn.Linear(1280, 640)
        elif model_name == 'poolformer_m36':
            model.head.fc = nn.Identity()
        elif model_name == 'vit_b_16':
            model.head = nn.Linear(768, 768)
        elif model_name == 'eva02':
            model.head = torch.nn.Linear(1024, 1024)
        elif model_name == 'convnext_xxlarge':
            model.head.fc = torch.nn.Linear(3072, 1024)

        model = model.to(local_rank)

        patch_selection = DPS(n_channel=3, high_size=(int(5100/2), int(5100/2)), low_size=(850, 850), score_size=(52, 52), k=num_patches, num_samples=500, sigma=sigma, patch_size=patch_size, model_name=model_name, device=local_rank)
        
        patch_selection = patch_selection.to(local_rank)
        
        model = torch.nn.Sequential(patch_selection, model)
        
        
        if model_name == 'efficientnetv2_m':
            input_dim = 640
        if model_name == 'inception_next_base':
            input_dim = 3072
        if model_name == 'poolformer_m36':
            input_dim = 768
        if model_name == 'vit_b_16':
            input_dim = 768
        if model_name == 'eva02':
            input_dim = 1024
        if model_name == 'convnext_xxlarge':
            input_dim = 1024
        if model_name == 'dinov3_vits16':
            input_dim = 384
        
        ##Transformer
        
        transformer = TransformerClassifier(input_dim, num_classes,
                                            n_layer=n_layer,
                                            n_token=(n_token,) * n_layer,
                                            n_head=n_head,
                                            d_k=d_k,
                                            d_v=d_v,
                                            d_model=d_model,
                                            d_inner=d_inner,
                                            attn_dropout=attn_dropout,
                                            dropout=dropout).to(local_rank)
        
        
        model = torch.nn.Sequential(model, transformer)
        
        if (model_name == 'vit_b_16') | (model_name == 'dinov3_vits16'):
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

        if model_name == 'dinov3_vits16':
            backbone_params = []
            patch_selector_params = []
            head_params = []

            for name, param in model.named_parameters():
                if 'module.0.1.' in name:  # DINOv3 backbone
                    backbone_params.append(param)
                elif 'module.0.0.' in name:  # Patch selector
                    patch_selector_params.append(param)
                else:  # Linear head
                    head_params.append(param)
            
            # Create optimizer with parameter groups
            optimizer_ft = optim.Lamb([
                {'params': backbone_params, 'lr': 1e-5},      # Very low LR for backbone (0.000005)
                {'params': patch_selector_params, 'lr': 2e-4}, # Medium LR for patch selector (0.0001)
                {'params': head_params, 'lr': 1e-3}           # Higher LR for transformer head (0.0005)
            ], weight_decay=weight_decay)
        else:
            # Regular optimizer for other models
            optimizer_ft = optim.Lamb(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        
        from torch.optim.lr_scheduler import OneCycleLR

        # Replace your current scheduler with this
        if model_name == 'dinov3_vits16':
            # For models with 3 parameter groups
            scheduler = OneCycleLR(
                optimizer_ft,
                max_lr=[5e-6, 1e-4, 5e-4],
                steps_per_epoch=len(dataloaders_dict['train']),
                epochs=num_epochs,
                pct_start=0.3,
                div_factor=25,
                final_div_factor=1000,
                anneal_strategy='cos',
            )
        else:
            # For models with a single parameter group
            scheduler = OneCycleLR(
                optimizer_ft,
                max_lr=lr,  # Single value for single parameter group
                steps_per_epoch=len(dataloaders_dict['train']),
                epochs=num_epochs,
                pct_start=0.3,
                div_factor=25,
                final_div_factor=1000,
                anneal_strategy='cos'
            )
        
        ## Weight loss function based on class imbalance
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        if rank == 0:
            print("Model and optimizer initialized")
                

        # Train the model
        test_samples = 1
        model, acc, trainloss, pred_ids, ground_truth_ids, best_val_loss, best_val_acc = ml_models.train_model(model, model_name, dataloaders_dict, image_datasets, criterion, optimizer_ft, batch_size, 
                                                                                                               class_names, main_dir, test_samples, local_rank, rank, scheduler, n_train_samples, n_val_samples, 
                                                                                                               val_sampler, sigma, num_epochs=num_epochs, jigsaw=jigsaw, train_sampler=train_sampler)
        # Print results
        if rank == 0:
            if best_val_acc > 0.90:
                torch.save(model.state_dict(), model_path)
            results_save_dir = 'data/Results/'+ str(c) + '.txt'
            file_result =  open(results_save_dir, 'w')
            print('loss: ', file=file_result)
            print(trainloss, file=file_result)
            print('acc: ', file=file_result)
            print(acc, file=file_result)
            print('best_acc is: ', file=file_result)
            print(best_val_acc, file=file_result)

            # log metrics to wandb
            wandb.log({'max_acc': best_val_acc})
            wandb.log({'min_val_loss': best_val_loss})

            wandb.finish()

    except RuntimeError as e:
        print(f"RuntimeError: {e}")
        print(traceback.format_exc())
        if rank == 0:
            run_id = wandb.run.id
            wandb.finish()

            api = wandb.Api()
            run = api.run("wpklab/AIMS/" + run_id)
            run.delete()

        model = 0

        gc.collect()
        torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.destroy_process_group()
        subprocess.run(["scancel", os.environ["SLURM_JOB_ID"]])
        os._exit(1)
        raise RuntimeError(e)

def run_agent(local_rank, world_size, rank):
    wandb.agent(sweep_id="AIMS/mq5r4ud3", count=1, function=lambda: run_model(local_rank, world_size, rank))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='train_fingerprint_dpp.log', filemode='a')
    os.environ["NCCl_DEBUG"] = "INFO"

    
    rank          = int(os.environ["SLURM_PROCID"])
    world_size    = int(os.environ["WORLD_SIZE"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])

    assert gpus_per_node == torch.cuda.device_count()
    print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
        f" {gpus_per_node} allocated GPUs per node.", flush=True)
    
    print(f"MASTER_PORT: {os.environ['MASTER_PORT']}", flush=True)

    try:
        if dist.is_initialized(): #Close initialized process group
            dist.destroy_process_group()

        dist.init_process_group(backend="nccl", init_method='env://', rank=rank, world_size=world_size)
        logging.info(f"Process group initialized successfully for rank {rank} with world size {world_size}.")
        
        if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

        local_rank = rank % gpus_per_node
        torch.cuda.set_device(local_rank)

        if rank == 0:
            run_agent(local_rank, world_size, rank)
        else:
            run_model(local_rank, world_size, rank)

        if dist.is_initialized():
            dist.destroy_process_group()
        
    except RuntimeError as e:
        print(f"RuntimeError: {e}")
        print(traceback.format_exc())
        if "NCCL error" in str(e):
            print("NCCL error detected. Cleaning up and terminating the process.")
            if dist.is_initialized():
                dist.destroy_process_group()
            gc.collect()
            torch.cuda.empty_cache()
            subprocess.run(["scancel", os.environ["SLURM_JOB_ID"]])
            os._exit(1)
            raise RuntimeError(e)
        else:
            if dist.is_initialized():
                dist.destroy_process_group()
            gc.collect()
            torch.cuda.empty_cache()
            subprocess.run(["scancel", os.environ["SLURM_JOB_ID"]])
            os._exit(1)
            raise RuntimeError(e)