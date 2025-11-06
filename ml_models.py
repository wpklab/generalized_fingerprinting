import copy # someone on stack overflow says copy is built into python, and thus requires no installation.
import os
import shutil # fine
import time # fine
from PIL import Image # seems fine
import wandb
import pandas as pd
import random


import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision import transforms
import numpy as np

import timm

from collections import Counter
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision import transforms
import numpy as np

import timm
import torch.distributed as dist

#print(device1)
torch.hub.set_dir('data/Models')

os.environ["HUGGINGFACE_HUB_CACHE"] = "data/Models"

def set_parameter_requires_grad(model, feature_extracting, freeze_layers):
    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False #I think this line was freezing a lot of parameters
    else:
        for param in model.parameters():
            param.requires_grad = True

# To initialize the model
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True, freeze_layers=True, classifier_layer_config=0, input_size=224):
    if model_name == "resnet18":
        model_ft = models.resnet18(weights='DEFAULT')
    if model_name == "resnet34":
        model_ft = models.resnet34(weights='DEFAULT')
    if model_name == "resnet50":
        model_ft = models.resnet50(weights='DEFAULT')
    if model_name == "resnet101":
        model_ft = models.resnet101(weights='DEFAULT')
    if model_name == "resnet152":
        model_ft = models.resnet152(weights='DEFAULT')
    if model_name == "convnext_base":
        model_ft = models.convnext_base(weights='DEFAULT')
    if model_name == "wideresnet50":
        model_ft = models.wide_resnet50_2(weights='DEFAULT')
    if model_name == "efficientnetv2_s":
        model_ft = models.efficientnet_v2_s(weights='DEFAULT')
    if model_name == "efficientnetv2_m":
        model_ft = models.efficientnet_v2_m(weights='DEFAULT')
    if model_name == "swin_v2_t":
        model_ft = models.swin_v2_t(weights='DEFAULT')
    if model_name == "swin_v2_s":
        model_ft = models.swin_v2_s(weights='DEFAULT')
    if model_name == 'convnext_small':
        model_ft = models.convnext_small(weights='DEFAULT')
    if model_name == 'convnext_tiny':
        model_ft = models.convnext_small(weights='DEFAULT')
    if model_name == 'maxvit':
        model_ft = models.maxvit_t(weights='DEFAULT')
    if model_name == 'mobilenet_large':
        model_ft = models.mobilenet_v3_large(weights='DEFAULT')
    if model_name == 'mobilenet_small':
        model_ft = models.mobilenet_v3_small(weights='DEFAULT')
    if model_name == 'vit_b_16':
        #model_ft = models.vit_b_16(weights='DEFAULT')
        model_ft = timm.create_model('vit_base_patch16_384', pretrained=True, img_size=448, num_classes=num_classes)
    if model_name == 'vit_b_32':
        model_ft = models.vit_b_32(weights='DEFAULT')
    if model_name == 'vit_l_16':
        model_ft = models.vit_l_16(weights='DEFAULT')
    if model_name == 'vit_l_32':
        model_ft = models.vit_l_32(weights='DEFAULT')
    if model_name == 'vit_h_14':
        model_ft = models.vit_h_14(weights='DEFAULT')
    if model_name == 'poolformer_m36':
        model_ft = timm.create_model('poolformer_m36.sail_in1k', pretrained=True, num_classes=num_classes)
    if model_name == 'poolformer_m48':
        model_ft = timm.create_model('poolformer_m48', pretrained=True, num_classes=num_classes)
    if model_name == 'poolformer_s36':
        model_ft = timm.create_model('poolformer_s36', pretrained=True, num_classes=num_classes)
    if model_name == 'efficient_vit_m5':
        model_ft = timm.create_model('efficientvit_m5', pretrained=True, num_classes=num_classes)
    if model_name == 'inception_next_base':
        model_ft = timm.create_model('inception_next_base', pretrained=True, num_classes=num_classes)
    if model_name == 'convnextv2_base':
        model_ft = timm.create_model('convnextv2_base.fcmae', pretrained=True, num_classes=num_classes)
    if model_name == 'efficientnetv2_l':
        model_ft = timm.create_model('efficientnetv2_l', num_classes=num_classes)
    if model_name == 'efficientnetv2_xl':
        model_ft = timm.create_model('efficientnetv2_xl', num_classes=num_classes)


    set_parameter_requires_grad(model_ft, feature_extract, freeze_layers)

    if model_name == "convnext_base":
        sequential_layers = nn.Sequential(
            nn.LayerNorm((1024, 1, 1,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1)
        )
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(1024, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(1024, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(1024, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.classifier = sequential_layers
    
    elif model_name == "convnext_small":
        sequential_layers = nn.Sequential(
            nn.LayerNorm((768, 1, 1,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
        )
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(768, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.classifier = sequential_layers
    
    elif model_name == "convnext_tiny":
        sequential_layers = nn.Sequential(
            nn.LayerNorm((768, 1, 1,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
        )
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(768, 384),
                                            nn.ReLU(),
                                            nn.Linear(384, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.classifier = sequential_layers
    
    elif model_name == "maxvit":
        sequential_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.LayerNorm((512,), eps=1e-05, elementwise_affine=True),
            nn.Linear(512, 512),
            nn.Tanh(),
        )
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(512, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(512, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(512, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)

        model_ft.classifier = sequential_layers
    
    elif model_name == "mobilenet_large":
        sequential_layers = nn.Sequential(
            nn.Linear(960, 1280, bias=True),
            nn.Hardshrink(),
            nn.Dropout(p=0.2, inplace=True),
        )
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(1280, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.classifier = sequential_layers

    elif model_name == "mobilenet_small":
        sequential_layers = nn.Sequential(
            nn.Linear(576, 1024, bias=True),
            nn.Hardshrink(),
            nn.Dropout(p=0.2, inplace=True),
        )
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(1024, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(1024, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(1024, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)

        model_ft.classifier = sequential_layers

    elif model_name == "efficientnetv2_s":
        sequential_layers = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
        )
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(1280, 640),
                                            nn.ReLU(),
                                            nn.Linear(640, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.classifier = sequential_layers

    elif model_name == "efficientnetv2_m":
        
        sequential_layers = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
        )
        if classifier_layer_config == 0:
            #classifier_layers=nn.Sequential(nn.Linear(1280, num_classes))
            classifier_layers=nn.Sequential(nn.Linear(1280, 640),
                                            nn.ReLU(),
                                            nn.Linear(640, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.classifier = sequential_layers

    elif model_name == 'swin_v2_t':
        n_inputs = model_ft.head.in_features
        sequential_layers = nn.Sequential()

        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(n_inputs, int(n_inputs/2)),
                                            nn.ReLU(),
                                            nn.Linear(int(n_inputs/2), num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(n_inputs, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(n_inputs, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.head = sequential_layers
    
    elif model_name == 'swin_v2_s':
        n_inputs = model_ft.head.in_features
        sequential_layers = nn.Sequential()
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(n_inputs, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(n_inputs, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(n_inputs, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.head = sequential_layers

    elif model_name == 'vit_b_16':
        sequential_layers = nn.Sequential()
        if classifier_layer_config == 0:
            # classifier_layers=nn.Sequential(nn.Linear(768, 512),
            #                                 nn.ReLU(),
            #                                 nn.Linear(512, 256),
            #                                 nn.ReLU(),
            #                                 nn.Linear(256, 128),
            #                                 nn.ReLU(),
            #                                 nn.Linear(128, num_classes))
            classifier_layers=nn.Sequential(nn.Linear(768, 384),
                                            nn.ReLU(),
                                            nn.Linear(384, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.heads = sequential_layers

    elif model_name == 'vit_b_32':
        sequential_layers = nn.Sequential()
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(768, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(768, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.heads = sequential_layers

    elif model_name == 'vit_l_16':
        sequential_layers = nn.Sequential()
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(1024, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(1024, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(1024, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.heads = sequential_layers

    elif model_name == 'vit_l_32':
        sequential_layers = nn.Sequential()
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(1024, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(1024, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(1024, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.heads = sequential_layers

    elif model_name == 'vit_h_14':
        sequential_layers = nn.Sequential()
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(1280, num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(1280, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.heads = sequential_layers

    elif model_name == 'efficient_vit_m5':
        sequential_layers = nn.Sequential()
        if classifier_layer_config == 0:
            classifier_layers=model_ft.head
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(384, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(384, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.heads = sequential_layers

    elif model_name == 'inception_next_base':
        sequential_layers = model_ft.head
        model_ft.head = sequential_layers

    elif model_name == 'convnextv2_base':
        sequential_layers = nn.Sequential(nn.Linear(1024, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, num_classes))
        model_ft.head.fc = sequential_layers

    elif (model_name == 'poolformer_m36') | (model_name == 'poolformer_m48') | (model_name == 'poolformer_s36'):
        sequential_layers = model_ft.head
        model_ft.head = sequential_layers

    elif model_name == 'transfg':
        pass

    elif model_name == 'pmg_50':
        pass

    elif model_name == 'pmg_101':
        pass

    elif model_name == 'apcnn':
        pass
    elif model_name == 's3n':
        pass

    elif model_name == 'efficientnetv2_l':
        pass

    elif model_name == 'efficientnetv2_xl':
        pass

    else:       
        num_ftrs = model_ft.fc.in_features
        sequential_layers = nn.Sequential()
        if classifier_layer_config == 0:
            classifier_layers=nn.Sequential(nn.Linear(num_ftrs, int(num_ftrs/2)),
                                            nn.ReLU(),
                                            nn.Linear(int(num_ftrs/2), num_classes))
        elif classifier_layer_config == 1:
            classifier_layers=nn.Sequential(nn.Linear(num_ftrs, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes),
                nn.LogSoftmax(dim=1))
        else:
            classifier_layers=nn.Sequential(nn.Linear(num_ftrs, 2048, bias=True),
                nn.ReLU())
            for ilayer in range(classifier_layer_config-1):
                classifier_layers.append(nn.Linear(2048, 2048, bias=True))
                classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Linear(2048, num_classes))

        sequential_layers.append(classifier_layers)
        model_ft.fc = sequential_layers

    input_size = 448

    return model_ft, input_size

def gather_tensors(tensor):
    gather_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_list, tensor)
    return torch.cat(gather_list, dim=0)

def gather_indices(indices):
    indices_tensor = torch.tensor(indices, dtype=torch.int64).cuda()
    gather_list = [torch.zeros_like(indices_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_list, indices_tensor)
    return torch.cat(gather_list, dim=0)

# To train the model
def train_model(model, model_name, dataloaders, image_datasets, criterion, optimizer, batch_size, class_names, data_dir, test_samples, img_size, device1, rank, scheduler, n_train_samples, n_val_samples, val_sampler, num_epochs, jigsaw=False, train_sampler=None, log_interval=20):
    
    model.cuda()

    since = time.time()
    val_acc_history = []
    train_loss_history = []
    best_acc = 0.0
    best_loss = np.inf
    
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(num_epochs):
        
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
            
        if rank == 0:
            epoch_since = time.time()
            print("Epoch {}/{}".format(epoch, num_epochs - 1))
            print("-" * 30)
            print('lr {}'.format(scheduler.get_last_lr()))
        running_loss = 0.
        running_corrects = 0.
        model.train()
        phase = 'train'

        for batch_id, (inputs, labels) in enumerate(dataloaders[phase], start=epoch * len(dataloaders[phase])):
            
            inputs, labels = inputs.cuda(), labels.cuda()

            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()

            if rank == 0:
                
                if batch_id % log_interval == 0: #Remove log interval
                    num_samples_processed = (batch_id + 1 - (epoch * len(dataloaders[phase]))) * batch_size
                    print("Train Epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}\tAcc: {}/{}".format(
                        epoch,
                        num_samples_processed,
                        len(dataloaders['train'].dataset),
                        100. * num_samples_processed / len(dataloaders['train'].dataset),
                        running_loss / num_samples_processed,
                        int(running_corrects),
                        num_samples_processed
                    ))

        scheduler.step()
        if rank == 0:
            epoch_loss = running_loss / n_train_samples
            epoch_acc = running_corrects / n_train_samples
            time_elapsed = time.time() - epoch_since
            print("Training Epoch compete in {}m   {}s".format(time_elapsed // 60, time_elapsed % 60))
            print("{} Loss: {} Acc: {}".format('Train', epoch_loss, epoch_acc))

        ##Log Validation Statistics
        if (epoch % 1 == 0) | (epoch == num_epochs-1): #Remove log interval
            val_loss, val_acc, pred_array_final, label_array_final = test_model(model, model_name, dataloaders['val'], image_datasets, criterion, epoch, class_names, data_dir, test_samples, n_val_samples, val_sampler, img_size, device1, rank, epoch_end=True)
            epoch_loss = running_loss / n_train_samples
            epoch_acc = running_corrects / n_train_samples
            
            world_size = dist.get_world_size()

            val_loss_tensor = torch.tensor(val_loss).cuda()
            
            if val_loss_tensor.dim() == 0:
                val_loss_tensor = val_loss_tensor.unsqueeze(0)

            val_loss_tensor = gather_tensors(val_loss_tensor)
            
            val_loss_tensor = val_loss_tensor[~torch.isnan(val_loss_tensor)]
            
            if len(val_loss_tensor) > 0:
                val_loss = torch.mean(val_loss_tensor).item()
            else:
                val_loss = 100.0  # or some default value


            # Concatenate pred_array_final and label_array_final across all nodes
            pred_array_final = gather_tensors(pred_array_final)
            label_array_final = gather_tensors(label_array_final)

            dist.barrier()
            
            val_acc = torch.sum(pred_array_final == label_array_final).item() / len(label_array_final)
            
            val_acc_history.append(val_acc)
            train_loss_history.append(epoch_loss)
            
            indices = list(val_sampler)
            all_indices = gather_indices(indices)
            
            if rank == 0:
                print(f"Averaged Validation Loss: {val_loss:.4f}")
                print(f"Averaged Validation Accuracy: {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                best_loss = val_loss
                
                pred_array_best = pred_array_final
                label_array_best = label_array_final
                
                if rank == 0:
                    best_model_wts = copy.deepcopy(model.state_dict())

                    wandb.log({'max_acc': best_acc})

                    wandb.log({"conf_mat_" : wandb.plot.confusion_matrix( 
                        preds=np.array(pred_array_final.cpu()), y_true=np.array(label_array_final.cpu()), class_names=np.array(class_names))})
                    
                    
                    ##Hardcoded in for now
                    torch.save(model.state_dict(), 'data/Models/aims_old_' + wandb.run.id + '.pth')

                    df = pd.DataFrame(columns=['img', 'label', 'pred'])
                    df.img = [image_datasets['val'].imgs[idx] for idx in all_indices]

                    df.pred = pred_array_best.cpu()
                    df.label = label_array_best.cpu()
                    
                    csv_path = data_dir + "/csv_outputs/ddp_model_" + str(os.environ["MASTER_PORT"]) + '_' + str(epoch) + ".csv"

                    df.to_csv(csv_path, index=False)
                
            if rank == 0:
                print("{} Loss: {} Acc: {}".format(phase, epoch_loss, epoch_acc))
                wandb.log({'epoch_train': epoch, 'train acc': epoch_acc, 'train loss': epoch_loss, 'val acc': val_acc, 'val loss': val_loss})

    if rank == 0:
        time_elapsed = time.time() - since
        print("Training compete in {}m   {}s".format(time_elapsed // 60, time_elapsed % 60))
        print("Best val Acc: {}".format(best_acc))

        model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_loss_history, pred_array_best, label_array_best, best_loss, best_acc

def mode_and_filter_raw(array, class_names, device1, filter_vals=None):
    array = array.to(device1)
    result = torch.sum(array, dim=1)
    max_val, pred = result.max(dim=1)

    return pred

def test_model(model, model_name, dataloaders, image_datasets, criterion, epoch, class_names, data_dir, test_samples, n_val_samples, val_sampler, img_size, device1, rank, epoch_end=False):
    
    model.to(device1)

    running_loss = 0.
    running_corrects = 0.
    model.eval()
    since = time.time()
    pred_array = torch.zeros(0).cuda()
    label_array = torch.zeros(0).cuda()
    pred_part_array = torch.zeros(0).cuda()
    label_part_array = torch.zeros(0).cuda()
    output_array = torch.zeros(0).cuda()
    softmax = nn.Softmax(dim=1)

    if rank == 0:
        print('Testing model')  

    for batch_id, (inputs, labels) in enumerate(dataloaders):
        inputs, labels = inputs.cuda(), labels.cuda()
        inputs = inputs.squeeze()
        
        # print(inputs.size())
        
        inputs = inputs.view(-1, 3, img_size, img_size)

        if test_samples > 1:
            labels = (np.repeat(np.array(labels.cpu()), test_samples))
        labels = torch.Tensor(labels).cuda().long()

        with torch.amp.autocast('cuda'):  # Enable mixed precision
            with torch.autograd.set_grad_enabled(False):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                outputs = softmax(outputs)

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * (inputs.size(0) / test_samples)

        pred_array = torch.cat((pred_array, preds.view(-1)), 0)
        label_array = torch.cat((label_array, labels.view(-1)), 0)
        output_array = torch.cat((output_array, outputs.view(-1)), 0)
    
    pred_img_array = pred_array.view(-1, test_samples)
    label_img_array = label_array.view(-1, test_samples)
    output_img_array = output_array.view(-1, test_samples, len(class_names))
    
    
    pred_img_array_mode = mode_and_filter_raw(output_img_array, class_names, device1).cuda()
    label_img_array_mode, _ = torch.mode(label_img_array, 1)
    label_img_array_mode = label_img_array_mode.cuda()
    
    running_corrects += torch.sum(pred_img_array_mode == label_img_array_mode).detach().cpu().numpy()
    pred_part_array = torch.cat((pred_part_array, pred_img_array_mode), 0)
    label_part_array = torch.cat((label_part_array, label_img_array_mode), 0)

    epoch_loss = running_loss / n_val_samples

    # epoch_acc = np.sqrt((running_corrects))
    epoch_acc = (running_corrects) / n_val_samples
    
    time_elapsed = time.time() - since
    
    if rank == 0:
        print("Testing compete in {}m   {}s".format(time_elapsed // 60, time_elapsed % 60))
        print("{} Loss: {} Acc: {}".format('val', epoch_loss, epoch_acc))

    return epoch_loss, epoch_acc, pred_part_array, label_part_array
