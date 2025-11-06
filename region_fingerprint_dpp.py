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
import torch.distributed as dist

import matplotlib.pyplot as plt
from fingerprint_proposal import adjust_sigma

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
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True, freeze_layers=True, classifier_layer_config=0, input_size=448):
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
        # model_ft = timm.create_model('vit_base_patch16_384', pretrained=True, img_size=448, num_classes=num_classes)
        model_ft = timm.create_model('vit_base_patch16_384', pretrained=True, img_size=input_size, num_classes=num_classes)
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
    """Gather tensors of different sizes from all processes"""
    local_size = torch.tensor([tensor.size(0)], dtype=torch.long).cuda()
    
    all_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(all_sizes, local_size)
    max_size = max([size.item() for size in all_sizes])
    
    if tensor.size(0) < max_size:
        if tensor.dim() > 1:
            padding = torch.zeros(max_size - tensor.size(0), *tensor.shape[1:], 
                                 dtype=tensor.dtype, device=tensor.device)
        else:
            padding = torch.zeros(max_size - tensor.size(0), 
                                 dtype=tensor.dtype, device=tensor.device)
        padded_tensor = torch.cat([tensor, padding], dim=0)
    else:
        padded_tensor = tensor
    
    padded_output = [torch.zeros_like(padded_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(padded_output, padded_tensor)
    
    output = []
    for i, size in enumerate(all_sizes):
        output.append(padded_output[i][:size.item()])
    
    return torch.cat(output, dim=0)

def gather_indices(indices):
    indices_tensor = torch.tensor(indices, dtype=torch.int64).cuda()
    
    local_size = torch.tensor([indices_tensor.size(0)], dtype=torch.long).cuda()
    
    all_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(all_sizes, local_size)
    max_size = max([size.item() for size in all_sizes])
    
    if indices_tensor.size(0) < max_size:
        padding = torch.zeros(max_size - indices_tensor.size(0), dtype=indices_tensor.dtype, 
                             device=indices_tensor.device)
        padded_tensor = torch.cat([indices_tensor, padding], dim=0)
    else:
        padded_tensor = indices_tensor
    
    padded_output = [torch.zeros_like(padded_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(padded_output, padded_tensor)
    
    output = []
    for i, size in enumerate(all_sizes):
        output.append(padded_output[i][:size.item()])
    
    return torch.cat(output, dim=0)

class EarlyStopping:
    def __init__(self, patience=4, verbose=False, delta=0.00001):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        score = val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


# To train the model
def train_model(model, model_name, dataloaders, image_datasets, criterion, optimizer, batch_size, class_names, data_dir, test_samples, device1, rank, scheduler, n_train_samples, n_val_samples, val_sampler, sigma, num_epochs, jigsaw=False, train_sampler=None, log_interval=5):
    early_stopping = EarlyStopping(patience=50, verbose=True)
    model.cuda()

    since = time.time()
    val_acc_history = []
    train_loss_history = []
    best_acc = 0.0
    best_loss = np.inf
    softmax = nn.LogSoftmax(dim=1)
    
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(num_epochs):
        
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        if rank == 0:
            epoch_since = time.time()
            print("Epoch {}/{}".format(epoch, num_epochs - 1))
            print("-" * 30)
            print('lr {}'.format(scheduler.get_last_lr()))
            print("Sigma: ", model.module[0][0].TOPK.sigma)
        running_loss = 0.
        running_corrects = 0.
        model.train()
        phase = 'train'
        
        for batch_id, (inputs, labels) in enumerate(dataloaders[phase], start=epoch * len(dataloaders[phase])):
            
            inputs, labels = inputs.cuda(), labels.cuda()

            labels = torch.Tensor(labels).cuda().long()
            
            adjust_sigma(0, num_epochs, sigma, model.module[0][0], dataloaders[phase], batch_id+1)
            

            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Clip gradients to prevent exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer
            scaler.update()
            running_loss += loss.item() * (inputs.size(0))
            running_corrects += (torch.sum(preds.view(-1) == labels.view(-1)).item())

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
        if (epoch % 5 == 0) | (epoch == num_epochs-1): #Remove log interval
            val_loss, val_acc, pred_array_final, label_array_final = test_model(model, model_name, dataloaders['val'], image_datasets, criterion, epoch, class_names, data_dir, test_samples, n_val_samples, val_sampler, device1, rank, epoch_end=True)
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
                val_loss = 100.0
                


            pred_array_final = gather_tensors(pred_array_final)
            label_array_final = gather_tensors(label_array_final)
            
            print(f"Rank {rank}: Gathered pred_array_final size={pred_array_final.size(0)}, label_array_final size={label_array_final.size(0)}", flush=True)
            
            val_acc = torch.sum(pred_array_final == label_array_final).item() / len(label_array_final)
            
            val_acc_history.append(val_acc)
            train_loss_history.append(epoch_loss)
            
            indices = list(val_sampler)
            all_indices = gather_indices(indices)
            
            early_stopping(val_acc, model) ## Early stoppage based on validation loss
            
            if rank == 0:
                print(f"Averaged Validation Loss: {val_loss:.4f}")
                print(f"Averaged Validation Accuracy: {val_acc:.4f}")
                
            if early_stopping.early_stop:
                print("Early stopping")
                return model, val_acc_history, train_loss_history, pred_array_best, label_array_best, best_loss, best_acc
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_loss = val_loss
                
                pred_array_best = pred_array_final
                label_array_best = label_array_final
                
                if rank == 0:
                    best_model_wts = copy.deepcopy(model.state_dict())

                    wandb.log({'max_acc': best_acc})

                    class_names_list = [str(name) for name in class_names]
                    wandb.log({"conf_mat_" : wandb.plot.confusion_matrix(
                        preds=pred_array_final.cpu().numpy(), 
                        y_true=label_array_final.cpu().numpy(), 
                        class_names=class_names_list)})
                
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
    array = array.cuda()

    result = torch.sum(array, dim=1)

    max_val, pred = result.max(dim=1)

    return pred

def visualize_map(model, sample_imgs, labels, output_dir, device1, class_names):
    with torch.amp.autocast('cuda'):
        with torch.autograd.set_grad_enabled(False):
            outputs = model.module[0][0].scorer(transforms.Resize((650, 850))(sample_imgs))

    if outputs.dim() == 4:
        outputs = outputs.permute((0, 2, 3, 1))
        
    outputs = outputs.cpu().numpy()
    labels = labels.cpu().numpy()

    # save the visualization images in output_dir/xxx
    for i in range(outputs.shape[0]):
        curr_map = (outputs[i] - np.min(outputs[i])) / (np.max(outputs[i]) - np.min(outputs[i]))
        plt.figure(frameon=False)
        plt.imshow(curr_map, cmap='plasma')
        plt.axis('off')
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, str(class_names[int(labels[i])] + '_id_' +  str(i) + '_Device_' + str(device1))))
        plt.close("all")
        
    indicators = model.module[0][0].indicators
                
    indicators_resized = torch.nn.functional.interpolate(indicators, size=(sample_imgs.size(2), sample_imgs.size(3)), mode='bilinear', align_corners=False)
    square_size = 448
    for i in range(indicators_resized.size(0)):  # Iterate over batch size
        for j in range(indicators_resized.size(1)):  # Iterate over the number of indicators

            indicator = indicators_resized[i, j]

            center = torch.nonzero(indicator == indicator.max(), as_tuple=False)[0]
            center_y, center_x = center[0].item(), center[1].item()

            top_left_y = max(center_y - square_size // 2, 0)
            top_left_x = max(center_x - square_size // 2, 0)

            bottom_right_y = min(top_left_y + square_size, indicator.size(0))
            bottom_right_x = min(top_left_x + square_size, indicator.size(1))

            indicators_resized[i, j, top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 1
            
    indicator_images = indicators_resized.sum(dim=1).cpu().numpy()
    
    mean = [0.2606, 0.2654, 0.2964]
    std =[0.1024, 0.1033, 0.1126]
    
    for image_id in range(indicator_images.shape[0]):

        plot_image = sample_imgs[image_id].cpu().numpy().transpose(1, 2, 0)
        plot_image = (np.clip(plot_image * std[0] + mean[0], 0, 1) * 255).astype(np.uint8)
        
        indicator_np = indicator_images[image_id]
        indicator_np = (indicator_np - indicator_np.min()) / (indicator_np.max() - indicator_np.min() + 1e-8)
        indicator_np = indicator_np.astype(np.float32)
        
        alpha_mask = np.where(indicator_np > 0.05, 0.5, 0.0).astype(np.float32)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(plot_image)
        plt.imshow(indicator_np, cmap='plasma', alpha=alpha_mask) 
        plt.axis('off')
        plt.savefig(output_dir + f'/test_indicator_{image_id}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

def test_model(model, model_name, dataloaders, image_datasets, criterion, epoch, class_names, data_dir, test_samples, n_val_samples, val_sampler, device1, rank, epoch_end=False):
    
    model.cuda()

    running_loss = 0.
    running_corrects = 0.
    running_report_corrects = 0.
    model.eval()
    since = time.time()
    pred_array = torch.zeros(0).cuda()
    label_array = torch.zeros(0).cuda()
    pred_part_array = torch.zeros(0).cuda()
    label_part_array = torch.zeros(0).cuda()
    output_array = torch.zeros(0).cuda()
    softmax = nn.LogSoftmax(dim=1)
    CELoss = nn.CrossEntropyLoss()
    
    if rank == 0:
        print('Testing model')  

    for batch_id, (inputs, labels) in enumerate(dataloaders):
        
        inputs, labels = inputs.cuda(), labels.cuda()
        
        label_plot = labels

        labels = torch.Tensor(labels).cuda().long()
        
        with torch.amp.autocast('cuda'):  # Enable mixed precision
            with torch.autograd.set_grad_enabled(False):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                outputs = softmax(outputs)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * (inputs.size(0))

        batch_corrects = torch.sum(preds.view(-1) == labels.view(-1)).item()
        running_report_corrects += batch_corrects

        if rank == 0:
            if batch_id % 10 == 0:
                num_samples_processed = (batch_id + 1) * inputs.size(0)
                print("Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {}/{}".format(
                    epoch,
                    num_samples_processed,
                    n_val_samples,
                    100. * num_samples_processed / n_val_samples,
                    running_loss / num_samples_processed,
                    int(running_report_corrects),
                    num_samples_processed
                ))
        

        pred_array = torch.cat((pred_array, preds.view(-1)), 0)
        label_array = torch.cat((label_array, labels.view(-1)), 0)
        output_array = torch.cat((output_array, outputs.view(-1)), 0)
        
        
    print('Rank:', rank, 'Completed testing all samples')
        
    
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

    epoch_acc = (running_corrects) / n_val_samples
    
    time_elapsed = time.time() - since
    
    if rank == 0:

        print("Testing compete in {}m   {}s".format(time_elapsed // 60, time_elapsed % 60))
        print("{} Loss: {} Acc: {}".format('val', epoch_loss, epoch_acc))

    return epoch_loss, epoch_acc, pred_part_array, label_part_array
