import torchvision
import torch.nn as nn
import os
import torch
import torchvision.transforms as tfs
import torch.utils.data as data
import random
from PIL import Image
import torch.nn.functional as F
import numpy as np
import torch
import numpy as np
import pickle
import datetime
import ray
from ray import tune
from collections import OrderedDict
from networks import CNN_Digits

def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    return _alr

def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            if lr<base_lr * 0.05:
                lr = base_lr * 0.05
        return lr
    
    return lr_policy(_lr_fn)

def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if key_item_1[1].device == key_item_2[1].device  and torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                _device = f'device {key_item_1[1].device}, {key_item_2[1].device}' if key_item_1[1].device != key_item_2[1].device else ''
                print(f'Mismtach {_device} found at', key_item_1[0], 'difference:', torch.sum(key_item_1[1]-key_item_2[1], 0))
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')
        
def trial_name_string(trial):
    if hasattr(trial, "config"):
        config = trial.config
    else:
        config = trial
    name = 'trial'        
    for k in config['hps_list']:
        v = config[k]
        if not v and v!=0: continue
        if "/" in str(k) or "/" in str(v): continue #filter out path
        if isinstance(v, bool):
            if v:
                name += f'_{k}'
        elif isinstance(v, str):
            name += f'_{v}'
        else:
            if k=='batch_size':
                name += f'_bs_{v}'
            else:
                name += f'_{v}'
    return name
    
def create_model(backbone, class_num, pretrained=True):
    if backbone=='resnet18':
        T = torchvision.models.resnet18(pretrained=pretrained)
        T.fc = nn.Linear(512, class_num)
        T.feature_dim = 512
    elif backbone=='resnet34':
        T = torchvision.models.resnet34(pretrained=pretrained)
        T.fc = nn.Linear(512, class_num)
        T.feature_dim = 512
    elif backbone=='resnet50':
        T = torchvision.models.resnet50(pretrained=pretrained)
        T.fc = nn.Linear(2048, class_num)
        T.feature_dim = 2048
        
    elif backbone=='efficientnet' or 'Efficient' in backbone:
        T = torchvision.models.efficientnet_b0(pretrained=True)
        T.classifier = nn.Sequential(OrderedDict({
            f'dropout': nn.Dropout(p=0.2, inplace=True),
            f'fc': nn.Linear(1280, class_num, bias=True)
            }))
        T.feature_dim = 1280

    elif backbone=='CNN_Digits':
        T = CNN_Digits(class_num)

    return T
    
def setup_seed(seed, deterministic=True):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: True.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  
        
def kl_loss(y, teacher_scores, temp=1, softmax_applied=False):
    p = F.log_softmax(y/temp, dim=1)
    
    if softmax_applied:
        q = teacher_scores
    else:
        q = F.softmax(teacher_scores/temp, dim=1)
        
    l_kl = F.kl_div(p, q, reduction='batchmean')
    l_kl = l_kl * temp ** 2          
    return l_kl
