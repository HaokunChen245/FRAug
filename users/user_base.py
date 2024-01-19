import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import json
from torch.utils.data import DataLoader, random_split
import numpy as np
import copy
from datasets import PACS, Digits, VLCS, OfficeHome, miniDomainNet, DomainNet_FedBN

class User:
    """
    Base class for users in federated learning.
    """
    def __init__(self, config, id, model):
        self.model_type = config['model']
        self.model = copy.deepcopy(model)
        self.feature_dim = self.model.feature_dim
        self.id = id  # integer        
        self.server_model_path = os.path.join(config['model_path'], 'server.pt')
        self.model_path = os.path.join(config['model_path'], f"user_{config['source_domains'][id//config['user_per_domain']]}_{id%config['user_per_domain']}.pt")       
        self.batch_size = config['batch_size']
        self.dataset = config['dataset']     
        self.device = config['device']     
        self.class_num = config['class_num'] 
        self.learning_rate = config['lr']
        self.local_epochs = config['local_epochs']
        self.algorithm = config['algorithm']
        self.portion = config['portion']

        self.init_loss_fn()
        self.prepare_data_loader(config, self.id)
        if config['optimizer']=='Adam':
            self.optimizer=torch.optim.Adam(
                params=self.model.parameters(),
                lr=self.learning_rate, betas=(0.9, 0.999),
                eps=1e-08, weight_decay=1e-2, amsgrad=False)
        elif config['optimizer']=='SGD':
            self.optimizer=torch.optim.SGD(
                params=self.model.parameters(),
                lr=self.learning_rate, 
                momentum=0.5)
            #check whether with 0.5 works better??
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.99)
        self.label_counts = {}
        
    def prepare_data_loader(self, config, id):
        dataset_mapping = {
            'PACS': PACS,
            'VLCS': VLCS,
            'OfficeHome': OfficeHome,
            'Digits': Digits,
            'miniDomainNet': miniDomainNet,
            'DomainNet_FedBN': DomainNet_FedBN,
        }
        dataset = dataset_mapping[config['dataset']]
        
        trainset = dataset(config['dataset_dir'], mode = 'train', img_size=config['img_size'], domain=config['source_domains'][id//config["user_per_domain"]])             
        len_valset = int(len(trainset) * 0.1) #train: val = 9:1, split the original trainset for model selection
        trainset_base, valset = random_split(trainset, [len(trainset)-len_valset, len_valset], 
                            generator=torch.Generator().manual_seed(config['seed'])) #fix the split
        len_trainset_portion = int(len(trainset_base)*self.portion)
        trainset_portion, _= random_split(trainset_base, [len_trainset_portion, len(trainset_base)-len_trainset_portion], 
                            generator=torch.Generator().manual_seed(config['seed'])) #fix the split

        self.train_loader = DataLoader(dataset=trainset_portion, batch_size=config['batch_size'], collate_fn=trainset.collate_fn, shuffle=True)     
        self.val_loader = DataLoader(dataset=valset, batch_size=config['batch_size'], collate_fn=trainset.collate_fn)
        
        self.num_train_samples = len(trainset_portion)
        self.iter_trainloader = iter(self.train_loader)    

    def init_loss_fn(self):
        self.loss = nn.CrossEntropyLoss()
        self.adv_loss = torch.nn.MSELoss() 

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def test(self, personalized=False):
        self.model.eval()
        if 'pFed' in self.algorithm:
            self.local_model.eval()
        correct_count = 0
        tot_count = 0
        loss = 0
        for x, y in self.val_loader:
            output = self.model(x)
            if 'pFed' in self.algorithm:
                output = self.local_model(x)
            loss += self.loss(output, y)
            correct_count += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            tot_count += y.shape[0]

        val_loss_path = self.model_path.replace('user', 'user_val_loss').replace('pt', 'txt')
        with open(val_loss_path, 'a+') as f:
            f.write(f"{loss} \n")

        val_acc_path = self.model_path.replace('user', 'user_val_acc').replace('pt', 'txt')
        with open(val_acc_path, 'a+') as f:
            f.write(f"{1.0*correct_count/tot_count} \n")
            
        return correct_count, tot_count, loss

    def save_model(self):
        torch.save(self.model, self.model_path)
        if 'pFed' in self.algorithm:
            torch.save(self.local_model, self.model_path) 

    def load_model(self):
        #load_from_server
        self.model = torch.load(self.server_model_path)
