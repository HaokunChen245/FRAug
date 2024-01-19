import torch
import os
import numpy as np
import copy
import torch.nn.functional as F
import time
import torch.nn as nn
from ray import tune
import yaml
            
class Server:
    def __init__(self, config, model):

        # Set up the main attributes
        self.model = copy.deepcopy(model).to(config['device'])
        self.feature_dim = self.model.feature_dim
        
        self.dataset = config['dataset']
        self.model_path = os.path.join(config['model_path'], 'server.pt')
        self.batch_size = config['batch_size']
        self.algorithm = config['algorithm']
        self.device = config['device']  
        self.personalized = 'pFed' in self.algorithm.lower()
        self.users = [] 
        self.global_epochs = config['global_epochs']   
        self.local_model_pretrain = 'local_epochs_pretrain' in config.keys() and config['local_epochs_pretrain']>0      
        self.num_users = config['num_users']
        self.class_num = config['class_num']
        self.metrics = {}
        self.best_glob_acc = 0     
         
        with open(f"{config['model_path']}/training_setting.yml", 'w') as f:        
            yaml.dump(config, f)  
       
    def communicate(self, model_name=None, mode='all'):
        assert (self.selected_users is not None and len(self.selected_users) > 0)        

        if not model_name: 
            server_model = self.model
            user_models = [user.model for user in self.selected_users]            
        elif model_name == 'G': 
            server_model = self.G
            user_models = [user.G for user in self.selected_users]
        elif model_name == 'D': 
            server_model = self.D
            user_models = [user.D for user in self.selected_users]
        elif model_name == 'featurizer': 
            server_model = self.featurizer
            user_models = [user.featurizer for user in self.selected_users]
        elif model_name == 'classifier': 
            server_model = self.classifier
            user_models = [user.classifier for user in self.selected_users]
        user_num_train_samples = [user.num_train_samples for user in self.selected_users]
        total_train_samples = sum(user_num_train_samples)

        for key in server_model.state_dict().keys():
            #update server model
            # num_batches_tracked is a non trainable LongTensor and
            # num_batches_tracked are the same for all clients for the given datasets 
            if 'num_batches_tracked' in key:                
                server_model.state_dict()[key].data.copy_(user_models[0].state_dict()[key])
            else:
                temp = torch.zeros_like(server_model.state_dict()[key])        
                for user_model, user_num_train_sample in zip(user_models, user_num_train_samples):
                    temp += user_num_train_sample / total_train_samples * user_model.state_dict()[key]
                server_model.state_dict()[key].data.copy_(temp)

            #update client model
            for user_model, user_num_train_sample in zip(user_models, user_num_train_samples):                  
                if mode=='no_sharing':
                    break
                if mode=='all':
                    pass
                if 'no_fc' in mode and  'fc' in key: 
                    continue
                if 'no_local' in mode and 'local' in key: 
                    continue
                if 'no_bn_running_stats' in mode and 'bn' in key and 'running' in key: 
                    continue
                elif 'no_bn' in mode and 'bn' in key: 
                    continue
                user_model.state_dict()[key].data.copy_(server_model.state_dict()[key])
                        
    def save_model(self, model_path=None):
        self.model.train()
        if model_path:
            torch.save(self.model, model_path)
        else:
            torch.save(self.model, self.model_path)

    def load_model(self, model_path=None):
        if model_path:
            assert os.path.exists(model_path)
            self.model = torch.load(model_path)
        else:
            assert os.path.exists(self.model_path)
            self.model = torch.load(self.model_path)

    def select_users(self, num_users, return_idx=False):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        Return:
            list of selected clients objects
        '''
        if (num_users >= len(self.users) or num_users==-1):
            print("All users are selected")
            return self.users
        
        num_users = min(num_users, len(self.users))
        if return_idx:
            user_idxs = np.random.choice(range(len(self.users)), num_users, replace=False)  # , p=pk)
            return [self.users[i] for i in user_idxs], user_idxs
        else:
            return np.random.choice(self.users, num_users, replace=False)

    def init_loss_fn(self):
        self.ensemble_loss=nn.KLDivLoss(reduction="batchmean")#,log_target=True)
        self.loss = nn.CrossEntropyLoss()

    def report_results(self):
        #two stars used dict with keys
        tune.report(**self.metrics)

    def evaluate(self, selected=True, logging=True, personalized=False):
        test_samples_num = []
        test_tot_correct = []
        test_losses = []
        users = self.selected_users if selected else self.users
        with torch.no_grad():
            for idx, user in enumerate(users):
                tc, sn, loss = user.test(personalized=personalized)   
                
                test_tot_correct.append(int(tc))            
                test_samples_num.append(int(sn))
                test_losses.append(loss)                

                local_acc = float(tc/sn)
                self.metrics[f'local_acc_{idx}'] = local_acc
                if local_acc>self.best_local_acc[idx]:
                    self.best_local_acc[idx] = local_acc
                    user.save_model()
                    if 'FedFA' in self.algorithm:
                        user.save_G()
                        user.save_FTNet()
                
        num_test_samples = np.sum(test_samples_num)
        num_correct_test_samples = np.sum(test_tot_correct)
        glob_acc = num_correct_test_samples*1.0/num_test_samples
        test_loss = np.sum([x * float(y) for (x, y) in zip(test_samples_num, test_losses)]).item() / num_test_samples
        
        if logging:
            self.metrics['glob_acc'] = float(glob_acc)
            self.metrics['glob_loss'] = float(test_loss)
            
        if float(glob_acc)>self.best_glob_acc:
            #store the model with best source domain validation acc.
            self.save_model()
            self.best_glob_acc = float(glob_acc)
