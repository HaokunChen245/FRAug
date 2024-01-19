from users.user_FRAug import UserFRAug
from servers.server_base import Server
from networks import G
import time
import torch
from ray import tune

class FRAug(Server):
    #pFederated Generative Domain Augmentation for feature space non-i.i.d. data
    def __init__(self, config, model):
        super().__init__(config, model)

        # Initialize data for all  users
        total_users = len(config['source_domains']) * config['user_per_domain']
        self.best_local_acc= [0 for _ in range(total_users)]
        if 'latent_dim' in config.keys():
            self.latent_dim = config['latent_dim']
        else:
            self.latent_dim = int(self.feature_dim/2)
        self.G = G(self.feature_dim, self.latent_dim, self.class_num).to(self.device)

        for i in range(total_users):            
            user = UserFRAug(config, i, model)            
            self.users.append(user)
                
    def train(self, config):
        for ep in range(self.global_epochs):
            self.metrics['comm_round'] = ep
            
            self.timestamp = time.time() # log communicate time
            self.selected_users = self.select_users(self.num_users)
            with torch.no_grad():
                self.communicate(mode=config['model_share'])
                self.communicate(model_name='G', mode=config['model_share_G'])
                self.evaluate()
            curr_timestamp=time.time()  # log  server-agg end time
            agg_time = curr_timestamp - self.timestamp
            self.metrics['comm_time'] = agg_time
            
            self.timestamp = time.time() # log user-training start time
            for user in self.selected_users: # allow selected users to train
                user.train(ep, personalized=self.personalized) #* user.train_samples                
            curr_timestamp = time.time() # log  user-training end time            
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
            self.metrics['user_avg_training_time'] = train_time
            
            self.report_results()
