import torchvision
import torch.nn as nn
import os
import torch
import random
import argparse
import ray
from ray import tune
from utils.main_utils import *
from utils.dataset_utils import *
import time
from servers import FRAug

def get_config_from_args(args):
    config = {}
    config['device'] = args.device
    config['dataset'] = args.dataset
    config['num_users'] = args.num_users
    config['global_epochs'] = args.global_epochs  
    config['local_epochs'] = args.local_epochs    
    config['class_names'] = get_class_names(args.dataset)
    config['class_num'] = get_class_number(args.dataset)
    config['source_domains'] = get_source_domain_names(args.dataset)
    config['img_size'] = get_image_size(args.dataset)
    if 'DomainNet' in config['dataset']:
        config['dataset_dir'] = os.path.join(args.dataset_dir, "DomainNet")
    else:
        config['dataset_dir'] = os.path.join(args.dataset_dir, config['dataset'])
    
    ################### for general model tuning#############################
    config['lr'] = tune.grid_search([0.01])
    config['optimizer'] = tune.grid_search(['SGD'])
    config['local_momentum'] = 0.5  
    config['algorithm'] = tune.grid_search(['FRAug']) #args.algorithm  
    config['cls_mean_momentum'] = tune.grid_search([0.3])
    config['steps_using_G'] = tune.grid_search([1])
    
    #Digits best: no feature interpolation, inter_form = 0.5 
    #Avg does not work for OfficeHome
    
    config['hps_list'] = ['algorithm', 'batch_size', 'local_epochs', 'seed', 'L_G_dist_lambda', 'L_FT_dist_lambda', 'lr_G', 'lr_FTNet', 'distance_metric_G', 'distance_metric_FT', 'use_FTNet',  'use_homo_label_aug', 'use_feature_inter',  'start_add_G', 'latent_dim' ] 
    
    # use start_add_G to control whether to add G
    config['start_add_G'] = tune.grid_search([0])
    config['use_G'] = tune.grid_search([True])
    config['use_FTNet'] = tune.grid_search([True])

    if config['dataset'] == 'OfficeHome':  
        config['user_per_domain'] = tune.grid_search([1])
        config['model_share'] = tune.grid_search(['no_bn_running_stats']) #also try all!!!!!!
        config['model_share_G'] = tune.grid_search(['no_local']) #all doesn't work! 
        config['use_homo_label_aug'] = tune.grid_search([2]) #2 works the best!
        config['cls_criterion_type_FT'] = tune.grid_search(['entropy'])
        config['use_feature_inter'] = tune.grid_search([True])   

        config['model'] = tune.grid_search(['resnet18']) #args.model    
        config['batch_size'] = tune.grid_search([16]) #96 not working for OfficeHome
        config['local_epochs'] = tune.grid_search([10])
        config['FTNet_with_BN'] = tune.grid_search([True])
        config['FTNet_with_cls'] = tune.grid_search([True]) 
        config['seed'] = tune.grid_search([42])
        config['portion'] = tune.grid_search([0.1])

        config['lr_G'] = tune.grid_search([0.05]) #0.02 is the best 
        config['lr_FTNet'] = tune.grid_search([0.025]) #0.05 is the best
        config['latent_dim'] = tune.grid_search([64, 128, 256, 512])
        config['L_G_dist_lambda'] = tune.grid_search([0, 0.75, 1.25, 1.5])         
        config['L_FT_dist_lambda'] = tune.grid_search([0, 0.75, 1.25, 1.5]) 

        config['inter_gamma'] = tune.grid_search([0.01])
        config['inter_form'] = tune.grid_search(['exp'])
        config['distance_metric_G'] = tune.grid_search(['MMD_kernel']) #MMD and MMD_kernel doesn't difference too much
        config['distance_metric_FT'] = tune.grid_search(['MMD_kernel']) 

    elif config['dataset'] == 'Digits':        
        config['user_per_domain'] = tune.grid_search([1])
        config['model_share'] = tune.grid_search(['no_bn_running_stats']) #all not working for Digits!!!!!
        config['model_share_G'] = tune.grid_search(['all'])
        config['use_homo_label_aug'] = tune.grid_search([2])
        config['cls_criterion_type_FT'] = tune.grid_search(['entropy'])
        config['use_feature_inter'] = tune.grid_search([False]) #Digits don't use feature interpolation

        config['model'] = tune.grid_search(['CNN_Digits']) #args.model  
        config['batch_size'] = tune.grid_search([64])
        config['local_epochs'] = tune.grid_search([20])
        config['portion'] = tune.grid_search([0.01])
        
        config['lr_G'] = tune.grid_search([0.05, 0.025]) #best already
        config['lr_FTNet'] = tune.grid_search([0.05, 0.025]) #best already
        config['L_G_dist_lambda'] = tune.grid_search([1]) #best already 
        config['L_FT_dist_lambda'] = tune.grid_search([0.1, 1, 10]) #best already 
        config['FTNet_with_BN'] = tune.grid_search([True])
        config['FTNet_with_cls'] = tune.grid_search([True]) 
        config['seed'] = tune.grid_search([42])
        
        config['inter_gamma'] = tune.grid_search([0.002])
        config['inter_form'] = tune.grid_search(['exp'])
        config['distance_metric_G'] = tune.grid_search(['L2']) #MMD and MMD_kernel doesn't difference too much
        config['distance_metric_FT'] = tune.grid_search(['MMD_kernel']) 

    elif config['dataset'] == 'PACS':        
        config['user_per_domain'] = tune.grid_search([1])
        config['model_share'] = tune.grid_search(['no_bn_running_stats']) 
        config['model_share_G'] = tune.grid_search(['all']) 
        config['use_homo_label_aug'] = tune.grid_search([2])
        config['cls_criterion_type_FT'] = tune.grid_search(['entropy'])
        config['use_feature_inter'] = tune.grid_search([True])  #True leads to better results for PACS

        config['model'] = tune.grid_search(['resnet18']) #args.model    
        config['batch_size'] = tune.grid_search([32]) #32 is the best 
        config['local_epochs'] = tune.grid_search([5]) #5 is the best 
        config['portion'] = tune.grid_search([0.1]) 
        
        #find already the best
        config['lr_G'] = tune.grid_search([0.05]) #0.05 is the best 
        config['lr_FTNet'] = tune.grid_search([0.05]) #0.05 is the best
        config['L_G_dist_lambda'] = tune.grid_search([10]) #10 is the best
        config['L_FT_dist_lambda'] = tune.grid_search([1]) #1 is the best
        config['FTNet_with_BN'] = tune.grid_search([True])
        config['FTNet_with_cls'] = tune.grid_search([True]) 
        config['seed'] = tune.grid_search([42, 84])
        
        config['inter_gamma'] = tune.grid_search([0.01])
        config['inter_form'] = tune.grid_search(['exp'])
        config['distance_metric_G'] = tune.grid_search(['MMD_kernel']) #MMD and MMD_kernel doesn't difference too much
        config['distance_metric_FT'] = tune.grid_search(['L2']) 
        
  
    return config

def create_server_n_user(config):    
    model = create_model(config['model'], config['class_num']).to(config['device'])   
    
    if 'FRAug' in config['algorithm']:
        server=FRAug(config, model)

    else:
        print(f"Algorithm {config['algorithm']} has not been implemented.")
        exit()
    return server

def run_one_trial(config, checkpoint_dir=None):
    setup_seed(config['seed'])
    config['trial_name'] = trial_name_string(config)
    config['model_path'] = os.path.join(config['logs_local_dir'], config['log_run_name'], config['trial_name'])
    # Generate model
    server = create_server_n_user(config)
    server.train(config)

def main():     
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="PACS")
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--gpu_per_trial", type=float, default=1.0)
    parser.add_argument("--cpu_per_trial", type=float, default=3.0)
    parser.add_argument("--use_aws_server", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda:0", choices=["cpu","cuda"], help="run device (cpu | cuda)")
    parser.add_argument("--algorithm", type=str, default="FedAvg")
    parser.add_argument("--lr", type=float, default=0.001, help="Local learning rate")
    parser.add_argument("--global_epochs", type=int, default=300, help="Training epochs for server")
    parser.add_argument("--local_epochs", type=int, default=1, help="Training epochs at each local client")
    parser.add_argument("--num_users", type=int, default=-1, help="Number of Users per round")
    parser.add_argument("--log_dir", type=str)
    args = parser.parse_args()
    config = get_config_from_args(args)
    
    if args.use_aws_server:
        n_cpu_per_trial = os.cpu_count()
        resources_per_trial = {
            "cpu": n_cpu_per_trial,
            "gpu": 1 if torch.cuda.is_available() else 0,
        }
        try:
            on_autoscale_cluster = False
            ray.init(
                address="auto",
                _redis_password=os.getenv(
                    "RAY_REDIS_PASSWD", ray.ray_constants.REDIS_DEFAULT_PASSWORD
                ),
            )
            on_autoscale_cluster = True
        except ConnectionError as e:
            print(f"ConnectionError: {e} --> Starting ray in single node mode.")
        config['logs_local_dir'] = os.path.join(args.log_dir, "logs")          
    else:        
        on_autoscale_cluster = False      
        config['logs_local_dir'] =  os.path.join(args.log_dir, "logs")
        ray.init()
        resources_per_trial = {"gpu": args.gpu_per_trial, "cpu": args.cpu_per_trial}
    
    try:
        config['log_run_name'] = args.dataset + '_' + time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))
        analysis = tune.run(
                run_one_trial,
                config = config,
                name = config['log_run_name'],
                resources_per_trial= resources_per_trial,
                local_dir = config['logs_local_dir'],
                raise_on_failed_trial=False if on_autoscale_cluster else True,
                trial_name_creator=trial_name_string,
                trial_dirname_creator=trial_name_string,
            
                # resume = 'PROMPT',
                # name = 'Digits_20220424_051705',
        )
    finally:
        if on_autoscale_cluster:
            print("Downscaling cluster in 1 minutes...")
            time.sleep(60)  # Wait for any syncing to complete.
        
if __name__=='__main__':
    main()
