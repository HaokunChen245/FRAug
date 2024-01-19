import torchvision
import torch.nn as nn
import os
import torch
import torchvision.transforms as tfs
import torch.utils.data as data
import random
import argparse
from datasets import PACS, OfficeHome, VLCS, Digits, miniDomainNet, DomainNet_FedBN
from utils.main_utils import *
from utils.dataset_utils import *
import ray
from ray import tune
from torch.utils.data import ConcatDataset, random_split
import time

def evaluate(model, loader):
    model.eval()
    ans = 0
    tot = 0
    with torch.no_grad():
        for (imgs, labels) in loader:
            o = model(imgs).argmax(1)
            tmp_out = labels==o
            ans += int(tmp_out.sum())
            tot += imgs.shape[0]
    return ans/tot

def get_train_val_split(dataset, config, d, portion):
    trainset = dataset(config['dataset_dir'], mode = 'train', img_size=config['img_size'], domain=d)
    len_valset = int(len(trainset) * 0.1) #train: val = 9:1, split the original trainset for model selection
    trainset_base, valset = random_split(trainset, [len(trainset)-len_valset, len_valset],
                        generator=torch.Generator().manual_seed(config['seed'])) #fix the split
    len_trainset_portion = int(len(trainset_base)*portion)
    trainset_portion, _= random_split(trainset_base, [len_trainset_portion, len(trainset_base)-len_trainset_portion],
                        generator=torch.Generator().manual_seed(config['seed'])) #fix the split

    return trainset_portion, valset

def get_config(args):
    train_args = {}
    train_args['algorithm'] = args.algorithm
    train_args['dataset'] = args.dataset
    train_args['img_size'] = args.img_size
    train_args['dataset_dir'] = args.dataset_dir
    train_args['max_ep_train'] = args.max_ep_train
    train_args['student_backbone'] = args.model

    train_args['num_class'] = get_class_number(train_args['dataset'])
    train_args['source_domains'] = get_source_domain_names(train_args['dataset'])

    if train_args['dataset'] == 'Digits':
        train_args['eval_batch_size'] = 512
        train_args['teacher_backbone'] = 'CNN_Digits'
        train_args['batch_size'] = tune.grid_search([256])
        train_args['lr'] = tune.grid_search([0.01])
        train_args['portion'] = tune.grid_search([0.05, 0.1, 0.2, 0.5, 1.0])

    elif train_args['dataset'] == 'miniDomainNet':
        train_args['eval_batch_size'] = 128
        train_args['teacher_backbone'] = 'resnet18'
        train_args['batch_size'] = tune.grid_search([128])
        train_args['lr'] = tune.grid_search([0.01])
        train_args['portion'] = tune.grid_search([0.1])

    elif train_args['dataset'] == 'DomainNet_FedBN':
        train_args['eval_batch_size'] = 128
        train_args['teacher_backbone'] = 'resnet18'
        train_args['batch_size'] = tune.grid_search([64])
        train_args['lr'] = tune.grid_search([0.01])
        train_args['portion'] = tune.grid_search([0.1])

    else:
        train_args['eval_batch_size'] = 128
        train_args['teacher_backbone'] = 'resnet18'
        train_args['batch_size'] = tune.grid_search([64])
        train_args['lr'] = tune.grid_search([0.01])
        train_args['portion'] = tune.grid_search([0.2, 0.4, 0.6, 0.8, 1.0])

    train_args['seed'] = tune.grid_search([84, 168, 210])
    if train_args['algorithm']=='Single':
        train_args['target_domain'] = tune.grid_search(train_args['source_domains'])
    else:
        train_args['target_domain'] = None

    train_args['hps_list'] = ['algorithm', 'dataset', 'teacher_backbone', 'target_domain', 'lr', 'batch_size', 'seed', 'portion']
    return train_args

def run_one_trial(config, checkpoint_dir=None):
    setup_seed(config['seed'])

    trainsets = []
    valsets = []
    best_acc = 0
    dataset_mapping = {
        'PACS': PACS,
        'VLCS': VLCS,
        'OfficeHome': OfficeHome,
        'Digits': Digits,
        'miniDomainNet': miniDomainNet,
        'DomainNet_FedBN': DomainNet_FedBN,
    }
    assert config['dataset'] in dataset_mapping.keys()
    dataset = dataset_mapping[config['dataset']]

    if config['target_domain']:
        best_acc=  0

        trainset = dataset(config['dataset_dir'], mode = 'train', img_size=config['img_size'], domain=config['target_domain'])
        trainset_portion, valset = get_train_val_split(dataset, config, config['target_domain'], config['portion'])

        valset_loader = torch.utils.data.DataLoader(dataset=valset, shuffle=True, batch_size=config['eval_batch_size'], collate_fn=trainset.collate_fn)
        trainset_loader = torch.utils.data.DataLoader(dataset=trainset_portion, shuffle=True, batch_size=config['batch_size'], collate_fn=trainset.collate_fn)

    else:
        best_local_acc = [0 for _ in range(len(config['source_domains']))]
        trainsets_portion = []
        valset_loaders = []

        for d in config['source_domains']:
            trainset = dataset(config['dataset_dir'], mode = 'train', img_size=config['img_size'], domain=d)
            trainset_portion, valset = get_train_val_split(dataset, config, d, config['portion'])
            trainsets_portion.append(trainset_portion)

            valset_loader = torch.utils.data.DataLoader(dataset=valset, shuffle=True, batch_size=config['eval_batch_size'], collate_fn=trainset.collate_fn)
            valset_loaders.append(valset_loader)

        trainset_portion = ConcatDataset(trainsets_portion)
        trainset_loader = torch.utils.data.DataLoader(dataset=trainset_portion, shuffle=True, batch_size=config['batch_size'], collate_fn=trainset.collate_fn)

    #########Here start from ImageNet pretrained network.##################
    net = create_model(config['teacher_backbone'], config['num_class'], pretrained=True)
    net = net.cuda().train()

    # net = torch.load(f"")
    # net = net.cuda().train()
    # for k, v in net.named_parameters():
    #     if not 'fc' in k and 'resnet' in config['teacher_backbone']:
    #         v.requires_grad=False
    #     if  'classifier' not in k and 'resnet' not in config['teacher_backbone']:
    #         v.requires_grad=False

    config['trial_name'] = trial_name_string(config)
    config['model_path'] = os.path.join(config['logs_local_dir'], config['log_run_name'], config['trial_name'])

    optimizer = torch.optim.SGD(net.parameters(), weight_decay=0.0005, momentum=.9, nesterov=True, lr=config['lr'])
    # lr_scheduler = lr_cosine_policy(config['lr'], 1 * len(trainset_loader), config['max_ep_train'] * len(trainset_loader))
    taskL = nn.CrossEntropyLoss()
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(max_ep*0.6), int(max_ep*0.8)], gamma=0.9)

    it = 0
    metrics = {}
    stop_criterion = 0
    for ep in range(config['max_ep_train']):
        for batch in trainset_loader:
            it += 1
            net.train()
            # cur_lr = lr_scheduler(optimizer, it, it)
            imgs, labels = batch
            optimizer.zero_grad()
            outputs = net(imgs)
            targets = labels
            L = taskL(outputs, targets)
            L.backward()
            optimizer.step()
            metrics['loss'] = float(L)
            # metrics['lr'] = float(cur_lr)
            if stop_criterion>15:
                tune.report(done=True)
            if it%20==0:
                if config['target_domain']:
                    val_acc = evaluate(net, valset_loader)
                    if best_acc < float(val_acc):
                        torch.save(net, os.path.join(config['model_path'], f"server.pt"))
                        best_acc = val_acc
                        stop_criterion = 0
                    else:
                        stop_criterion += 1
                    metrics['acc'] = float(val_acc)
                else:
                    for idx, valset_loader in enumerate(valset_loaders):
                        val_acc = evaluate(net, valset_loader)
                        if best_local_acc[idx] < float(val_acc):
                            torch.save(net, os.path.join(config['model_path'], f"user_{config['source_domains'][idx]}.pt"))
                            best_local_acc[idx] = val_acc
                        metrics[f'acc_{idx}'] = float(val_acc)
            tune.report(**metrics)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="PACS")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--algorithm", type=str)
    parser.add_argument("--dataset_dir", type=str, default='/home/')
    parser.add_argument("--gpu_per_trial", type=float, default=0.5)
    parser.add_argument("--cpu_per_trial", type=float, default=2.0)
    parser.add_argument("--use_aws_server", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda:0", choices=["cpu","cuda"], help="run device (cpu | cuda)")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--max_ep_train", type=int, default=40)

    args = parser.parse_args()
    config = get_config(args)

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
        config['logs_local_dir'] =  "/home/ubuntu/logs"
    else:
        on_autoscale_cluster = False
        config['logs_local_dir'] =  "/home/ubuntu/fedlda/logs"
        ray.init()
        resources_per_trial = {"gpu": args.gpu_per_trial, "cpu": args.cpu_per_trial}

    try:
        config['log_run_name'] = config['algorithm'] + '_' + config['dataset'] + '_' +time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))
        analysis = tune.run(
                run_one_trial,
                config = config,
                name = config['log_run_name'],
                resources_per_trial= resources_per_trial,
                local_dir = config['logs_local_dir'],
#                 queue_trials=True if on_autoscale_cluster else False,
                raise_on_failed_trial=False if on_autoscale_cluster else True,
                trial_name_creator=trial_name_string,
                trial_dirname_creator=trial_name_string,

                #resume = 'ERRORED_ONLY',
                #name = 'train_one_setting_2021-09-09_20-10-52',
        )

    finally:
        if on_autoscale_cluster:
            print("Downscaling cluster in 2 minutes...")
            time.sleep(120)  # Wait for any syncing to complete.

if __name__=='__main__':
    main()
