import torch.nn as nn
import os
import torch
import torch.utils.data as data
import random
import argparse
from datasets import PACS, OfficeHome, VLCS, Digits, miniDomainNet, DomainNet_FedBN
from utils.dataset_utils import *
from utils.main_utils import *
import numpy as np
import json
import collections

#for efficiency
global_valsets = {}
global_testsets = {}

def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs).cuda()
    labels = torch.stack(labels).cuda()
    return imgs, labels
        
def evaluate(model, mode, domain):
    if 'test' in mode:
        set = global_testsets[domain]
    elif 'val' in mode:
        set = global_valsets[domain]
        
    loader = torch.utils.data.DataLoader(dataset=set, batch_size=128, collate_fn=collate_fn)
    if isinstance(model, list):
        F = model[0].eval()
        C = model[1].eval()
    else:
        model.eval()

    ans = 0
    tot = 0
    with torch.no_grad():
        for (imgs, labels) in loader:
            if isinstance(model, list):
                o = F(imgs).squeeze(-1).squeeze(-1)
                o = C(o).argmax(1)
            else:        
                o = model(imgs).argmax(1)
                
            tmp_out = labels==o
            ans += int(tmp_out.sum())
            tot += imgs.shape[0]
    return ans/tot

def eval_on_one_setting(args):
    print('--------------------------------------------------')
    print('path:', args.model_dir, 'evaluation_type:', args.eval_type)
    acc = collections.OrderedDict()
    std = {}
    count_d = {}
    count_all = {}
    flag_for_print = False

    results = ""
    use_best_fc = False    

    for trial in os.listdir(args.model_dir):
        if not 'trial' in trial: continue
        backbone = trial.split('_')[1]
        
        args.num_class = get_class_number(args.dataset)
        args.source_domains = get_source_domain_names(args.dataset)     
        lists = trial.split('_')     

        if False:
            flag_for_print = True
            setting_default = trial    
            acc[setting_default] = {}
            count_d[setting_default] = 0        
            for idx, d in enumerate(args.source_domains):
                temp_acc = 0
                for j in range(int(lists[-2])):
                    model_snapshot_dir = os.path.join(args.model_dir, trial, f'user_{d}_{j}.pt')
                    if not os.path.exists(model_snapshot_dir): 
                        continue
                    T = torch.load(model_snapshot_dir)
                    temp_acc += evaluate(T, 'val', d)*100
                acc[setting_default][d] = temp_acc/int(lists[-2])
                count_d[setting_default] += 1
            continue
        
        if not 'Single' in lists:
            #evaluate with local model  
            flag_for_print = True
            setting_default = trial    
            acc[setting_default] = {}
            count_d[setting_default] = 0        
            for idx, d in enumerate(args.source_domains):
                model_snapshot_dir = os.path.join(args.model_dir, trial, f'user_{d}_0.pt')  
                if not os.path.exists(model_snapshot_dir): 
                    model_snapshot_dir = os.path.join(args.model_dir, trial, f'user_{d}.pt')
                    if not os.path.exists(model_snapshot_dir): continue
                T = torch.load(model_snapshot_dir)

                setting_all = setting_default.replace('42', 'allseed').replace('84', 'allseed').replace('168', 'allseed').replace('210', 'allseed')
                if not setting_all in acc.keys():
                    acc[setting_all] = {}
                if not d in acc[setting_all].keys():
                    acc[setting_all][d] = []

                temp_acc = evaluate(T, 'val', d)*100
                acc[setting_all][d].append(temp_acc)

                # count_d[setting_all] += 1
                acc[setting_default][d] = temp_acc
                count_d[setting_default] += 1
            continue
        
        for target_domain in args.source_domains:      
            if target_domain in lists:
                break
            elif target_domain=='art_painting' and 'art' in lists:
                break
            elif target_domain=='Real_World' and 'Real' in lists:
                break   
        args.source_domains.remove(target_domain)
                
        model_snapshot_dir = os.path.join(args.model_dir, trial, 'server.pt')     
        if not os.path.exists(model_snapshot_dir): continue
        T = torch.load(model_snapshot_dir)
        setting_default = trial.replace(target_domain, 'domain')            
        
        if not setting_default in acc.keys():
            acc[setting_default] = {}
            count_d[setting_default] = 0               
        
        if 'source' in args.eval_type:
            #evaluate on source domain validation set.
            for d in args.source_domains:
                temp_acc = evaluate(T, args.eval_type, d)
                print(f'{trial} evaluated on {args.eval_type} of {d}:', temp_acc)
                
        else:
            temp_acc = evaluate(T, args.eval_type, target_domain)*100
            acc[setting_default][target_domain] = temp_acc
            count_d[setting_default] += 1

            setting_all = setting_default.replace('42', 'allseed').replace('84', 'allseed').replace('168', 'allseed').replace('210', 'allseed')
            if not setting_all in acc.keys():
                acc[setting_all] = {}
            if not target_domain in acc[setting_all].keys():
                acc[setting_all][target_domain] = []

            print(setting_all)
            acc[setting_all][target_domain].append(temp_acc)
    
    if 'target' in args.eval_type or flag_for_print:  
        portion = {}
        for setting_default in acc.keys():          
            method = setting_default.split('_')[1]
            # if 14>len(setting_default.split('_')):
            #     p = setting_default.split('_')[-1]                
            # else:
            #     p = setting_default.split('_')[14]
            # for d in acc[setting_default].keys(): 
            #     tname = d+ '_' + method
            #     if tname not in portion.keys():                    
            #         portion[tname] = {}
            #     portion[tname][p] = acc[setting_default][d]
            
            if 'allseed' in setting_default:
                print(setting_default)
                acc_all = []
                for k in sorted(acc[setting_default].keys()):    
                    ct = len(acc[setting_default][k])

                    avg = np.mean(acc[setting_default][k])
                    std = np.std(acc[setting_default][k])
                    print(k, ":", avg, std)

                    for i in range(ct):
                        if i>=len(acc_all): acc_all.append(0)
                        acc_all[i] += acc[setting_default][k][i]                
                    acc[setting_default][k].append(avg)
                    acc[setting_default][k].append(std)
                    
                acc_all = [i/len(acc[setting_default].keys()) for i in acc_all]
                print('avg:', np.mean(acc_all), np.std(acc_all))

            else:
                if not count_d[setting_default]: continue
                print(setting_default)
                print(sorted(acc[setting_default].items(), key=lambda x:x[0]))
                acc_sum = 0
                for d in acc[setting_default].keys():                   
                    acc_sum += acc[setting_default][d]               
                print('avg:', acc_sum/count_d[setting_default])  
                acc[setting_default]['avg'] = acc_sum/count_d[setting_default]

def main():
    parser = argparse.ArgumentParser(description='PyTorch Generator training code')
    parser.add_argument('--dataset', type=str, help="name of the dataset.")
    parser.add_argument('--dataset_dir', type=str, help="directory of the dataset.")
    parser.add_argument('--model_dir', type=str, help="directory of the saved model.")
    parser.add_argument('--eval_type', type=str, default="target_testset", help='how to evaluate the model')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    global global_valsets       
    global global_testsets  
    
    dataset_mapping = {
        'PACS': PACS,
        'VLCS': VLCS,
        'OfficeHome': OfficeHome,
        'Digits': Digits,
        'DomainNet_FedBN': DomainNet_FedBN,
        'miniDomainNet': miniDomainNet,
    }
    assert args.dataset in dataset_mapping.keys()
    dataset = dataset_mapping[args.dataset]
    
    args.eval_batch_size = 128
    args.source_domains = get_source_domain_names(args.dataset)
    args.img_size = get_image_size(args.dataset)
    args.dataset_dir = os.path.join(args.dataset_dir, args.dataset)

    for d in args.source_domains:  
        if args.dataset=='Digits':
            global_valsets[d] = dataset(args.dataset_dir, mode = 'val', img_size=args.img_size, domain=d)           
            global_testsets[d] = dataset(args.dataset_dir, mode = 'test', img_size=args.img_size, domain=d)   
        else:
            trainset = dataset(args.dataset_dir, mode = 'train', img_size=args.img_size, domain=d)             
            len_valset = int(len(trainset) * 0.1) #train: val = 9:1, split the original trainset for model selection
            trainset_base, valset = random_split(trainset, [len(trainset)-len_valset, len_valset], 
                                generator=torch.Generator().manual_seed(args.seed)) #fix the split            
            global_valsets[d] = valset                         
            global_testsets[d] = dataset(args.dataset_dir, mode = 'test', img_size=args.img_size, domain=d) 
    eval_on_one_setting(args)    

if __name__=='__main__':
    main()
    
