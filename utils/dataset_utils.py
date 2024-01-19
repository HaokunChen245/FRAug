import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as tfs
from PIL import Image
from PIL import ImageFile
from torchvision.datasets import USPS, MNIST, SVHN
from torch.utils.data import ConcatDataset, random_split

def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)
    # print (b)
    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    ratio = random.randint(1,10)/10

    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    # a_src[:,h1:h2,w1:w2] = a_src[:,h1:h2,w1:w2] * ratio + a_trg[:,h1:h2,w1:w2] * (1- ratio)
    # a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    # a_trg[:,h1:h2,w1:w2] = a_src[:,h1:h2,w1:w2]
    # a_trg = np.fft.ifftshift( a_trg, axes=(-2, -1) )
    return a_src

def source_to_target_freq( src_img, amp_trg, L=0.1 ):
    # exchange magnitude
    # input: src_img, trg_img
    src_img_np = src_img.cpu().numpy()
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )
    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )
    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg

# if self.apply_FedDG:            
#     img_raw = img
#     ava_frq_target = self.source_domains
#     if self.domain in ava_frq_target: ava_frq_target.remove(self.domain)
#     if self.target_domain in ava_frq_target: ava_frq_target.remove(self.target_domain)
#     frq_amp_target = random.choice(self.source_domains)

#     frq_amp_root = os.path.join(self.root_dir.replace('VLCS', f"VLCS_freq_amp"), frq_amp_target)
#     with open(os.path.join(frq_amp_root, 'file_list.txt')) as f:
#         frq_amp_dir = random.choice(list(f.readlines()))

#     print(img_raw.min(), img_raw.max())
#     tar_freq = np.load(frq_amp_dir[:-1])
#     img_trans = source_to_target_freq(img_raw, tar_freq, L=0)
#     img_trans = np.clip(img_trans, 0, 255)

#     return img_raw, img_trans, label

def get_source_domain_names(dataset):
    if dataset=='PACS':
        return ['art_painting', 'cartoon', 'photo', 'sketch']
    elif dataset=='Digits':
        return ['MNIST', 'MNISTM', 'SVHN', 'USPS']
    elif dataset=='VLCS':
        return ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
    elif dataset=='OfficeHome':
        return ['Art', 'Clipart', 'Product', 'Real_World']
    elif dataset=='miniDomainNet':
        return ['real', 'clipart', 'painting', 'sketch']
    elif dataset=='DomainNet_FedBN':
        return ['real', 'clipart', 'painting', 'sketch', 'infograph', 'quickdraw']
    
def get_source_domain_name_abbr(dataset):
    if dataset=='PACS':
        return ['A', 'C', 'P', 'S']
    elif dataset=='Digits':
        return ['MT', 'MM', 'SV', 'UP']
    elif dataset=='OfficeHome':
        return ['A', 'C', 'P', 'R']
    elif dataset=='miniDomainNet':
        return ['R', 'C', 'P', 'S']
    elif dataset=='DomainNet_FedBN':
        return ['R', 'C', 'P', 'S', 'I', 'Q']
    
def get_image_size(dataset):
    if dataset=='PACS':
        return 224
    elif dataset=='VLCS':
        return 224
    elif dataset=='OfficeHome':
        return 224
    elif dataset=='Digits':
        return 32
    elif dataset=='miniDomainNet':
        return 96
    elif dataset=='DomainNet_FedBN':
        return 224
    
def get_class_number(dataset):
    if dataset=='PACS':
        return 7
    elif dataset=='VLCS':
        return 5
    elif dataset=='OfficeHome':
        return 65
    elif dataset=='Digits':
        return 10
    elif dataset=='miniDomainNet':
        return 126
    elif dataset=='DomainNet_FedBN':
        return 10
    
def get_class_names(dataset):
    if dataset=='PACS':
        return ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
    elif dataset=='VLCS':
        return ['bird', 'car', 'chair', 'dog', 'person']
    elif dataset=='OfficeHome':
        CLASSES = 'Alarm Clock, Backpack, Batteries, Bed, Bike, Bottle, Bucket, Calculator, Calendar, Candles, Chair, Clipboards, Computer, Couch, Curtains, Desk Lamp, Drill, Eraser, Exit Sign, Fan, File Cabinet, Flipflops, Flowers, Folder, Fork, Glasses, Hammer, Helmet, Kettle, Keyboard, Knives, Lamp Shade, Laptop, Marker, Monitor, Mop, Mouse, Mug, Notebook, Oven, Pan, Paper Clip, Pen, Pencil, Postit Notes, Printer, Push Pin, Radio, Refrigerator, ruler, Scissors, Screwdriver, Shelf, Sink, Sneakers, Soda, Speaker, Spoon, Table, Telephone, Toothbrush, Toys, Trash Can, TV, Webcam'
        return [c.upper() for c in CLASSES.split(', ')]
    elif dataset=='Digits':
        return [0,1,2,3,4,5,6,7,8,9]
    elif dataset=='miniDomainNet':
        return ['umbrella', 'television', 'potato', 'see_saw', 'zebra', 'dragon', 'chair', 'carrot', 'sea_turtle', 'helicopter', 'teddy-bear', 'sheep', 'coffee_cup', 'grapes', 'helmet', 'dolphin', 'squirrel', 'drums', 'guitar', 'basket', 'pillow', 'crocodile', 'mushroom', 'dog', 'table', 'anvil', 'peanut', 'fish', 'bottlecap', 'mosquito', 'camera', 'elephant', 'ant', 'bathtub', 'butterfly', 'dumbbell', 'asparagus', 'streetlight', 'cat', 'purse', 'penguin', 'calculator', 'crab', 'duck', 'string_bean', 'giraffe', 'lion', 'ceiling_fan', 'The_Great_Wall_of_China', 'fence', 'alarm_clock', 'monkey', 'goatee', 'pencil', 'speedboat', 'truck', 'mug', 'screwdriver', 'train', 'pear', 'hammer', 'castle', 'flower', 'skateboard', 'feather', 'raccoon', 'cactus', 'panda', 'lipstick', 'The_Eiffel_Tower', 'aircraft_carrier', 'bee', 'rabbit', 'eyeglasses', 'kangaroo', 'bus', 'banana', 'horse', 'shoe', 'saxophone', 'cannon', 'onion', 'submarine', 'computer', 'flamingo', 'cruise_ship', 'lantern', 'blueberry', 'strawberry', 'canoe', 'spider', 'compass', 'foot', 'broccoli', 'axe', 'bird', 'blackberry', 'laptop', 'bear', 'candle', 'pineapple', 'peas', 'rifle', 'fork', 'mouse', 'toe', 'watermelon', 'power_outlet', 'cake', 'chandelier', 'cow', 'vase', 'snake', 'frog', 'whale', 'microphone', 'tiger', 'cell_phone', 'camel', 'leaf', 'pig', 'rhinoceros', 'swan', 'lobster', 'teapot', 'cello']
    elif dataset=='DomainNet_FedBN':
        return ['bird', 'feather', 'headphones', 'ice_cream', 'teapot', 'tiger', 'whale', 'windmill', 'wine_glass', 'zebra']  
