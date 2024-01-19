from utils.dataset_utils import *
from datasets.base import BaseDataset
ImageFile.LOAD_TRUNCATED_IMAGES = True

class miniDomainNet(BaseDataset):   
    def __init__(self, dataset_root_dir, mode, domain, img_size=96):
        BaseDataset.__init__(self, dataset_root_dir, mode, domain, img_size)
        
        self.CLASSES = ['umbrella', 'television', 'potato', 'see_saw', 'zebra', 'dragon', 'chair', 'carrot', 'sea_turtle', 'helicopter', 'teddy-bear', 'sheep', 'coffee_cup', 'grapes', 'helmet', 'dolphin', 'squirrel', 'drums', 'guitar', 'basket', 'pillow', 'crocodile', 'mushroom', 'dog', 'table', 'anvil', 'peanut', 'fish', 'bottlecap', 'mosquito', 'camera', 'elephant', 'ant', 'bathtub', 'butterfly', 'dumbbell', 'asparagus', 'streetlight', 'cat', 'purse', 'penguin', 'calculator', 'crab', 'duck', 'string_bean', 'giraffe', 'lion', 'ceiling_fan', 'The_Great_Wall_of_China', 'fence', 'alarm_clock', 'monkey', 'goatee', 'pencil', 'speedboat', 'truck', 'mug', 'screwdriver', 'train', 'pear', 'hammer', 'castle', 'flower', 'skateboard', 'feather', 'raccoon', 'cactus', 'panda', 'lipstick', 'The_Eiffel_Tower', 'aircraft_carrier', 'bee', 'rabbit', 'eyeglasses', 'kangaroo', 'bus', 'banana', 'horse', 'shoe', 'saxophone', 'cannon', 'onion', 'submarine', 'computer', 'flamingo', 'cruise_ship', 'lantern', 'blueberry', 'strawberry', 'canoe', 'spider', 'compass', 'foot', 'broccoli', 'axe', 'bird', 'blackberry', 'laptop', 'bear', 'candle', 'pineapple', 'peas', 'rifle', 'fork', 'mouse', 'toe', 'watermelon', 'power_outlet', 'cake', 'chandelier', 'cow', 'vase', 'snake', 'frog', 'whale', 'microphone', 'tiger', 'cell_phone', 'camel', 'leaf', 'pig', 'rhinoceros', 'swan', 'lobster', 'teapot', 'cello'] #126
        self.source_domains = get_source_domain_names('miniDomainNet') 
        
        #follow Dassl
        if mode=='train':
            self.split_images('train')  
        if mode=='val':
            self.split_images('test')  
        if mode=='test':
            self.split_images('train')  
            self.split_images('test')  
        
    def split_images(self, mode):            
        split_file = os.path.join(
            self.root_dir, f'miniDomainNet_split/{self.domain}_{mode}.txt')
        
        with open(split_file, "r") as f:
            for l in f.readlines():
                self.imgs.append(l.split(' ')[0])   
                self.labels.append(self.CLASSES.index(l.split('/')[1]))
    
    def __getitem__(self, index):     
        imgs = []
        labels = []   
        p = self.imgs[index]
        img = self.transforms(Image.open(os.path.join(self.root_dir,p)).convert('RGB'))
        label = torch.tensor(self.labels[index])
        
        return img, label          
 
