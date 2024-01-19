from utils.dataset_utils import *
from datasets.base import BaseDataset
import pickle
ImageFile.LOAD_TRUNCATED_IMAGES = True

class DomainNet_FedBN(BaseDataset):   
    def __init__(self, dataset_root_dir, mode, domain, img_size=224):
        if 'DomainNet' in dataset_root_dir:
            dataset_root_dir = dataset_root_dir.replace('/DomainNet', '')
        BaseDataset.__init__(self, dataset_root_dir, mode, domain, img_size)        
        
        self.source_domains = get_source_domain_names('DomainNet_FedBN') 
        self.CLASSES = ['bird', 'feather', 'headphones', 'ice_cream', 'teapot', 'tiger', 'whale', 'windmill', 'wine_glass', 'zebra']  
        
        #follow FedBN
        if mode=='train':
            self.split_images('train')  
        if mode=='val':
            self.split_images('test')  
        if mode=='test':
            self.split_images('train')  
            self.split_images('test')  
        
    def split_images(self, mode): 
        split_file = os.path.join(
            self.root_dir, f'DomainNet/FedBN_split/{self.domain}_{mode}.pkl')
        
        with open(split_file, "rb") as f:
            for img in pickle.load(f)[0]:
                self.imgs.append(img)    
                self.labels.append(self.CLASSES.index(img.split('/')[2]))            
    
    def __getitem__(self, index):     
        imgs = []
        labels = []   
        p = self.imgs[index]
        img = self.transforms(Image.open(os.path.join(self.root_dir,p)).convert('RGB'))
        label = torch.tensor(self.labels[index])
        
        return img, label          
 

