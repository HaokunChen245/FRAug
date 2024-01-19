from utils.dataset_utils import *
from datasets.base import BaseDataset
ImageFile.LOAD_TRUNCATED_IMAGES = True

class PACS(BaseDataset):   
    def __init__(self, dataset_root_dir, mode, domain, img_size=224):
        BaseDataset.__init__(self, dataset_root_dir, mode, domain, img_size)
        
        self.CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
        self.split_images()        
        self.source_domains = get_source_domain_names('PACS') 
        
    def split_images(self):    
        split_dir = self.root_dir + '/splits/'
        for p in os.listdir(split_dir):
            if self.mode in p and self.domain in p:
                with open(split_dir + p) as f:                    
                    for l in f.readlines():
                        self.imgs.append(l.split(' ')[0])                            
    
    def __getitem__(self, index):     
        imgs = []
        labels = []   
        p = self.imgs[index]
        img = self.transforms(Image.open(self.root_dir + '/images/kfold/' + p).convert('RGB'))
        tag = p.split('/')[1]
        label = torch.tensor(self.CLASSES.index(tag))
        
        return img, label  
