from utils.dataset_utils import *
from datasets.base import BaseDataset
ImageFile.LOAD_TRUNCATED_IMAGES = True

class OfficeHome(BaseDataset):
    def __init__(self, dataset_root_dir, mode, domain, img_size=224):
        BaseDataset.__init__(self, dataset_root_dir, mode, domain, img_size)
        
        CLASSES = 'Alarm Clock, Backpack, Batteries, Bed, Bike, Bottle, Bucket, Calculator, Calendar, Candles, Chair, Clipboards, Computer, Couch, Curtains, Desk Lamp, Drill, Eraser, Exit Sign, Fan, File Cabinet, Flipflops, Flowers, Folder, Fork, Glasses, Hammer, Helmet, Kettle, Keyboard, Knives, Lamp Shade, Laptop, Marker, Monitor, Mop, Mouse, Mug, Notebook, Oven, Pan, Paper Clip, Pen, Pencil, Postit Notes, Printer, Push Pin, Radio, Refrigerator, ruler, Scissors, Screwdriver, Shelf, Sink, Sneakers, Soda, Speaker, Spoon, Table, Telephone, Toothbrush, Toys, Trash Can, TV, Webcam'
        self.CLASSES = [c.upper() for c in CLASSES.split(', ')]
        self.source_domains = get_source_domain_names('OfficeHome') 
        
        f = os.path.join(self.root_dir, self.domain)
        for ff in os.listdir(f):
            for fff in os.listdir(os.path.join(f, ff)):
                self.imgs.append(os.path.join(self.domain, ff, fff))
        
        trainset_len = int(len(self.imgs)*0.9)        
        if mode=='test':
            pass
        else:
            trainset, valset = random_split(self.imgs, [trainset_len, len(self.imgs)-trainset_len], 
                               generator=torch.Generator().manual_seed(45)) #fix the split
            if mode=='train':
                self.imgs = trainset
            elif 'val' in mode:
                self.imgs = valset
                
    def __getitem__(self, index):
        imgs = []
        labels = []
        p = self.imgs[index]
        img = self.transforms(Image.open(
            os.path.join(self.root_dir, p)).convert('RGB'))
        tag = p.split('/')[1].replace('_', ' ').upper()
        label = torch.tensor(self.CLASSES.index(tag))     
        
        return img, label
