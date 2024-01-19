from utils.dataset_utils import *
from datasets.base import BaseDataset
ImageFile.LOAD_TRUNCATED_IMAGES = True
        
class MNISTM(data.Dataset):
    def __init__(self, root, transform):
        self.CLASSES = [0,1,2,3,4,5,6,7,8,9]
        self.transforms = transform
        self.imgs = []
        self.labels = []
        self.root = root
        
        with open(os.path.join(root, 'mnist_m/mnist_m_train_labels.txt')) as f:
            for l in f.readlines():
                l = l.strip('\n')
                self.imgs.append(os.path.join('mnist_m/mnist_m_train/', l.split(' ')[0]))
                self.labels.append(int(l.split(' ')[1]))
                
        with open(os.path.join(root, 'mnist_m/mnist_m_test_labels.txt')) as f:
            for l in f.readlines():
                l = l.strip('\n')
                self.imgs.append(os.path.join('mnist_m/mnist_m_test/', l.split(' ')[0]))
                self.labels.append(int(l.split(' ')[1]))
        
    def __getitem__(self, index):
        imgs = []
        labels = []
        p = self.imgs[index]
        img = self.transforms(
            Image.open(os.path.join(self.root, p)).convert('RGB'))        
        label = torch.tensor(self.labels[index])     
        
        return img, label       
    
    def __len__(self):
        return len(self.imgs)
                
class Digits(BaseDataset): 
    def __init__(self, dataset_root_dir, mode, domain, img_size=32):
        BaseDataset.__init__(self, dataset_root_dir, mode, domain, img_size)
        
        self.CLASSES = [0,1,2,3,4,5,6,7,8,9]
        if mode == 'test' or mode == 'val':
            transforms=[
                tfs.Resize((img_size, img_size)),
                tfs.ToTensor(),
            ]
        else:
            transforms=[
                tfs.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                #tfs.RandomHorizontalFlip(),
                tfs.ColorJitter(0.3, 0.3, 0.3, 0.3),
                tfs.RandomGrayscale(),
                tfs.ToTensor(),
            ]
        if domain=='USPS' or domain=='MNIST':
            transforms.append(
                tfs.Lambda(lambda x: x.repeat(3, 1, 1)))            
        transforms.append(
            tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        
        self.transforms = tfs.Compose(transforms)
        self.source_domains = get_source_domain_names('Digits') 
        
        if domain=='USPS':
            dataset_train = USPS(root=dataset_root_dir, train=True, transform=self.transforms)
            dataset_test = USPS(root=dataset_root_dir, train=False, transform=self.transforms)
            dataset = ConcatDataset([dataset_train, dataset_test])
        elif domain=='MNIST':
            dataset_train = MNIST(root=dataset_root_dir, train=True, transform=self.transforms)
            dataset_test = MNIST(root=dataset_root_dir, train=False, transform=self.transforms)
            dataset = ConcatDataset([dataset_train, dataset_test])
        elif domain=='SVHN':
            dataset_train = SVHN(root=dataset_root_dir, split='train', transform=self.transforms)
            dataset_test = SVHN(root=dataset_root_dir, split='test', transform=self.transforms)
            dataset = ConcatDataset([dataset_train, dataset_test])
        elif domain=='MNISTM' or domain=='MNIST-M':
            dataset = MNISTM(root=dataset_root_dir, transform=self.transforms)
        
        trainset_len = int(len(dataset)*0.9)        
        if mode=='test':
            self.dataset = dataset
        else:
            trainset, valset = random_split(dataset, [trainset_len, len(dataset)-trainset_len], 
                               generator=torch.Generator().manual_seed(42)) #fix the split
            if mode=='train':
                self.dataset = trainset
            elif 'val' in mode:
                self.dataset = valset
        
    def __getitem__(self, index):  
        img, label = self.dataset[index]
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)
                    
        return img, label
        
    def __len__(self):
        return len(self.dataset)   
