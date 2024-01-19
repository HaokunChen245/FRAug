from utils.dataset_utils import *

class BaseDataset(data.Dataset):
    def __init__(self, dataset_root_dir, mode, domain, img_size):
        self.root_dir = dataset_root_dir
        self.imgs = []
        self.labels = []
        self.domain = domain
        self.mode = mode
        if mode == 'test' or mode == 'val':
            self.transforms = tfs.Compose([tfs.Resize((img_size, img_size)),
                                           tfs.ToTensor(),
                                           tfs.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
        elif mode == 'train':
            self.transforms = tfs.Compose([
                tfs.RandomResizedCrop(img_size, scale=(0.7, 1.0)), 
                tfs.RandomHorizontalFlip(),
                tfs.ColorJitter(0.3, 0.3, 0.3, 0.3),
                tfs.RandomGrayscale(),
                tfs.ToTensor(),
                tfs.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
    def __len__(self):
        return len(self.imgs)

    def collate_fn(self, batch):
        imgs, labels = zip(*batch)
        imgs = torch.stack(imgs).cuda()
        labels = torch.stack(labels).cuda()
        return imgs, labels

    
    def get_images_with_cls(self, label, max_num=256):
        imgs = []
        labels = []
        for i in range(len(self.imgs)):            
            p = self.imgs[i]
            if self.CLASSES[label] not in p: continue
            temp = self.__getitem__(i)
            imgs.append(temp[0])
            labels.append(temp[1])
            if len(imgs)>max_num: break
        imgs = torch.stack(imgs, 0).cuda()
        labels = torch.stack(labels, 0).cuda()
        return imgs, labels
