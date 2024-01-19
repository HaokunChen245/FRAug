from utils.dataset_utils import *
from datasets.base import BaseDataset
ImageFile.LOAD_TRUNCATED_IMAGES = True
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader

class rxrx1(BaseDataset):   
    def __init__(self, dataset_root_dir, mode, domain, client_id=None, img_size=256):
        BaseDataset.__init__(self, dataset_root_dir, mode, domain, img_size)
        dataset = get_dataset(dataset="rxrx1", root_dir=dataset_root_dir)
        self.selected_classes = [13, 51, 54, 61, 65, 88, 161, 178, 189, 191, 198, 209, 228, 255, 285, 318, 326, 407, 440, 447, 451, 457, 476, 501, 541, 563, 569, 689, 696, 704, 735, 775, 778, 859, 864, 865, 919, 940, 1034, 1098, 1110, 1112, 1120, 1122, 1126, 1131, 1135, 1138]
        if not client_id:
            client_id = domain * 4
        self.client_id = client_id
        self.mode = mode
        self.length = 0
        self.dataset_root_dir = dataset_root_dir
        PATH = os.path.join(dataset_root_dir, 'rxrx1_FL')
        for d in os.listdir(PATH):
            if f'client_{client_id}_' in d and mode in d and 'img' in d:
                self.length += 1
                
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        PATH = os.path.join(self.dataset_root_dir, 'rxrx1_FL')     
        img_id = str(index).zfill(4)        
        img = torch.load(os.path.join(PATH, f'client_{self.client_id}_img_{img_id}_{self.mode}.pt'))
        label = torch.load(os.path.join(PATH, f'client_{self.client_id}_gt_{img_id}_{self.mode}.pt'))
        return img.cuda(), torch.LongTensor([label]).squeeze().cuda()
