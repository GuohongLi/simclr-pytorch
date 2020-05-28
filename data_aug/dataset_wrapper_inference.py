import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

np.random.seed(0)

# Dataset
class FP_Dataset(Dataset):
    
    def __init__(self, root, dataset, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform

        with open(txt) as f:
            for line in f:
                tokens = line.strip().split('\t')
                if len(tokens) < 1:
                    continue
                imgfile_path = '%s/%s/%s'%(root, dataset, tokens[0])
                self.img_path.append(imgfile_path)
        
    def __len__(self):
        return len(self.img_path)
        
    def __getitem__(self, index):
        path = self.img_path[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, path

class DataSetWrapper(object):

    def __init__(self, batch_size, num_workers , input_shape, data_root, dataset_name):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_shape = eval(input_shape)
        self.data_root = data_root
        self.dataset_name = dataset_name

    def get_data_loaders(self):
        data_augment = self._get_simclr_pipeline_transform()
        valid_loader = self.load_data(self.data_root, self.dataset_name, 'test', self.batch_size, self.num_workers, shuffle=False, 
                                     transform=data_augment, sampler=None)
            
        return valid_loader

    def load_data(self, data_root, dataset, phase, batch_size, num_workers=4, shuffle=True, transform=None, sampler=None):
        txt = '%s/%s/%s.txt'%(data_root, dataset, phase)
        print('Loading data from %s' % (txt))
        print('Use data transformation:', transform)
        set_ = FP_Dataset(data_root, dataset, txt, transform)
    
        if sampler and phase == 'train':
            print('Using sampler.')
            return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False,
                               sampler=sampler,num_workers=num_workers,drop_last=True)
        elif phase == 'test':
            #test
            print('No sampler.')
            print('Shuffle is %s.' % (shuffle))
            return DataLoader(dataset=set_, batch_size=batch_size,
                              shuffle=shuffle, num_workers=num_workers,drop_last=False)
        else:
            print('No sampler.')
            print('Shuffle is %s.' % (shuffle))
            return DataLoader(dataset=set_, batch_size=batch_size,
                              shuffle=shuffle, num_workers=num_workers,drop_last=True)

    def _get_simclr_pipeline_transform(self):
        data_transforms = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(size=self.input_shape[0]),
                                              transforms.ToTensor()])
        return data_transforms

class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj
