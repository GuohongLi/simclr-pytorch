from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np

# Dataset
class FP_Dataset(Dataset):
    
    def __init__(self, root, dataset, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.transform_label = transforms.Compose([transforms.ToTensor()])

        with open(txt) as f:
            for line in f:
                tokens = line.strip().split('\t')
                if len(tokens) != 2:
                    continue
                imgfile_path = '%s/%s/%s'%(root, dataset, tokens[0])
                gtfile_path = '%s/%s/%s'%(root, dataset, tokens[1])
                self.img_path.append(imgfile_path)
                #self.labels.append(gtfile_path)
        
    def __len__(self):
        return len(self.img_path)
        
    def __getitem__(self, index):
        path = self.img_path[index]
        '''
        label_path = self.labels[index]
        with open(label_path, 'r') as f:
           label_str = f.read().strip().split('\t')[0].strip().split(' ')
           label = np.array([10*float(x) for x in label_str], dtype=np.float32)
        '''
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        #return sample, label
        return sample, 0

# Load datasets
def load_data(data_root, dataset, phase, batch_size, num_workers=4, shuffle=True, transform=None, sampler=None):
    txt = '%s/%s/%s.txt'%(data_root, dataset, phase)
    print('Loading data from %s' % (txt))
    print('Use data transformation:', transform)
    set_ = FP_Dataset(data_root, dataset, txt, transform)

    if sampler and phase == 'train':
        print('Using sampler.')
        return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False,
                           sampler=sampler,num_workers=num_workers,drop_last=True)
    else:
        print('No sampler.')
        print('Shuffle is %s.' % (shuffle))
        return DataLoader(dataset=set_, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers,drop_last=True)
        
if __name__ == "__main__":
    data_root = './data'
    dataset = 'select_base_200e'
    phase = 'train'
    txt = '%s/%s/%s.txt'%(data_root, dataset, phase)
    d_loader = load_data(data_root, dataset, phase, batch_size=1, sampler_dic=None, num_workers=4, shuffle=True)
    num = 10
    for i, (img, label, imgpath) in enumerate(d_loader):
        if i>=num:
            break
        print(imgpath[0])
        print(img.shape)
        print(label.shape)
        print(label[0])
        print('----')
        #img, label = img.cuda(), label.cuda()
