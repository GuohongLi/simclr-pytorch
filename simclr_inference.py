import torch
import torch.nn as nn
from models.resnet_simclr import ResNetSimCLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import os
import yaml
from data_aug.dataset_wrapper_inference import DataSetWrapper

import os
import shutil
import sys
import time

import numpy as np

torch.manual_seed(0)

class SimCLR(object):

    def __init__(self, dataset, config):
        self.config = config
        self.device = 'cuda'
        self.dataset = dataset
        self.batch_size = config['train']['train_batch_size_per_gpu']
        if self.device == 'cuda':
            self.batch_size = self.batch_size * config['gpu']['gpunum']
        self.outfile = config['result_file']

    def _step(self, model, xis):

        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)

        return ris, zis

    def predict(self):
        #Data
        valid_loader = self.dataset.get_data_loaders()

        #Model
        model = ResNetSimCLR(**self.config["model"])
        if self.device == 'cuda':
            model = nn.DataParallel(model, device_ids=[i for i in range(self.config['gpu']['gpunum'])])
        model = model.cuda()
        #print(model)
        model = self._load_pre_trained_weights(model)
        
        # validate the model if requested
        self._validate(model, valid_loader)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs', self.config['train']['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):

        # validation steps
        with torch.no_grad():
            model.eval()
            with open(self.outfile,'w') as fpout:
                for counter, (xis, path) in enumerate(valid_loader):
                    xis = xis.cuda()
                    #h,x
                    ris, zis = self._step(model, xis)
                    #
                    ris_out = ris.data.cpu().numpy()
                    zis_out = zis.data.cpu().numpy()
                    print(ris_out.shape, zis_out.shape)
                    for i in range(ris_out.shape[0]):
                        fpout.write("%s\t%s\t%s\n" % (path[i], ' '.join([str(x) for x in ris_out[i]]), ' '.join([str(x) for x in zis_out[i]])))

        return

def main():
    config = yaml.load(open("config_inference.yaml", "r"), Loader=yaml.FullLoader)
   
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']['gpu_ids']
    print(os.environ['CUDA_VISIBLE_DEVICES'])

    batch_size = config['train']['train_batch_size_per_gpu']* config['gpu']['gpunum']
    dataset = DataSetWrapper(batch_size, **config['dataset'])

    simclr = SimCLR(dataset, config)
    simclr.predict()


if __name__ == "__main__":
    main()
