import torch
import torch.nn as nn
from models.resnet_simclr import ResNetSimCLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from loss.nt_xent import NTXentLoss
from util import AverageMeter

from lars_opt import LARS 

import os
import shutil
import sys
import time

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

import numpy as np

torch.manual_seed(0)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


class SimCLR(object):

    def __init__(self, dataset, config):
        self.config = config
        #self.device = self._get_device()
        self.device = 'cuda'
        self.writer = SummaryWriter()
        self.dataset = dataset
        self.batch_size = config['train']['train_batch_size_per_gpu']
        if self.device == 'cuda':
            self.batch_size = self.batch_size * config['gpu']['gpunum']
        self.nt_xent_criterion = NTXentLoss(self.device, self.batch_size, **config['loss'])

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _step(self, model, xis, xjs, n_iter):

        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs)
        return loss

    def train(self):
        #Data
        train_loader, valid_loader = self.dataset.get_data_loaders()

        #Model
        model = ResNetSimCLR(**self.config["model"])
        if self.device == 'cuda':
            model = nn.DataParallel(model, device_ids=[i for i in range(self.config['gpu']['gpunum'])])
        #model = model.to(self.device)
        model = model.cuda()
        print(model)
        model = self._load_pre_trained_weights(model)
        
        each_epoch_steps = len(train_loader)
        total_steps = each_epoch_steps * self.config['train']['epochs'] 
        warmup_steps = each_epoch_steps * self.config['train']['warmup_epochs']
        scaled_lr = eval(self.config['train']['lr']) * self.batch_size / 256.

        optimizer = torch.optim.Adam(
                     model.parameters(), 
                     scaled_lr, 
                     weight_decay=eval(self.config['train']['weight_decay']))
       
        '''
        optimizer = LARS(params=model.parameters(),
                     lr=eval(self.config['train']['lr']),
                     momentum=self.config['train']['momentum'],
                     weight_decay=eval(self.config['train']['weight_decay'],
                     eta=0.001,
                     max_epoch=self.config['train']['epochs'])
        '''

        # scheduler during warmup stage
        lambda1 = lambda epoch:epoch*1.0 / int(warmup_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

        if apex_support and self.config['train']['fp16_precision']:
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        lr = eval(self.config['train']['lr']) 

        end = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        
        for epoch_counter in range(self.config['train']['epochs']):
            model.train()
            for i, ((xis, xjs), _) in enumerate(train_loader):
                data_time.update(time.time() - end)
                optimizer.zero_grad()

                xis = xis.cuda()
                xjs = xjs.cuda()

                loss = self._step(model, xis, xjs, n_iter)

                #print("Loss: ",loss.data.cpu())
                losses.update(loss.item(), 2 * xis.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                print('Epoch: [{epoch}][{step}/{each_epoch_steps}] Loss {loss.val:.4f} Avg Loss {loss.avg:.4f} DataTime {datatime.val:.4f} BatchTime {batchtime.val:.4f} LR {lr})'.format(epoch=epoch_counter, step=i, each_epoch_steps=each_epoch_steps, loss=losses, datatime=data_time, batchtime=batch_time, lr=lr))

                if n_iter % self.config['train']['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)

                if apex_support and self.config['train']['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

                #adjust lr
                if n_iter == warmup_steps:
                    # scheduler after warmup stage
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps-warmup_steps, eta_min=0, last_epoch=-1)
                scheduler.step()
                lr = scheduler.get_lr()[0]
                self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)
                sys.stdout.flush()

            # validate the model if requested
            if epoch_counter % self.config['train']['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

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

            valid_loss = 0.0
            for counter, ((xis, xjs), _) in enumerate(valid_loader):
                #xis = xis.to(self.device)
                #xjs = xjs.to(self.device)
                xis = xis.cuda()
                xjs = xjs.cuda()

                loss = self._step(model, xis, xjs, counter)
                valid_loss += loss.item()

            valid_loss /= counter
        model.train()
        return valid_loss
