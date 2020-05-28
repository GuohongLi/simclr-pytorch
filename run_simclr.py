import torch
import os
from simclr import SimCLR
import yaml
from data_aug.dataset_wrapper import DataSetWrapper


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
   
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']['gpu_ids']
    print(os.environ['CUDA_VISIBLE_DEVICES'])

    batch_size = config['train']['train_batch_size_per_gpu']* config['gpu']['gpunum']
    #if torch.cuda.is_available():
    #    batch_size = batch_size * config['gpu']['gpunum']
    dataset = DataSetWrapper(batch_size, **config['dataset'])

    simclr = SimCLR(dataset, config)
    simclr.train()


if __name__ == "__main__":
    main()
