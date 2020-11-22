import argparse
import os
import copy
import sys

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models import ESPCN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr
import yaml

import train


def printconfig(config_dict):
    print('\nConfiguration parameters-\n')
    for i in config_dict:
        print(i,':')
        for key in config_dict[i]:
            print('  ',key, ':', config_dict[i][key])
    print('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ESPCN')
    parser.add_argument('-pc', '--print-config', dest= 'print_config', default=None, action='store_true', help= 'print configuration file')
    parser.add_argument('--train', dest= 'train', default=None, action='store_true', help= 'train the model')
    parser.add_argument('-c', '--config-file', dest= 'config_file', default='config.yaml', action='store_true', help= 'path to configuration file')
    args = parser.parse_args()

    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    if args.print_config:
        printconfig(config_dict)
        sys.exit()

    if args.train:
        train.training(config_dict['training'])
