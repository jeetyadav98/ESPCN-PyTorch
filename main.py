import argparse
import os
import copy
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import yaml

from math import sqrt

from source.models import ESPCN
from source.datasets import TrainDataset, EvalDataset
from source.utils import AverageMeter, calc_psnr

from source.train import training
from source.test_image import testing_image
from source.test_video import testing_video


def visualize_filters(weights_file):
    device = torch.device('cpu')
    model = ESPCN(scale_factor=3)

    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model_weights= []   # To store weights
    conv_layers= []     # To store the conv2d layers

    model_children= list(model.children())
    counter = 0 
    for i in range(len(model_children)):
        for j in range(len(model_children[i])):
            child= model_children[i][j]
            if type(child) == nn.Conv2d:
                counter += 1
                model_weights.append(child.weight)
                conv_layers.append(child)

    out_path= 'data/visualize_filters'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    sizes= [(8,8), (4,8), (3,3)]
    k_sizes= [5,3,3]
    plt.figure(figsize=(20, 17))
    for n in range(len(model_weights)):
        for i, filter in enumerate(model_weights[n]):
            plt.subplot(sizes[n][0], sizes[n][1], i+1)
            plt.imshow(filter[0, :, :].detach(), cmap='gray')
            plt.axis('off')
        plt.suptitle('Convolutional Layer ' + str(n+1) + ': Filter visualization', fontsize=40)
        plt.savefig('data/visualize_filters/filter'+str(n+1)+'.png')
        plt.clf()
    print('Filter images saved to data/visualize_filters')
    sys.exit()

def printconfig(config_dict):
    print('\nConfiguration parameters-\n')
    for i in config_dict:
        print(i,':')
        for key in config_dict[i]:
            print('  ',key, ':', config_dict[i][key])
        print()
    sys.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ESPCN')
    parser.add_argument('-pc', '--print-config', dest= 'print_config', default=None, action='store_true', help= 'print configuration file')
    parser.add_argument('-c', '--config-file', dest= 'config_file', default='config.yaml', action='store_true', help= 'path to configuration file')
    parser.add_argument('-t', '--train', dest= 'train', default=None, action='store_true', help= 'train the model')
    parser.add_argument('-im','--test-image', dest= 'test_image', default=None, action='store_true', help= 'test an image using ESPCN')
    parser.add_argument('-vi','--test-video', dest= 'test_video', default=None, action='store_true', help= 'test a video using ESPCN')
    parser.add_argument('-b','--batch', dest= 'batch_mode', default=None, action='store_true', help= 'process entire directory of images/videos')
    parser.add_argument('-v','--vis-filters', dest= 'vis_filters', default=None, action='store_true', help= 'visualize filters of each conv layer')
    parser.add_argument('-p','--plot', dest= 'plot', default=None, action='store_true', help= 'plot psnr for image batches or videos')
    args = parser.parse_args()
    
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    printconfig(config_dict) if args.print_config else None
    visualize_filters(config_dict['visualize filters']['weights file']) if args.vis_filters else None

    if not (args.train or args.test_image or args.test_video):
        print('Please provide argument to train/test')
        sys.exit()

    training(config_dict['training']) if args.train else None
    
    testing_image(config_dict['test image'], args.batch_mode, args.plot) if args.test_image else None

    testing_video(config_dict['test video'], args.batch_mode, args.plot) if args.test_video else None