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

from source.models import ESPCN


def printconfig(config_dict):
    """ Print configuration dictionary to console

    Configuration values in yaml file (default= config.yaml) passed as a dictionary and printed to the console for convenient inspection. Command line arg is (-pc, --print-config). Terminates execution after printing.

    :param config_dict: Nested dictionary of configuration values for using ESPCN
    :return: None

    """

    print('\nConfiguration parameters-\n')
    for i in config_dict:
        print(i,':')
        for key in config_dict[i]:
            print('  ',key, ':', config_dict[i][key])
        print()
    sys.exit()


def visualize_filters(dict_vis):
    """ Visualize and save filters of all the convolutional layers

    Plot filters of the conv layers using matplotlib. Weights are loaded, after which the function extracts the weights to 'model_weights'. Filters visuals are plotted for each layer and saved in data/visualize_filters. Command line arg is (-f, --filter-vis). Terminates execution after plotting and saving.

    :param dict_vis: dictionary containing scale value and path to weights file
    :return: None

    """

    weights_file= dict_vis['weights file']
    scale= dict_vis['scale']
    
    device = torch.device('cpu')
    model = ESPCN(scale_factor=scale)

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


def is_image_file(filename):
    """ Check if file is an image
    :param filename: file name string
    :return: Boolean toggle

    """

    return any(filename.endswith(extension) for extension in ['.bmp', '.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG'])


def is_video_file(filename):
    """ Check if file is a video
    :param filename: file name string
    :return: Boolean toggle

    """

    return any(filename.endswith(extension) for extension in ['.mp4', '.avi', '.mpg', '.mkv', '.wmv', '.flv'])


def convert_rgb_to_y(img, dim_order='hwc'):
    """ Get Y(CbCr) value from RGB image (standard conversion)

    :param img: input image array in RGB form
    :return: array of Y values

    """

    if dim_order == 'hwc':
        return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    else:
        return 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.


def convert_rgb_to_ycbcr(img, dim_order='hwc'):
    """ Convert to YCbCr from RGB (standard conversion)

    :param img: input image array in RGB form
    :return: out image array in YCbCr form

    """

    if dim_order == 'hwc':
        y = 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
        cb = 128. + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 * img[..., 2]) / 256.
        cr = 128. + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[..., 2]) / 256.
    else:
        y = 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.
        cb = 128. + (-37.945 * img[0] - 74.494 * img[1] + 112.439 * img[2]) / 256.
        cr = 128. + (112.439 * img[0] - 94.154 * img[1] - 18.285 * img[2]) / 256.
    return np.array([y, cb, cr]).transpose([1, 2, 0])


def convert_ycbcr_to_rgb(img, dim_order='hwc'):
    """ Convert to RGB from YCbCr (standard conversion)

    :param img: input image array in YCbCr form
    :return: out image array in RGB form

    """

    if dim_order == 'hwc':
        r = 298.082 * img[..., 0] / 256. + 408.583 * img[..., 2] / 256. - 222.921
        g = 298.082 * img[..., 0] / 256. - 100.291 * img[..., 1] / 256. - 208.120 * img[..., 2] / 256. + 135.576
        b = 298.082 * img[..., 0] / 256. + 516.412 * img[..., 1] / 256. - 276.836
    else:
        r = 298.082 * img[0] / 256. + 408.583 * img[2] / 256. - 222.921
        g = 298.082 * img[0] / 256. - 100.291 * img[1] / 256. - 208.120 * img[2] / 256. + 135.576
        b = 298.082 * img[0] / 256. + 516.412 * img[1] / 256. - 276.836
    return np.array([r, g, b]).transpose([1, 2, 0])


def preprocess(img, device):
    """ Process image into torch tensor

    :param img: input image array in RGB form
    :return: tensor, array

    """
    img = np.array(img).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(img)
    x = ycbcr[..., 0]
    x /= 255.
    x = torch.from_numpy(x).to(device)
    x = x.unsqueeze(0).unsqueeze(0)
    return x, ycbcr


def calc_psnr(img1, img2):
    """ PSNR calculator

    :param img1: true/estimated image
    :param img2: estimated/true image
    :return: PSNR value

    """

    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


class AverageMeter(object):
    """ Simple object to keep track of a parameter average over time

    Object initialized to zero, and stores the average, count, sum and last value variables. Used in training to track best average PSNR.

    """

    def __init__(self):
        """ Constructor
        """
        self.reset()

    def reset(self):
        """ Reset to zero
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update and compute val, sum, count, avg
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count