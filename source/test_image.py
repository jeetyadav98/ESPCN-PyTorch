import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import matplotlib.pyplot as plt

from source.models import ESPCN
from source.utils import convert_ycbcr_to_rgb, preprocess, calc_psnr, is_image_file
from os import listdir
import os


def testing_image(dict_image, batch_mode, psnr_plot):

    weights_file= dict_image['weights file']
    scale= dict_image['scale']
    image_dir= dict_image['image dir']
    
    image_file= dict_image['image file'] if not batch_mode else None
        
    cudnn.benchmark = True
    
    # device = torch.device('cuda:0')
    device = torch.device('cpu')

    model = ESPCN(scale_factor=scale).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    ###########################
    if not batch_mode:
        images= [image_file]
    else:
        images = [x for x in listdir(image_dir) if is_image_file(x)]
    
    out_path = image_dir + '_x{}/'.format(scale)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    espcn_psnr= np.zeros(len(images))
    bc_psnr= np.zeros(len(images))
    count=0

    for image_name in images:
        image = pil_image.open(image_dir + '/'+ image_name).convert('RGB')

        image_width = (image.width // scale) * scale
        image_height = (image.height // scale) * scale

        hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // scale, hr.height // scale), resample=pil_image.BICUBIC)
        bicubic = lr.resize((lr.width * scale, lr.height * scale), resample=pil_image.BICUBIC)

        hr, _ = preprocess(hr, device)
        lr, _ = preprocess(lr, device)
        bc, _ = preprocess(bicubic, device)
        _, ycbcr = preprocess(bicubic, device)

        with torch.no_grad():
            espcn_out = model(lr).clamp(0.0, 1.0)

        # Printing PSNR Values
        print(image_name)
        espcn_psnr[count] = calc_psnr(hr, espcn_out)
        print('PSNR ESPCN  : {:.2f}'.format(espcn_psnr[count]))

        bc_psnr[count] = calc_psnr(hr, bc)
        print('PSNR Bicubic: {:.2f}\n'.format(bc_psnr[count]))
        count+=1

        ##########
        espcn_out = espcn_out.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

        output = np.array([espcn_out, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
        output = pil_image.fromarray(output)

        # Saving bicubic, espcn outputs
        bc_out= image_name.replace('.', '_bicubic_x{}.'.format(scale))
        espcn_out= image_name.replace('.', '_espcn_x{}.'.format(scale))

        bicubic.save(out_path+bc_out)
        output.save(out_path+espcn_out)

    if psnr_plot:
        x= np.arange(1, len(espcn_psnr)+1, dtype=int)

        plt.figure(figsize=(10, 7))
        plt.plot(x, espcn_psnr)
        plt.plot(x,bc_psnr)
        plt.title('PSNR versus image number')
        plt.legend(['ESPCN  (Average: '+ '{:.2f}'.format(np.mean(espcn_psnr)) + ' dB)', 'Bicubic (Average: '+ '{:.2f}'.format(np.mean(bc_psnr)) + ' dB)'])
        plt.xlabel('Image number')
        plt.ylabel('PSNR (dB)')
        plt.axis([0, len(x), 0, 60])

        plt.savefig(out_path + 'psnr_batchplot.png')
        plt.show()