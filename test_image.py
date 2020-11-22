import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import ESPCN
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr


def testing_image(dict_image):

    weights_file= dict_image['weights file']
    image_file= dict_image['image file']
    scale= dict_image['scale']

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

    image = pil_image.open(image_file).convert('RGB')

    image_width = (image.width // scale) * scale
    image_height = (image.height // scale) * scale

    hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr.width // scale, hr.height // scale), resample=pil_image.BICUBIC)
    bicubic = lr.resize((lr.width * scale, lr.height * scale), resample=pil_image.BICUBIC)
    bicubic.save(image_file.replace('.', '_bicubic_x{}.'.format(scale)))

    hr, _ = preprocess(hr, device)
    lr, _ = preprocess(lr, device)
    bc, _ = preprocess(bicubic, device)
    _, ycbcr = preprocess(bicubic, device)

    with torch.no_grad():
        espcn_out = model(lr).clamp(0.0, 1.0)

    # Printing PSNR Values
    psnr1 = calc_psnr(hr, espcn_out)
    print('\nPSNR ESPCN  : {:.2f}'.format(psnr1))

    psnr2 = calc_psnr(hr, bc)
    print('PSNR Bicubic: {:.2f}\n'.format(psnr2))

    espcn_out = espcn_out.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([espcn_out, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(image_file.replace('.', '_espcn_x{}.'.format(scale)))