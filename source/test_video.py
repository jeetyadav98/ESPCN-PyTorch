import argparse

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from source.models import ESPCN
from source.utils import convert_ycbcr_to_rgb, preprocess, calc_psnr, is_video_file
import os
from os import listdir

def testing_video(dict_video, batch_mode, psnr_plot):
    """ Process video(s) through ESPCN

    This function processes a video (file mode), or videos (batch mode) through ESPCN. Videos are processed frame by frame. These are first downscaled to lower resolution images. These are upscaled and saved using both (a) Bicubic interpolation and (b) ESPCN. The former is used for comparison. The function can optionally plot the PSNR values of each frame, corresponding to both Bicubic and ESPCN outputs for a video.

    :param dict_video: dictionary for configuration values (scale, location of weights file...)
    :param batch_mode: Boolean toggle; will process all images in 'image_dir' if True
    :param psnr_plot: Boolean toggle; will plot and save Bicubic and ESPCN PSNR for a batch of images if True
    :return: None

    """

    # Initialize configuration values from input dictionary
    weights_file= dict_video['weights file']
    scale= dict_video['scale']
    video_dir= dict_video['video dir']

    video_file= dict_video['video file'] if not batch_mode else None

    # Initialize model in eval, load weights
    cudnn.benchmark = True    
    device = torch.device('cpu')    # OR device = torch.device('cuda:0')

    model = ESPCN(scale_factor=scale).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    # Identify input/output paths
    if not batch_mode:
        videos= [video_file]
    else:
        videos = [x for x in listdir(video_dir) if is_video_file(x)]
    
    out_path = video_dir + '_x{}/'.format(scale)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Iterate over all videos (processed frame by frame)
    for video_name in videos:
    
        videoCapture= cv2.VideoCapture(video_dir + '/' +video_name)

        if (videoCapture.isOpened()== False): 
            print("Error opening video stream or file")

        # Get video metadata
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        width= (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))// scale )*scale 
        height= (int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))//scale )*scale
        frame_count = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))

        # Constructing videowriter objects for bicubic and espcn outputs

        espcn_out_name = video_name.replace('.','_espcn_x{}.'.format(scale))
        espcn_videoWriter = cv2.VideoWriter(out_path + espcn_out_name, cv2.VideoWriter_fourcc(*'MPEG'), fps, (width, height))

        bic_out_name = video_name.replace('.','_bicubic_x{}.'.format(scale))
        bic_videoWriter = cv2.VideoWriter(out_path + bic_out_name, cv2.VideoWriter_fourcc(*'MPEG'), fps, (width, height))

        # Read first frame from video, initialize arrays for PSNR values (per video)
        success, frame = videoCapture.read()
        espcn_psnr= np.zeros(frame_count)
        bc_psnr= np.zeros(frame_count)
        count= 0

        # Iterate over all
        while success:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')

            image_width = (image.width // scale) * scale
            image_height = (image.height // scale) * scale

            hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
            lr = hr.resize((hr.width // scale, hr.height // scale), resample=pil_image.BICUBIC)
            bicubic = lr.resize((lr.width * scale, lr.height * scale), resample=pil_image.BICUBIC)

            # Preprocess to tensors
            hr, _ = preprocess(hr, device)
            lr, _ = preprocess(lr, device)
            bc, _ = preprocess(bicubic, device)
            _, ycbcr = preprocess(bicubic, device)

            with torch.no_grad():
                espcn_out = model(lr).clamp(0.0, 1.0)

            # PSNR Values
            espcn_psnr[count] = calc_psnr(hr, espcn_out)
            bc_psnr[count] = calc_psnr(hr, bc)
            count+=1
            
            # Convert back to image (frame)
            espcn_out = espcn_out.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

            output = np.array([espcn_out, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
            output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
            output = pil_image.fromarray(output)
            
            # Saving frames to bicubic, espcn videos
            espcn_img = cv2.cvtColor(np.asarray(output), cv2.COLOR_RGB2BGR)
            bic_img = cv2.cvtColor(np.asarray(bicubic), cv2.COLOR_RGB2BGR)

            espcn_videoWriter.write(espcn_img)
            bic_videoWriter.write(bic_img)
            
            # Next frame
            success, frame = videoCapture.read()
        
        # Conditionally plot PSNR and save them in the output directory (one plot per video)
        if psnr_plot:
            x= np.arange(1, len(espcn_psnr)+1, dtype=int)
        
            plt.figure(figsize=(10, 7))
            plt.plot(x, espcn_psnr)
            plt.plot(x,bc_psnr)
            plt.title(video_name + ': PSNR versus frame number')
            plt.legend(['ESPCN  (Average: '+ '{:.2f}'.format(np.mean(espcn_psnr)) + ' dB)', 'Bicubic (Average: '+ '{:.2f}'.format(np.mean(bc_psnr)) + ' dB)'])
            plt.xlabel('Frame number')
            plt.ylabel('PSNR (dB)')
            plt.axis([0, len(x), 0, 60])

            plt.savefig(out_path + video_name.split('.')[0] + '_psnr_plot.png')
    
    # Show all plots together 
    plt.show() if psnr_plot else None