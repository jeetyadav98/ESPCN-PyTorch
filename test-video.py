import argparse

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor

from models import ESPCN
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--video-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = ESPCN(scale_factor=args.scale).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    ################
    video_name= args.video_file
    videoCapture= cv2.VideoCapture(video_name)

    if (videoCapture.isOpened()== False): 
        print("Error opening video stream or file")

    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    width= (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))// args.scale )*args.scale 
    height= (int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))//args.scale )*args.scale

    # Constructing videowriter objects for bicubic and espcn outputs
    espcn_out_name = args.video_file.replace('.','_espcn_x{}.'.format(args.scale))
    espcn_videoWriter = cv2.VideoWriter(espcn_out_name, cv2.VideoWriter_fourcc(*'MPEG'), fps, (width, height))

    bic_out_name = args.video_file.replace('.','_bicubic_x{}.'.format(args.scale))
    bic_videoWriter = cv2.VideoWriter(bic_out_name, cv2.VideoWriter_fourcc(*'MPEG'), fps, (width, height))

    # Read frame from video
    success, frame = videoCapture.read()

    while success:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')

        image_width = (image.width // args.scale) * args.scale
        image_height = (image.height // args.scale) * args.scale

        hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
        bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)

        hr, _ = preprocess(hr, device)
        lr, _ = preprocess(lr, device)
        bc, _ = preprocess(bicubic, device)
        _, ycbcr = preprocess(bicubic, device)

        with torch.no_grad():
            espcn_out = model(lr).clamp(0.0, 1.0)

        # Printing PSNR Values
        psnr1 = calc_psnr(hr, espcn_out)
        print('PSNR ESPCN  : {:.2f}'.format(psnr1))

        psnr2 = calc_psnr(hr, bc)
        print('PSNR Bicubic: {:.2f}\n'.format(psnr2))

        espcn_out = espcn_out.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

        output = np.array([espcn_out, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
        output = pil_image.fromarray(output)
        
        # Saving frames to bicubic, espcn videos
        espcn_img = cv2.cvtColor(np.asarray(output), cv2.COLOR_RGB2BGR)
        bic_img = cv2.cvtColor(np.asarray(bicubic), cv2.COLOR_RGB2BGR)

        espcn_videoWriter.write(espcn_img)
        bic_videoWriter.write(bic_img)
        
        # next frame
        success, frame = videoCapture.read()