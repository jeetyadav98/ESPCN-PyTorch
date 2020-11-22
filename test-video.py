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
    # parser.add_argument('--image-file', type=str, required=True)
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
    ####################
    video_name= 'data/vid1.mp4'
    videoCapture= cv2.VideoCapture(video_name)

    if (videoCapture.isOpened()== False): 
        print("Error opening video stream or file")

    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    width= (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))// args.scale )*args.scale 
    height= (int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))//args.scale )*args.scale

    output_name = 'data/testvid.avi'
    videoWriter = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'MPEG'), fps, (width, height))

    # image = pil_image.open(args.image_file).convert('RGB')
    success, frame = videoCapture.read()

    while success:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')

        image_width = (image.width // args.scale) * args.scale
        image_height = (image.height // args.scale) * args.scale

        hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
        bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        # bicubic.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

        lr, _ = preprocess(lr, device)
        hr, _ = preprocess(hr, device)
        bc, _ = preprocess(bicubic, device)
        _, ycbcr = preprocess(bicubic, device)

        with torch.no_grad():
            preds = model(lr).clamp(0.0, 1.0)

        psnr1 = calc_psnr(hr, preds)
        print('PSNR SR: {:.2f}'.format(psnr1))

        psnr2 = calc_psnr(hr, bc)
        print('PSNR Bicubic: {:.2f}'.format(psnr2))

        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

        output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
        output = pil_image.fromarray(output)
        # output.save(args.image_file.replace('.', '_espcn_x{}.'.format(args.scale)))
        out_img = cv2.cvtColor(np.asarray(output), cv2.COLOR_RGB2BGR)

        videoWriter.write(out_img)
        # next frame
        success, frame = videoCapture.read()
        
        hr = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('YCbCr')
        image_width = (hr.width // 3) * 3
        image_height = (hr.height // 3) * 3

        hr = hr.resize((image_width, image_height), resample=pil_image.BICUBIC)

        y1, cb1, cr1 = hr.split()
        hr_image = Variable(ToTensor()(y1)).view(1, -1, y1.size[1], y1.size[0])