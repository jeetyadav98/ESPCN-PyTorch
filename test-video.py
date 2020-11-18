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
    width= int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))*args.scale 
    height= int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))*args.scale

    output_name = 'data/testvid.avi'
    videoWriter = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'MPEG'), fps, (width, height))

    # image = pil_image.open(args.image_file).convert('RGB')
    success, frame = videoCapture.read()

    while success:
        # image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')

        # image_width = (image.width // args.scale) * args.scale
        # image_height = (image.height // args.scale) * args.scale

        # hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        # lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
        # bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        # # bicubic.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

        # lr, _ = preprocess(lr, device)
        # hr, _ = preprocess(hr, device)
        # _, ycbcr = preprocess(bicubic, device)

        # with torch.no_grad():
        #     preds = model(lr).clamp(0.0, 1.0)

        # psnr1 = calc_psnr(hr, preds)
        # print('PSNR SR: {:.2f}'.format(psnr1))

        # # psnr2 = calc_psnr(hr, bicubic)
        # # print('PSNR Bicubic: {:.2f}'.format(psnr2))

        # preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

        # output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        # output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
        # output = pil_image.fromarray(output)
        # # output.save(args.image_file.replace('.', '_espcn_x{}.'.format(args.scale)))
        # out_img = cv2.cvtColor(np.asarray(output), cv2.COLOR_RGB2BGR)

        # videoWriter.write(out_img)
        #     # next frame
        # success, frame = videoCapture.read()
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('YCbCr')
        y, cb, cr = img.split()
        image = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
        if torch.cuda.is_available():
            image = image.cuda()

        out = model(image)
        out = out.cpu()
        out_img_y = out.data[0].numpy()
        out_img_y *= 255.0
        out_img_y = out_img_y.clip(0, 255)
        out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
        out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
        out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
        out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
        out_img = cv2.cvtColor(np.asarray(out_img), cv2.COLOR_RGB2BGR)

        # if IS_REAL_TIME:
        #     cv2.imshow('LR Video ', frame)
        #     cv2.imshow('SR Video ', out_img)
        #     cv2.waitKey(DELAY_TIME)
        # else:
            # save video
        videoWriter.write(out_img)
        # next frame
        success, frame = videoCapture.read()
