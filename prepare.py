import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from source.utils import convert_rgb_to_y


def train(args):
    """ Write and save train dataset

    :param args: arparser input
    :returns: None

    """

    h5_file = h5py.File(args.output_path, 'w')
    lr_patches = []
    hr_patches = []

    for image_path in sorted(glob.glob('{}/*'.format(args.images_dir))):
        hr = pil_image.open(image_path).convert('RGB')
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

        for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
            for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
                lr_patches.append(lr[i:i + args.patch_size, j:j + args.patch_size])
                hr_patches.append(hr[i * args.scale:i * args.scale + args.patch_size * args.scale, j * args.scale:j * args.scale + args.patch_size * args.scale])

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)

    h5_file.close()


def val(args):
    """ Write and save eval dataset

    :param args: arparser input
    :returns: None

    """
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir)))):
        hr = pil_image.open(image_path).convert('RGB')
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

    h5_file.close()


if __name__ == '__main__':
    """ Prepares data sets for training in .h5 format

    This script converts multi file data sets in a single .h5 file for training. Train and eval datasets are both supposed to be in .h5 format. The argparser takes command line arguments to prepare the datasets. As compared to main, this takes the config values from the command line itself, due to the one-time nature of this operation. 

    :returns: None

    """

    # Initialize argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, required=True, help= 'path to images directory')
    parser.add_argument('--output-path', type=str, required=True, help= 'path to output file')
    parser.add_argument('--scale', type=int, default=3, help= 'super resolution scale factor')
    parser.add_argument('--patch-size', type=int, default=17, help= 'patch size')
    parser.add_argument('--stride', type=int, default=13, help= 'stride')
    parser.add_argument('--eval', action='store_true', help= 'eval dataset')
    args = parser.parse_args()

    val(args) if args.eval else train(args)