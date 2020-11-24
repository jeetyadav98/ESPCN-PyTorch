import argparse
import sys
import yaml

from source.utils import visualize_filters, printconfig
from source.train import training
from source.test_image import testing_image
from source.test_video import testing_video


if __name__ == "__main__":
    """ Main script - runs everything from here

    main.py is the primary command console for all operations. The argparser takes command line arguments to run different modes of this project. Run "python3 main.py --help" for automatic description of the arguments.
    
    The arguments only indicate the mode of operation, configuration values are instead taken from a .yaml file (default = config.yaml). Each mode has its own dictionary in the .yaml file.

    All functions/classes present in files in the source/ folder. 

    """

    # Initialize argparser
    parser = argparse.ArgumentParser(description='ESPCN')
    parser.add_argument('-pc', '--print-config', dest= 'print_config', default=None, action='store_true', help= 'print configuration file')
    parser.add_argument('-c', '--config-file', dest= 'config_file', default='config.yaml', action='store_true', help= 'path to configuration file')
    parser.add_argument('-t', '--train', dest= 'train', default=None, action='store_true', help= 'train the model')
    parser.add_argument('-im','--test-image', dest= 'test_image', default=None, action='store_true', help= 'test an image using ESPCN')
    parser.add_argument('-vi','--test-video', dest= 'test_video', default=None, action='store_true', help= 'test a video using ESPCN')
    parser.add_argument('-b','--batch', dest= 'batch_mode', default=None, action='store_true', help= 'process entire directory of images/videos')
    parser.add_argument('-f','--filters-vis', dest= 'filters_vis', default=None, action='store_true', help= 'visualize filters of each conv layer')
    parser.add_argument('-p','--plot', dest= 'plot', default=None, action='store_true', help= 'plot psnr for image batches or videos')
    args = parser.parse_args()
    
    # Load configuration dictionary
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    # if conditions on command line arguments decide operation(s)
    printconfig(config_dict) if args.print_config else None
    visualize_filters(config_dict['visualize filters']) if args.filters_vis else None

    if not (args.train or args.test_image or args.test_video):
        print('Please provide argument to train/test')
        sys.exit()

    training(config_dict['training']) if args.train else None
    
    testing_image(config_dict['test image'], args.batch_mode, args.plot) if args.test_image else None

    testing_video(config_dict['test video'], args.batch_mode, args.plot) if args.test_video else None