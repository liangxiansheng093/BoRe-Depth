import configargparse


def get_opts():
    parser = configargparse.ArgumentParser()

    # configure file
    parser.add_argument('--config', is_config_file=True, help='config file path')

    # dataset
    parser.add_argument('--dataset_name', type=str,
                        default='nyu', choices=['kitti', 'nyu', 'iBims'])

    # ckpt
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/nyu.ckpt',
                        help='checkpoint path to load')

    # inference options
    parser.add_argument('--input_dir', type=str, default='demo', help='input image path')
    parser.add_argument('--output_dir', type=str, default='output', help='output depth path')
    parser.add_argument('--save-vis', action='store_true',
                        help='save depth visualization')
    parser.add_argument('--save-depth', action='store_true',
                        help='save depth with factor 1000')

    return parser.parse_args()


def get_size(dataset_name):

    if dataset_name == 'kitti':
        image_size = [256, 832]
    elif dataset_name in ['nyu', 'iBims']:
        image_size = [256, 320]
    else:
        print('unknown dataset type')

    return image_size
