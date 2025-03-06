import os
import argparse
from pathlib import Path
import warnings

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms

# load attack
from autoattack import AutoAttack
from adversarial_training.utils import setup_dataset_models_eval, save_checkpoint
import glob

def str2bool(str):
    return True if str.lower() == 'true' else False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--epsilon', type=float, default=8. / 255.)
    parser.add_argument('--n_ex', type=int, default=10000)
    parser.add_argument('--individual', action='store_true')
    parser.add_argument('--save_dir', type=str, default='/home/snns/codes/data_dst_diff/adversarial_training/results')
    parser.add_argument('--log_path', type=str, default='./results/log_file.txt')
    parser.add_argument('--version', type=str, default='standard')
    parser.add_argument('--state-path', type=Path, default=None)

    parser.add_argument('--batch-size', type=int, default=500, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument("--data", help="data to train", default='cifar10', type=str)
    parser.add_argument('--model', type=str, default='ResNet18') #vgg16, ResNet18, ResNet34, vgg19
    parser.add_argument('--root_path', type=str, default='/data/snns/models/data_dst_diff')
    parser.add_argument("--data_dir", help='path for data', default='/data/snns/data/data_dst/data_bias', type=str)
    parser.add_argument('--data_path', type=str, default='/data/snns/data/sparse_robust')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--seed', type=int, default=15, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--max-threads', type=int, default=10, help='How many threads to.')
    parser.add_argument('--gpu', type=int, default=0, help='device ids of gpus')

    parser.add_argument('--method', type=str, default='SET', help='dst explicit') ## SET, rigl, GMP, prune; finetune, fix
    parser.add_argument('--density', type=float, default=0.2, help='The density of the overall sparse network.')
    parser.add_argument('--sparse', type=str2bool, default=True, help='Enable sparse mode. Default: True.')
    parser.add_argument('--p_data', type=float, default=1.0)
    parser.add_argument('--task', type=str, default='dataadv')


    args = parser.parse_args()

    device = torch.device('cuda:{}'.format(args.gpu))

    models_seed = 'models-seed{}'.format(args.seed)
    path = '{}/{}/{}'.format(args.root_path, args.task, models_seed)

    '''
    filename = "adv-{}-{}-{}-{}-{}-{}-{}-{}".format(args.data, args.model, args.method, args.density, args.sparse,
                                                    args.p_data, 'linf', args.epochs)
    save_path = '{}/{}'.format(path, filename)

    filepath_final = '{}_{}'.format(save_path, 'checkpoint_adv_final.pth.tar')
    '''

    search_pattern = os.path.join(path, '*checkpoint_adv_*')  ## checkpoint_adv_, *final.pth*
    file_paths = glob.glob(search_pattern)

    # if args.best_check:
    #     filepath_best = '{}_{}'.format(save_path, 'checkpoint_adv_best.pth.tar')

    for file_path in file_paths:
        base_name = os.path.basename(file_path)

        ### print best, final output
        checkpoint = torch.load(file_path, map_location='cpu')
        if 'adv_best' in base_name:
            test_bn = checkpoint['result']['test_bn'][-1]
            test_adv = checkpoint['result']['test_adv'][-1]
        if 'adv_final' in base_name:
            test_bn = checkpoint['result']['test_bn'][-1]
            test_adv = checkpoint['result']['test_adv'][-1]
        print('{}----test_bn: {}, test_adv: {}'.format(base_name, test_bn, test_adv))
