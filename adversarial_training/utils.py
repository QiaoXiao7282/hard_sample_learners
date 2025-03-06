import os
import time
import torch
import random
import shutil
import numpy as np
from advertorch.attacks import LinfPGDAttack, L2PGDAttack
from advertorch.context import ctx_noparamgrad
from advertorch.utils import NormalizeByChannelMeanStd
from adversarial_training.datasets import *
from sparselearning.models_de import AlexNet, VGG16, LeNet_300_100, LeNet_5_Caffe, WideResNet, MLP_CIFAR10, ResNet34, ResNet18, VGG
from torch.utils.data import DataLoader, Subset, Dataset

import hashlib
import logging

__all__ = ['save_checkpoint', 'setup_dataset_models', 'print_args', 'get_save_path', 'getinfo']

def save_checkpoint(state, save_path, filename='checkpoint.pth.tar'):
    
    filepath = '{}_{}'.format(save_path, filename)
    torch.save(state, filepath)


#print training configuration
def print_args(args):
    print('*'*50)
    print('Dataset: {}'.format(args.dataset))
    print('Model: {}'.format(args.arch))
    if args.arch == 'wideresnet':
        print('Depth {}'.format(args.depth_factor))
        print('Width {}'.format(args.width_factor))
    print('*'*50)        
    print('Attack Norm {}'.format(args.norm))  
    print('Test Epsilon {}'.format(args.test_eps))
    print('Test Steps {}'.format(args.test_step))
    print('Train Steps Size {}'.format(args.test_gamma))
    print('Test Randinit {}'.format(args.test_randinit))
    if args.eval:
        print('Evaluation')
        print('Loading weight {}'.format(args.pretrained))
    else:
        print('Training')
        print('Train Epsilon {}'.format(args.train_eps))
        print('Train Steps {}'.format(args.train_step))
        print('Train Steps Size {}'.format(args.train_gamma))
        print('Train Randinit {}'.format(args.train_randinit))

def setup_dataset_models_eval(args):

    # prepare dataset
    if args.data == 'cifar10':
        classes = 10
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        train_set, val_set, test_set = cifar10_dataloaders_eval(batch_size = args.batch_size, args=args)
    
    elif args.data == 'cifar100':
        classes = 100
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
        train_set, val_set, test_set = cifar100_dataloaders_eval(batch_size = args.batch_size, args=args)
    
    elif args.data == 'tinyimagenet':
        classes = 200
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
        train_set, val_set, test_set = tiny_imagenet_dataloaders_eval(batch_size = args.batch_size, args=args)
    
    else:
        raise ValueError("Unknown Dataset")

    len_dataset = len(train_set)
    rnd = np.random.RandomState(seed=12345)  # Create a random generator with a fixed seed
    num_seq = int(len_dataset * args.p_data)
    data_set_index = np.arange(len_dataset)
    selected_indices = rnd.choice(data_set_index, num_seq, replace=False)
    train_set = Subset(train_set, selected_indices)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.max_threads, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.max_threads, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.max_threads, pin_memory=True)

    if args.model == 'ResNet18':
        model = ResNet18(c = classes)
        model.normalize = dataset_normalization

    elif args.model == 'wideresnet':
        model = WideResNet(args.depth_factor, classes, widen_factor=args.width_factor, dropRate=0.0)
        model.normalize = dataset_normalization

    elif args.model == 'vgg16':
        model = VGG16(config='D', num_classes = classes)
        model.normalize = dataset_normalization

    else:
        raise ValueError("Unknown Model")
    
    return train_loader, val_loader, test_loader, model

# prepare dataset and models
def setup_dataset_models(args):

    # prepare dataset
    if args.data == 'cifar10':
        classes = 10
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        train_set, val_set, test_set = cifar10_dataloaders(batch_size = args.batch_size, args=args)
    
    elif args.data == 'cifar100':
        classes = 100
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
        train_set, val_set, test_set = cifar100_dataloaders(batch_size = args.batch_size, args=args)
    
    elif args.data == 'tinyimagenet':
        classes = 200
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
        train_set, val_set, test_set = tiny_imagenet_dataloaders(batch_size = args.batch_size, args=args)
    
    else:
        raise ValueError("Unknown Dataset")

    len_dataset = len(train_set)
    rnd = np.random.RandomState(seed=12345)  # Create a random generator with a fixed seed
    num_seq = int(len_dataset * args.p_data)
    data_set_index = np.arange(len_dataset)
    selected_indices = rnd.choice(data_set_index, num_seq, replace=False)
    train_set = Subset(train_set, selected_indices)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.max_threads, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.max_threads, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.max_threads, pin_memory=True)


    #prepare model
    if args.model == 'ResNet18':
        model = ResNet18(c = classes)
        model.normalize = dataset_normalization

    elif args.model == 'wideresnet':
        model = WideResNet(args.depth_factor, classes, widen_factor=args.width_factor, dropRate=0.0)
        model.normalize = dataset_normalization

    elif args.model == 'vgg16':
        model = VGG16(config='D', num_classes = classes)
        model.normalize = dataset_normalization

    else:
        raise ValueError("Unknown Model")
    
    return train_loader, val_loader, test_loader, model

def get_save_path(args):
    dir = ""
    if(args.fb):
        dir_format = 'fb_{args.arch}_{args.dataset}_d{args.density}_{args.growth}_T{args.update_frequency}_b{args.batch_size}_r{args.death_rate}_{flag}'
    elif(args.fbp):
        dir_format = 'fbp_{args.arch}_{args.dataset}_{args.sparse_init}_T{args.update_frequency}_d{args.density}_dr{args.death_rate}_{args.growth}_p{args.prune_ratio}_g{args.growth_ratio}_b{args.batch_size}_e{args.epoch_range}_r{args.update_threshold}_seed{args.seed}{flag}'
    else:
        dir_format = 'dense_{args.arch}_{args.dataset}_b{args.batch_size}_{flag}'

    dir = dir_format.format(args = args, flag = hashlib.md5(str(args).encode('utf-8')).hexdigest()[:4])
    save_path = os.path.join(args.save_dir, dir)
    return save_path

        
def input_a_sample(model, criterion, optimizer, args, data_sample):

    if args.norm == 'linf':
        adversary = LinfPGDAttack(
            model, loss_fn=criterion, eps=args.train_eps, nb_iter=args.train_step, eps_iter=args.train_gamma,
            rand_init=args.train_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )
    elif args.norm == 'l2':
        adversary = L2PGDAttack(
            model, loss_fn=criterion, eps=args.train_eps, nb_iter=args.train_step, eps_iter=args.train_gamma,
            rand_init=args.train_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )  

    model.eval()
    input, target = data_sample

    input = input.unsqueeze(dim = 0)
    target = torch.Tensor([target]).long()

    input = input.cuda()
    target = target.cuda()

    #adv samples
    with ctx_noparamgrad(model):
        input_adv = adversary.perturb(input, target)
    # compute output
    output_adv = model(input_adv)
    loss = criterion(output_adv, target)

    optimizer.zero_grad()
    loss.backward()


def getinfo(checkpoint):
    best_sa = checkpoint['best_sa']
    best_ra = checkpoint['best_ra']
    end_epoch = checkpoint['epoch']
    print('end_epoch', end_epoch)
    all_result = checkpoint['result']

    best_val_ra_index = all_result['val_ra'].index(best_ra)

    best_test_ra =  all_result['test_ra'][best_val_ra_index]
    final_test_ra = all_result['test_ra'][-1]
    diff1 = best_test_ra - final_test_ra


    best_test_sa =  all_result['test_sa'][best_val_ra_index]
    final_test_sa = all_result['test_sa'][-1]
    diff2 = best_test_sa - final_test_sa

    print('* Model best ra = {:.2f}, final_ra = {:.2f}, Diff1 = {:.2f}'.format(best_test_ra, final_test_ra, diff1))
    print('* Model best sa = {:.2f}, final_sa = {:.2f}, Diff2 = {:.2f}'.format(best_test_sa, final_test_sa, diff2))
