import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from sparselearning.utils import DatasetSplitter
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
import argparse

def get_cifar10_dataloaders_new(args, validation_split=0.1, max_threads=0):
    """Creates augmented train, validation, and test data loaders."""

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))

    train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                    (4,4,4,4),mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
         normalize
    ])

    full_dataset = datasets.CIFAR10('_dataset', True, train_transform, download=True)
    test_dataset = datasets.CIFAR10('_dataset', False, test_transform, download=False)


    # we need at least two threads
    max_threads = 2 if max_threads < 2 else max_threads
    if max_threads >= 6:
        val_threads = 2
        train_threads = max_threads - val_threads
    else:
        val_threads = 1
        train_threads = max_threads - 1


    valid_loader = None
    if validation_split > 0.0:
        ## choose specific percent train_dataset
        split = int(np.floor((1.0 - validation_split) * len(full_dataset)))
        val_dataset = DatasetSplitter(full_dataset, split_start=split)

        train_idx, val_idx = choose_dataset(dataset=full_dataset, p=args.p, split=split)
        train_sampler = SubsetRandomSampler(train_idx)
        train_batch_sampler = BatchSampler(train_sampler, args.batch_size,
                                           drop_last=True)
        train_loader = DataLoader(full_dataset,
                                  batch_sampler=train_batch_sampler,
                                  num_workers=train_threads,
                                  pin_memory=True)

        valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            args.test_batch_size,
            num_workers=val_threads,
            pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            full_dataset,
            args.batch_size,
            num_workers=8,
            pin_memory=True, shuffle=True)

    print('Train loader length', len(train_loader))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        args.test_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True)

    return train_loader, valid_loader, test_loader

def get_cifar100_dataloaders_new(args, validation_split=0.0, max_threads=10):
    """Creates augmented train, validation, and test data loaders."""
    cifar_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    cifar_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                             transform=transform_train)

    train_idx, val_idx = choose_dataset(dataset=trainset, p=args.p, split=50000)
    train_sampler = SubsetRandomSampler(train_idx)
    train_batch_sampler = BatchSampler(train_sampler, args.batch_size,
                                       drop_last=True)
    train_loader = DataLoader(trainset,
                              batch_sampler=train_batch_sampler,
                              num_workers=2,
                              pin_memory=True)


    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, test_loader


def choose_dataset(dataset, random_flag=False, p=0.2, split=45000):
    """ Return the labeled indexes
    """
    labels = np.array(dataset.train_labels)
    classes = np.unique(labels)

    train_labels = labels[:split]

    train_idxs = []
    val_idxs = np.array([i for i in range(split, len(dataset))])
    for id in classes:
        indexes = np.where(train_labels==id)[0]
        n_per_class = int(len(indexes) * p)
        if random_flag:
            np.random.shuffle(indexes)
        indexes_label = np.array(indexes[:n_per_class].tolist())
        train_idxs.extend(indexes_label)

    return train_idxs, val_idxs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--p', type=float, default=0.5)

    args = parser.parse_args()
    get_cifar10_dataloaders_new(args)