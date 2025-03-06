import os 
import numpy as np 
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

__all__ = ['cifar10_dataloaders', 'cifar100_dataloaders', 'tiny_imagenet_dataloaders',
            'cifar10_dataloaders_eval', 'cifar100_dataloaders_eval', 'adv_image_dataset', 'tiny_imagenet_dataloaders_eval']

def cifar10_dataloaders(batch_size=64, args=None):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = Subset(CIFAR10('{}/dataset'.format(args.data_path), train=True, transform=train_transform, download=True), list(range(45000)))
    val_set = Subset(CIFAR10('{}/dataset'.format(args.data_path), train=True, transform=test_transform, download=True), list(range(45000, 50000)))

    test_set = CIFAR10('{}/dataset'.format(args.data_path), train=False, transform=test_transform, download=True)

    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    #
    # return train_loader, val_loader, test_loader
    return train_set, val_set, test_set


def cifar100_dataloaders(batch_size=64, args=None):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = Subset(CIFAR100('{}/dataset'.format(args.data_path), train=True, transform=train_transform, download=True), list(range(45000)))
    val_set = Subset(CIFAR100('{}/dataset'.format(args.data_path), train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR100('{}/dataset'.format(args.data_path), train=False, transform=test_transform, download=True)

    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    #
    # return train_loader, val_loader, test_loader
    return train_set, val_set, test_set

def tiny_imagenet_dataloaders(batch_size=64, args=None, permutation_seed=10):

    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    data_dir = '{}/tiny-imagenet-200'.format(args.data_path_tiny)

    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'val')

    np.random.seed(permutation_seed)
    split_permutation = list(np.random.permutation(100000))

    train_set = Subset(ImageFolder(train_path, transform=train_transform), split_permutation[:90000])
    val_set = Subset(ImageFolder(train_path, transform=test_transform), split_permutation[90000:])
    test_set = ImageFolder(val_path, transform=test_transform)

    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    #
    # return train_loader, val_loader, test_loader
    return train_set, val_set, test_set


def tiny_imagenet_dataloaders_eval(batch_size=64, args=None, permutation_seed=10):

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    data_dir = '{}/tiny-imagenet-200'.format(args.data_path_tiny)

    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'val')

    np.random.seed(permutation_seed)
    split_permutation = list(np.random.permutation(100000))

    train_set = Subset(ImageFolder(train_path, transform=test_transform), split_permutation[:90000])
    val_set = Subset(ImageFolder(train_path, transform=test_transform), split_permutation[90000:])
    test_set = ImageFolder(val_path, transform=test_transform)

    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    #
    # return train_loader, val_loader, test_loader
    return train_set, val_set, test_set



def cifar100_dataloaders_eval(batch_size=64, args=None, data_num=45000):

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = Subset(CIFAR100('{}/dataset'.format(args.data_path), train=True, transform=test_transform, download=True), list(range(data_num)))
    val_set = Subset(CIFAR100('{}/dataset'.format(args.data_path), train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR100('{}/dataset'.format(args.data_path), train=False, transform=test_transform, download=True)

    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    #
    # return train_loader, val_loader, test_loader
    return train_set, val_set, test_set


def cifar10_dataloaders_eval(batch_size=64, args=None, data_num=45000):

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = Subset(CIFAR10('{}/dataset'.format(args.data_path), train=True, transform=test_transform, download=True), list(range(data_num)))
    val_set = Subset(CIFAR10('{}/dataset'.format(args.data_path), train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR10('{}/dataset'.format(args.data_path), train=False, transform=test_transform, download=True)

    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    #
    # return train_loader, val_loader, test_loader
    return train_set, val_set, test_set


class adv_image_dataset(Dataset):
    
    def __init__(self, data):
        super(adv_image_dataset, self).__init__()

        self.image = data['data']
        self.target = data['label']

        self.number = self.image.shape[0]

    def __len__(self):

        return self.number

    def __getitem__(self, index):

        img = self.image[index]
        label = self.target[index]

        return img, label




