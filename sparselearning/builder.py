import math

import torch
import torch.nn as nn

# from sparselearning.sparse_type import SparseConv, SparseLin


class Builder(object):
    def __init__(self):
        self.M = 12

    # make everything sparse_conv
    def conv(self, kernel_size, in_planes, out_planes, stride=1, padding=1, bias=False):

        if kernel_size == 3:
            conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=bias)
        elif kernel_size == 1:
            conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias)
        elif kernel_size == 5:
            conv = nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=padding, bias=bias)
        elif kernel_size == 7:
            conv = nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=padding, bias=bias)
        else:
            return None

        return conv

    def conv3x3(self, in_planes, out_planes, stride=1, padding=1, bias=False):
        """3x3 convolution with padding"""
        c = self.conv(3, in_planes, out_planes, stride=stride, padding=padding, bias=bias)
        return c

    def conv1x1(self, in_planes, out_planes, stride=1, padding=0, bias=False):
        """1x1 convolution with padding"""
        c = self.conv(1, in_planes, out_planes, stride=stride, padding=padding, bias=bias)
        return c

    def conv7x7(self, in_planes, out_planes, stride=1, padding=3, bias=False):
        """7x7 convolution with padding"""
        c = self.conv(7, in_planes, out_planes, stride=stride, padding=padding, bias=bias)
        return c

    def conv5x5(self, in_planes, out_planes, stride=1, padding=2, bias=False):
        """5x5 convolution with padding"""
        c = self.conv(5, in_planes, out_planes, stride=stride, padding=padding, bias=bias)
        return c

    def batchnorm(self, planes, affine=True):
        bn = nn.BatchNorm2d(planes, affine=affine)
        # gamma_init_val = 1
        # nn.init.constant_(bn.weight, gamma_init_val)
        # nn.init.constant_(bn.bias, 0)
        return bn

    def batchnorm1d(self, planes, affine=True):
        bn = nn.BatchNorm1d(planes, affine=affine)
        # gamma_init_val = 1
        # nn.init.constant_(bn.weight, gamma_init_val)
        # nn.init.constant_(bn.bias, 0)
        return bn

    def activation(self):
        return (lambda: nn.ReLU(inplace=True))()

    def linear(self, in_planes, out_planes, bias=True):
        Linear = nn.Linear(in_planes, out_planes, bias=bias)
        return Linear


def get_builder():

    builder = Builder()

    return builder
