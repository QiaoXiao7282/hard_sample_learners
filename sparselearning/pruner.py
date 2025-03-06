import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

import numpy as np
import math


'''
        DEATH
'''

def find_first_pos(weight, value):
    idx = (torch.abs(weight - value)).argmin()
    return idx # (1,0,0,1,1) 1

def find_last_pos(weight, value):
    idx = torch.abs(weight - value).flip(dims=[0]).argmin()
    return weight.shape[0] - idx   #-1 (1,0,0,1,1) ---> (1,1,0,0,1) 5-2=3 BUT ACTUALLY IT IS 2

def threshold_death(mask, weight, name, threshold):
    return (torch.abs(weight.data) > threshold)

def taylor_FO(mask, weight, name, num_remove):

    num_zeros = mask.numel() - mask.sum().item()
    k = math.ceil(num_zeros + num_remove)

    x, idx = torch.sort((weight.data * weight.grad).pow(2).flatten())
    mask.data.view(-1)[idx[:k]] = 0.0

    return mask

def magnitude_death(mask, weight, name, num_remove):

    if num_remove == 0.0: return weight.data != 0.0
    num_zeros = mask.numel() - mask.sum().item()

    x, idx = torch.sort(torch.abs(weight.data.view(-1)))
    n = idx.shape[0]
    k = math.ceil(num_zeros + num_remove)
    threshold = x[k-1].item()

    return (torch.abs(weight.data) > threshold)

def magnitude_set_death(mask, weight, name, death_rate):
    values, _ = torch.sort(weight.data.view(-1))  # bq: weight.data.view(-1)
    firstZeroPos = find_first_pos(values, 0) # the number of negative value
    lastZeroPos = find_last_pos(values, 0) # values.shape[0] - lastZeroPos: the number of positive value
    largestNegative = values[int((1 - death_rate) * firstZeroPos)]
    smallestPositive = values[int(min(values.shape[0] - 1, lastZeroPos + death_rate * (values.shape[0] - lastZeroPos)))]

    rewiredWeights = weight.data.clone()
    rewiredWeights[rewiredWeights > smallestPositive] = 1
    rewiredWeights[rewiredWeights < largestNegative] = 1
    rewiredWeights[rewiredWeights != 1] = 0
    weightMaskCore = rewiredWeights.clone()

    #weightMaskCore = (weight.data > smallestPositive) | (weight.data < largestNegative)

    return weightMaskCore


def magnitude_and_negativity_death(mask, weight, name, num_remove):
    # num_zeros = mask.numel() - mask.sum().item()

    # find magnitude threshold
    # remove all weights which absolute value is smaller than threshold
    x, idx = torch.sort(weight[weight > 0.0].data.view(-1))
    k = math.ceil(num_remove/2.0)
    if k >= x.shape[0]:
        k = x.shape[0]

    threshold_magnitude = x[k-1].item()

    # find negativity threshold
    # remove all weights which are smaller than threshold
    x, idx = torch.sort(weight[weight < 0.0].view(-1))
    k = math.ceil(num_remove/2.0)
    if k >= x.shape[0]:
        k = x.shape[0]
    threshold_negativity = x[k-1].item()

    pos_mask = (weight.data > threshold_magnitude) & (weight.data > 0.0)
    neg_mask = (weight.data < threshold_negativity) & (weight.data < 0.0)

    new_mask = pos_mask | neg_mask
    return new_mask