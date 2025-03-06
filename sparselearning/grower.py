
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

import numpy as np
import math

'''
                    GROWTH
'''

def random_unfired_growth(name, new_mask, weight, num_regrow, fired_masks):
    total_regrowth = num_regrow
    n = (new_mask == 0).sum().item()
    if n == 0: return new_mask
    num_nonfired_weights = (fired_masks[name]==0).sum().item()

    if total_regrowth <= num_nonfired_weights:
        idx = (fired_masks[name].flatten() == 0).nonzero()
        indices = torch.randperm(len(idx))[:total_regrowth]

        # idx = torch.nonzero(self.fired_masks[name].flatten())
        new_mask.data.view(-1)[idx[indices]] = 1.0
    else:
        new_mask[fired_masks[name]==0] = 1.0
        n = (new_mask == 0).sum().item()
        expeced_growth_probability = ((total_regrowth-num_nonfired_weights) / n)
        new_weights = torch.rand(new_mask.shape).to(weight.device) < expeced_growth_probability
        new_mask = new_mask.byte() | new_weights
    return new_mask

def random_growth(name, new_mask, weight, num_regrow):
    total_regrowth = num_regrow
    n = (new_mask==0).sum().item()
    if n == 0: return new_mask
    expeced_growth_probability = (total_regrowth/n)
    new_weights = torch.rand(new_mask.shape).to(weight.device) < expeced_growth_probability
    new_mask_ = new_mask.byte() | new_weights  # bq:new_mask_ = new_mask.bool()
    if (new_mask_!=0).sum().item() == 0:
        new_mask_ = new_mask
    return new_mask_

def random_growth_new(name, new_mask, weight, num_regrow):
    total_regrowth = num_regrow
    n = (new_mask==0).sum().item()
    if n == 0: return new_mask

    idx = (new_mask.flatten() == 0).nonzero()
    indices = torch.randperm(len(idx))[:total_regrowth]
    new_mask.data.view(-1)[idx[indices]] = 1.0
    return new_mask


def momentum_growth(name, new_mask, weight, num_regrow, optimizer):
    total_regrowth = num_regrow
    grad = get_momentum_for_weight(weight, optimizer)
    grad = grad*(new_mask==0).float()
    y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
    new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

    return new_mask

def gradient_growth(name, new_mask, weight, num_regrow):
    total_regrowth = num_regrow
    grad = get_gradient_for_weights(weight)

    abs_grad = torch.abs(grad) + 1e-6
    abs_grad = abs_grad * (new_mask == 0).float()

    y, idx = torch.sort(abs_grad.flatten(), descending=True)
    new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

    return new_mask

def momentum_neuron_growth(name, new_mask, weight, num_regrow, optimizer):
    total_regrowth = num_regrow
    grad = get_momentum_for_weight(weight, optimizer)

    M = torch.abs(grad)
    if len(M.shape) == 2: sum_dim = [1]
    elif len(M.shape) == 4: sum_dim = [1, 2, 3]

    v = M.mean(sum_dim).data
    v /= v.sum()

    slots_per_neuron = (new_mask==0).sum(sum_dim)

    M = M*(new_mask==0).float()
    for i, fraction  in enumerate(v):
        neuron_regrowth = math.floor(fraction.item()*total_regrowth)
        available = slots_per_neuron[i].item()

        y, idx = torch.sort(M[i].flatten())
        if neuron_regrowth > available:
            neuron_regrowth = available
        threshold = y[-(neuron_regrowth)].item()
        if threshold == 0.0: continue
        if neuron_regrowth < 10: continue
        new_mask[i] = new_mask[i] | (M[i] > threshold)

    return new_mask

'''
            UTILITY
'''
def get_momentum_for_weight(weight, optimizer):
    if 'exp_avg' in optimizer.state[weight]:
        adam_m1 = optimizer.state[weight]['exp_avg']
        adam_m2 = optimizer.state[weight]['exp_avg_sq']
        grad = adam_m1/(torch.sqrt(adam_m2) + 1e-08)
    elif 'momentum_buffer' in optimizer.state[weight]:
        grad = optimizer.state[weight]['momentum_buffer']
    return grad

def get_gradient_for_weights(weight):
    grad = weight.grad.clone()
    return grad