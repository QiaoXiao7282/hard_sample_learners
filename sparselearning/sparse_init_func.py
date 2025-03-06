import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sparselearning.snip import SNIP, GraSP
import numpy as np
import math
import copy


def sparse_init_func(modules, masks, mode='ERK', density=0.05, erk_power_scale=1.0, train_loader=None, device=None, model_lth=None, args=None):

    if mode == 'global_magnitude':
        print('initialize by global magnitude')
        baseline_nonzero = 0
        weight_abs = []
        for module in modules:
            for name, weight in module.named_parameters():
                if name not in masks: continue
                weight_abs.append(torch.abs(weight))

        # Gather all scores in a single vector and normalise
        all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
        num_params_to_keep = int(len(all_scores) * density)

        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]

        for module in modules:
            for name, weight in module.named_parameters():
                if name not in masks: continue
                masks[name] = ((torch.abs(weight)) >= acceptable_score).float()
                baseline_nonzero += (masks[name] != 0).sum().int().item()

    elif mode == 'snip':
        print('initialize by snip')
        layer_wise_sparsities,_ = SNIP(modules[0], density, train_loader, device, args)
        # re-sample mask positions
        for sparsity_, name in zip(layer_wise_sparsities, masks):
            masks[name][:] = (torch.rand(masks[name].shape) < (1 - sparsity_)).float().data

    elif mode == 'GraSP':
        print('initialize by GraSP')
        layer_wise_sparsities = GraSP(modules[0], density, train_loader, device)
        # re-sample mask positions
        for sparsity_, name in zip(layer_wise_sparsities, masks):
            masks[name][:] = (torch.rand(masks[name].shape) < (1 - sparsity_)).float().data

    if mode == 'GMP':
        baseline_nonzero = 0
        for module in modules:
            for name, weight in module.named_parameters():
                if name not in masks: continue
                masks[name] = torch.ones_like(weight, dtype=torch.float32, requires_grad=False)
                baseline_nonzero += (masks[name] != 0).sum().int().item()

    elif mode == 'lottery_ticket':
        print('initialize by lottery ticket')
        baseline_nonzero = 0
        weight_abs = []
        for name, weight in model_lth.named_parameters():
            if name not in masks: continue
            weight_abs.append(torch.abs(weight))

        # Gather all scores in a single vector and normalise
        all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
        num_params_to_keep = int(len(all_scores) * density)

        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]

        for name, weight in model_lth.named_parameters():
            if name not in masks: continue
            masks[name] = ((torch.abs(weight)) >= acceptable_score).float()
            baseline_nonzero += (masks[name] != 0).sum().int().item()

    elif mode == 'uniform':
        baseline_nonzero = 0
        for module in modules:
            for name, weight in module.named_parameters():
                if name not in masks: continue
                masks[name][:] = (torch.rand(weight.shape) < density).float().data  # lsw
                # masks[name][:] = (torch.rand(weight.shape) < density).float().data #lsw
                baseline_nonzero += weight.numel() * density

    elif mode == 'uniform_pre':
        baseline_nonzero = 0
        for module in modules:
            for name, weight in module.named_parameters():
                if name not in masks: continue

                flattened_weights = weight.abs().flatten()
                num_to_retain = int(weight.numel() * density)
                threshold = torch.topk(flattened_weights, num_to_retain, largest=True).values.min()
                masks[name][:] = (weight.abs() >= threshold).float().data
                baseline_nonzero += torch.sum(masks[name]).item()

    elif mode == 'half_pre':
        baseline_nonzero = 0
        # sorted_layer_names = sorted([name for name in masks.keys() if "classifier" not in name])
        sorted_layer_names = list([name for name in masks.keys()])

        half_length = len(sorted_layer_names) // 2
        first_half_layer_names = sorted_layer_names[:half_length]
        second_half_layer_names = sorted_layer_names[half_length:]

        weight_abs_first_half = []
        for module in modules:
            for name, weight in module.named_parameters():
                if name not in first_half_layer_names: continue
                weight_abs_first_half.append(torch.abs(weight))

        # Gather all scores in a single vector and normalise
        all_scores = torch.cat([torch.flatten(x) for x in weight_abs_first_half])
        num_params_to_keep = int(len(all_scores) * density)

        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]

        for module in modules:
            for name, weight in module.named_parameters():
                if name not in first_half_layer_names: continue
                masks[name] = ((torch.abs(weight)) >= acceptable_score).float()
                # baseline_nonzero += (masks[name] != 0).sum().int().item()

        weight_abs_second_half = []
        for module in modules:
            for name, weight in module.named_parameters():
                if name not in second_half_layer_names: continue
                weight_abs_second_half.append(torch.abs(weight))

        # Gather all scores in a single vector and normalise
        all_scores_second_half = torch.cat([torch.flatten(x) for x in weight_abs_second_half])
        num_params_to_keep = int(len(all_scores_second_half) * density)

        threshold_second_half, _ = torch.topk(all_scores_second_half, num_params_to_keep, sorted=True)
        acceptable_score = threshold_second_half[-1]

        for module in modules:
            for name, weight in module.named_parameters():
                if name not in second_half_layer_names: continue
                masks[name] = ((torch.abs(weight)) >= acceptable_score).float()
                baseline_nonzero += (masks[name] != 0).sum().int().item()

        aa = 0


    elif mode == 'ERK':
        print('initialize by ERK')
        total_params = 0
        for name, weight in masks.items():
            total_params += weight.numel()
        is_epsilon_valid = False
        # # The following loop will terminate worst case when all masks are in the
        # custom_sparsity_map. This should probably never happen though, since once
        # we have a single variable or more with the same constant, we have a valid
        # epsilon. Note that for each iteration we add at least one variable to the
        # custom_sparsity_map and therefore this while loop should terminate.
        dense_layers = set()
        while not is_epsilon_valid:
            # We will start with all layers and try to find right epsilon. However if
            # any probablity exceeds 1, we will make that layer dense and repeat the
            # process (finding epsilon) with the non-dense layers.
            # We want the total number of connections to be the same. Let say we have
            # for layers with N_1, ..., N_4 parameters each. Let say after some
            # iterations probability of some dense layers (3, 4) exceeded 1 and
            # therefore we added them to the dense_layers set. Those layers will not
            # scale with erdos_renyi, however we need to count them so that target
            # paratemeter count is achieved. See below.
            # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
            #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
            # eps * (p_1 * N_1 + p_2 * N_2) =
            #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
            # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

            divisor = 0
            rhs = 0
            raw_probabilities = {}
            for name, mask in masks.items():
                n_param = np.prod(mask.shape)
                n_zeros = n_param * (1 - density)
                n_ones = n_param * density

                if name in dense_layers:
                    rhs -= n_zeros
                else:
                    rhs += n_ones
                    raw_probabilities[name] = (np.sum(mask.shape) / np.prod(mask.shape)) ** erk_power_scale
                    divisor += raw_probabilities[name] * n_param
            epsilon = rhs / divisor
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for mask_name, mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:
                        print(f"Sparsity of var:{mask_name} had to be set to 0.")
                        dense_layers.add(mask_name)
            else:
                is_epsilon_valid = True

        density_dict = {}
        total_nonzero = 0.0
        # With the valid epsilon, we can set sparsities of the remaning layers.
        for name, mask in masks.items():
            n_param = np.prod(mask.shape)
            if name in dense_layers:
                density_dict[name] = 1.0
            else:
                probability_one = epsilon * raw_probabilities[name]
                density_dict[name] = probability_one
            print(
                f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
            )
            masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data

            total_nonzero += density_dict[name] * mask.numel()
        print(f"Overall denstity {total_nonzero / total_params}")

    total_params = 0
    total_nonzero = 0
    for name, weight in masks.items():
        total_params += weight.numel()
        total_nonzero += (weight != 0).sum().int().item()
        print(name, 'density:', (weight != 0).sum().item() / weight.numel())
    print('Total Model parameters:', total_params, 'Overall denstity:', total_nonzero / total_params)

    return masks
