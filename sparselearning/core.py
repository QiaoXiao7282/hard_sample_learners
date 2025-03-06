from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

import numpy as np
import math

import sparselearning.pruner as pruner
import sparselearning.grower as grower
from sparselearning.sparse_init_func import sparse_init_func
from sparselearning.funcs import strategy_dict

def str2bool(str):
    return True if str.lower() == 'true' else False

def add_sparse_args(parser):
    parser.add_argument('--sparse', type=str2bool, default=True, help='Enable sparse mode. Default: True.')
    parser.add_argument('--fix', type=str2bool, default=False, help='Fix sparse connectivity during training. Default: True.')
    parser.add_argument('--sparse_init', type=str, default='ERK', help='sparse initialization')
    parser.add_argument('--method', type=str, default='GMP', help='dst explicit') ## SET, rigl, GMP, finetune, fix, lth
    parser.add_argument('--growth', type=str, default='random', help='Growth mode. Choose from: momentum, random, random_unfired, and gradient.')
    parser.add_argument('--death', type=str, default='set_prune', help='Death mode / pruning mode. Choose from: magnitude, set_prune, threshold.')
    parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--death-rate', type=float, default=0.20, help='The pruning rate / death rate.')
    parser.add_argument('--density', type=float, default=0.3, help='The density of the overall sparse network.')
    parser.add_argument('--update_frequency', type=int, default=2, metavar='N', help='how many iterations to train between parameter exploration')
    parser.add_argument('--decay-schedule', type=str, default='cosine', help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
    parser.add_argument('--momentum_reset', type=str2bool, default=False, help='Masking the momentum yes or not')
    parser.add_argument('--init-prune-epoch', type=int, default=0, help='The pruning rate / death rate.')
    parser.add_argument('--final-prune-epoch', type=int, default=140, help='The density of the overall sparse network.')

class CosineDecay(object):
    def __init__(self, death_rate, T_max, eta_min=0.001, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self):
        return self.sgd.param_groups[0]['lr']

class LinearDecay(object):
    def __init__(self, death_rate, factor=0.99, frequency=600):
        self.factor = factor
        self.steps = 0
        self.frequency = frequency

    def step(self):
        self.steps += 1

    def get_dr(self, death_rate):
        if self.steps > 0 and self.steps % self.frequency == 0:
            return death_rate*self.factor
        else:
            return death_rate


class Masking(object):
    def __init__(self, optimizer=None, death_rate=0.3, growth_death_ratio=1.0, death_rate_decay=None, death_mode='magnitude', growth_mode='momentum', redistribution_mode='momentum', threshold=0.001, train_loader=None, device='cpu', args=None):
        growth_modes = ['random', 'momentum', 'momentum_neuron', 'gradient']
        if growth_mode not in growth_modes:
            print('Growth mode: {0} not supported!'.format(growth_mode))
            print('Supported modes are:', str(growth_modes))

        self.args = args
        self.device = device
        self.growth_death_ratio = growth_death_ratio
        self.redistribution_mode = redistribution_mode
        self.death_rate_decay = death_rate_decay
        self.train_loader = train_loader

        ## death grow strategy
        if self.args.method == 'SET' or self.args.method == 'rigl':
            self.death_mode, self.growth_mode = strategy_dict[self.args.method]
        else:
            self.growth_mode = growth_mode
            self.death_mode = death_mode

        self.masks = {}
        self.modules = []
        self.names = []
        self.optimizer = optimizer

        # stats
        self.name2zeros = {}
        self.num_remove = {}
        self.name2nonzeros = {}
        self.death_rate = death_rate
        self.baseline_nonzero = None
        self.steps = 0

        self.explore_step = 0
        self.pruned_masks = {}
        self.regrowed_masks = {}
        self.pre_masks = None

        # if fix, then we do not explore the sparse connectivity
        if self.args.fix:
            self.prune_every_k_steps = None
        else:
            self.prune_every_k_steps = self.args.update_frequency * len(self.train_loader)


    def step(self):
        self.optimizer.step()
        self.apply_mask()
        self.death_rate_decay.step()
        self.death_rate = self.death_rate_decay.get_dr()
        # self.death_rate = 0.2

        self.steps += 1

        if self.prune_every_k_steps is not None and self.steps % self.prune_every_k_steps == 0:

            if self.args.method == 'SET' or self.args.method == 'rigl':
                self.explore_step += 1
                self.truncate_weights()

            elif self.args.method == 'GMP':
                if self.steps >= (self.args.init_prune_epoch * len(self.train_loader)):
                    self.explore_step += 1
                    self.pruning(self.steps)

            self.cal_nonzero_counts()
            self.curr_density = self.total_nozeros / self.total_weights
            print('curr_density: {0:.4f}'.format(self.curr_density))

            _, _ = self.fired_masks_update()
            # if self.explore_step > 1:
            #     self.print_nonzero_counts()
            self.pre_masks = copy.deepcopy(self.pruned_masks)

    def add_module(self, module, density, sparse_init='ER', model_lth=None):
        self.density = density
        self.modules.append(module)
        for name, tensor in module.named_parameters():
            self.names.append(name)
            self.masks[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).to(self.device)

        print('Removing biases...')
        self.remove_weight_partial_name('bias')
        print('Removing act funcs...')
        self.remove_weight_partial_name('act')
        print('Removing 2D batch norms...')
        self.remove_type(nn.BatchNorm2d)
        print('Removing 1D batch norms...')
        self.remove_type(nn.BatchNorm1d)

        self.masks = sparse_init_func(self.modules, self.masks, mode=sparse_init, density=density, train_loader=self.train_loader, device=self.device, model_lth=model_lth, args=self.args)
        self.apply_mask()
        self.fired_masks = copy.deepcopy(self.masks)
        self.pruned_masks = copy.deepcopy(self.masks)
        self.init_masks = copy.deepcopy(self.masks)
        self.pre_pruned_masks = copy.deepcopy(self.masks)
        self.previous_masks = copy.deepcopy(self.masks)

    def cal_nonzero_counts(self):
        self.total_nozeros = 0
        self.total_weights = 0

        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks:
                    continue
                mask = self.masks[name]
                self.total_nozeros += (mask != 0).sum().to(self.device)
                self.total_weights += mask.numel()

    def remove_weight(self, name):
        if name in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name].shape,
                                                                      self.masks[name].numel()))
            self.masks.pop(name)
        elif name + '.weight' in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name + '.weight'].shape,
                                                                      self.masks[name + '.weight'].numel()))
            self.masks.pop(name + '.weight')
        else:
            print('ERROR', name)

    def remove_weight_partial_name(self, partial_name):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:

                print('Removing {0} of size {1} with {2} parameters...'.format(name, self.masks[name].shape,
                                                                                   np.prod(self.masks[name].shape)))
                removed.add(name)
                self.masks.pop(name)

        print('Removed {0} layers.'.format(len(removed)))

        i = 0
        while i < len(self.names):
            name = self.names[i]
            if name in removed:
                self.names.pop(i)
            else:
                i += 1

    def remove_type(self, nn_type):
        for module in self.modules:
            for name, module in module.named_modules():
                if isinstance(module, nn_type):
                    self.remove_weight(name)

    def apply_mask(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks:
                    tensor.data = tensor.data*self.masks[name]
                    # tensor.grad = tensor.grad * self.masks[name]  ##add
                    # reset momentum
                    if self.args.momentum_reset:
                        if 'momentum_buffer' in self.optimizer.state[tensor]:
                            self.optimizer.state[tensor]['momentum_buffer'] = self.optimizer.state[tensor]['momentum_buffer']*self.masks[name]

    def truncate_weights_GMP(self, epoch):
        '''
        Implementation  of GMP To prune, or not to prune: exploring the efficacy of pruning for model compression https://arxiv.org/abs/1710.01878
        :param epoch: current training epoch
        :return:
        '''
        prune_rate = 1 - self.density
        curr_prune_epoch = epoch
        total_prune_epochs = self.args.final_prune_epoch - self.args.init_prune_epoch + 1
        if epoch >= self.args.init_prune_epoch and epoch <= self.args.final_prune_epoch:

            prune_decay = (1 - ((curr_prune_epoch - self.args.init_prune_epoch) / total_prune_epochs)) ** 3
            curr_prune_rate = prune_rate - (prune_rate * prune_decay)

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue

                    x, idx = torch.sort(torch.abs(weight.data.view(-1)))
                    p = int(curr_prune_rate * weight.numel())
                    self.masks[name].data.view(-1)[idx[:p]] = 0.0
            self.apply_mask()
        total_size = 0
        for name, weight in self.masks.items():
            total_size += weight.numel()
        print('Total Model parameters:', total_size)

        sparse_size = 0
        for name, weight in self.masks.items():
            sparse_size += (weight != 0).sum().int().item()

        print('Total parameters under sparsity level of {0}: {1} after epoch of {2}'.format(self.density, sparse_size / total_size, epoch))

    def pruning(self, step):
        curr_prune_iter = int(step / self.prune_every_k_steps)
        final_iter = int((self.args.final_prune_epoch * len(self.train_loader)) / self.prune_every_k_steps)
        ini_iter = int((self.args.init_prune_epoch * len(self.train_loader)) / self.prune_every_k_steps)
        total_prune_iter = final_iter - ini_iter
        print('******************************************************')
        print(f'Pruning Progress is {curr_prune_iter - ini_iter} / {total_prune_iter}')
        print('******************************************************')

        prune_rate = 1 - self.density
        if curr_prune_iter >= ini_iter and curr_prune_iter <= final_iter - 1:
            prune_decay = (1 - ((curr_prune_iter - ini_iter) / total_prune_iter)) ** 3
            curr_prune_rate = prune_rate - (prune_rate * prune_decay)

            weight_abs = []
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    weight_abs.append(torch.abs(weight))

            # Gather all scores in a single vector and normalise
            all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
            num_params_to_keep = int(len(all_scores) * (1 - curr_prune_rate))

            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = ((torch.abs(weight)) > acceptable_score).float()  # must be > to prevent acceptable_score is zero, leading to dense tensors

            self.apply_mask()

            total_size = 0
            for name, weight in self.masks.items():
                total_size += weight.numel()
            print('Total Model parameters:', total_size)

            sparse_size = 0
            for name, weight in self.masks.items():
                sparse_size += (weight != 0).sum().int().item()

            print('Sparsity after pruning: {0}'.format((total_size - sparse_size) / total_size))


    def truncate_weights(self):

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                self.name2nonzeros[name] = mask.sum().item()
                # self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]

                # death
                num_death = math.ceil(self.death_rate * self.name2nonzeros[name])
                if self.death_mode == 'magnitude':
                    new_mask = pruner.magnitude_death(mask, weight, name, num_death)
                elif self.death_mode == 'set_prune':
                    new_mask = pruner.magnitude_set_death(mask, weight, name, self.death_rate)
                elif self.death_mode == 'Taylor_FO':
                    new_mask = pruner.taylor_FO(mask, weight, name, num_death)
                elif self.death_mode == 'threshold':
                    new_mask = pruner.threshold_death(mask, weight, name, self.threshold)

                self.num_remove[name] = int(self.name2nonzeros[name] - new_mask.sum().item())

                self.previous_masks[name] = copy.deepcopy(self.masks[name])
                self.masks[name][:] = new_mask

                self.pre_pruned_masks[name] = copy.deepcopy(self.pruned_masks[name])
                self.pruned_masks[name] = copy.deepcopy(new_mask)

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name].data.byte()

                # growth
                num_regrow = self.num_remove[name]
                if self.growth_mode == 'random':
                    new_mask = grower.random_growth_new(name, mask, weight, num_regrow)

                if self.growth_mode == 'random_unfired':
                    new_mask = grower.random_unfired_growth(name, mask, weight, num_regrow, self.fired_masks)

                elif self.growth_mode == 'momentum':
                    new_mask = grower.momentum_growth(name, mask, weight, num_regrow, self.optimizer)

                elif self.growth_mode == 'gradient':
                    new_mask = grower.gradient_growth(name, mask, weight, num_regrow)

                # exchanging masks
                self.masks.pop(name)
                self.masks[name] = new_mask.float()

                self.regrowed_masks[name] = new_mask.float()

        self.apply_mask()


    def print_nonzero_counts(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                num_nonzeros = (mask != 0).sum().item()

                diff_mask = self.pre_masks[name].data.eq(self.pruned_masks[name].data)
                diff = (diff_mask==False).sum()

                val = '{0}: {1}->{2}, density: {3:.3f}, diff: {4}'.format(name, self.name2nonzeros[name], num_nonzeros, num_nonzeros/float(mask.numel()), diff)
                print(val)


        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                print('Death rate: {0}\n'.format(self.death_rate))
                break

    def fired_masks_update(self):
        ntotal_fired_weights = 0.0
        ntotal_weights = 0.0
        layer_fired_weights = {}
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                self.fired_masks[name] = self.masks[name].data.byte() | self.fired_masks[name].data.byte()
                ntotal_fired_weights += float(self.fired_masks[name].sum().item())
                ntotal_weights += float(self.fired_masks[name].numel())
                layer_fired_weights[name] = float(self.fired_masks[name].sum().item())/float(self.fired_masks[name].numel())
                print('Layerwise percentage of the fired weights of', name, 'is:', layer_fired_weights[name])
        total_fired_weights = ntotal_fired_weights/ntotal_weights
        print('The percentage of the total fired weights is:', total_fired_weights)
        return layer_fired_weights, total_fired_weights


    def mask_diff(self):
        layer_stats = {}

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue

                # Initialize dictionary for the layer if it doesn't exist
                if name not in layer_stats:
                    layer_stats[name] = {
                        'pruned_mask_diff_ratio': 0,
                        'mask_diff_ratio': 0,
                        'pre_mask_diff_ratio': 0,
                    }

                if self.steps > 2 * self.prune_every_k_steps - 1:
                    num_params = self.init_masks[name].sum().item()
                    mask_diff = self.init_masks[name].sum().item() - (
                            self.init_masks[name].byte() & self.masks[name].byte()).sum().item()
                    pre_mask_diff = self.previous_masks[name].sum().item() - (
                            self.previous_masks[name].byte() & self.masks[name].byte()).sum().item()
                    pruned_mask_diff = self.pre_pruned_masks[name].sum().item() - (
                            self.pruned_masks[name].byte() & self.pre_pruned_masks[name].byte()).sum().item()

                    layer_stats[name]['mask_diff_ratio'] = mask_diff / num_params
                    layer_stats[name]['pre_mask_diff_ratio'] = pre_mask_diff / num_params
                    layer_stats[name]['pruned_mask_diff_ratio'] = pruned_mask_diff / num_params

        return layer_stats
