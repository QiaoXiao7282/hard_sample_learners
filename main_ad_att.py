from __future__ import print_function

import os
import time
import argparse
import logging
import hashlib
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import random, math
import socket
import pickle
import wandb
from utils.utils import visual_score_dict, evaluate_el2n_and_grand, cal_el2n_score, setup_seed

import sparselearning
# from sparselearning.core_ste import Masking, CosineDecay, LinearDecay
from sparselearning.core import Masking, CosineDecay, LinearDecay, str2bool
import warnings
import torch.nn as nn
from adversarial_training.utils import setup_dataset_models, setup_dataset_models_eval, save_checkpoint
from advertorch.attacks import LinfPGDAttack, L2PGDAttack
from advertorch.context import ctx_noparamgrad

warnings.filterwarnings("ignore", category=UserWarning)
cudnn.benchmark = True
cudnn.deterministic = True

if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
logger = None

criterion = nn.CrossEntropyLoss()

def train(args, model, device, train_loader, optimizer, epoch, mask=None):

    if args.norm == 'linf':
        adversary = LinfPGDAttack(
            model, loss_fn=criterion, eps=args.train_eps, nb_iter=args.train_step, eps_iter=args.train_gamma,
            rand_init=args.train_randinit, clip_min=0.0, clip_max=1.0, targeted=False)
    elif args.norm == 'l2':
        adversary = L2PGDAttack(
            model, loss_fn=criterion, eps=args.train_eps, nb_iter=args.train_step, eps_iter=args.train_gamma,
            rand_init=args.train_randinit, clip_min=0.0, clip_max=1.0, targeted=False)

    train_loss = 0
    correct = 0
    total = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        if args.fp16: data = data.half()
        optimizer.zero_grad()

        # adv samples
        with ctx_noparamgrad(model):
            input_adv = adversary.perturb(data, target)

        adv_outputs = model(input_adv)
        loss = criterion(adv_outputs, target)

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        if mask is not None: mask.step()
        else: optimizer.step()

        train_loss += loss.item()
        _, predicted = adv_outputs.max(1)

        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Adversarial Loss: {:.6f}  Adversarial Accuracy: {}/{} ({:.3f}%)'.format(
                epoch, batch_idx * len(data), len(train_loader)*args.batch_size,
                100. * batch_idx / len(train_loader), loss.item(), correct, total, 100. * correct / float(total)))

    # training summary
    print('\n{}: Average Adversarial loss: {:.4f}, Adversarial Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Training summary', train_loss/batch_idx, correct, total, 100. * correct / float(total)))

    return correct / float(total), train_loss / batch_idx

def eval_bn(args, model, device, test_loader, epoch=0):
    model.eval()
    benign_loss = 0
    benign_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            benign_loss += loss.item()

            _, predicted = outputs.max(1)
            benign_correct += predicted.eq(targets).sum().item()

    benign_loss /= float(batch_idx)

    print('\n{}  at {} epoch: Benign Average loss: {:.4f}, Benign Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Benign Test evaluation', epoch, benign_loss, benign_correct, total, 100. * benign_correct / float(total)))


    return benign_correct / total, benign_loss


def eval_adv(args, model, device, test_loader, epoch=0):

    if args.norm == 'linf':
        adversary = LinfPGDAttack(
            model, loss_fn=criterion, eps=args.test_eps, nb_iter=args.test_step, eps_iter=args.test_gamma,
            rand_init=args.test_randinit, clip_min=0.0, clip_max=1.0, targeted=False)
    elif args.norm == 'l2':
        adversary = L2PGDAttack(
            model, loss_fn=criterion, eps=args.test_eps, nb_iter=args.test_step, eps_iter=args.test_gamma,
            rand_init=args.test_randinit, clip_min=0.0, clip_max=1.0, targeted=False)

    adv_loss = 0
    adv_correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        # adv samples
        input_adv = adversary.perturb(inputs, targets)
        # compute output
        with torch.no_grad():
            adv_outputs = model(input_adv)
            loss = criterion(adv_outputs, targets)

        adv_loss += loss.item()
        total += targets.size(0)
        _, predicted = adv_outputs.max(1)
        adv_correct += predicted.eq(targets).sum().item()

    adv_loss /= float(batch_idx)
    print('\n{} at {} epoch: Adv Average loss: {:.4f}, Adv Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Adv Test evaluation', epoch, adv_loss, adv_correct, total, 100. * adv_correct / float(total)))

    return adv_correct / total, adv_loss


def get_generalization_gap(args, model, device):
    # final
    train_loader, val_loader, test_loader, _ = setup_dataset_models_eval(args)

    final_train_ra, _ = eval_adv(args, model, device, train_loader)
    final_test_ra, _ = eval_adv(args, model, device, test_loader)

    final_gap = final_train_ra - final_test_ra

    final_sparsity = 1 - args.density

    print('* Model final train RA = {:.2f}, final test RA = {:.2f}'.format(final_train_ra, final_test_ra))
    print('* Model final GAP = {:.2f}, final sparsity = {:.2f}'.format(final_gap, final_sparsity))

    return final_gap, final_sparsity

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--multiplier', type=int, default=1, metavar='N',
                        help='extend training time by multiplier times')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--fp16', action='store_true', help='Run in fp16 mode.')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=15, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--optimizer', type=str, default='sgd', help='The optimizer to use. sdg, adam.')
    parser.add_argument('--lr_s', type=str, default='step', help='learning rate scheduler')
    parser.add_argument("--data", help="data to train", default= 'cifar10', type=str)
    parser.add_argument('--decay_frequency', type=int, default=25000)
    parser.add_argument('--l1', type=float, default=0.0)
    parser.add_argument('--valid_split', type=float, default=0.0)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--model', type=str, default='ResNet18') #vgg16, ResNet18, ResNet34, vgg19
    parser.add_argument('--l2', type=float, default=5.0e-4)
    parser.add_argument('--iters', type=int, default=1, help='How many times the model. Default=1')
    parser.add_argument('--save-features', action='store_true', help='Resumes a saved  to disk for plotting.')
    parser.add_argument('--bench', action='store_true', help='Enables the benchmarking speedups')
    parser.add_argument('--max-threads', type=int, default=10, help='How many threads to.')
    parser.add_argument('--gpu', type=int, default=4, help='device ids of gpus')

    parser.add_argument('--root_path', type=str, default='/data/snns/models/data_dst_diff')
    parser.add_argument("--data_dir", help='path for loading data', default='/data/snns/data/data_dst/data_bias', type=str)
    parser.add_argument('--data_path', type=str, default='/data/snns/data/sparse_robust')

    parser.add_argument("--save_model", type=str2bool, default=False)

    ## data parameters
    parser.add_argument('--task', type=str, default='dataadv')
    parser.add_argument('--p_data', type=float, default=1.0)
    parser.add_argument('--data_retrain', type=str2bool, default=False)

    # ##################################### Attack setting #################################################
    # parser.add_argument('--norm', default='l2', type=str, help='linf or l2')
    # parser.add_argument('--epsilon', default=8/255, type=float, help='epsilon of attack during testing')
    # parser.add_argument('--k', default=7, type=int, help='itertion number of attack during testing')
    # parser.add_argument('--alpha', default=2/255, type=float, help='step size of attack during testing')

    ########################## attack setting ##########################
    parser.add_argument('--norm', default='linf', type=str, help='linf or l2')
    parser.add_argument('--train_eps', default=8./255, type=float, help='epsilon of attack during training')
    parser.add_argument('--train_step', default=10, type=int, help='itertion number of attack during training')
    parser.add_argument('--train_gamma', default=2./255, type=float, help='step size of attack during training')
    parser.add_argument('--train_randinit', action='store_false', help='randinit usage flag (default: on)')
    parser.add_argument('--test_eps', default=8./255, type=float, help='epsilon of attack during testing')
    parser.add_argument('--test_step', default=20, type=int, help='itertion number of attack during testing')
    parser.add_argument('--test_gamma', default=2./255, type=float, help='step size of attack during testing')
    parser.add_argument('--test_randinit', action='store_false', help='randinit usage flag (default: on)')

    ## wandb
    parser.add_argument("--team_name", type=str, default='qiaoqian')
    parser.add_argument("--project_name", type=str, default='data_diff')
    parser.add_argument("--experiment_name", type=str, default='test')
    # parser.add_argument("--group_name", type=str, default='seeds')
    parser.add_argument("--wandb", type=str2bool, default=False)

    # ITOP settings
    sparselearning.core.add_sparse_args(parser)
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda:{}'.format(args.gpu) if use_cuda else "cpu")
    print(args)

    setup_seed(args.seed)

    if not args.sparse:
        args.density = 1.0
        args.method = 'dense'

    # if args.fix:
    #     args.method = 'fix'

    if args.density == 1.0:
        args.sparse = False
        args.method = 'dense'

    if args.method == 'finetune':
        args.fix = True
        args.sparse_init = 'global_magnitude'

    if args.method == 'lth':
        args.fix = True
        args.sparse_init = 'lottery_ticket'

    if args.method == 'uniform':
        args.fix = True
        args.sparse_init = 'uniform'

    if args.method == 'fix':
        args.fix = True

    if args.method == 'GMP':
        args.sparse_init = 'GMP'

    if args.method == 'snip':
        args.sparse_init = 'snip'
        args.fix = True

    if args.method == 'pre_erk':
        args.sparse_init = 'ERK'
        args.fix = True

    if args.method == 'static':
        args.sparse_init = 'ERK'
        args.fix = True


    models_seed = 'models-seed{}'.format(args.seed)
    path = '{}/{}/{}'.format(args.root_path, args.task, models_seed)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    filename = "{}-{}-{}-{}-{}-{}-{}-{}-{}".format(args.task, args.data, args.model, args.method, args.density, args.sparse, args.p_data, args.norm, args.epochs)
    groupname = "{}-{}-{}-{}-{}-{}-{}-{}-{}".format(args.task, args.data, args.model, args.method, args.density, args.sparse,
                                                    args.p_data, args.norm, args.epochs)
    save_path = '{}/{}'.format(path, filename)

    if args.wandb:
        wandb.init(config=args,
                   project=args.project_name,
                   entity=args.team_name,
                   notes=socket.gethostname(),
                   name="{} {} seed{}".format(args.experiment_name, args.data, args.seed),
                   group=groupname,
                   job_type="{}-{}-{}-{}".format(args.data, args.model, 1-args.density, args.update_frequency),
                   reinit=True)

    all_result = {}
    all_result['train_acc'] = []
    all_result['val_bn'] = []
    all_result['test_bn'] = []
    all_result['val_adv'] = []
    all_result['test_adv'] = []

    for i in range(args.iters):
        train_loader, val_loader, test_loader, model = setup_dataset_models(args)
        model = model.to(device)

        print(model)
        print('=' * 60)
        print(args.model)
        print('=' * 60)
        print('=' * 60)
        print('Prune mode: {0}'.format(args.death))
        print('Growth mode: {0}'.format(args.growth))
        print('Redistribution mode: {0}'.format(args.redistribution))
        print('=' * 60)

        model_lth = None
        if args.method == 'lth' or args.method == 'finetune' or args.method == 'pre_erk':
            densename = "{}-{}-{}-{}-{}-{}-{}-{}-{}".format(args.task, args.data, args.model, 'dense', '1.0',
                                                            False, args.p_data, args.norm, args.epochs)

            filepath = '{}/{}_{}'.format(path, densename, 'checkpoint_adv_final.pth.tar')
            checkpoint = torch.load(filepath, map_location='cpu')
            print("loading checkpoint ...")
            if args.method == 'finetune' or args.method == 'pre_erk':
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model_lth = copy.deepcopy(model)
                model.load_state_dict(checkpoint['state_dict_init'])
                model_lth.load_state_dict(checkpoint['state_dict'])

        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.l2, nesterov=True)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.l2)
        else:
            print('Unknown optimizer: {0}'.format(args.optimizer))
            raise Exception('Unknown optimizer.')

        if args.method == 'dense':
            init_model = copy.deepcopy(model)

        lr_scheduler = None
        if args.lr_s == 'step':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[int(args.epochs / 2) * args.multiplier,
                                                                int(args.epochs * 3 / 4) * args.multiplier], last_epoch=-1)
        elif args.lr_s == 'cos':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * args.multiplier,
                                                                      args.lr / 100, last_epoch=-1)

        mask = None
        if args.sparse:
            epoch_end = args.epochs  ## args.epochs, args.final_prune_epoch
            T_max = int(len(train_loader)*(epoch_end*args.multiplier))
            decay = CosineDecay(args.death_rate, T_max)
            mask = Masking(optimizer, death_rate=args.death_rate, death_mode=args.death, death_rate_decay=decay,
                           growth_mode=args.growth, redistribution_mode=args.redistribution, train_loader=train_loader,
                           device=device, args=args)
            mask.add_module(model, sparse_init=args.sparse_init, density=args.density, model_lth=model_lth)


        bn_best_acc = 0.0
        adv_best_acc = 0.0
        for epoch in range(1, args.epochs*args.multiplier + 1):
            t0 = time.time()

            train_acc, train_loss = train(args, model, device, train_loader, optimizer, epoch, mask)

            all_result['train_acc'].append(train_acc)

            if args.wandb:
                wandb.log({'train loss': train_loss, 'train acc': train_acc}, step=epoch)

            lr_scheduler.step()
            benign_correct_val, benign_loss_val = eval_bn(args, model, device, val_loader, epoch=epoch)
            benign_correct_test, benign_loss_test = eval_bn(args, model, device, test_loader, epoch=epoch)
            adv_correct_val, adv_loss_val = eval_adv(args, model, device, val_loader, epoch=epoch)
            adv_correct_test, adv_loss_test = eval_adv(args, model, device, test_loader, epoch=epoch)

            all_result['val_bn'].append(benign_correct_val)
            all_result['test_bn'].append(benign_correct_test)
            all_result['val_adv'].append(adv_correct_val)
            all_result['test_adv'].append(adv_correct_test)

            if args.method == 'dense':
                checkpoint_state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'state_dict_init': init_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                    'result': all_result
                }

            else:
                checkpoint_state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                    'result': all_result
                }

            ## save best checpoint
            if adv_correct_val > adv_best_acc:
                adv_best_acc = adv_correct_val
                if args.save_model:
                    print('Saving Best Model')
                    save_checkpoint(checkpoint_state, save_path, filename='checkpoint_adv_best.pth.tar')

            if benign_correct_val > bn_best_acc:
                bn_best_acc = benign_correct_val
                if args.save_model:
                    print('Saving Best Model')
                    save_checkpoint(checkpoint_state, save_path, filename='checkpoint_bn_best.pth.tar')

            if args.wandb:
                wandb.log({'benign_test_loss': benign_loss_test, 'adv_test_loss': adv_loss_test}, step=epoch)
                wandb.log({'benign_test_correct': benign_correct_test, 'adv_test_correct': adv_correct_test}, step=epoch)
                wandb.log({'benign_val_correct': benign_correct_val, 'adv_val_correct': adv_correct_val}, step=epoch)


            print({'benign_test_loss': benign_loss_test, 'adv_test_loss': adv_loss_test})
            print({'benign_test_correct': benign_correct_test, 'adv_test_correct': adv_correct_test, 'bn_best': bn_best_acc, 'adv_best': adv_best_acc})

            print('Current learning rate: {}. Time taken for epoch: {:.2f} seconds\n'.format(
                           optimizer.param_groups[0]['lr'], time.time() - t0))

        if args.save_model:
            print('Saving Final Model')
            save_checkpoint(checkpoint_state, save_path, filename='checkpoint_adv_final.pth.tar')

        final_gap, _ = get_generalization_gap(args, model, device)
        print({'final gap': final_gap})
        if args.wandb:
            wandb.log({'final gap': final_gap}, step=args.epochs)

if __name__ == '__main__':
   main()
