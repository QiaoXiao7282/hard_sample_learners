import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from functorch import make_functional_with_buffers, vmap, grad
import copy
import numpy as np
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import torchvision
import torchvision.transforms as transforms
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def compute_el2n_score(unnormalized_model_outputs, targets):
  """
  The EL2N score the error-L2 norm score, ie the Brier score of a single sample with its target (one-hot) label.
  :param unnormalized_model_outputs: BXC the unnormalized model outputs, ie the logits
  :param targets: BxC the one-hot target labels
  :return:
      a tensor of shape B with the EL2N score for each sample
  """
  # compute the softmax of the unnormalized model outputs
  softmax_outputs = F.softmax(unnormalized_model_outputs, dim=1)
  # compute the squared L2 norm of the difference between the softmax outputs and the target labels
  el2n_score = torch.sum((softmax_outputs - targets) ** 2, dim=1)
  return el2n_score

def visual_score_dict(score_dict, save_path, epoch, epoch_sort, args, sorted_keys_in=None):

    if epoch == epoch_sort:
        sorted_keys = sorted(score_dict, key=score_dict.get, reverse=True)  ##大到小
        # sorted_keys = sorted(score_dict, key=lambda k: score_dict[k], reverse=True)
    else:
        sorted_keys = sorted_keys_in

    sorted_values = [score_dict[key].item() for key in sorted_keys]

    # 柱状图
    print('***plotting score value***')
    plt.bar(range(len(sorted_values)), sorted_values)
    # plt.xticks(range(len(sorted_values)), sorted_keys)
    plt.xlabel("Num Pics")
    plt.ylabel("Score Value")
    plt.title("Bar Plot of Score Values: {}".format(epoch))
    plt.savefig('{}/{}_{}_{}.png'.format(save_path, args.data, args.model, epoch), format='png', dpi=200)
    plt.close()

    return copy.deepcopy(sorted_keys)

def evaluate_el2n_and_grand(net, epoch, args, device, train_set, sequence=None):

    scores_dict = {}
    if args.data == 'cifar10':
        num_classes = 10
    if args.data == 'cifar100':
        num_classes = 100

    if args.query_str == 'el2n':
        print("Evaluating EL2N scores...")

        image_number = sequence.shape[0]
        trainset_permutation_inds = sequence

        batch_size = args.batch_size
        iteration_steps = int(image_number / batch_size)
        if (image_number / batch_size) > iteration_steps:
            iteration_steps += 1

        with torch.no_grad():
            for batch_idx in range(iteration_steps):
                batch_start_ind = batch_idx * batch_size
                batch_end_ind = min(batch_start_ind + batch_size, image_number)

                # get batch inputs and targets, transform them appropriately
                batch_inds = trainset_permutation_inds[batch_start_ind:batch_end_ind]

                transformed_trainset = []
                labels = []
                for ind in batch_inds:
                    img, label = train_set.__getitem__(ind)
                    transformed_trainset.append(img)
                    labels.append(label)

                data = torch.stack(transformed_trainset)
                target = torch.LongTensor(labels)
                inputs, targets = data.to(device), target.to(device)

                output = net(inputs)
                prob = F.softmax(output, dim=1)
                one_hot_target = F.one_hot(targets, num_classes=num_classes)
                el2n_scores = torch.sum((prob - one_hot_target.float()).detach().cpu() ** 2, dim=1)

                for j, index in enumerate(batch_inds):
                    scores_dict[index] = [el2n_scores[j], target[j]]

    if args.query_str == 'grand':
        fmodel, params, buffers = make_functional_with_buffers(net)

        fmodel.eval()

        def compute_loss_stateless_model(params, buffers, sample, target):
          batch = sample.unsqueeze(0)
          targets = target.unsqueeze(0)

          predictions = fmodel(params, buffers, batch)
          loss = F.cross_entropy(predictions, targets)
          return loss

        ft_compute_grad = grad(compute_loss_stateless_model)
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

        print("Evaluating GRAND scores...")
        grad_score_dict = {}

        image_number = sequence.shape[0]
        trainset_permutation_inds = sequence

        batch_size = args.batch_size
        iteration_steps = int(image_number / batch_size)
        if (image_number / batch_size) > iteration_steps:
            iteration_steps += 1

        for batch_idx in range(iteration_steps):
            batch_start_ind = batch_idx * batch_size
            batch_end_ind = min(batch_start_ind + batch_size, image_number)

            # get batch inputs and targets, transform them appropriately
            batch_inds = trainset_permutation_inds[batch_start_ind:batch_end_ind]

            transformed_trainset = []
            labels = []
            for ind in batch_inds:
                img, label = train_set.__getitem__(ind)
                transformed_trainset.append(img)
                labels.append(label)

            data = torch.stack(transformed_trainset)
            target = torch.LongTensor(labels)

            inputs, targets = data.to(device), target.to(device)
            ft_per_sample_grads = ft_compute_sample_grad(params, buffers, inputs, targets)

            squared_norm = 0
            for param_grad in ft_per_sample_grads:
                squared_norm += param_grad.flatten(1).square().sum(dim=-1)
            grad_norms = squared_norm.detach().cpu() ** 0.5
            for j, index in enumerate(batch_inds):
                scores_dict[index] = [grad_norms[j], target[j]]

    if args.query_str == 'uncertainty':
        print("Evaluating Uncertainty scores...")

        image_number = sequence.shape[0]
        trainset_permutation_inds = sequence

        batch_size = args.batch_size
        iteration_steps = int(image_number / batch_size)
        if (image_number / batch_size) > iteration_steps:
            iteration_steps += 1

        with torch.no_grad():
            for batch_idx in range(iteration_steps):
                batch_start_ind = batch_idx * batch_size
                batch_end_ind = min(batch_start_ind + batch_size, image_number)

                # get batch inputs and targets, transform them appropriately
                batch_inds = trainset_permutation_inds[batch_start_ind:batch_end_ind]

                transformed_trainset = []
                labels = []
                for ind in batch_inds:
                    img, label = train_set.__getitem__(ind)
                    transformed_trainset.append(img)
                    labels.append(label)

                data = torch.stack(transformed_trainset)
                target = torch.LongTensor(labels)
                inputs, targets = data.to(device), target.to(device)

                output = net(inputs)
                prob = F.softmax(output.clone().detach(), dim=1)
                target_prob = prob[torch.arange(len(target)), target]

                for j, index in enumerate(batch_inds):
                    scores_dict[index] = [target_prob[j].cpu().item(), target[j]]

    return scores_dict

def cal_el2n_score(output, target, args, all_correct_old=None):

    ## cal EL2N
    if args.data == 'cifar10':
        num_classes = 10
    if args.data == 'cifar100':
        num_classes = 100

    if args.query_str == 'el2n_ori' or args.query_str == 'random':
        prob = F.softmax(output.clone().detach(), dim=1)
        one_hot_target = F.one_hot(target, num_classes=num_classes)
        scores = torch.sum((prob - one_hot_target.float()).detach().cpu() ** 2, dim=1)

    elif args.query_str == 'forget':
        maxk = 1
        _, pred = output.clone().detach().topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        all_correct_epoch = correct.cpu().numpy().astype(int)
        scores = all_correct_epoch

    elif args.query_str == 'uncertainty':
        prob = F.softmax(output.clone().detach(), dim=1)
        scores = prob[torch.arange(len(target)), target]

    else:
        scores = F.cross_entropy(output.clone().detach(), target, reduction='none').detach().cpu()  ## loss value

    return scores



def kernel_remain_ratio(model, masks):
    act_kernel_ratio = {}
    for name, tensor in model.named_parameters():
        if name not in masks:
            continue
        mask = masks[name]

        if mask.dim() >= 4:
            mask_sum = mask.sum(dim=[2, 3])
            nonzero_ratio = (mask_sum > 0).sum().item()/mask_sum.numel()
            act_kernel_ratio[name] = round(nonzero_ratio, 4)

    return act_kernel_ratio

def _torch_entropy10(probs):
    e = 0
    for p in probs:
        e += p * torch.log10(p)
    return -e.item()


def _entropy_max_threshold(n):
    l, x0, k, b = (1.2618047, 2.30436435, 0.88767525, -0.31050834)  # min distribution
    return l / (1 + np.exp(-k * (np.log2(n) - x0))) + b


def _svd_variance_ratio(x):
    s = torch.linalg.svd((x - x.mean(dim=0)), full_matrices=False).S
    variance = s ** 2 / (len(x) - 1)
    den = variance.sum()
    if den == 0:
        den = 1
    return variance / den

def variance_entropy(w, mask= None):

    assert (
        len(w.shape) == 4
    ), "w must be a tensor of shape (out_channels, in_channels, kernel_size, kernel_size)"

    w = w.view(w.shape[0] * w.shape[1], w.shape[2] * w.shape[3])
    if mask is not None:
        # w = w[mask]
        idx = mask.nonzero().squeeze()
        sel_w = torch.index_select(w, dim=0, index=idx)
    else:
        sel_w = w

    n = sel_w.shape[0]
    ratio = _svd_variance_ratio(sel_w)
    entropy = 0.0
    if ratio.sum() != 0:
        entropy = _torch_entropy10(ratio)
    return entropy, entropy /_entropy_max_threshold(n)

def sparsity(w, sparsity_eps=0.01):

    assert (
        len(w.shape) == 4
    ), "w must be a tensor of shape (out_channels, in_channels, kernel_size, kernel_size)"

    w = w.view(w.shape[0] * w.shape[1], w.shape[2] * w.shape[3])
    n = w.shape[0]
    t = abs(w).max().item() * sparsity_eps

    new_layer = torch.zeros_like(w)
    new_layer[abs(w) > t] = 1
    sparse_mask = (new_layer.sum(dim=1) == 0)
    sparsity = sparse_mask.sum() / n
    return sparsity.item(), sparse_mask

def measure_conv_weight_quality(w, sparsity_eps):

    assert (
        len(w.shape) == 4
    ), "w must be a tensor of shape (out_channels, in_channels, kernel_size, kernel_size)"

    info_dict = {}
    n = w.shape[0] * w.shape[1]

    with torch.no_grad():
        sparsity_ratio, sparse_mask = sparsity(w, sparsity_eps)
        ve, ve_norm = variance_entropy(w, mask=~sparse_mask)
        # ve_clean, ve_clean_norm = variance_entropy(w, mask=~sparse_mask)

        info_dict["n"] = n
        info_dict["sparsity"] = sparsity_ratio
        info_dict["variance_entropy"] = ve
        # info_dict["variance_entropy_norm"] = ve_norm
        # info_dict["variance_entropy_clean"] = ve_clean
        # info_dict["variance_entropy_clean_norm"] = ve_clean_norm

    return info_dict, ve


def measure_layer_quality(conv_layer, sparsity_eps):
    w = conv_layer.weight.detach()
    info_dict = measure_conv_weight_quality(w, sparsity_eps)
    return info_dict


def measure_quality(model, sparsity_eps=0.01):
    info_dict = {}
    entropy_sum = 0
    for name, conv_layer in filter(
        lambda x: type(x[1]) == torch.nn.Conv2d and x[1].kernel_size == (3, 3),
        model.named_modules(),
    ):
        info_dict[name], entropy_ve = measure_layer_quality(conv_layer, sparsity_eps)
        entropy_sum += entropy_ve

    return info_dict, entropy_sum

def effective_filter_size(model, threshold=0.0):

    filter_size_count = {i: 0 for i in range(0, 10)}
    filter_num = 0
    filter_num_l = []
    for name, conv_layer in filter(
            lambda x: type(x[1]) == torch.nn.Conv2d and x[1].kernel_size == (3, 3),
            model.named_modules(),
    ):
        weight = conv_layer.weight.data.detach()
        mask = (torch.abs(weight) > threshold).float()
        filter_size = mask.sum(dim=[2, 3])
        for i in range(0, 10):
            filter_size_count[i] += (filter_size == i).sum().item()
        filter_num += filter_size.numel()
        filter_num_l.append(filter_size.flatten())

    filter_size_ratio = {key: filter_size_count[key] / filter_num for key in filter_size_count.keys()}

    return filter_size_ratio, filter_num_l


def save_to_image(train_loader, data_name='cifar100'):

    if data_name == 'cifar100':
        cifar_mean = (0.4914, 0.4822, 0.4465)
        cifar_std = (0.2023, 0.1994, 0.2010)
    elif data_name == 'tinyimagenet':
        cifar_mean = (0.4802, 0.4481, 0.3975)
        cifar_std = (0.2302, 0.2265, 0.2262)

    label_list = [11, 22, 31, 43, 52]
    label_to_images = {label: [] for label in label_list}

    # Gather images for each label
    for batch_images, batch_labels, index in train_loader:
        for img, label in zip(batch_images, batch_labels):
            if label.item() in label_list:
                label_to_images[label.item()].append(img)

    def denormalize(img):
        img = img.permute(1, 2, 0)  # (C, H, W) to (H, W, C)
        for i in range(3):
            img[:, :, i] = img[:, :, i] * cifar_std[i] + cifar_mean[i]
        return img

    # Display the images
    fig, axs = plt.subplots(len(label_list), 10, figsize=(15, 7.5))
    for ax_row, label in zip(axs, label_list):
        for ax, img in zip(ax_row, label_to_images[label][:10]):  # Displaying at most 10 images per label
            ax.imshow(denormalize(img))
            ax.set_title(label)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig("./results/batch_images_with_labels.png")






