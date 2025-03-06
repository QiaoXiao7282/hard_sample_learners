import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import math, random
import string
from torch.utils.data import DataLoader, Dataset, Subset

def load_checkpoint(model, checkpoint_path):
    """
	Loads weights from checkpoint
	:param model: a pytorch nn student
	:param str checkpoint_path: address/path of a file
	:return: pytorch nn student with weights loaded from checkpoint
	"""
    model_ckp = torch.load(checkpoint_path)
    model.load_state_dict(model_ckp)
    return model

def compute_diff_masks(pre_masks, curr_masks):

    total_diff = 0
    val = ''
    for name in pre_masks.keys():
        ## compare pruned mask
        pre_masks_neg = pre_masks[name].data < 1.0
        pruned_masks_neg = curr_masks[name].data < 1.0
        comp_1 = pre_masks[name].data.byte() & curr_masks[name].data.byte()
        comp_2 = pre_masks_neg.byte() & pruned_masks_neg.byte()
        diff = pre_masks[name].numel() - (comp_1.sum().item() + comp_2.sum().item())
        total_diff += diff

        val = '{0}: diff: {1}'.format(name, diff)
        print(val)

    return total_diff

def generate_random_str(randomlength):
    str_list = random.sample(string.digits+string.ascii_letters, randomlength)
    random_str = ''.join(str_list)
    return random_str

def weights_init(m):
    # print('=> weights init')
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.normal_(m.weight, 0, 0.1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        # nn.init.xavier_normal(m.weight)
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        # Note that BN's running_var/mean are
        # already initialized to 1 and 0 respectively.
        if m.weight is not None:
            m.weight.data.fill_(1.0)
        if m.bias is not None:
            m.bias.data.zero_()

class DatasetWithIndices(torch.utils.data.Dataset):
    def __init__(self, sub_dataset):
        self.sub_dataset = sub_dataset

    def __getitem__(self, index):
        data, target = self.sub_dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.sub_dataset)

class DatasetSplitter(torch.utils.data.Dataset):
    """This splitter makes sure that we always use the same training/validation split"""
    def __init__(self,parent_dataset,split_start=-1,split_end= -1):
        split_start = split_start if split_start != -1 else 0
        split_end = split_end if split_end != -1 else len(parent_dataset)
        assert split_start <= len(parent_dataset) - 1 and split_end <= len(parent_dataset) and     split_start < split_end , "invalid dataset split"

        self.parent_dataset = parent_dataset
        self.split_start = split_start
        self.split_end = split_end

    def __len__(self):
        return self.split_end - self.split_start


    def __getitem__(self,index):
        assert index < len(self),"index out of bounds in split_datset"
        return self.parent_dataset[index + self.split_start]


class GaussianNoise(object):
    """Add Gaussian noise to the image."""
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """__call__ method to add Gaussian noise to the input tensor."""
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


def get_cifar100_dataloaders(args, validation_split=0.0, max_threads=10, if_train_set=False):
    """Creates augmented train, validation, and test data loaders."""
    cifar_mean = (0.50707516, 0.48654887, 0.44091784)
    cifar_std = (0.26733429, 0.25643846, 0.27615047)
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

    trainset = torchvision.datasets.CIFAR100(root='{}/dataset'.format(args.root_path), train=True, download=True,
                                             transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='{}/dataset'.format(args.root_path), train=False, download=True,
                                            transform=transform_test)
    full_dataset_clr = datasets.CIFAR100('{}/dataset'.format(args.root_path), train=True, download=True,
                                             transform=transform_test)

    trainset_index = DatasetWithIndices(trainset)
    full_dataset_clr_index = DatasetWithIndices(full_dataset_clr)

    if validation_split > 0.0:
        split = int(np.floor((1.0 - validation_split) * len(trainset_index)))
        train_dataset = DatasetSplitter(trainset_index, split_end=split)
        train_dataset_clr = DatasetSplitter(full_dataset_clr_index, split_end=split)
        val_dataset = DatasetSplitter(trainset_index, split_start=split)

        valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            args.batch_size,
            num_workers=2,
            pin_memory=True)
    else:
        train_dataset = trainset_index
        train_dataset_clr = full_dataset_clr_index
        valid_loader = torch.utils.data.DataLoader(
            testset,
            args.batch_size,
            shuffle=False,
            num_workers=2)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    if not if_train_set:
        train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, num_workers=8, pin_memory=True, shuffle=True)
        return train_loader, valid_loader, test_loader

    return train_dataset, train_dataset_clr, valid_loader, test_loader


def get_cifar10_dataloaders(args, validation_split=0.0, max_threads=10, if_train_set=False):
    """Creates augmented train, validation, and test data loaders."""

    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std)
    ])

    full_dataset = datasets.CIFAR10('{}/dataset'.format(args.data_path), True, train_transform, download=True)
    test_dataset = datasets.CIFAR10('{}/dataset'.format(args.data_path), False, test_transform, download=True)
    full_dataset_clr = datasets.CIFAR10('{}/dataset'.format(args.data_path), True, test_transform, download=True)

    trainset_index = DatasetWithIndices(full_dataset)
    full_dataset_clr_index = DatasetWithIndices(full_dataset_clr)

    if validation_split > 0.0:
        split = int(np.floor((1.0-validation_split) * len(trainset_index)))
        train_dataset = DatasetSplitter(trainset_index,split_end=split)
        train_dataset_clr = DatasetSplitter(full_dataset_clr_index, split_end=split)
        val_dataset = DatasetSplitter(trainset_index,split_start=split)
        valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            args.batch_size,
            num_workers=2,
            pin_memory=True)
    else:
        train_dataset = trainset_index
        train_dataset_clr = full_dataset_clr_index
        valid_loader = torch.utils.data.DataLoader(
            test_dataset,
            args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True)

    if not if_train_set:
        train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, num_workers=8, pin_memory=True, shuffle=True)
        return train_loader, valid_loader, test_loader

    return train_dataset, train_dataset_clr, valid_loader, test_loader


def get_tinyimagenet_dataloaders_64(args, split_file=None, dataset=False, max_threads=2, if_train_set=False):

    data_dir = '{}/tiny-imagenet-200'.format(args.data_path_tiny)

    normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2302, 0.2265, 0.2262])

    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'val')

    if not split_file:
        split_file = '../sparse_robustness/npy_files/tiny-imagenet-train-val.npy'
    split_permutation = list(np.load(split_file))

    train_set = datasets.ImageFolder(train_path, transform=train_transform)
    # val_set = Subset(datasets.ImageFolder(train_path, transform=test_transform), split_permutation[90000:])
    test_set = datasets.ImageFolder(val_path, transform=test_transform)

    train_set.imgs.sort(key=lambda x: x[0])
    train_set.samples = train_set.imgs
    train_set.targets = [sample[1] for sample in train_set.samples]

    trainset_index = DatasetWithIndices(train_set)

    val_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    if not if_train_set:
        train_loader = torch.utils.data.DataLoader(trainset_index, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        return train_loader, val_loader, test_loader

    return trainset_index, val_loader, test_loader



def get_mnist_dataloaders(args, validation_split=0.0):
    """Creates augmented train, validation, and test data loaders."""
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    transform = transform=transforms.Compose([transforms.ToTensor(),normalize])

    full_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    valid_loader = None
    if validation_split > 0.0:
        split = int(np.floor((1.0-validation_split) * len(full_dataset)))
        train_dataset = DatasetSplitter(full_dataset,split_end=split)
        val_dataset = DatasetSplitter(full_dataset,split_start=split)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            args.batch_size,
            num_workers=8,
            pin_memory=True, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            args.test_batch_size,
            num_workers=2,
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

def seq_update(T_max, seq_set_loss, sequence, death_rate=0.0, data_remain=1.0, gradual_flag=False, first_up=False, args=None):

    seq_len = math.ceil(len(sequence)*data_remain)
    num_remove = math.ceil(seq_len*death_rate)
    # loss_np = torch.zeros([len(sequence), len(seq_set_loss[sequence[0]])])
    loss_np = torch.zeros([len(sequence), 1])
    for i, indx in enumerate(sequence):
        tmp_data = seq_set_loss[indx].item()
        loss_np[i] = tmp_data

    ## death
    num_remain = seq_len - num_remove
    # remain_index = np.argsort(loss_np)[-num_remain:]  ## 从小到大

    ## margin
    if args.query_str == 'margin':
        probs_sorted, idxs = loss_np.sort(descending=True)
        uncertainties = probs_sorted[:, 0] - probs_sorted[:, 1]

    ## entropy
    elif args.query_str == 'entropy':
        log_probs = torch.log(loss_np)
        uncertainties = (loss_np * log_probs).sum(1)

    # ## max confidence
    # if args.query_str == 'el2n' or args.query_str == 'grand' or args.query_str == 'el2n_ori':
    #     uncertainties = loss_np.max(1)[0]
    #
    # if args.query_str == 'min':
    #     uncertainties = loss_np.min(1)[0]

    ## random
    elif args.query_str == 'random':
        uncertainties = torch.rand(len(loss_np))

    else:
        uncertainties = loss_np[:, 0]

    if args.category == 'easy':
        remain_index = uncertainties.sort()[1][:num_remain]  ## 小到大
    else:
        remain_index = uncertainties.sort(descending=True)[1][:num_remain]

    remain_seq = sequence[remain_index]

    ## regrow
    if gradual_flag:
        num_grow = int(seq_len * data_remain) - num_remain
    else:
        num_grow = num_remove

    all_index = np.arange(T_max)
    rest_index = np.delete(all_index, sequence)

    sample_seq = np.random.choice(rest_index, num_grow, replace=True)
    new_seq = np.concatenate((remain_seq, sample_seq))

    print('Train Len: {}. Regrow Len: {} Final Len: {}\n'.format(seq_len, num_grow, len(new_seq)))

    return new_seq

def seq_update_all(args, seq_set_loss, sequence, data_remain=0.0, gradual_flag=True):

    seq_len = len(seq_set_loss)
    loss_np = torch.zeros([seq_len, len(seq_set_loss[sequence[0]])])  ## [num, classes]
    seq_all = np.arange(seq_len)
    for i, (indx, tmp_data) in enumerate(list(seq_set_loss.items())):
        loss_np[i] = tmp_data
        seq_all[i] = indx

    ## margin
    if args.query_strategy == 'margin':
        probs_sorted, idxs = loss_np.sort(descending=True)
        uncertainties = probs_sorted[:, 0] - probs_sorted[:, 1]

    ## entropy
    elif args.query_strategy == 'entropy':
        log_probs = torch.log(loss_np)
        uncertainties = (loss_np * log_probs).sum(1)

    ## least confidence
    elif args.query_strategy == 'confid':
        uncertainties = loss_np.max(1)[0]

    ## random
    elif args.query_strategy == 'random':
        uncertainties = torch.rand(len(loss_np))

    ##
    if gradual_flag:
        remain_num = int(seq_len * data_remain)
    else:
        remain_num = len(sequence)

    remain_index = uncertainties.sort()[1][:remain_num]
    new_seq = seq_all[remain_index]

    return new_seq


def forget_times(record_list):
    offset = 200000
    number = offset
    learned = False

    for i in range(record_list.shape[0]):

        if not learned:
            if record_list[i] == 1:
                learned = True
                if number == offset:
                    number = 0

        else:
            if record_list[i] == 0:
                learned = False
                number += 1

    return number

def false_times(record_list):

    number = (record_list == 0).sum()

    return number

def sorted_examples(example_wise_prediction, data_prune, data_rate, threshold=0, train_number=0):
    offset = 200000

    forgetting_events_number = np.zeros(example_wise_prediction.shape[0])
    for j in range(example_wise_prediction.shape[0]):
        tmp_data = example_wise_prediction[j, :]
        if tmp_data[0] < 0:
            forgetting_events_number[j] = -1  ## 有部分样本没有参与训练
        else:
            # forgetting_events_number[j] = forget_times(tmp_data)
            forgetting_events_number[j] = false_times(tmp_data)

    if data_prune == 'constant':
        print('* remain {} data'.format(data_rate))
        rest_number = int(train_number * data_rate)
    elif data_prune == 'zero_out':
        print('zero all unforgettable images out')
        rest_number = np.where(forgetting_events_number > threshold)[0].shape[0]
    else:
        print('error data_prune type')
        assert False

    # print('max forgetting times = {}'.format(np.max(forgetting_events_number)))
    selected_index = np.argsort(forgetting_events_number)[-rest_number:]

    return selected_index

def cosine_adjuster(epoch, epoch_num, start_value=0, end_value=0):

    e = epoch
    es = epoch_num
    lr = (start_value - end_value) * (np.cos(np.pi / 2.0 * e / es)) + end_value

    return lr

class TransformedSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        original_tuple = self.dataset[self.indices[idx]]
        if self.transform:
            x = self.transform(original_tuple[0])
            y = original_tuple[1]
            return x, y
        else:
            return original_tuple

    def __len__(self):
        return len(self.indices)

def plot_class_feature_histograms(args, model, device, test_loader, optimizer):
    if not os.path.exists('./results'): os.mkdir('./results')
    model.eval()
    agg = {}
    num_classes = 10
    feat_id = 0
    sparse = not args.dense
    model_name = 'alexnet'
    #model_name = 'vgg'
    #model_name = 'wrn'


    densities = None
    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx % 100 == 0: print(batch_idx,'/', len(test_loader))
        with torch.no_grad():
            #if batch_idx == 10: break
            data, target = data.to(device), target.to(device)
            for cls in range(num_classes):
                #print('=='*50)
                #print('CLASS {0}'.format(cls))
                model.t = target
                sub_data = data[target == cls]

                output = model(sub_data)

                feats = model.feats
                if densities is None:
                    densities = []
                    densities += model.densities

                if len(agg) == 0:
                    for feat_id, feat in enumerate(feats):
                        agg[feat_id] = []
                        #print(feat.shape)
                        for i in range(feat.shape[1]):
                            agg[feat_id].append(np.zeros((num_classes,)))

                for feat_id, feat in enumerate(feats):
                    map_contributions = torch.abs(feat).sum([0, 2, 3])
                    for map_id in range(map_contributions.shape[0]):
                        #print(feat_id, map_id, cls)
                        #print(len(agg), len(agg[feat_id]), len(agg[feat_id][map_id]), len(feats))
                        agg[feat_id][map_id][cls] += map_contributions[map_id].item()

                del model.feats[:]
                del model.densities[:]
                model.feats = []
                model.densities = []

    if sparse:
        np.save('./results/{0}_sparse_density_data'.format(model_name), densities)

    for feat_id, map_data in agg.items():
        data = np.array(map_data)
        #print(feat_id, data)
        full_contribution = data.sum()
        #print(full_contribution, data)
        contribution_per_channel = ((1.0/full_contribution)*data.sum(1))
        #print('pre', data.shape[0])
        channels = data.shape[0]
        #data = data[contribution_per_channel > 0.001]

        channel_density = np.cumsum(np.sort(contribution_per_channel))
        print(channel_density)
        idx = np.argsort(contribution_per_channel)

        threshold_idx = np.searchsorted(channel_density, 0.05)
        print(data.shape, 'pre')
        data = data[idx[threshold_idx:]]
        print(data.shape, 'post')

        #perc = np.percentile(contribution_per_channel[contribution_per_channel > 0.0], 10)
        #print(contribution_per_channel, perc, feat_id)
        #data = data[contribution_per_channel > perc]
        #print(contribution_per_channel[contribution_per_channel < perc].sum())
        #print('post', data.shape[0])
        normed_data = np.max(data/np.sum(data,1).reshape(-1, 1), 1)
        #normed_data = (data/np.sum(data,1).reshape(-1, 1) > 0.2).sum(1)
        #counts, bins = np.histogram(normed_data, bins=4, range=(0, 4))
        np.save('./results/{2}_{1}_feat_data_layer_{0}'.format(feat_id, 'sparse' if sparse else 'dense', model_name), normed_data)
        #plt.ylim(0, channels/2.0)
        ##plt.hist(normed_data, bins=range(0, 5))
        #plt.hist(normed_data, bins=[(i+20)/float(200) for i in range(180)])
        #plt.xlim(0.1, 0.5)
        #if sparse:
        #    plt.title("Sparse: Conv2D layer {0}".format(feat_id))
        #    plt.savefig('./output/feat_histo/layer_{0}_sp.png'.format(feat_id))
        #else:
        #    plt.title("Dense: Conv2D layer {0}".format(feat_id))
        #    plt.savefig('./output/feat_histo/layer_{0}_d.png'.format(feat_id))
        #plt.clf()


