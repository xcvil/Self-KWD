import argparse
import builtins
import math
import os
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import json

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--save-dir', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')

# options for mix precision training
parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')

parser.add_argument('--pretrained', default=None, type=str,
                    help='path to moco pretrained checkpoint')
# knn monitor
parser.add_argument('--knn-k', default=20, type=int, help='k in kNN monitor')
parser.add_argument('--knn-t', default=0.1, type=float, help='softmax temperature in kNN monitor; could be different with moco-t')
parser.add_argument('--knn-data', default='', type=str, metavar='PATH',
                    help='path to dataset of KNN')

def main():
    global args
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    path = os.path.join(args.save_dir, "config_encoding.json")
    with open(path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print("Full config saved to {}".format(path))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    print('there is/are {} GPUs per nodes'.format(ngpus_per_node))
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=2)
    print(model)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)

    # optionally load from a pretrained
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            if args.gpu is None:
                checkpoint = torch.load(args.pretrained)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.pretrained, map_location=loc)

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}' in epoch {}".format(args.pretrained, epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    model = nn.Sequential(*list(model.children())[:-1])

    cudnn.benchmark = True

    memdir = os.path.join(args.knn_data, 'train')
    testdir = os.path.join(args.knn_data, 'test')
    normalize = transforms.Normalize(mean=[0.34098161014906836, 0.47044207777359126, 0.5797972380147923],
                                     std=[0.10761384273454896, 0.11021859651496183, 0.12975552642180524])

    test_aug = [
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        normalize
    ]
    memory_dataset = datasets.ImageFolder(memdir, transforms.Compose(test_aug))
    memory_loader = torch.utils.data.DataLoader(memory_dataset, batch_size=args.batch_size, shuffle=False,
                                                num_workers=2, pin_memory=True)

    test_dataset = datasets.ImageFolder(testdir, transforms.Compose(test_aug))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                              pin_memory=True)

    # logging
    results = {'knn-k': [], 'test_acc@1': []}

    for i in range(600):
        args.knn_k += 1
        test_acc_1 = test(model.module.encoder_q, memory_loader, test_loader, epoch, args)
        results['knn-k'].append(args.knn_k)
        results['test_acc@1'].append(test_acc_1)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(args.start_epoch + 1, epoch + 2))
        data_frame.to_csv(args.save_dir + 'log.csv', index_label='epoch')

# test using a knn monitor
def test(model, memory_data_loader, test_data_loader, epoch, args):
    model.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_num, feature_bank = 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = model(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = model(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_description(
                'Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, total_top1 / total_num * 100))

    return total_top1 / total_num * 100


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels