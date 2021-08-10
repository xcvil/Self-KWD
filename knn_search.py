import argparse
import builtins
import os
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

import moco.builder as builder

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
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default=None, type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--save-dir', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')

# moco specific configs:
parser.add_argument('--bimoco', action='store_true',
                    help='use two branches MoCo')
parser.add_argument('--bimoco-gamma', default=0.5, type=float,
                    help='fraction of MoCo v2 loss')
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--mixup', action='store_true',
                    help='use mixup data augmentation')
parser.add_argument('--aug-color-only', action='store_true',
                    help='use only color data augmentation')
parser.add_argument('--aug-geo', action='store_true',
                    help='use only geometric data augmentation')
parser.add_argument('--geo-plus', action='store_true',
                    help='use only geometric data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

# MixUp augmentation configs:
parser.add_argument('--mixup-p', default=None, type=float,
                    help='the prob to apply a mixup aug in a certain iteration')
parser.add_argument('--replace', action='store_true',
                    help='whether replace the original loss with mixup loss or not')
parser.add_argument('--rui', action='store_true',
                    help='use Rui method')

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
    path = os.path.join(args.save_dir, "config.json")
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
    # Simply call main_worker function
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.bimoco:
        model = builder.BiMoCo(models.__dict__[args.arch],
                               args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    else:
        model = builder.MoCo(models.__dict__[args.arch],
                             args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = torch.nn.parallel.DataParallel(model, device_ids=[args.gpu])
    else:
        model = model.cuda()


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer = checkpoint['optimizer']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

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
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2,
                                              pin_memory=True)

    instfe, labelfe = encode(test_loader, model.module.encoder_q, args)
    np.save(args.save_dir + 'inst_feat.npy', instfe)
    np.save(args.save_dir + 'label.npy', labelfe)

    # logging
    results = {'knn-k': [], 'test_acc@1': []}

    for i in range(0,600):
        args.knn_k += 1
        test_acc_1 = test(model.module.encoder_q, memory_loader, test_loader, i, args)
        results['knn-k'].append(args.knn_k)
        results['test_acc@1'].append(test_acc_1)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, i + 2))
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
                'Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, 600, total_top1 / total_num * 100))

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

def encode(train_loader, model, args):
    # switch to train mode
    model.eval()
    num_data = len(train_loader)
    inst_feat = np.zeros((num_data, 128)) # store the features
    label_list = np.zeros((num_data,))

    for i, (images, labels) in enumerate(train_loader):
        # measure data loading time

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        # compute output
        feature = model(images)
        feature = F.normalize(feature, dim=1)
        instDis = feature.cpu().data.numpy()
        inst_feat[i] = instDis
        label_list[i] = labels
        print('encoding image {} is finished'.format(str(i)))

    return inst_feat, label_list

if __name__ == '__main__':
    main()