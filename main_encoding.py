#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import warnings
import numpy as np

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
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
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

    # Data loading code
    traindir = args.data
    normalize = transforms.Normalize(mean=[0.34098161014906836, 0.47044207777359126, 0.5797972380147923],
                                     std=[0.10761384273454896, 0.11021859651496183, 0.12975552642180524])

    augmentation = [
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        normalize
    ]

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(augmentation))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False)

    instfe, labelfe = encode(train_loader, model, args)
    np.save(args.save_dir+'inst_feat.npy', instfe)
    np.save(args.save_dir + 'label.npy', labelfe)


def encode(train_loader, model, args):
    # switch to train mode
    model.eval()
    num_data = len(train_loader)
    inst_feat = np.zeros((num_data, 2048, 1, 1)) # store the features
    label_list = np.zeros((num_data,))

    for i, (images, labels) in enumerate(train_loader):
        # measure data loading time

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        # compute output
        outputs = model(images)
        instDis = outputs.cpu().data.numpy()
        inst_feat[i] = instDis
        label_list[i] = labels
        print('encoding image {} is finished'.format(str(i)))

    return inst_feat, label_list

if __name__ == '__main__':
    main()
