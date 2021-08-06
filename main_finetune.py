#!/usr/bin/env python
# thanks to https://github.com/bearpaw/pytorch-classification.git
import argparse
import builtins
import os
import random
import shutil
import time
import math
import warnings

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

import datetime

import moco.loader as loader
import moco.builder as builder

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
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--gamma', '--learning-rate-decay', default=0.1, type=float,
                    metavar='GAMMA', help='learning rate decay coefficient', dest='gamma')
parser.add_argument('--schedule', default=[120, 160], nargs='+', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--save-dir', default='output/', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')
parser.add_argument('--pretrained', default=None, type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

parser.add_argument('--mixup', action='store_true',
                    help='both geo and color for finetuning because the pretrained model is geo- and color- invariant')

best_acc1 = 0


def main():
    args = parser.parse_args()

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
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    global best_acc1
    args.gpu = gpu

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=2)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            epoch = checkpoint['epoch']

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            print("Missed keys: ", msg.missing_keys)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}' in epoch {}".format(args.pretrained, epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

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
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}) with acc {}"
                  .format(args.resume, checkpoint['epoch'], best_acc1))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'test')
    print("valdir is: {}".format(valdir))
    normalize = transforms.Normalize(mean=[0.33872248711331276, 0.46409005915048474, 0.5697602907932671],
                                     std=[0.1060011105334372, 0.10909067324153679, 0.12906063885055458])

    if args.mixup:
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.RandomRotation([90, 90]),
                ], p=0.5),
                transforms.RandomApply([
                    transforms.RandomRotation([180, 180]),
                ], p=0.5),
                transforms.ToTensor(),
                normalize
            ]))
    else:
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]))

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            # transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    train_class_id = train_dataset.class_to_idx
    val_class_id = val_dataset.class_to_idx
    print('The class index for training is {}'.format(train_class_id))
    print('The class index for validating is {}'.format(val_class_id))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args, train_loader, val_class_id)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        # if epoch in [args.start_epoch, 0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75] or epoch>=80:
        #     acc1 = validate(val_loader, model, criterion, args, train_loader)
        # else:
        #     acc1 = best_acc1
        acc1, prec, recall, f1 = validate(val_loader, model, criterion, args, train_loader, val_class_id)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
            'precision' : prec,
            'recall' : recall,
            'F1-score' : f1,
            'current_acc1' : acc1
        }, is_best, save_dir=args.save_dir, epoch=epoch)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1, ))[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, args)


def validate(test_loader, model, criterion, args, train_loader, class_id):
    train_labels = torch.tensor(train_loader.dataset.targets).cuda(args.gpu)
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    precisions = AverageMeter('precision', ':6.2f')
    recalls = AverageMeter('recall', ':6.2f')
    f1_scores = AverageMeter('f1_score', ':6.2f')
    labels, label_counts = train_labels.unique(return_counts=True)
    assert torch.all(labels == torch.arange(labels.size(0), device=labels.device))

    progress = ProgressMeter(
        len(test_loader),
        [batch_time, losses, top1, precisions, recalls, f1_scores],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(test_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1, ))[0]
            precision, recall, F1 = f1_score(output, target, class_id)

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            precisions.update(precision.item(), images.size(0))
            recalls.update(recall.item(), images.size(0))
            f1_scores.update(F1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                args.current_lr = 0
                progress.display(i, args)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Precisions {precision.avg:.3f} Recalls {recall.avg:.3f} F1-score {F1.avg:.3f}'
              .format(top1=top1, precision=precisions, recall=recalls, F1=f1_scores))

    return top1.avg, precisions.avg, recalls.avg, f1_scores.avg


def save_checkpoint(state, is_best, filename='checkpoint_linear.pth.tar', save_dir='output/', epoch=0):
    filename = os.path.join(save_dir,filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_dir,'model_best_linear.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if n > 0:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, args):
        time = str(datetime.datetime.now())
        prefix = time + ' lr: {:.4f}\t'.format(args.current_lr) + self.prefix
        entries = [prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        message = '\t'.join(entries)
        print(message)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= args.gamma if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    args.current_lr = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True) # indices size: [batch_size, 5]
        pred = pred.t() # indices size: [5, batch_size]
        correct = pred.eq(target.view(1, -1).expand_as(pred)) #return True or False, size same as pred

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def f1_score(output, target, class_id):
    with torch.no_grad():
        pos_idx = class_id['foreground']
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))
        sum_pred_pos = (pred == pos_idx).cpu().sum(dim=1)
        sum_label_pos = (target.view(1, -1) == pos_idx).cpu().sum(dim=1)
        sum_ture_pos = ((pred+target.view(1, -1)) == 2*pos_idx).cpu().sum(dim=1)

        precision = (sum_ture_pos / sum_pred_pos) * 100
        recall = (sum_ture_pos/sum_label_pos)*100
        F1 = 2 * recall * precision / (recall + precision)

        return precision, recall, F1


if __name__ == '__main__':
    main()
