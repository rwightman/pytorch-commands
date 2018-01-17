import argparse
import csv
import os
import time
import numpy as np
from collections import OrderedDict
from datetime import datetime

import dataset
from models import model_factory
from lr_scheduler import ReduceLROnPlateau
from utils import AverageMeter, CheckpointSaver, get_outdir, accuracy
from optim import nadam

import torch
import torch.autograd as autograd
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.utils

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', default='resnet18', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--gp', default='avg', type=str, metavar='POOL',
                    help='Type of global pool, "avg", "max", "avgmax", "avgmaxc" (default: "avg")')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument('--fold', type=int, default=0, metavar='N',
                    help='Train/valid fold #. (default: 0')
parser.add_argument('--labels', default='all', type=str, metavar='NAME',
                    help='Label set (default: "all"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--multi-target', '--mt', type=int, default=0, metavar='N',
                    help='multi-target classifier count (default: 0)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=int, default=15, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')
parser.add_argument('--ft-epochs', type=float, default=0., metavar='LR',
                    help='Number of finetuning epochs (final layer only)')
parser.add_argument('--ft-opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--ft-lr', type=float, default=0.0001, metavar='N',
                    help='Finetune learning rates.')
parser.add_argument('--drop', type=float, default=0.5, metavar='DROP',
                    help='Dropout rate (default: 0.1)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0005, metavar='M',
                    help='weight decay (default: 0.0001)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='path to init checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save-batches', action='store_true', default=False,
                    help='save images of batch inputs and targets every log interval for debugging/verification')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--class-weights', action='store_true', default=False,
                    help='Use class weights for specified labels as loss penalty')


def main():
    args = parser.parse_args()

    if args.output:
        output_base = args.output
    else:
        output_base = './output'
    exp_name = '-'.join([
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        args.model,
        args.gp,
        'f'+str(args.fold)])
    output_dir = get_outdir(output_base, 'train', exp_name)

    train_input_root = os.path.join(args.data)
    batch_size = args.batch_size
    num_epochs = args.epochs
    wav_size = (16000,)
    num_classes = len(dataset.get_labels())

    torch.manual_seed(args.seed)

    model = model_factory.create_model(
        args.model,
        in_chs=1,
        pretrained=args.pretrained,
        num_classes=num_classes,
        drop_rate=args.drop,
        global_pool=args.gp,
        checkpoint_path=args.initial_checkpoint)
    #model.reset_classifier(num_classes=num_classes)

    dataset_train = dataset.CommandsDataset(
        root=train_input_root,
        mode='train',
        fold=args.fold,
        wav_size=wav_size,
        format='spectrogram',
    )

    loader_train = data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=args.workers
    )

    dataset_eval = dataset.CommandsDataset(
        root=train_input_root,
        mode='validate',
        fold=args.fold,
        wav_size=wav_size,
        format='spectrogram',
    )

    loader_eval = data.DataLoader(
        dataset_eval,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=args.workers
    )

    train_loss_fn = validate_loss_fn = torch.nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = validate_loss_fn.cuda()

    opt_params = list(model.parameters())
    if args.opt.lower() == 'sgd':
        optimizer = optim.SGD(
            opt_params, lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif args.opt.lower() == 'adam':
        optimizer = optim.Adam(
            opt_params, lr=args.lr, weight_decay=args.weight_decay, eps=args.opt_eps)
    elif args.opt.lower() == 'nadam':
        optimizer = nadam.Nadam(
            opt_params, lr=args.lr, weight_decay=args.weight_decay, eps=args.opt_eps)
    elif args.opt.lower() == 'adadelta':
        optimizer = optim.Adadelta(
            opt_params, lr=args.lr, weight_decay=args.weight_decay, eps=args.opt_eps)
    elif args.opt.lower() == 'rmsprop':
        optimizer = optim.RMSprop(
            opt_params, lr=args.lr, alpha=0.9, eps=args.opt_eps,
            momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        assert False and "Invalid optimizer"
    del opt_params

    if not args.decay_epochs:
        print('No decay epoch set, using plateau scheduler.')
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=10)
    else:
        lr_scheduler = None

    # optionally resume from a checkpoint
    start_epoch = 0 if args.start_epoch is None else args.start_epoch
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                if 'args' in checkpoint:
                    print(checkpoint['args'])
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    if k.startswith('module'):
                        name = k[7:] # remove `module.`
                    else:
                        name = k
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
                if 'optimizer' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                if 'loss' in checkpoint:
                    train_loss_fn.load_state_dict(checkpoint['loss'])
                print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
                start_epoch = checkpoint['epoch'] if args.start_epoch is None else args.start_epoch
            else:
                model.load_state_dict(checkpoint)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit(1)

    saver = CheckpointSaver(checkpoint_dir=output_dir)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model.cuda()

    # Optional fine-tune of only the final classifier weights for specified number of epochs (or part of)
    if not args.resume and args.ft_epochs > 0.:
        if isinstance(model, torch.nn.DataParallel):
            classifier_params = model.module.get_classifier().parameters()
        else:
            classifier_params = model.get_classifier().parameters()
        if args.opt.lower() == 'adam':
            finetune_optimizer = optim.Adam(
                classifier_params,
                lr=args.ft_lr, weight_decay=args.weight_decay)
        else:
            finetune_optimizer = optim.SGD(
                classifier_params,
                lr=args.ft_lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

        finetune_epochs_int = int(np.ceil(args.ft_epochs))
        finetune_final_batches = int(np.ceil((1 - (finetune_epochs_int - args.ft_epochs)) * len(loader_train)))
        print(finetune_epochs_int, finetune_final_batches)
        for fepoch in range(0, finetune_epochs_int):
            if fepoch == finetune_epochs_int - 1 and finetune_final_batches:
                batch_limit = finetune_final_batches
            else:
                batch_limit = 0
            train_epoch(
                fepoch, model, loader_train, finetune_optimizer, train_loss_fn, args,
                output_dir=output_dir, batch_limit=batch_limit)

    best_loss = None
    stats = []
    try:
        for epoch in range(start_epoch, num_epochs):
            if args.decay_epochs:
                adjust_learning_rate(
                    optimizer, epoch, initial_lr=args.lr,
                    decay_rate=args.decay_rate, decay_epochs=args.decay_epochs)

            train_metrics = train_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                saver=saver, output_dir=output_dir)

            stats.append(train_metrics['stats'])
            print(np.mean(stats, axis=0))

            # save a recovery in case validation blows up
            saver.save_recovery({
                'epoch': epoch + 1,
                'arch': args.model,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': train_loss_fn.state_dict(),
                'args': args,
                'gp': args.gp,
                },
                epoch=epoch + 1,
                batch_idx=0)

            step = epoch * len(loader_train)
            eval_metrics = validate(
                step, model, loader_eval, validate_loss_fn, args,
                output_dir=output_dir)

            if lr_scheduler is not None:
                lr_scheduler.step(eval_metrics['eval_loss'])

            rowd = OrderedDict(epoch=epoch)
            rowd.update(train_metrics)
            rowd.update(eval_metrics)
            with open(os.path.join(output_dir, 'summary.csv'), mode='a') as cf:
                dw = csv.DictWriter(cf, fieldnames=rowd.keys())
                if best_loss is None:  # first iteration (epoch == 1 can't be used)
                    dw.writeheader()
                dw.writerow(rowd)

            # save proper checkpoint with eval metric
            best_loss = saver.save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.model,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args,
                'gp': args.gp,
                },
                epoch=epoch + 1,
                metric=eval_metrics['eval_loss'])

    except KeyboardInterrupt:
        pass
    print('*** Best loss: {0} (epoch {1})'.format(best_loss[1], best_loss[0]))


def train_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        saver=None, output_dir='', batch_limit=0):

    epoch_step = (epoch - 1) * len(loader)
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    sample_idx = 0
    stats = []
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == len(loader) - 1
        step = epoch_step + batch_idx
        data_time_m.update(time.time() - end)

        input_var = autograd.Variable(input.cuda())
        if isinstance(target, list):
            target_var = [autograd.Variable(t.cuda()) for t in target]
        else:
            target_var = autograd.Variable(target.cuda())

        input_mean = input_var.data.mean()
        input_std = input_var.data.std()
        stats.append([input_mean, input_std])

        output = model(input_var)

        loss = loss_fn(output, target_var)
        losses_m.update(loss.data[0], input_var.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sample_idx += input_var.size(0)
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]  '
                  'Loss: {loss.val:.6f} ({loss.avg:.4f})  '
                  'Time: {batch_time.val:.3f}s, {rate:.3f}/s  '
                  '({batch_time.avg:.3f}s, {rate_avg:.3f}/s)  '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                epoch,
                sample_idx, len(loader.sampler),
                100. * sample_idx / len(loader.sampler),
                loss=losses_m,
                batch_time=batch_time_m,
                rate=input_var.size(0) / batch_time_m.val,
                rate_avg=input_var.size(0) / batch_time_m.avg,
                data_time=data_time_m))

            if args.save_batches:
                torchvision.utils.save_image(
                    input,
                    os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                    padding=0,
                    normalize=True)

        if saver is not None and batch_idx % args.recovery_interval == 0:
            saver.save_recovery({
                'epoch': epoch,
                'arch': args.model,
                'state_dict':  model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args,
                'gp': args.gp,
                },
                epoch=epoch,
                batch_idx=batch_idx)

        end = time.time()

        if batch_limit and batch_idx >= batch_limit:
            break

    return OrderedDict([('train_loss', losses_m.avg), ('stats', np.mean(stats, axis=0))])


def validate(step, model, loader, loss_fn, args, output_dir=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    prec1_m = AverageMeter()
    prec3_m = AverageMeter()

    model.eval()

    end = time.time()
    sample_idx = 0
    for i, (input, target) in enumerate(loader):
        last_batch = i == len(loader) - 1
        input_var = autograd.Variable(input.cuda(), volatile=True)
        if isinstance(target, list):
            target = target[0]
        target_var = autograd.Variable(target.cuda(), volatile=True)

        output = model(input_var)

        if isinstance(output, list):
            output = output[0]

        # augmentation reduction
        #reduce_factor = loader.dataset.get_aug_factor()
        #if reduce_factor > 1:
        #    output.data = output.data.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
        #    target_var.data = target_var.data[0:target_var.size(0):reduce_factor]

        # calc loss
        loss = loss_fn(output, target_var)
        losses_m.update(loss.data[0], input.size(0))

        # metrics
        prec1, prec3 = accuracy(output.data, target_var.data, topk=(1, 3))
        prec1_m.update(prec1[0], output.size(0))
        prec3_m.update(prec3[0], output.size(0))

        batch_time_m.update(time.time() - end)
        end = time.time()
        sample_idx += input.size(0)
        if last_batch or i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                  'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                  'Prec@1 {top1.val:.4f} ({top1.avg:.4f})  '
                  'Prec@3 {top5.val:.4f} ({top5.avg:.4f})'.format(
                sample_idx, len(loader.sampler),
                batch_time=batch_time_m, loss=losses_m,
                top1=prec1_m, top5=prec3_m))

            if args.save_batches:
                torchvision.utils.save_image(
                    input,
                    os.path.join(output_dir, 'validate-batch-%d.jpg' % i),
                    padding=0,
                    normalize=True)

    metrics = OrderedDict([('eval_loss', losses_m.avg), ('eval_prec1', prec1_m.avg)])

    return metrics


def adjust_learning_rate(optimizer, epoch, initial_lr, decay_rate=0.1, decay_epochs=30):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (decay_rate ** (epoch // decay_epochs))
    print('Setting LR to', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_batch_size(epoch, initial_bs, target_bs, decay_epochs=1):
    batch_size = min(target_bs, initial_bs * (2 ** (epoch // decay_epochs)))
    return batch_size


if __name__ == '__main__':
    main()
