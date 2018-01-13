import argparse
import zlib
import io
import os
import csv
import struct
import shutil
import time
import glob
import numpy as np
import pandas as pd
from collections import OrderedDict
from datetime import datetime

from dataset import CommandsDataset
from models import model_factory
from utils import AverageMeter, get_outdir

import torch
import torch.autograd as autograd
import torch.nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.utils

try:
    import lmdb
    has_lmdb = True
except ImportError:
    has_lmdb = False

parser = argparse.ArgumentParser(description='Inference')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--gp', default='avg', type=str, metavar='POOL',
                    help='Type of global pool, "avg", "max", "avgmax", "avgmaxc" (default: "avg")')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--img-size', type=int, default=180, metavar='N',
                    help='Image patch size (default: 180)')
parser.add_argument('-b', '--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 512)')
parser.add_argument('-j', '--workers', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to restore checkpoint (default: none)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save-batches', action='store_true', default=False,
                    help='save images of batch inputs and targets every log interval for debugging/verification')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')


def main():
    args = parser.parse_args()

    img_size = (args.img_size, args.img_size)
    num_classes = 12

    #if 'inception' in args.model:
    #    num_classes_init = 1001
    #    normalize = 'le'
    #else:
    #    num_classes_init = 1000
    #    normalize = 'torchvision'

    test_time_pool = 0 #5 if 'dpn' in args.model else 0

    model = model_factory.create_model(
        args.model,
        in_chs=1,
        num_classes=num_classes,
        global_pool=args.gp,
        test_time_pool=test_time_pool)
    #model.reset_classifier(num_classes=num_classes)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model.cuda()

    if not os.path.exists(args.checkpoint):
        print("=> no checkpoint found at '{}'".format(args.checkpoint))
        exit(1)
    print("=> loading checkpoint '{}'".format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint, checkpoint['epoch']))
    else:
        model.load_state_dict(checkpoint)

    csplit = os.path.normpath(args.checkpoint).split(sep=os.path.sep)
    if len(csplit) > 1:
        exp_name = csplit[-2] + '-' + csplit[-1].split('.')[0]
    else:
        exp_name = ''

    if args.output:
        output_base = args.output
    else:
        output_base = './output'

    output_dir = get_outdir(output_base, 'predictions', exp_name)

    dataset = CommandsDataset(
        root=args.data,
        mode='test',
        format='spectrogram'
    )

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=args.workers
    )

    model.eval()

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()


    try:
        # open CSV for writing predictions
        cf = open(os.path.join(output_dir, 'results.csv'), mode='w')
        res_writer = csv.writer(cf)
        res_writer.writerow(['fname'] + dataset.id_to_label)

        # open CSV for writing submission
        cf = open(os.path.join(output_dir, 'submission.csv'), mode='w')
        sub_writer = csv.writer(cf)
        sub_writer.writerow(['fname', 'label', 'prob'])

        end = time.time()
        batch_sample_idx = 0
        for batch_idx, (input, target) in enumerate(loader):
            data_time_m.update(time.time() - end)
            input_var = autograd.Variable(input.cuda(), volatile=True)
            output = model(input_var)

            # augmentation reduction
            #reduce_factor = loader.dataset.get_aug_factor()
            #if reduce_factor > 1:
            #    output.data = output.data.unfold(0, reduce_factor, reduce_factor).mean(dim=2).squeeze(dim=2)
            #    index = index[0:index.size(0):reduce_factor]

            # move data to CPU and collect)
            output_logprob = F.log_softmax(output, dim=1).data.cpu().numpy()
            output = F.softmax(output, dim=1)
            output_prob, output_idx = output.max(1)
            output_prob = output_prob.data.cpu().numpy()
            output_idx = output_idx.data.cpu().numpy()
            for i in range(output_logprob.shape[0]):
                index = batch_sample_idx + i
                pred_label = dataset.id_to_label[output_idx[i]]
                pred_prob = output_prob[i]
                filename = dataset.filename(index)
                res_writer.writerow([filename] + list(output_logprob[i]))
                sub_writer.writerow([filename] + [pred_label, pred_prob])

            batch_sample_idx += input_var.size(0)
            batch_time_m.update(time.time() - end)
            if batch_idx % args.print_freq == 0:
                print('Inference: [{}/{} ({:.0f}%)]  '
                      'Time: {batch_time.val:.3f}s, {rate:.3f}/s  '
                      '({batch_time.avg:.3f}s, {rate_avg:.3f}/s)  '
                      'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                    batch_sample_idx, len(loader.sampler),
                    100. * batch_idx / len(loader),
                    batch_time=batch_time_m,
                    rate=input_var.size(0) / batch_time_m.val,
                    rate_avg=input_var.size(0) / batch_time_m.avg,
                    data_time=data_time_m))

            end = time.time()
            # end iterating through dataset

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(str(e))

if __name__ == '__main__':
    main()
