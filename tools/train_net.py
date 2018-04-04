#!/usr/bin/env python
#-*- coding:utf-8 -
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import datasets.imdb
import caffe
import argparse
import pprint
import numpy as np
import sys
import os

LIGHT_CLASSES = ('__background__',
                 'red_on', 'red_off', 'yellow_on', 'yellow_off', 'green_on', 'green_off')
BG_CLASSES = ('__background__',
              'bg')
NUM_CLASSES = ('__background__',
               'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero')

DEVKIT_DIR = {'light': '../data/VOCdevkit2007_light/',
              'bg': '../data/VOCdevkit2007_bg/',
              'num': '../data/VOCdevkit2007_num/'}

OUTPUT_DIR = {'light': '../output/faster_rcnn_end2end/voc_2007_trainval/light/',
              'bg': '../output/faster_rcnn_end2end/voc_2007_trainval/bg/',
              'num': '../output/faster_rcnn_end2end/voc_2007_trainval/num/'}


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=4000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def combined_roidb(imdb_names, devkit_path, CLASSES):
    def get_roidb(imdb_name, devkit_path, CLASSES):
        imdb = get_imdb(imdb_name, devkit_path, CLASSES)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s, devkit_path, CLASSES) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = datasets.imdb.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names, devkit_path, CLASSES)
    return imdb, roidb


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    # customized parameter
    with open("trainning_config.txt", "r") as f:
        class_type = f.readline().split(':')[-1].strip('\n').strip()
        max_iters = int(f.readline().split(':')[-1].strip('\n').strip())

    devkit_dir = DEVKIT_DIR[class_type]
    output_dir = OUTPUT_DIR[class_type]

    # choose class
    CLASSES = None
    if class_type == 'light':
        CLASSES = LIGHT_CLASSES
    elif class_type == 'bg':
        CLASSES = BG_CLASSES
    elif class_type == 'num':
        CLASSES = NUM_CLASSES
    else:
        print("class type does not match any CLASSES")

    # devkit
    if not devkit_dir:
        devkit_dir = None

    # get imdb and roidb
    imdb, roidb = combined_roidb(args.imdb_name, devkit_dir, CLASSES)
    print '{:d} roidb entries'.format(len(roidb))

    if not output_dir:
        output_dir = get_output_dir(imdb)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_net(solver_prototxt=args.solver, roidb=roidb, output_dir=output_dir,
              pretrained_model=args.pretrained_model, max_iters=max_iters)
