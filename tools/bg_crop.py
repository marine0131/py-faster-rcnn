#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
# import matplotlib.pyplot as plt
import numpy as np
# import scipy.io as sio
import caffe, os, cv2
import argparse

DEBUG = 0
CLASSES = ('__background__',
           'red_on', 'red_off', 'yellow_on', 'yellow_off',
           'green_on', 'green_off', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'bg')

NETS = {'vgg16': ('VGG16', 'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF', 'ZF_faster_rcnn_final.caffemodel')}


def demo(net, image_name, sumnum):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the raw image
    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    np.set_printoptions(threshold='nan')

    timer.toc()

    if DEBUG:
        print ('Detection took {:.3f}s for {:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    im = im[:, :, (2, 1, 0)]
    # fig, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(im, aspect='equal')
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]

        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)

        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        for i in inds:
            bbox = dets[i, :4]
            # score = dets[i, -1]

            roiImg = im[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            roiImg = roiImg[:, :, (2, 1, 0)]

            namepic = './croped_bg/'+str(sumnum).zfill(6)+'.jpg'
            cv2.imwrite(namepic, roiImg)
            sumnum = sumnum+1

    del im, keep, dets
    # plt.close()
    return sumnum


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    # prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
    #                       'faster_rcnn_end2end', 'test.prototxt')  #
    # caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
    #                          NETS[args.demo_net][1])  #NETS[args.demo_net][1]
    prototxt = '../models/pascal_voc/ZF/faster_rcnn_end2end/test.prototxt'
    caffemodel = '../output/faster_rcnn_end2end/voc_2007_trainval/bg/zf_faster_rcnn_iter_30000_purebg.caffemodel'

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    # print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _ = im_detect(net, im)

    # process images
    with open('./bg_crop_config.txt') as f:
        start_num = f.readline().split(':')[-1].strip('\n').strip()
        write_num = int(f.readline().split(':')[-1].strip('\n').strip())

    im_names = []
    for imname in os.listdir('../data/VOCdevkit2007_bg/VOC2007/JPEGImages/'):
        im_names.append(imname)
    im_names.sort()

    im_names = im_names[im_names.index(start_num+'.jpg'):]

    # print im_names
    for im_name in im_names:
        # print 'Demo for data/demo/{}'.format(im_name)
        print im_name
        write_num = demo(net, os.path.join('../data/VOCdevkit2007_bg/VOC2007/JPEGImages/', im_name), write_num)
