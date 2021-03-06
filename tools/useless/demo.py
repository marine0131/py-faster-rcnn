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
import matplotlib
matplotlib.use('Agg')
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from bottle import route, run
DEBUG=0
CLASSES = ('__background__',
           'red_on', 'red_off', 'yellow_on', 'yellow_off',
           'green_on', 'green_off', 'white_on', 'white_off',
	   'grey_on', 'grey_off')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo_1(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = './test/'+image_name
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.01
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)
def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join('test', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    np.set_printoptions(threshold='nan') 
    
    timer.toc()
    #detsu=[]
    #detsu=detsu.append(2009)
    #detsu=np.ndarray(detsu)
    #detsu=detsu.shape(40,5)
    #detsu =detsu.reshape(40,5)
   
    if DEBUG:
        print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.85
    NMS_THRESH = 0.01
    print CONF_THRESH,NMS_THRESH
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    #boxes = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
	
   
	cls_ind += 1 # because we skipped background
	if DEBUG:
            print '~~~~~~~~~~~~~~~cls_ind :{}'.format(cls_ind)
	    print '~~~~~~~~~~~~~~~cls :{}'.format(cls)
	cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
	cls_scores = scores[:, cls_ind]
	
	dets = np.hstack((cls_boxes,
			  cls_scores[:, np.newaxis])).astype(np.float32)
	
	keep = nms(dets, NMS_THRESH)
	dets = dets[keep, :]
	
		
	
	inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if DEBUG:
            print '~~~~~~~~~~~~~~~inds :{}'.format(inds)
      

    
        for i in inds:
	    bbox = dets[i, :4]
	    score = dets[i, -1]
	    if DEBUG:
                print '~~~~~~~~~~~~~~~imagename :{}'.format(image_name)
	        print '~~~~~~~~~~~~~~~ssbbox :{}'.format(bbox)
		print '~~~~~~~~~~~~~~~ssscore  :{}'.format(score)
	    ax.add_patch(
	        plt.Rectangle((bbox[0], bbox[1]),
	                      bbox[2] - bbox[0],
	                      bbox[3] - bbox[1], fill=False,
	                      edgecolor='red', linewidth=3.5)
	        )
	    ax.text(bbox[0], bbox[1] - 2,
	            '{:s} {:.3f}'.format(cls, score),
	            bbox=dict(facecolor='blue', alpha=0.5),
	            fontsize=14, color='white')

	    ax.set_title(('{} detections with '
			  'p({} | box) >= {:.1f}').format(image_name, cls,
			   CONF_THRESH),
			   fontsize=14)

	    #plt.axis('off')
	    #plt.tight_layout()
	    #plt.draw()
    x = np.random.randn(60)
    y = np.random.randn(60)

    plt.scatter(x, y, s=20)
    plt.savefig('/opt/gxxj_robot/upload/image/'+image_name)
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

#if __name__ == '__main__':

@route('/')                               
def index():
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = './model_test/test.prototxt'
    caffemodel = './model_test/zf_faster_rcnn_iter_30000_deng.caffemodel'

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

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)
    im_names=[]
    for filename in os.listdir('./test'):
        im_names.append(filename)
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {}'.format(im_name)
        demo(net, im_name)                      
                                         
run(host='192.168.10.252', port=6666)
