#! /usr/bin/env python
import os
os.system('python train_net.pyc \
           --solver ../models/pascal_voc/ZF/faster_rcnn_end2end/solver.prototxt \
           --imdb voc_2007_trainval \
           --cfg ../experiments/cfgs/faster_rcnn_end2end.yml \
           --weight ../prepare/ZF.v2.caffemodel')
