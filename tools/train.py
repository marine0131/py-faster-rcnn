import os
os.popen('python train_net.pyc  --solver ../../py-faster-rcnn/models/pascal_voc/ZF/faster_rcnn_end2end/solver.prototxt --imdb voc_2007_trainval --cfg ../../py-faster-rcnn/experiments/cfgs/faster_rcnn_end2end.yml --weight /home/wsy/trainnew/py-faster-rcnn/prepare/ZF.v2.caffemodel')

