# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.pascal_voc import pascal_voc
from datasets.coco import coco

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = ("voc", split, year)

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = ("coco", split, year)

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = ("coco", split, year)


def get_imdb(name, devkit_path, CLASSES):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))

    if __sets[name][0] == "voc":
        return pascal_voc(__sets[name][1], __sets[name][2], CLASSES, devkit_path)
    elif __sets[name][0] == "coco":
        return coco(__sets[name][1], __sets[name][2], devkit_path)


def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()

if __name__ == "__main__":
    get_imdb("voc_2007_trainval", "/opt/trainnew/light/py-faster-rcnn")
