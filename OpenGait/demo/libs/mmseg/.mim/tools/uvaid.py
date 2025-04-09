# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import time

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmseg.core.evaluation import get_palette,mean_dice,mean_iou
from mmcv.runner import get_dist_info, init_dist

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import cv2
import os

CLASSES = ('Building', 'Road', 'Tree', 'LowVeg', 'Moving_Car',  'Static_Car', 'Human', 'Clutter')
PALETTE = [[128, 0, 0], [128, 64, 128], [0, 128, 0], [128, 128, 0], [64, 0, 128], [192, 0, 192], [64, 64, 0], [0, 0, 0]]

@DATASETS.register_module()
class UAVID(CustomDataset):
    CLASSES = classes
    PALETTE = palette
    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png', 
                    split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None