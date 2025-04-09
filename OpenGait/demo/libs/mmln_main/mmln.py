"""
UnetFormer for uavid datasets with supervision training
Libo Wang, 2022.02.22
"""
from torch.utils.data import DataLoader
from mmln_main.losses import *
from mmln_main.datasets.uavid_dataset import *
from mmln_main.models.MMLN import mmln_small
from catalyst.contrib.nn import Lookahead
from catalyst import utils

# training hparam
max_epoch = 50
ignore_index = 255
train_batch_size = 1
val_batch_size = 1
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
accumulate_n = 1

num_classes = len(CLASSES)
classes = CLASSES

weights_name = "uavid666-v2.ckpt"
weights_path = './demo/checkpoints/MMLNseg/uavid666-v2.ckpt'
test_weights_name = ""
log_name = 'uavid/{}'.format(weights_name)
log_dir="runs/uavid/{}".format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 3
save_last = True
check_val_every_n_epoch = 5
gpus = [0,1,2,3]
# strategy = None
# gpus = 4
strategy = "ddp"
pretrained_ckpt_path = None
resume_ckpt_path = None

#  define the network
net = mmln_small(num_classes=num_classes, weight_path = weights_path)

# define the loss
# loss = UnetFormerLoss(ignore_index=ignore_index)
loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)

# use_aux_loss = True
use_aux_loss=False

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)

