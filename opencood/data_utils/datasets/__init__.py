# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# Modified by: WeiJia Li <vjiali@stu.xmu.edu.cn>
# License: MIT License 2020-2024. All Rights Reserved.
import copy

from opencood.data_utils.datasets.late_fusion_dataset import LateFusionDataset
from opencood.data_utils.datasets.early_fusion_dataset import EarlyFusionDataset
from opencood.data_utils.datasets.intermediate_fusion_dataset import IntermediateFusionDataset
from opencood.data_utils.datasets.intermediate_fusion_dataset_DSRC import IntermediateFusionDataset_DSRC

__all__ = {
    'LateFusionDataset': LateFusionDataset,
    'EarlyFusionDataset': EarlyFusionDataset,
    'IntermediateFusionDataset': IntermediateFusionDataset,
    'IntermediateFusionDataset_DSRC': IntermediateFusionDataset_DSRC,
}

# Ground-truth ranges used by training/validation and inference/test workflows.
GT_RANGE_TRAIN_VAL = [-100.8, -80, -4, 100.8, 80, 4]
GT_RANGE_TEST = [-15, -80, -4, 100, 80, 4]

# The communication range for cavs
COM_RANGE = 100


def build_dataset(dataset_cfg, visualize=False, train=True, mode=None):
    dataset_name = dataset_cfg['fusion']['core_method']
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name in opencood/" \
                    f"data_utils/datasets/init.py"
    assert dataset_name in ['LateFusionDataset', 'EarlyFusionDataset',
                            'IntermediateFusionDataset', 'IntermediateFusionDataset_DSRC'], error_message
    if mode is None:
        mode = 'train' if train else 'val'
    mode = mode.lower()

    dataset_cfg = copy.deepcopy(dataset_cfg)

    if mode in ['train', 'val', 'valid', 'validate', 'train_val']:
        dataset_cfg['postprocess']['GT_RANGE'] = GT_RANGE_TRAIN_VAL
    elif mode in ['test', 'predict', 'inference']:
        dataset_cfg['postprocess']['GT_RANGE'] = GT_RANGE_TEST
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train
    )

    return dataset
