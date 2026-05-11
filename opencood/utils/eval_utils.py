# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# Modified by: WeiJia Li <vjiali@stu.xmu.edu.cn>
# License: MIT


import os

import numpy as np
import torch

from opencood.utils import common_utils
from opencood.hypes_yaml import yaml_utils


def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def calculate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh):
    """
    Calculate the true positive and false positive numbers of the current
    frames.

    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each predicted bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    """
    # fp, tp and gt in the current frame
    fp = []
    tp = []
    gt = gt_boxes.shape[0]
    if det_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend] # from high to low
        det_polygon_list = list(common_utils.convert_format(det_boxes))
        gt_polygon_list = list(common_utils.convert_format(gt_boxes))

        # match prediction and gt bounding box
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)

            gt_index = np.argmax(ious)
            gt_polygon_list.pop(gt_index)

        result_stat[iou_thresh]['score'] += det_score.tolist()

    result_stat[iou_thresh]['fp'] += fp
    result_stat[iou_thresh]['tp'] += tp
    result_stat[iou_thresh]['gt'] += gt


# Backward-compatible alias for older scripts.
caluclate_tp_fp = calculate_tp_fp


def calculate_ap(result_stat, iou, global_sort_detections):
    """
    Calculate the average precision and recall, and save them into a txt.

    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.
        
    iou : float
        The threshold of iou.

    global_sort_detections : bool
        Whether to sort the detection results globally.
    """
    iou_5 = result_stat[iou]

    if global_sort_detections:
        fp = np.array(iou_5['fp'])
        tp = np.array(iou_5['tp'])
        score = np.array(iou_5['score'])

        assert len(fp) == len(tp) and len(tp) == len(score)
        sorted_index = np.argsort(-score)
        fp = fp[sorted_index].tolist()
        tp = tp[sorted_index].tolist()
        
    else:
        fp = iou_5['fp']
        tp = iou_5['tp']
        assert len(fp) == len(tp)

    gt_total = iou_5['gt']

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec


def eval_final_results(result_stat, save_path, global_sort_detections, scenario_result_stat=None):
    """
    Calculate and save the final evaluation results.
    
    Parameters
    ----------
    result_stat : dict
        Overall statistics for all scenarios.
    save_path : str
        Path to save the evaluation results.
    global_sort_detections : bool
        Whether to sort detections globally.
    scenario_result_stat : dict, optional
        Statistics grouped by scenario. Format: {scenario_name: {0.25: {...}, 0.5: {...}, 0.7: {...}}}
    """
    dump_dict = {}

    ap_25, _, _ = calculate_ap(result_stat, 0.25, global_sort_detections)
    ap_50, _, _ = calculate_ap(result_stat, 0.50, global_sort_detections)
    ap_70, _, _ = calculate_ap(result_stat, 0.70, global_sort_detections)

    dump_dict.update({'ap_25': ap_25,
                        'ap_50': ap_50,
                        'ap_70': ap_70,
                        })
        
    if scenario_result_stat is not None:
        scenario_aps = {0.25: [], 0.5: [], 0.7: []}

        for scenario_name, scenario_stat in scenario_result_stat.items():
            scenario_prefix = f"{scenario_name}_"
            
            scenario_gt_25 = scenario_stat[0.25]['gt']
            scenario_gt_50 = scenario_stat[0.5]['gt']
            scenario_gt_70 = scenario_stat[0.7]['gt']
            
            scenario_ap_25 = 0.0
            scenario_ap_50 = 0.0
            scenario_ap_70 = 0.0

            if scenario_gt_25 > 0:
                scenario_ap_25, _, _ = calculate_ap(scenario_stat, 0.25, global_sort_detections)
            if scenario_gt_50 > 0:
                scenario_ap_50, _, _ = calculate_ap(scenario_stat, 0.50, global_sort_detections)
            if scenario_gt_70 > 0:
                scenario_ap_70, _, _ = calculate_ap(scenario_stat, 0.70, global_sort_detections)
            
            if scenario_gt_25 > 0:
                scenario_aps[0.25].append((scenario_ap_25, scenario_gt_25))
            if scenario_gt_50 > 0:
                scenario_aps[0.5].append((scenario_ap_50, scenario_gt_50))
            if scenario_gt_70 > 0:
                scenario_aps[0.7].append((scenario_ap_70, scenario_gt_70))
            
            if 'scenarios' not in dump_dict:
                dump_dict['scenarios'] = {}
            dump_dict['scenarios'][scenario_name] = {
                'ap_25': scenario_ap_25,
                'ap_50': scenario_ap_50,
                'ap_70': scenario_ap_70,
            }

    output_file = 'eval.yaml' if not global_sort_detections else 'eval_global_sort.yaml'
    output_path = os.path.join(save_path, output_file)
    
    yaml_utils.save_yaml(dump_dict, output_path)

    print('The Average Precision at IOU 0.25 is %.2f, '
          'The Average Precision at IOU 0.5 is %.2f, '
          'The Average Precision at IOU 0.7 is %.2f' % (ap_25, ap_50, ap_70))
    
    if scenario_result_stat is not None:
        for scenario_name in sorted(scenario_result_stat.keys()):
            scenario_prefix = f"{scenario_name}_"
            if f'{scenario_prefix}ap_50' in dump_dict:
                print(f'  {scenario_name}: AP@0.25=%.2f, AP@0.5=%.2f, AP@0.7=%.2f' % (
                    dump_dict[f'{scenario_prefix}ap_25'],
                    dump_dict[f'{scenario_prefix}ap_50'],
                    dump_dict[f'{scenario_prefix}ap_70']
                ))
