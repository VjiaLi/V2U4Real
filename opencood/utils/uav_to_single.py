#!/usr/bin/env python3
"""
V2U4Real → KITTI-style simplified label converter (single-ego + UAV union GT, UAV priority)

Features
--------
• Support multiple LiDAR types: ouster, ruby, m1
• Merge annotations from both ego and UAV (UAV priority)
• Automatically match YAML annotation subfolder for the selected LiDAR
• Export:
    - points/*.npy                 (float32 x,y,z,intensity)
    - labels/*.txt                 (x y z dx dy dz heading_angle category_name)

Usage
--------
python uav_to_single.py \
    --src v2u4real \
    --dst custom_data \
    --split train \
    --write-points \
    --write-labels
"""

import os
import argparse
import glob
import yaml
import numpy as np

from opencood.utils.pcd_utils import pcd_to_np
from opencood.utils.transformation_utils import x1_to_x2
from opencood.utils import box_utils

CLASS_MAP = {
    "car": "Vehicle",
    "truck": "Vehicle",
    "bus": "Vehicle",
    "pedestrian": "Pedestrian",
    "cyclist": "Cyclist"
}

EGO_FOLDER = "1"
UAV_FOLDER = "2"


def parse_args():
    parser = argparse.ArgumentParser("V2U4Real → simplified KITTI converter (ego+uav union, UAV priority)")
    parser.add_argument("--src", required=True, help="Path to V2U4Real dataset root")
    parser.add_argument("--dst", required=True, help="Path to output dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--lidar", default="ouster", choices=["ouster", "ruby", "m1"], help="Which LiDAR to convert")
    parser.add_argument("--write-points", action="store_true", help="Export point clouds (.npy)")
    parser.add_argument("--write-labels", action="store_true", help="Export simplified labels")
    parser.add_argument("--dry-run", action="store_true", help="Preview only (no file write)")
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_annotation_objects(yaml_file, transform=None):
    """
    读取一个 YAML 标注文件中的所有对象，并可选地进行坐标变换（UAV → ego）
    返回 dict[obj_id] = {loc, dims, heading, cls}
    """
    with open(yaml_file, "r") as f:
        ann = yaml.safe_load(f)

    if not ann or "vehicles" not in ann:
        return {}

    objs = {}
    for obj_id, obj in ann["vehicles"].items():
        # ---- 修复这里 ----
        location = obj['location']
        rotation = obj['angle']
        center = obj['center']
        extent = obj['extent']

        cls = obj.get("obj_type", "Car").lower()
        cls = CLASS_MAP.get(cls, "Vehicle")

        # 构造物体在自身 LiDAR 坐标系下的姿态（位置 + 角度）
        object_pose = [
            location[0] + center[0],
            location[1] + center[1],
            location[2] + center[2],
            rotation[0], rotation[1], rotation[2]
        ]

        # LiDAR 坐标下的物体矩阵
        object2lidar = x1_to_x2(object_pose, np.eye(4))

        # 如果提供了 transform（UAV→EGO），则在外层再乘一次变换矩阵
        if transform is not None:
            object2lidar = transform @ object2lidar

        # shape (3, 8)
        bbx = box_utils.create_bbx(extent).T
        bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]  # 4x8
        bbx_lidar = np.dot(object2lidar, bbx).T     # 8x4
        bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0)
        bbx_lidar = box_utils.corner_to_center(bbx_lidar, order='lwh')

        # 写入 KITTI-style txt
        x, y, z, l, w, h, yaw = bbx_lidar[0]

        objs[obj_id] = {
            "loc": [x, y, z],
            "dims": [l, w, h],
            "heading": yaw,
            "cls": cls
        }

    return objs


def write_annotation_file(objects, out_file):
    """Write merged objects to KITTI-style txt label."""
    lines = []
    for _, o in objects.items():
        line = f"{o['loc'][0]:.8f} {o['loc'][1]:.8f} {o['loc'][2]:.8f} " \
               f"{o['dims'][0]:.8f} {o['dims'][1]:.8f} {o['dims'][2]:.8f} " \
               f"{o['heading']:.8f} {o['cls']}"
        lines.append(line)
    with open(out_file, "w") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()

    src_split_dir = os.path.join(args.src, args.split)
    out_split_dir = os.path.join(args.dst, args.split)
    ensure_dir(out_split_dir)

    sample_ids = []

    # 遍历场景
    scenario_dirs = sorted(glob.glob(os.path.join(src_split_dir, "*")))
    for scen in scenario_dirs:
        ego_path = os.path.join(scen, EGO_FOLDER)
        uav_path = os.path.join(scen, UAV_FOLDER)
        if not os.path.exists(ego_path) or not os.path.exists(uav_path):
            continue

        scene_name = os.path.basename(scen)
        out_scene_dir = os.path.join(out_split_dir, scene_name)
        points_dir = os.path.join(out_scene_dir, "points")
        label_dir = os.path.join(out_scene_dir, "labels")
        for d in [points_dir, label_dir]:
            ensure_dir(d)

        ego_yaml_path = os.path.join(ego_path, "yaml", args.lidar)
        uav_yaml_path = os.path.join(uav_path, "yaml", args.lidar)
        if not os.path.exists(ego_yaml_path) or not os.path.exists(uav_yaml_path):
            continue

        ego_yaml_files = sorted(glob.glob(os.path.join(ego_yaml_path, "*.yaml")))
        uav_yaml_files = sorted(glob.glob(os.path.join(uav_yaml_path, "*.yaml")))

        for frame_idx, (ego_y, uav_y) in enumerate(zip(ego_yaml_files, uav_yaml_files)):
            base_id = f"{frame_idx + 1:06d}"
            sample_ids.append(f"{args.split}/{scene_name}/points/{base_id}")

            # 读取 ego/uav pose
            with open(ego_y, "r") as f:
                ego_ann = yaml.safe_load(f)
            ego_pose = ego_ann['lidar_pose']

            with open(uav_y, "r") as f:
                uav_ann = yaml.safe_load(f)
            uav_pose = uav_ann['lidar_pose']

            # UAV → ego 变换矩阵
            transform = x1_to_x2(uav_pose, ego_pose)

            # UAV 点云投影
            if args.write_points:
                uav_pcd_file = os.path.join(uav_path, args.lidar, os.path.basename(uav_y).replace(".yaml", ".pcd"))
                if os.path.exists(uav_pcd_file) and not args.dry_run:
                    out_npy = os.path.join(points_dir, base_id + ".npy")
                    uav_points = pcd_to_np(uav_pcd_file)
                    uav_points[:, :3] = box_utils.project_points_by_matrix_torch(uav_points[:, :3], transform)
                    np.save(out_npy, uav_points)

            # 合并 ego + UAV 标签（UAV 优先）
            if args.write_labels and not args.dry_run:
                merged_objs = {}

                # ego 不需要变换
                ego_objs = load_annotation_objects(ego_y)
                merged_objs.update(ego_objs)

                # UAV 要变换到 ego 坐标
                uav_objs = load_annotation_objects(uav_y, transform=transform)
                for oid, obj in uav_objs.items():
                    merged_objs[oid] = obj  # UAV 优先覆盖

                out_txt = os.path.join(label_dir, base_id + ".txt")
                write_annotation_file(merged_objs, out_txt)


        print(f"✅ Done. {len(sample_ids)} frames processed for split={args.split}, lidar={args.lidar}")


if __name__ == "__main__":
    main()
