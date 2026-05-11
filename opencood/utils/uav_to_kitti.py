#!/usr/bin/env python3
"""
V2U4Real → KITTI-style simplified label converter (single-ego + UAV)

Features
--------
• Support multiple LiDAR types: ouster, ruby, m1
• Automatically match YAML annotation subfolder for the selected LiDAR
• Export:
    - points/*.npy                 (float32 x,y,z,intensity)
    - labels/*.txt                 (x y z dx dy dz heading_angle category_name)

Usage
--------
python uav_to_kitti.py \
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
    parser = argparse.ArgumentParser("V2U4Real → simplified KITTI converter")
    parser.add_argument("--src", required=True, help="Path to V2U4Real dataset root")
    parser.add_argument("--dst", required=True, help="Path to output dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--lidar", default="ouster", choices=["ouster", "ruby", "m1"], help="Which LiDAR to convert")
    parser.add_argument("--write-points", action="store_true", help="Export point clouds (.npy)")
    parser.add_argument("--write-labels", action="store_true", help="Export simplified labels")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_annotation(uav_yaml_file, out_txt, transformation_matrix):
    with open(uav_yaml_file, "r") as f:
        uav_ann = yaml.safe_load(f)

    if "vehicles" not in uav_ann:
        return

    for obj_id, object_content in uav_ann["vehicles"].items():
        location = object_content['location']
        rotation = object_content['angle']
        center = object_content['center']
        extent = object_content['extent']

        # UAV → ego 的 4x4 矩阵
        object_pose = [location[0] + center[0],
                       location[1] + center[1],
                       location[2] + center[2],
                       rotation[0], rotation[1], rotation[2]]
        object2lidar = x1_to_x2(object_pose, transformation_matrix)

        # shape (3, 8)
        bbx = box_utils.create_bbx(extent).T
        bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]  # 4x8
        bbx_lidar = np.dot(object2lidar, bbx).T     # 8x4
        bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0)
        bbx_lidar = box_utils.corner_to_center(bbx_lidar, order='lwh')

        # 写入 KITTI-style txt
        x, y, z, l, w, h, yaw = bbx_lidar[0]
        cls = CLASS_MAP.get(object_content.get("obj_type","Car").lower(), "Vehicle")
        line = f"{x:.8f} {y:.8f} {z:.8f} {l:.8f} {w:.8f} {h:.8f} {yaw:.8f} {cls}"
        with open(out_txt, "a") as f:
            f.write(line + "\n")


def main():
    args = parse_args()

    src_split_dir = os.path.join(args.src, args.split)
    out_split_dir = os.path.join(args.dst, args.split)  # train/val/test
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

        # YAML 文件路径
        ego_yaml_path = os.path.join(ego_path, "yaml", args.lidar)
        uav_yaml_path = os.path.join(uav_path, "yaml", args.lidar)
        if not os.path.exists(ego_yaml_path) or not os.path.exists(uav_yaml_path):
            continue

        ego_yaml_files = sorted(glob.glob(os.path.join(ego_yaml_path, "*.yaml")))
        uav_yaml_files = sorted(glob.glob(os.path.join(uav_yaml_path, "*.yaml")))

        # 假设 ego 与 UAV 的 yaml 文件数量相同，按顺序对齐
        for frame_idx, (ego_y, uav_y) in enumerate(zip(ego_yaml_files, uav_yaml_files)):
            base_id = f"{frame_idx+1:06d}"
            sample_ids.append(f"{args.split}/{scene_name}/points/{base_id}")

            # 读取 ego 和 UAV 的 lidar_pose
            with open(ego_y, "r") as f:
                ego_ann = yaml.safe_load(f)
            ego_pose = ego_ann['lidar_pose']

            with open(uav_y, "r") as f:
                uav_ann = yaml.safe_load(f)
            uav_pose = uav_ann['lidar_pose']

            # UAV → ego 坐标变换
            transformation_matrix = x1_to_x2(uav_pose, ego_pose)

            # 导出 UAV 点云（投影到 ego 坐标系）
            if args.write_points:
                uav_pcd_file = os.path.join(uav_path, args.lidar, os.path.basename(uav_y).replace(".yaml", ".pcd"))
                if os.path.exists(uav_pcd_file) and not args.dry_run:
                    out_npy = os.path.join(points_dir, base_id + ".npy")
                    uav_points = pcd_to_np(uav_pcd_file)
                    uav_points[:, :3] = box_utils.project_points_by_matrix_torch(uav_points[:, :3], transformation_matrix)
                    np.save(out_npy, uav_points)

            # 导出 uav 标签
            # UAV 标签投影到 ego
            if args.write_labels:
                out_txt = os.path.join(label_dir, base_id + ".txt")
                if not args.dry_run:
                    parse_annotation(uav_y, out_txt, transformation_matrix)


        print(f"Done. {len(sample_ids)} frames processed for split={args.split}, lidar={args.lidar}")


if __name__ == "__main__":
    main()
