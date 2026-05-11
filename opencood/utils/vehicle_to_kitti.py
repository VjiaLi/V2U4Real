#!/usr/bin/env python3
"""
V2U4Real → KITTI-style simplified label converter (single-ego, selectable LiDAR)

Features
--------
• Support multiple LiDAR types: ouster, ruby, m1
• Automatically match YAML annotation subfolder for the selected LiDAR
• Export:
    - points/*.npy                 (float32 x,y,z,intensity)
    - labels/*.txt                 (x y z dx dy dz heading_angle category_name)

Usage
--------
python vehicle_to_kitti.py \
    --src v2u4real \
    --dst custom_data \
    --split train \
    --ego-id 1 \
    --lidar ouster \
    --write-points \
    --write-labels
"""

import os
import argparse
import glob
import yaml
import numpy as np
import open3d as o3d

from opencood.utils.pcd_utils import pcd_to_np


CLASS_MAP = {
    "car": "Vehicle",
    "truck": "Vehicle",
    "bus": "Vehicle",
    "pedestrian": "Pedestrian",
    "cyclist": "Cyclist"
}


def parse_args():
    parser = argparse.ArgumentParser("V2U4Real → simplified KITTI converter")
    parser.add_argument("--src", required=True, help="Path to V2U4Real dataset root")
    parser.add_argument("--dst", required=True, help="Path to output dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--ego-id", type=int, default=1, help="Which CAV_x to treat as ego vehicle")
    parser.add_argument("--lidar", default="ouster", choices=["ouster", "ruby", "m1"], help="Which LiDAR to convert")
    parser.add_argument("--write-points", action="store_true", help="Export point clouds (.npy)")
    parser.add_argument("--write-labels", action="store_true", help="Export simplified labels")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_annotation(yaml_file, out_file):
    with open(yaml_file, "r") as f:
        ann = yaml.safe_load(f)

    if "vehicles" not in ann:
        return

    lines = []
    for obj_id, obj in ann["vehicles"].items():
        cls = obj.get("obj_type", "Car").lower()
        cls = CLASS_MAP.get(cls, "Vehicle")
        loc = obj["location"]       # x, y, z
        dims = obj["extent"]        # dx, dy, dz
        heading = np.deg2rad(obj["angle"][1])  # yaw in radians
        if loc is None or any(v is None for v in loc):
            print("Warning: invalid loc, skip:", loc)
            return

        line = f"{loc[0]:.8f} {loc[1]:.8f} {loc[2]:.8f} " \
               f"{dims[0]:.8f} {dims[1]:.8f} {dims[2]:.8f} " \
               f"{heading:.8f} {cls}"
        lines.append(line)

    with open(out_file, "w") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()

    ego_folder = f"{args.ego_id}"
    src_split_dir = os.path.join(args.src, args.split)

    # prepare output dirs
    points_dir = os.path.join(args.dst, "points")
    label_dir = os.path.join(args.dst, "labels")

    sample_ids = []
    scenario_dirs = sorted(glob.glob(os.path.join(src_split_dir, "*")))

    for scen in scenario_dirs:
        ego_path = os.path.join(scen, ego_folder)
        if not os.path.exists(ego_path):
            continue

        scene_name = os.path.basename(scen) 
        out_split_dir = os.path.join(args.dst, args.split)  # train/val/test
        out_scene_dir = os.path.join(out_split_dir, scene_name)

        points_dir = os.path.join(out_scene_dir, "points")
        label_dir = os.path.join(out_scene_dir, "labels")
        for d in [points_dir, label_dir]:
            ensure_dir(d)

        # YAML path for the chosen LiDAR
        yaml_path = os.path.join(ego_path, "yaml", args.lidar)
        if not os.path.exists(yaml_path):
            continue

        yaml_files = sorted(glob.glob(os.path.join(yaml_path, "*.yaml")))
        for frame_idx, y in enumerate(yaml_files):
            base_id = f"{frame_idx+1:06d}"  
            sample_ids.append(f"{args.split}/{scene_name}/points/{base_id}")

            if args.write_points:
                pcd_file = os.path.join(ego_path, args.lidar, os.path.basename(y).replace(".yaml", ".pcd"))
                if os.path.exists(pcd_file) and not args.dry_run:
                    out_npy = os.path.join(points_dir, base_id + ".npy")
                    points = pcd_to_np(pcd_file)
                    np.save(out_npy, points)

            if args.write_labels:
                out_txt = os.path.join(label_dir, base_id + ".txt")
                if not args.dry_run:
                    parse_annotation(y, out_txt)


    print(f"Done. {len(sample_ids)} frames processed for split={args.split}, lidar={args.lidar}")


if __name__ == "__main__":
    main()
