#!/usr/bin/env python3
"""
V2U4Real → KITTI-style simplified label converter (multi-agent union GT version)

Features
--------
• Support multiple LiDAR types: ouster, ruby, m1
• Automatically merge annotations from all agents (ego + others)
• Remove duplicate object IDs, keeping ego's label if conflict
• Export:
    - points/*.npy                 (float32 x,y,z,intensity)
    - labels/*.txt                 (x y z dx dy dz heading_angle category_name)

Usage
--------
python vehicle_to_single.py \
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
    parser = argparse.ArgumentParser("V2U4Real → simplified KITTI converter (union GT)")
    parser.add_argument("--src", required=True, help="Path to V2U4Real dataset root")
    parser.add_argument("--dst", required=True, help="Path to output dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--ego-id", type=int, default=1, help="Which CAV_x to treat as ego vehicle")
    parser.add_argument("--lidar", default="ouster", choices=["ouster", "ruby", "m1"], help="Which LiDAR to convert")
    parser.add_argument("--write-points", action="store_true", help="Export point clouds (.npy)")
    parser.add_argument("--write-labels", action="store_true", help="Export simplified labels")
    parser.add_argument("--dry-run", action="store_true", help="Preview only (no file write)")
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_annotation_objects(yaml_file):
    """Load object annotations from a YAML file."""
    with open(yaml_file, "r") as f:
        ann = yaml.safe_load(f)

    if not ann or "vehicles" not in ann:
        return {}

    objs = {}
    for obj_id, obj in ann["vehicles"].items():
        cls = obj.get("obj_type", "Car").lower()
        cls = CLASS_MAP.get(cls, "Vehicle")

        loc = obj["location"]       # x, y, z
        dims = obj["extent"]        # dx, dy, dz
        heading = np.deg2rad(obj["angle"][1])  # yaw (degree → radian)

        objs[obj_id] = {
            "loc": loc,
            "dims": dims,
            "heading": heading,
            "cls": cls
        }
    return objs


def write_annotation_file(objects, out_file):
    """Write merged objects to KITTI-style txt label."""
    lines = []
    for obj_id, o in objects.items():
        line = f"{o['loc'][0]:.8f} {o['loc'][1]:.8f} {o['loc'][2]:.8f} " \
               f"{o['dims'][0]:.8f} {o['dims'][1]:.8f} {o['dims'][2]:.8f} " \
               f"{o['heading']:.8f} {o['cls']}"
        lines.append(line)
    with open(out_file, "w") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()

    ego_folder = f"{args.ego_id}"
    src_split_dir = os.path.join(args.src, args.split)

    sample_ids = []
    scenario_dirs = sorted(glob.glob(os.path.join(src_split_dir, "*")))

    for scen in scenario_dirs:
        ego_path = os.path.join(scen, ego_folder)
        if not os.path.exists(ego_path):
            continue

        scene_name = os.path.basename(scen)
        out_split_dir = os.path.join(args.dst, args.split)
        out_scene_dir = os.path.join(out_split_dir, scene_name)

        points_dir = os.path.join(out_scene_dir, "points")
        label_dir = os.path.join(out_scene_dir, "labels")
        for d in [points_dir, label_dir]:
            ensure_dir(d)

        # list all CAV folders (e.g., 1, 2, 3, ...)
        cav_folders = sorted(
            [d for d in os.listdir(scen) if os.path.isdir(os.path.join(scen, d)) and d.isdigit()]
        )

        # ego yaml path as reference timeline
        ego_yaml_dir = os.path.join(ego_path, "yaml", args.lidar)
        if not os.path.exists(ego_yaml_dir):
            continue

        yaml_files = sorted(glob.glob(os.path.join(ego_yaml_dir, "*.yaml")))
        for frame_idx, ego_yaml in enumerate(yaml_files):
            base_name = os.path.basename(ego_yaml)
            base_id = f"{frame_idx+1:06d}"
            sample_ids.append(f"{args.split}/{scene_name}/points/{base_id}")

            # export ego point cloud
            if args.write_points:
                pcd_file = os.path.join(ego_path, args.lidar, base_name.replace(".yaml", ".pcd"))
                if os.path.exists(pcd_file) and not args.dry_run:
                    out_npy = os.path.join(points_dir, base_id + ".npy")
                    points = pcd_to_np(pcd_file)
                    np.save(out_npy, points)

            # merge GT from all CAVs
            if args.write_labels and not args.dry_run:
                merged_objs = {}
                for cav_id in cav_folders:
                    yaml_path = os.path.join(scen, cav_id, "yaml", args.lidar, base_name)
                    if not os.path.exists(yaml_path):
                        continue

                    objs = load_annotation_objects(yaml_path)
                    for oid, o in objs.items():
                        # ego version overrides duplicates
                        if oid not in merged_objs or cav_id == ego_folder:
                            merged_objs[oid] = o

                out_txt = os.path.join(label_dir, base_id + ".txt")
                write_annotation_file(merged_objs, out_txt)

    print(f"✅ Done. {len(sample_ids)} frames processed for split={args.split}, lidar={args.lidar}")


if __name__ == "__main__":
    main()
