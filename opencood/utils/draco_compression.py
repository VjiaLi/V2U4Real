"""
To use this script, draco [https://github.com/google/draco.git] should be installed.
1. During test, keypoint coordinates and features should be saved as .ply files
   using the save_ply function.
2. Compress and analyze the CPM size using draco_compression.
"""
import random, os, re
import numpy as np
import torch
from glob import glob
import subprocess

DRACO_ENCODER = os.environ.get("DRACO_ENCODER", "draco_encoder")


def save_ply(path, batch_coords, batch_features):
    dirname = "{:06d}".format(random.randint(0, 999999))
    os.mkdir(path + dirname)
    for bi, (coords, features) in enumerate(zip(batch_coords[1:],
                                                batch_features[1:])):
        header = "ply\n" \
                 "format ascii 1.0\n" \
                 f"element vertex {len(coords)}\n" \
                 "property float x\n" \
                 "property float y\n" \
                 "property float z\n"
        header = header + "".join([f"property float feat{i}\n" for i in range(32)]) + "end_header"
        data = torch.cat([coords, features], dim=1).detach().cpu().numpy()
        np.savetxt(path + dirname + f"/{bi + 1}.ply", data,
                   delimiter=' ', header=header, comments='')


def draco_compression(ply_path):
    files = glob(os.path.join(ply_path, '*/*.ply'))
    cpm_sizes = list(map(draco_compression_one, files))
    return cpm_sizes


def draco_compression_one(file):
    out_file = file.replace('ply', 'drc')
    std_out = subprocess.getoutput(
        f"{DRACO_ENCODER} -point_cloud -i {file} -o {out_file}")
    size_str = re.findall('[0-9]+ bytes', std_out)
    if len(size_str)<1:
        print("Compression failed:", file)
        cpm_size = 0
    else:
        cpm_size = int(size_str[0].split(' ')[0])

    return cpm_size


def cal_avg_num_kpts(ply_path):
    files = glob(os.path.join(ply_path, '*/*.ply'))

    def read_vertex_num(file):
        with open(file, 'r') as f:
            size_str = re.findall('element vertex [0-9]+', f.read())[0]
        return float(size_str.split(' ')[-1]) * 4 * 32 / 1024

    sizes = list(map(read_vertex_num, files))

    return sizes


if __name__ == "__main__":
    cpm_path = "opencood/logs/fpvrcnn_intermediate_fusion/cpms"
    cpm_sizes = cal_avg_num_kpts(cpm_path)
    # cpm_sizes = draco_compression(cpm_path)
    print(np.array(cpm_sizes).mean())
