<p align="center">
  <h1 align="center">V2U4Real: A Real-world Large-scale Dataset for Vehicle-to-UAV Cooperative Perception</h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=lrF4Nx8AAAAJ&hl=zh-CN">Weijia Li</a>,
    <a href="https://github.com/VjiaLi/V2U4Real">Haoen Xiang</a>,
    <a href="https://github.com/VjiaLi/V2U4Real">Tianxu Wang</a>,
    <a href="https://github.com/VjiaLi/V2U4Real">Shuaibing Wu</a>,
    <a href="https://scholar.google.com/citations?user=A6spPv_n5qUC&hl=zh-CN&oi=sra">Qiming Xia</a>,
    <a href="https://scholar.google.com/citations?user=kAnv3SkAAAAJ&hl=zh-CN">Cheng Wang</a>,
    <a href="https://scholar.google.com/citations?user=JOoZUmUAAAAJ&hl=zh-CN&oi=sra">Chenglu Wen</a>
  </p>
</p>

<p align="center">
  <strong>CVPR 2026</strong>
</p>

<div align="center">
  <img src="./images/teaser.png" alt="V2U4Real Teaser" width="100%">
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2603.25275">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg" alt="arXiv Paper">
  </a>
  <a href="https://github.com/VjiaLi/V2U4Real">
    <img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page">
  </a>
  <a href="https://huggingface.co/datasets/VJiaLi/V2U4Real/tree/main">
    <img src="https://img.shields.io/badge/Dataset-Hugging%20Face-yellow.svg" alt="Dataset">
  </a>
  <a href="#citation">
    <img src="https://img.shields.io/badge/Citation-BibTeX-green.svg" alt="Citation">
  </a>
  <a href="https://opensource.org/license/MIT">
    <img src="https://img.shields.io/badge/License-MIT-orange.svg" alt="License">
  </a>
</p>

## Dataset Demo
<div align="center">
  <img src="./images/demo_1.gif" alt="V2U4Real demo 1" width="49%">
  <img src="./images/demo_2.gif" alt="V2U4Real demo 2" width="49%">
</div>

## News

- **[2026.02]** V2U4Real is accepted to **CVPR 2026**:boom::boom::boom:.
- **[2026.05]** Codebase is released.
- **[2026.07]** Dataset download links were released.

## TODO

- [x] Release training and inference code
- [x] Release pretrained models
- [x] Release benchmark leaderboard
- [x] Release dataset download links

## Installation
### 1. Clone the repository

```bash
git clone https://github.com/VjiaLi/V2U4Real.git
cd V2U4Real
```
### 2. Create the environment
We recommend using Conda:
```sh
conda env create -f environment.yml
conda activate v2u4real
python setup.py develop
```

If Conda installation fails, you can install the dependencies with pip instead:
```sh
pip install -r requirements.txt
python setup.py develop
```

### 3. Pytorch Installation (>=1.8, tested on 1.8-1.12.0)
Go to https://pytorch.org/ to install pytorch cuda version.

### 4. Install Spconv
```bash
pip install spconv-cu113
```
#### Notes for installing spconv 1.2.1:
1. Make sure your cmake version >= 3.13.2
2. CUDNN and CUDA runtime library (use `nvcc --version` to check) needs to be installed on your machine.


### 4. Compile CUDA ops for 3D box IoU / NMS
  
  ```bash
  python opencood/utils/setup.py build_ext --inplace
  ```

## Data Download
Please check [Hugging Face](https://huggingface.co/datasets/VJiaLi/V2U4Real/tree/main) to download the data.

After downloading the data, please put the data in the following structure:
```shell
├── V2U4Real
│   ├── train
|       |── 2025-07-17-16-12_1
|       |   |──1  # Vehicle Side
|       |   |  |── camera
|       |   |  |   |── left     # left camera images
|       |   |  |   |   |── 000001.jpg
|       |   |  |   |── middle   # center camera images
|       |   |  |   |   |── 000001.jpg
|       |   |  |   |── right    # right camera images
|       |   |  |   |   |── 000001.jpg
|       |   |  |── ouster       # OS-128 LiDAR point clouds
|       |   |  |   |── 000001.pcd
|       |   |  |── ruby         # RS-128 LiDAR point clouds
|       |   |  |   |── 000001.pcd
|       |   |  |── m1           # M1-PLUS LiDAR point clouds
|       |   |  |   |── 000001.pcd
|       |   |  |── yaml         # metadata for each timestamp
|       |   |  |   |── ouster
|       |   |  |   |   |── 000001.yaml
|       |   |  |   |── ruby
|       |   |  |   |   |── 000001.yaml
|       |   |  |   |── m1
|       |   |  |   |   |── 000001.yaml
|       |   |──2  # UAV Side
|       |   |  |── camera       # downward camera images
|       |   |  |   |── 000001.jpg
|       |   |  |── ouster       # OS-128 LiDAR point clouds
|       |   |  |   |── 000001.pcd
|       |   |  |── yaml         # metadata for each timestamp
|       |   |  |   |── ouster
|       |   |  |   |   |── 000001.yaml
|       |── 2025-07-17-16-12_2
│   ├── val
│   ├── test
```

## Quick Start
### Data sequence visualization
To quickly visualize the LiDAR stream in the V2U4Real dataset, first modify the `validate_dir`
in your `opencood/hypes_yaml/visualization.yaml` to the v2u4real data path on your local machine, e.g. `v2u4real/val`,
and then run the following command:
```python
cd ~/V2U4Real
python opencood/visualization/vis_data_sequence.py [--color_mode ${COLOR_RENDERING_MODE}]
```
Arguments Explanation:
- `color_mode` : str type, indicating the lidar color rendering mode. You can choose from 'constant', 'intensity' or 'z-value'.


### Train your model
V2U4Real uses yaml file to configure all the parameters for training. To train your own model
from scratch or a continued checkpoint, run the following command:
```python
python opencood/tools/train.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER} --half]
```
Arguments Explanation:
- `hypes_yaml`: the path of the training configuration file, e.g. `opencood/hypes_yaml/pointpillar_early_fusion.yaml`, meaning you want to train
an early fusion model which utilizes Pointpillar as the backbone.
- `model_dir` (optional) : the path of the checkpoints. This is used to fine-tune the trained models. When the `model_dir` is
given, the trainer will discard the `hypes_yaml` and load the `config.yaml` in the checkpoint folder.
- `half` (optional): If set, the model will be trained with half precision. It cannot be set together with multi-gpu training.

To train on **multiple gpus**, run the following command:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --use_env opencood/tools/train.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}]
```


### Test the model
Before you run the following command, first make sure the `validation_dir` in config.yaml under your checkpoint folder
refers to the testing dataset path, e.g. `v2u4real/test`.

```python
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} --fusion_method ${FUSION_STRATEGY} [--show_vis] [--show_sequence]
```
Arguments Explanation:
- `model_dir`: the path to your saved model.
- `fusion_method`: indicate the fusion strategy, currently support 'early', 'late', and 'intermediate'.
- `show_vis`: whether to visualize the detection overlay with point cloud.
- `show_sequence` : the detection results will visualized in a video stream. It can NOT be set with `show_vis` at the same time.

The evaluation results  will be dumped in the model directory. 

## Benchmark and Model Zoo
### Results of Cooperative 3D object detection
| Method        | Backbone    | Sync AP@0.5 | Sync AP@0.7 | Async AP@0.5 | Async AP@0.7 | Bandwidth | Download Link                                                            |
|--------------|-------------|----------------|----------------|--------------|--------------|-----------|--------------------------------------------------------------------------|
| Late Fusion  | PointPillar | 43.53           |   21.46    |  24.36       | 9.88         |      0.009     |        [url](https://drive.google.com/drive/folders/1ojZE-o_EGa1-KTr8ic-EvM7xDuquHpu0?usp=drive_link)                                                                  |
| Early Fusion | PointPillar  | 51.31         | 27.74         | 30.94        | 13.99      |      3.18     |         [url](https://drive.google.com/drive/folders/1-lBTw0oMd6-S1sV-SLTuCgmW8CSl-eDD?usp=drive_link)              |
| [Where2Comm](https://arxiv.org/abs/2209.12836) | PointPillar | 53.85         | 29.71         | 48.99        | 28.36       |      0.65     |     [url](https://drive.google.com/drive/folders/1OEM_tprtb2J_O4VRkFJ5v0kla_KPPwRP?usp=drive_link)                                                                    |
| [AttFuse](https://arxiv.org/abs/2109.07644)     | PointPillar | 50.20         | 27.33          | 40.51        | 21.92      |     0.65      |      [url](https://drive.google.com/drive/folders/15nE5vHkaxTV-RQDtO75RSIAasMlcHZtV?usp=drive_link)                                                       |
| [V2X-ViT](https://arxiv.org/pdf/2203.10638.pdf)         |PointPillar | 45.51         | 25.99          | 45.74       | 26.00     |     0.65      |     [url](https://drive.google.com/drive/folders/1a_Xeje60e2lJXKbOr_SzsXhywDRDHMtJ?usp=drive_link)                                                                  |
| [CoBEVT](https://arxiv.org/abs/2207.02202)    | PointPillar | 41.99          | 22.38          | 31.47       | 16.05      |   0.65        | [url](https://drive.google.com/drive/folders/1hHYqThIru-Tnw1_CEn9OEAB-nuev0RV1?usp=drive_link)
| [CoAlign](https://arxiv.org/abs/2211.07214)      | PointPillar |    **56.67**     |  **36.61**   | **50.81**  | **33.33**  | 0.65| [url](https://drive.google.com/drive/folders/1h5121PKO8Aljg0ozwHSFqi-kr5tO8B38?usp=drive_link)|
| [ERMVP](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_ERMVP_Communication-Efficient_and_Collaboration-Robust_Multi-Vehicle_Perception_in_Challenging_Environments_CVPR_2024_paper.pdf)      | PointPillar |    45.56     |  22.64  | 28.06  | 14.22  | 0.65| [url](https://drive.google.com/drive/folders/1BS4EOHv9mMFFU04zDDl70hyNWJRl6Jl3?usp=drive_link)|
| [DSRC](https://arxiv.org/abs/2412.10739)      | PointPillar |    54.64     |  31.77   | 47.63  | 26.05  | 0.65| [url](https://drive.google.com/drive/folders/1SfW63JP1WqUgCi2g-KEOokevuxZ3chn_?usp=drive_link)|
| No Fusion (Vehicle only)    | PointPillar | 27.53          | 12.75          | 27.53          | 12.75          |      0.0     |                                                                      |
| No Fusion (UAV only)   | PointPillar | 32.44           | 14.31          | 32.44           | 14.31          |      0.0     |                                                                      |

### Results of Cooperative 3D object tracking

| Method | AMOTA(↑) | AMOTP(↑) | sAMOTA(↑) | MOTA(↑) | MOTP(↑) | MT(↑) | ML(↓) |
|--------|----------|----------|-----------|---------|---------|-------|-------|
| Early Fusion | 19.22 | 39.41 | 56.79 | 59.26 | 64.67 | 67.94 | 23.81 |
| Late Fusion | 14.82 | 34.39 | 51.12 | 50.64 | 64.90 | 48.41 | 37.30 |
| AttFuse | 20.98 | 41.31 | **61.55** | 60.40 | 64.89 | 57.94 | 23.81 |
| Where2comm | 20.07 | 41.49 | 58.52 | 62.07 | 63.04 | 65.08 | 15.87 |
| V2X-ViT | 14.74 | 35.25 | 50.38 | 50.76 | 64.76 | 57.94 | 30.16 |
| CoBEVT | 15.96 | 36.53 | 52.62 | 53.60 | 64.54 | 51.59 | 35.71 |
| CoAlign | **22.08** | **43.11** | 59.03 | **63.49** | **65.43** | 69.05 | 14.29 |
| ERMVP | 19.02 | 37.42 | 56.82 | 58.27 | 63.31 | 35.71 | 49.21 |
| DSRC | 18.67 | 39.72 | 53.20 | 59.74 | 65.03 | **73.02** | **13.49** |
| No Fusion (Vehicle only) | 11.73 | 25.84 | 46.56 | 45.33 | 44.59 | 34.13 | 52.38 |
| No Fusion (UAV only)  | 7.00 | 21.58 | 33.56 | 35.72 | 55.40 | 3.17 | 81.75 |


## Citation
If you find this dataset or code useful in your research, please consider citing our paper.
```bash
@inproceedings{li2026v2u4real,
  title={V2U4Real: A Real-world Large-scale Dataset for Vehicle-to-UAV Cooperative Perception},
  author={Li, Weijia and Xiang, Haoen and Wang, Tianxu and Wu, Shuaibing and Xia, Qiming and Wang, Cheng and Wen, Chenglu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4728--4737},
  year={2026}
}
```

## Acknowledgements
Thanks for the excellent cooperative perception datasets [OPV2V](https://mobility-lab.seas.ucla.edu/opv2v/) and [V2V4Real](https://github.com/ucla-mobility/V2V4Real).

Thanks for the dataset and code support from [DerrickXu](https://github.com/DerrickXuNu).
