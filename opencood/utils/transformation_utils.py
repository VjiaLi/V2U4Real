# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: MIT


"""
Transformation utils
"""

import numpy as np
from scipy.spatial.transform import Rotation

def x_to_world(pose, degree=True):
    """
    The transformation matrix from local coordinate system to world system 
    based on position and Euler angles.

    Parameters
    ----------
    pose : list or tuple
        [x, y, z, roll, yaw, pitch]
    degree : bool, optional
        If True, the Euler angles are in degrees. Default is True.

    Returns
    -------
    matrix : np.ndarray
        The 4x4 homogeneous transformation matrix.
    """
    x, y, z, roll, yaw, pitch = pose[:]

    # Euler angles → rotation matrix (ZYX order, NED coordinate system)
    R = Rotation.from_euler('ZYX', [yaw, pitch, roll], degrees=degree).as_matrix()
    
    # construct homogeneous transformation matrix
    matrix = np.eye(4)
    matrix[:3, :3] = R
    matrix[:3, 3] = [x, y, z]
    
    return matrix


def x1_to_x2(x1, x2):
    """
    Transformation matrix from x1 to x2.

    Parameters
    ----------
    x1 : list or np.ndarray
        The pose of x1 under world coordinates or
        transformation matrix x1->world
    x2 : list or np.ndarray
        The pose of x2 under world coordinates or
         transformation matrix x2->world

    Returns
    -------
    transformation_matrix : np.ndarray
        The transformation matrix.

    """
    if isinstance(x1, list) and isinstance(x2, list):
        x1_to_world = x_to_world(x1)
        x2_to_world = x_to_world(x2)
        world_to_x2 = np.linalg.inv(x2_to_world)
        transformation_matrix = np.dot(world_to_x2, x1_to_world)

    # object pose is list while lidar pose is transformation matrix
    elif isinstance(x1, list) and not isinstance(x2, list):
        x1_to_world = x_to_world(x1)
        world_to_x2 = x2
        transformation_matrix = np.dot(world_to_x2, x1_to_world)
    # both are numpy matrix
    else:
        world_to_x2 = np.linalg.inv(x2)
        transformation_matrix = np.dot(world_to_x2, x1)

    return transformation_matrix


def dist_to_continuous(p_dist, displacement_dist, res, downsample_rate):
    """
    Convert points discretized format to continuous space for BEV representation.
    Parameters
    ----------
    p_dist : numpy.array
        Points in discretized coorindates.

    displacement_dist : numpy.array
        Discretized coordinates of bottom left origin.

    res : float
        Discretization resolution.

    downsample_rate : int
        Dowmsamping rate.

    Returns
    -------
    p_continuous : numpy.array
        Points in continuous coorindates.

    """
    p_dist = np.copy(p_dist)
    p_dist = p_dist + displacement_dist
    p_continuous = p_dist * res * downsample_rate
    return p_continuous


def get_pairwise_transformation(base_data_dict, max_cav, proj_first):
    """
    Get pair-wise transformation matrix accross different agents.

    Parameters
    ----------
    base_data_dict : dict
        Key : cav id, item: transformation matrix to ego, lidar points.

    max_cav : int
        The maximum number of cav, default 5

    Return
    ------
    pairwise_t_matrix : np.array
        The pairwise transformation matrix across each cav.
        shape: (L, L, 4, 4), L is the max cav number in a scene
        pairwise_t_matrix[i, j] is Tji, i_to_j
    """
    pairwise_t_matrix = np.tile(np.eye(4), (max_cav, max_cav, 1, 1)) # (L, L, 4, 4)

    if proj_first:
        # if lidar projected to ego first, then the pairwise matrix
        # becomes identity
        # no need to warp again in fusion time.

        # pairwise_t_matrix[:, :] = np.identity(4)
        return pairwise_t_matrix
    else:
        t_list = []

        # save all transformation matrix in a list in order first.
        for cav_id, cav_content in base_data_dict.items():
            lidar_pose = cav_content['params']['lidar_pose']
            t_list.append(x_to_world(lidar_pose))  # Twx

        for i in range(len(t_list)):
            for j in range(len(t_list)):
                # identity matrix to self
                if i != j:
                    # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                    # t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                    t_matrix = np.linalg.solve(t_list[j], t_list[i])  # Tjw*Twi = Tji
                    pairwise_t_matrix[i, j] = t_matrix

    return pairwise_t_matrix


def normalize_pairwise_tfm(pairwise_t_matrix, H, W, discrete_ratio, downsample_rate=1):
    """
    normalize the pairwise transformation matrix to affine matrix need by torch.nn.functional.affine_grid()

    pairwise_t_matrix: torch.tensor
        [B, L, L, 4, 4], B batchsize, L max_cav
    H: num.
        Feature map height
    W: num.
        Feature map width
    discrete_ratio * downsample_rate: num.
        One pixel on the feature map corresponds to the actual physical distance
    """

    pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
    pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
    pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
    pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (downsample_rate * discrete_ratio * W) * 2
    pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (downsample_rate * discrete_ratio * H) * 2

    normalized_affine_matrix = pairwise_t_matrix

    return normalized_affine_matrix