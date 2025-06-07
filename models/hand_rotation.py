"""
Created by: Xiaoyi Xiong
Date: 02/05/2025
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

# def compute_hand_rotation(wrist, thumb, index):
#     # calculate two vectors
#     v1 = np.array(thumb) - np.array(wrist)
#     v2 = np.array(index) - np.array(wrist)
#
#     # normal vector
#     normal = np.cross(v1, v2)
#
#     # normalization
#     normal = normal / np.linalg.norm(normal)
#
#     # 计算旋转矩阵
#     ref_vector = np.array([0, 0, 1])  # 参考方向（Z 轴向上）
#     rot_matrix = np.linalg.inv(R.align_vectors([normal], [ref_vector])[0].as_matrix())
#
#     # 转换为欧拉角 (yaw, pitch, roll)
#     yaw, pitch, roll = R.from_matrix(rot_matrix).as_euler('zyx', degrees=True)
#     # z-axis, y-axis, x-axis
#
#     return normal, yaw, pitch, roll


def palm_nomal_vector(wrist, thumb, index):
    # calculate two vectors
    v1 = np.array(thumb) - np.array(wrist)
    v2 = np.array(index) - np.array(wrist)

    # normal vector
    normal = np.cross(v1, v2)

    # normalization
    normal = normal / np.linalg.norm(normal)
    return normal


def compute_hand_rotation(wrist, thumb, index):
    v1 = np.array(thumb) - np.array(wrist)
    v2 = np.array(index) - np.array(wrist)
    normal = np.cross(v1, v2)

    # 防止零向量
    if np.linalg.norm(normal) == 0:
        return [np.nan, np.nan, np.nan, np.nan]

    normal = normal / np.linalg.norm(normal)
    ref_vector = np.array([0, 0, 1])  # Z轴向上

    try:
        rot_matrix = np.linalg.inv(R.align_vectors([normal], [ref_vector])[0].as_matrix())
        yaw, pitch, roll = R.from_matrix(rot_matrix).as_euler('zyx', degrees=True)
        return [normal, yaw, pitch, roll]
    except Exception:
        return [np.nan, np.nan, np.nan, np.nan]

def calculate_hand_orientations(wrist_seq, thumb_seq, index_seq, keyframes, midpoints):
    """
    wrist_seq, thumb_seq, index_seq: 每一帧的关键点坐标序列（list of [x, y, z]）
    keyframes: 所有关键帧的集合，如set([0, 25, 56, ...])
    midpoints: 所有中点帧的集合，如set([13, 41, ...])
    """
    num_frames = len(wrist_seq)
    hand_orientation = []

    # 合并所有需要计算角度的帧索引
    # valid_frames = set(keyframes).union(midpoints)
    valid_frames = set(midpoints) | {f for pair in keyframes for f in pair}
    for i in range(num_frames):
        if i in valid_frames:
            print(f"[Frame {i}] wrist: {wrist_seq[i]}, thumb: {thumb_seq[i]}, index: {index_seq[i]}")
            result = compute_hand_rotation(wrist_seq[i], thumb_seq[i], index_seq[i])
        else:
            result = [np.nan, np.nan, np.nan, np.nan]
        hand_orientation.append(result)

    return hand_orientation