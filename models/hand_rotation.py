"""
Created by: Xiaoyi Xiong
Date: 02/05/2025
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

#
# def palm_nomal_vector(wrist, thumb, index):
#     # calculate two vectors
#     v1 = np.array(thumb) - np.array(wrist)
#     v2 = np.array(index) - np.array(wrist)
#
#     # normal vector
#     normal = np.cross(v1, v2)
#
#     # normalization
#     normal = normal / np.linalg.norm(normal)
#     return normal
#
#
# def compute_hand_rotation(wrist, thumb, index):
#     v1 = np.array(thumb) - np.array(wrist)
#     v2 = np.array(index) - np.array(wrist)
#     normal = np.cross(v1, v2)
#
#     # 防止零向量
#     if np.linalg.norm(normal) == 0:
#         return [np.nan, np.nan, np.nan, np.nan]
#
#     normal = normal / np.linalg.norm(normal)
#     # ref_vector = np.array([0, 0, 1])  # Z轴向上
#     ref_vector = np.array([-1, 0, 0])   # 手掌朝前为参考方向
#
#     def round15(v):
#         return int(np.round(v / 15.0)) * 15
#
#     try:
#         rot_matrix = np.linalg.inv(R.align_vectors([normal], [ref_vector])[0].as_matrix())
#         yaw, pitch, roll = R.from_matrix(rot_matrix).as_euler('xyz', degrees=True)
#         # yaw, pitch, roll = R.from_matrix(rot_matrix).as_euler('zyx', degrees=True)
#         return [normal, round15(roll), round15(pitch), round15(yaw)]
#         # return [normal, round15(yaw), round15(pitch), round15(roll)]
#     except Exception:
#         return [np.nan, np.nan, np.nan, np.nan]
#
# def calculate_hand_orientations(wrist_seq, thumb_seq, index_seq, keyframes, midpoints):
#     """
#     wrist_seq, thumb_seq, index_seq: 每一帧的关键点坐标序列（list of [x, y, z]）
#     keyframes: 所有关键帧的集合，如set([0, 25, 56, ...])
#     midpoints: 所有中点帧的集合，如set([13, 41, ...])
#     """
#     num_frames = len(wrist_seq)
#     hand_orientation = []
#
#     # 合并所有需要计算角度的帧索引
#     # valid_frames = set(keyframes).union(midpoints)
#     # valid_frames = set(midpoints) | {f for pair in keyframes for f in pair}
#     for i in range(num_frames):
#         # print(f"[Frame {i}] wrist: {wrist_seq[i]}, thumb: {thumb_seq[i]}, index: {index_seq[i]}")
#         result = compute_hand_rotation(wrist_seq[i], thumb_seq[i], index_seq[i])
#         # result = [np.nan, np.nan, np.nan, np.nan]
#         hand_orientation.append(result)
#     # for i in range(num_frames):
#     #     if i in valid_frames:
#     #         # print(f"[Frame {i}] wrist: {wrist_seq[i]}, thumb: {thumb_seq[i]}, index: {index_seq[i]}")
#     #         result = compute_hand_rotation(wrist_seq[i], thumb_seq[i], index_seq[i])
#     #     else:
#     #         result = [np.nan, np.nan, np.nan, np.nan]
#     #     hand_orientation.append(result)
#
#     return hand_orientation

def compute_hand_rotation(wrist, thumb, index, ref_matrix=None):
    if any(np.any(np.isnan(p)) for p in [wrist, thumb, index]):
        return [np.nan, np.nan, np.nan, np.nan]

    # Step 1: 构造手掌法向量（z轴：掌心方向）
    v1 = np.array(thumb) - np.array(wrist)
    v2 = np.array(index) - np.array(wrist)
    normal = np.cross(v1, v2)
    if np.linalg.norm(normal) == 0:
        return [np.nan, np.nan, np.nan, np.nan]
    z_axis = normal / np.linalg.norm(normal)

    # Step 2: 手部局部 y 轴（指向食指）
    y_axis = np.array(index) - np.array(wrist)
    if np.linalg.norm(y_axis) == 0:
        return [np.nan, np.nan, np.nan, np.nan]
    y_axis = y_axis / np.linalg.norm(y_axis)

    # Step 3: 手部局部 x 轴（保持正交性）
    x_axis = np.cross(y_axis, z_axis)
    if np.linalg.norm(x_axis) == 0:
        return [np.nan, np.nan, np.nan, np.nan]
    x_axis = x_axis / np.linalg.norm(x_axis)

    # Step 4: 构建手部旋转矩阵（每列是局部坐标轴在世界坐标下的表示）
    R_hand = np.stack([x_axis, y_axis, z_axis], axis=1)  # shape: (3, 3)

    # Step 5: 定义参考坐标系（stop手势）
    if ref_matrix is None:
        ref_matrix = np.stack([
            np.array([1, 0, 0]),   # x: 向右
            np.array([0, 1, 0]),   # y: 向上
            np.array([0, 0, 1])    # z: 向前（掌心朝前）
        ], axis=1)

    # Step 6: 相对旋转矩阵
    R_delta = ref_matrix.T @ R_hand

    try:
        euler_deg = R.from_matrix(R_delta).as_euler('xyz', degrees=True)

        def round15(v):
            return int(np.round(v / 15.0)) * 15

        x_angle = round15(euler_deg[0])
        y_angle = round15(euler_deg[1])
        z_angle = round15(euler_deg[2])

        return [z_axis, x_angle, y_angle, z_angle]
    except Exception:
        return [np.nan, np.nan, np.nan, np.nan]


def calculate_hand_orientations(wrist_seq, thumb_seq, index_seq, keyframes=None, midpoints=None):
    num_frames = len(wrist_seq)
    hand_orientation = []

    for i in range(num_frames):
        wrist = wrist_seq[i]
        thumb = thumb_seq[i]
        index = index_seq[i]
        result = compute_hand_rotation(wrist, thumb, index)
        hand_orientation.append(result)

    return hand_orientation