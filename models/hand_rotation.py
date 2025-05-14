"""
Created by: Xiaoyi Xiong
Date: 02/05/2025
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

def compute_hand_rotation(wrist, thumb, index):
    # calculate two vectors
    v1 = np.array(thumb) - np.array(wrist)
    v2 = np.array(index) - np.array(wrist)

    # normal vector
    normal = np.cross(v1, v2)

    # normalization
    normal = normal / np.linalg.norm(normal)

    # 计算旋转矩阵
    ref_vector = np.array([0, 0, 1])  # 参考方向（Z 轴向上）
    rot_matrix = np.linalg.inv(R.align_vectors([normal], [ref_vector])[0].as_matrix())

    # 转换为欧拉角 (yaw, pitch, roll)
    yaw, pitch, roll = R.from_matrix(rot_matrix).as_euler('zyx', degrees=True)
    # z-axis, y-axis, x-axis

    return normal, yaw, pitch, roll


