# """
# Created by: Xiaoyi Xiong
# Date: 02/05/2025
# """
#
# import numpy as np
# from scipy.spatial.transform import Rotation as R
#
# def compute_hand_rotation(wrist, index, pinky, is_right=True):
#     if any(np.any(np.isnan(p)) for p in [wrist, index, pinky]):
#         return [np.nan, np.nan, np.nan, np.nan]
#
#     # Step 1: 构造局部坐标轴
#     v_index = np.array(index) - np.array(wrist)
#     v_pinky = np.array(pinky) - np.array(wrist)
#
#     # z轴：掌心法向量（右手：index × pinky，左手反过来）
#     # normal = np.cross(v_index, v_pinky)
#     normal = np.cross(v_index, v_pinky) if is_right else np.cross(v_pinky, v_index)
#     if np.linalg.norm(normal) == 0:
#         return [np.nan, np.nan, np.nan, np.nan]
#     z_axis = normal / np.linalg.norm(normal)
#
#     # y轴：指向 index
#     if np.linalg.norm(v_index) == 0:
#         return [np.nan, np.nan, np.nan, np.nan]
#     y_axis = v_index / np.linalg.norm(v_index)
#
#     # # x轴：正交
#     # x_axis = np.cross(y_axis, z_axis)
#     # if np.linalg.norm(x_axis) == 0:
#     #     return [np.nan, np.nan, np.nan, np.nan]
#     # x_axis = x_axis / np.linalg.norm(x_axis)
#
#     # x-axis: from index_mcp to pinky_mcp
#     x_v = np.array(pinky) - np.array(index)
#     if np.linalg.norm(x_v) == 0:
#         return [np.nan, np.nan, np.nan, np.nan]
#     x_axis = x_v / np.linalg.norm(x_v)
#
#     # 手部姿态旋转矩阵
#     R_hand = np.stack([x_axis, y_axis, z_axis], axis=1)
#
#     # 参考坐标系：stop 手势
#     x_ref = np.array([1, 0, 0]) if is_right else np.array([-1, 0, 0])
#     y_ref = np.array([0, -1, 0])  # 始终竖直向上
#     z_ref = np.array([0, 0, 1])  # 朝向镜头
#     R_ref = np.stack([x_ref, y_ref, z_ref], axis=1)
#
#     # 相对旋转矩阵
#     R_delta = R_ref.T @ R_hand
#
#     try:
#         euler_deg = R.from_matrix(R_delta).as_euler('yxz', degrees=True)
#
#         def round15(v):
#             return int(np.round(v / 15.0)) * 15
#
#         y_angle = round15(euler_deg[0])
#         x_angle = round15(euler_deg[1])
#         z_angle = round15(euler_deg[2])
#
#         return [z_axis, x_angle, y_angle, z_angle]
#     except Exception:
#         return [np.nan, np.nan, np.nan, np.nan]
#
#
# def calculate_hand_orientations(wrist_seq, index_seq, pinky_seq, is_right=True):
#     num_frames = len(wrist_seq)
#     hand_orientation = []
#
#     for i in range(num_frames):
#         wrist = wrist_seq[i]
#         index = index_seq[i]
#         pinky = pinky_seq[i]
#         result = compute_hand_rotation(wrist, index, pinky, is_right=is_right)
#         hand_orientation.append(result)
#
#     return hand_orientation

"""
Created by: Xiaoyi Xiong
Date: 02/05/2025
Fixed version with proper left/right hand symmetry
"""

# import numpy as np
# from scipy.spatial.transform import Rotation as R
# def mediapipe_to_standard_coords(point):
#     """
#     将MediaPipe坐标系转换为标准右手坐标系
#     MediaPipe: x右, y下, z远离摄像头
#     标准系统: x右, y上, z朝向摄像头
#     """
#     x, y, z = point
#     return np.array([x, -y, -z])  # y和z都取反
# def normalize_vector(v):
#     """归一化向量"""
#     norm = np.linalg.norm(v)
#     if norm == 0:
#         return v
#     return v / norm
#
# def compute_hand_rotation(wrist, index, pinky, is_right=True):
#     """
#     基于MediaPipe坐标系的手部旋转计算 (YXZ顺序)
#
#     参数:
#     wrist: 手腕坐标 [x, y, z]
#     index: 食指MCP关节坐标 [x, y, z]
#     pinky: 小指MCP关节坐标 [x, y, z]
#     is_right: 是否为右手
#
#     返回:
#     [z_axis, x_angle, y_angle, z_angle] 或 [np.nan, np.nan, np.nan, np.nan]
#     """
#
#     # 检查输入数据的有效性
#     if any(np.any(np.isnan(p)) for p in [wrist, index, pinky]):
#         return [np.nan, np.nan, np.nan, np.nan]
#
#     # 转换为numpy数组
#     wrist = np.array(wrist)
#     index = np.array(index)
#     pinky = np.array(pinky)
#
#     try:
#         # Step 1: 构造当前帧的局部坐标系
#         # Y轴：从wrist指向index
#         v_index = index - wrist
#         if np.linalg.norm(v_index) == 0:
#             return [np.nan, np.nan, np.nan, np.nan]
#         y_axis = normalize_vector(v_index)
#
#         # 从wrist到pinky的向量
#         v_pinky = pinky - wrist
#         if np.linalg.norm(v_pinky) == 0:
#             return [np.nan, np.nan, np.nan, np.nan]
#
#         # Z轴：掌心法向量（使用叉积）
#         # 对于右手和左手，使用不同的叉积顺序来保证掌心方向一致
#         if is_right:
#             # 右手：y_axis × v_pinky，掌心向外
#             z_axis = np.cross(y_axis, v_pinky)
#         else:
#             # 左手：v_pinky × y_axis，掌心向外
#             z_axis = np.cross(v_pinky, y_axis)
#
#         if np.linalg.norm(z_axis) == 0:
#             return [np.nan, np.nan, np.nan, np.nan]
#         z_axis = normalize_vector(z_axis)
#
#         # X轴：通过y轴和z轴的叉积得到，确保右手坐标系
#         x_axis = normalize_vector(np.cross(y_axis, z_axis))
#
#         # 构建当前帧的旋转矩阵
#         R_hand = np.column_stack([x_axis, y_axis, z_axis])
#
#         # Step 2: 定义参考坐标系（Stop手势）
#         # 左右手使用相同的参考坐标系
#         x_ref = np.array([1, 0, 0])       # X轴向右
#         y_ref = np.array([0, 1, 0])       # Y轴向上
#         z_ref = np.array([0, 0, 1])       # Z轴向前（掌心向外）
#
#         R_ref = np.column_stack([x_ref, y_ref, z_ref])
#
#         # Step 3: 计算从参考坐标系到当前坐标系的相对旋转
#         R_delta = R_ref.T @ R_hand
#
#         # Step 4: 使用scipy提取YXZ顺序的Euler角度
#         # 抑制万向锁警告
#         import warnings
#         with warnings.catch_warnings():
#             warnings.filterwarnings("ignore", message="Gimbal lock detected")
#             euler_rad = R.from_matrix(R_delta).as_euler('yxz', degrees=False)
#
#         y_angle = euler_rad[0]  # Y轴旋转
#         x_angle = euler_rad[1]  # X轴旋转
#         z_angle = euler_rad[2]  # Z轴旋转
#
#         # 角度标准化到[-180, 180]度范围
#         def normalize_angle(angle_rad):
#             angle_deg = np.degrees(angle_rad)
#             # 标准化到[-180, 180]
#             while angle_deg > 180:
#                 angle_deg -= 360
#             while angle_deg < -180:
#                 angle_deg += 360
#             return angle_deg
#
#         y_angle_deg = normalize_angle(y_angle)
#         x_angle_deg = normalize_angle(x_angle)
#         z_angle_deg = normalize_angle(z_angle)
#
#         # 对于右手手，调整Y轴角度使其与左手在掌心相对时保持一致
#         # 当左手Y轴为正值时，左手Y轴取负值，反之亦然
#         if is_right:
#             y_angle_deg = -y_angle_deg
#             # 重新标准化到[-180, 180]范围
#             while y_angle_deg > 180:
#                 y_angle_deg -= 360
#             while y_angle_deg < -180:
#                 y_angle_deg += 360
#
#         # 四舍五入到15度的倍数
#         def round15_deg(angle_deg):
#             return int(np.round(angle_deg / 15.0)) * 15
#
#         y_angle_deg = round15_deg(y_angle_deg)
#         x_angle_deg = round15_deg(x_angle_deg)
#         z_angle_deg = round15_deg(z_angle_deg)
#
#         return [z_axis, x_angle_deg, y_angle_deg, z_angle_deg]
#
#     except Exception as e:
#         print(f"计算手部旋转时出错: {e}")
#         return [np.nan, np.nan, np.nan, np.nan]
#
# def calculate_hand_orientations(wrist_seq, index_seq, pinky_seq, is_right=True):
#     """
#     计算一序列帧的手部朝向
#
#     参数:
#     wrist_seq: 手腕坐标序列
#     index_seq: 食指MCP关节坐标序列
#     pinky_seq: 小指MCP关节坐标序列
#     is_right: 是否为右手
#
#     返回:
#     手部朝向结果列表
#     """
#     num_frames = len(wrist_seq)
#     hand_orientation = []
#
#     for i in range(num_frames):
#         wrist = wrist_seq[i]
#         index = index_seq[i]
#         pinky = pinky_seq[i]
#         result = compute_hand_rotation(wrist, index, pinky, is_right=is_right)
#         hand_orientation.append(result)
#
#     return hand_orientation
#
# def verify_rotation(wrist, index, pinky, is_right=True):
#     """
#     验证旋转计算的正确性
#     通过重构坐标系并与原始坐标系比较
#     """
#     result = compute_hand_rotation(wrist, index, pinky, is_right)
#
#     if any(np.isnan(x) if not isinstance(x, np.ndarray) else np.any(np.isnan(x)) for x in result):
#         print("计算失败，返回NaN值")
#         return
#
#     z_axis, x_angle, y_angle, z_angle = result
#
#     print(f"原始坐标:")
#     print(f"  Wrist: {wrist}")
#     print(f"  Index: {index}")
#     print(f"  Pinky: {pinky}")
#     print(f"  是否右手: {is_right}")
#
#     print(f"\n计算得到的旋转角度:")
#     print(f"  Y轴旋转: {y_angle:.2f}°")
#     print(f"  X轴旋转: {x_angle:.2f}°")
#     print(f"  Z轴旋转: {z_angle:.2f}°")
#
#     print(f"\n计算得到的Z轴向量:")
#     print(f"  Z轴: {z_axis}")
#
# # 使用示例和测试
# def example_usage():
#     """使用示例"""
#
#     print("=== 手部旋转计算示例 ===")
#
#     # 测试对称性：左右手执行相同旋转应该得到相同角度
#     print("\n=== 对称性测试 ===")
#
#     # Stop手势（参考位置）
#     print("1. Stop手势测试:")
#     # 右手：wrist在中心，index向上，pinky向左上
#     wrist_r = [0.3, 0.5, 0.1]
#     index_r = [0.35, 0.3, 0.1]  # 向上
#     pinky_r = [0.2, 0.3, 0.1]  # 向左上
#
#     # 左手：wrist在中心，index向上，pinky向右上（镜像位置）
#     wrist_l = [0.5, 0.5, 0.0]
#     index_l = [0.5, 0.7, 0.0]  # 向上
#     pinky_l = [0.7, 0.6, 0.0]  # 向右上
#
#     result_r = compute_hand_rotation(wrist_r, index_r, pinky_r, is_right=True)
#     result_l = compute_hand_rotation(wrist_l, index_l, pinky_l, is_right=False)
#
#     print(f"  右手: Y={result_r[2]:.1f}°, X={result_r[1]:.1f}°, Z={result_r[3]:.1f}°")
#     print(f"  左手: Y={result_l[2]:.1f}°, X={result_l[1]:.1f}°, Z={result_l[3]:.1f}°")
#
#     # 掌心相对（沿Y轴内旋）
#     print("\n2. 掌心相对测试（预期Y轴旋转约-90°）:")
#     # 右手向左转：pinky相对wrist向前
#     wrist_r = [0.5, 0.5, 0.0]
#     index_r = [0.5, 0.7, 0.0]   # 向上
#     pinky_r = [0.5, 0.6, 0.2]   # 向前（右手内旋）
#
#     # 左手向右转：pinky相对wrist向前
#     wrist_l = [0.5, 0.5, 0.0]
#     index_l = [0.5, 0.7, 0.0]   # 向上
#     pinky_l = [0.5, 0.6, 0.2]   # 向前（左手内旋）
#
#     result_r = compute_hand_rotation(wrist_r, index_r, pinky_r, is_right=True)
#     result_l = compute_hand_rotation(wrist_l, index_l, pinky_l, is_right=False)
#
#     print(f"  右手: Y={result_r[2]:.1f}°, X={result_r[1]:.1f}°, Z={result_r[3]:.1f}°")
#     print(f"  左手: Y={result_l[2]:.1f}°, X={result_l[1]:.1f}°, Z={result_l[3]:.1f}°")
#
#     # 掌心向下（沿X轴旋转）
#     print("\n3. 掌心向下测试（预期X轴旋转约+90°）:")
#     # 右手：index向前，pinky在index左侧向前
#     wrist_r = [0.5, 0.5, 0.0]
#     index_r = [0.5, 0.5, 0.2]   # 向前
#     pinky_r = [0.3, 0.5, 0.2]   # 左前方
#
#     # 左手：index向前，pinky在index右侧向前
#     wrist_l = [0.5, 0.5, 0.0]
#     index_l = [0.5, 0.5, 0.2]   # 向前
#     pinky_l = [0.7, 0.5, 0.2]   # 右前方
#
#     result_r = compute_hand_rotation(wrist_r, index_r, pinky_r, is_right=True)
#     result_l = compute_hand_rotation(wrist_l, index_l, pinky_l, is_right=False)
#
#     print(f"  右手: Y={result_r[2]:.1f}°, X={result_r[1]:.1f}°, Z={result_r[3]:.1f}°")
#     print(f"  左手: Y={result_l[2]:.1f}°, X={result_l[1]:.1f}°, Z={result_l[3]:.1f}°")
#
#     # 额外的测试：掌心向上
#     print("\n4. 掌心向上测试（预期X轴旋转约-90°）:")
#     # 右手：index向后，pinky在index左侧向后
#     wrist_r = [0.5, 0.5, 0.0]
#     index_r = [0.5, 0.5, -0.2]  # 向后
#     pinky_r = [0.3, 0.5, -0.2]  # 左后方
#
#     # 左手：index向后，pinky在index右侧向后
#     wrist_l = [0.5, 0.5, 0.0]
#     index_l = [0.5, 0.5, -0.2]  # 向后
#     pinky_l = [0.7, 0.5, -0.2]  # 右后方
#
#     result_r = compute_hand_rotation(wrist_r, index_r, pinky_r, is_right=True)
#     result_l = compute_hand_rotation(wrist_l, index_l, pinky_l, is_right=False)
#
#     print(f"  右手: Y={result_r[2]:.1f}°, X={result_r[1]:.1f}°, Z={result_r[3]:.1f}°")
#     print(f"  左手: Y={result_l[2]:.1f}°, X={result_l[1]:.1f}°, Z={result_l[3]:.1f}°")
#
#     # 额外的测试：掌心向上
#     print("\n5. 00623测试（预期y轴旋转约-180°）:")
#     # 右手：index向后，pinky在index左侧向后
#
#
#     wrist_r = [4.64052796e-01,  5.10482192e-01, - 2.24953794e-07]
#     index_r = [0.48069167, 0.42277145, - 0.01834386]
#     pinky_r =  [0.51938581,0.44671547, - 0.01553667]
#
#
#
#     result_r = compute_hand_rotation(wrist_r, index_r, pinky_r, is_right=True)
#     # result_l = compute_hand_rotation(wrist_l, index_l, pinky_l, is_right=False)
#
#     print(f"  右手: Y={result_r[2]:.1f}°, X={result_r[1]:.1f}°, Z={result_r[3]:.1f}°")
#     # print(f"  左手: Y={result_l[2]:.1f}°, X={result_l[1]:.1f}°, Z={result_l[3]:.1f}°")
#
#
#
# if __name__ == "__main__":
#     example_usage()

"""
Created by: Xiaoyi Xiong
Date: 02/05/2025
Modified for MediaPipe coordinate system with middle finger
Fixed version with proper left/right hand symmetry for MediaPipe
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def normalize_vector(v):
    """归一化向量"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def convert_mediapipe_coords(coords):
    """
    将MediaPipe坐标系转换为标准坐标系
    MediaPipe: x右为正, y下为正, z远离摄像头为正
    标准坐标系: x右为正, y上为正, z向前为正
    """
    x, y, z = coords
    return np.array([x, -y, -z])  # y和z都取反


def compute_hand_rotation(wrist, middle, pinky, is_right=True):
    """
    计算手部旋转的Euler角度 (YXZ顺序)

    参数:
    wrist: 手腕坐标 [x, y, z] (MediaPipe坐标系)
    middle: 中指MCP关节坐标 [x, y, z] (MediaPipe坐标系)
    pinky: 小指MCP关节坐标 [x, y, z] (MediaPipe坐标系)
    is_right: 是否为右手

    返回:
    [z_axis, x_angle, y_angle, z_angle] 或 [np.nan, np.nan, np.nan, np.nan]
    """

    # 检查输入数据的有效性
    if any(np.any(np.isnan(p)) for p in [wrist, middle, pinky]):
        return [np.nan, np.nan, np.nan, np.nan]

    # 转换为标准坐标系
    wrist = convert_mediapipe_coords(wrist)
    middle = convert_mediapipe_coords(middle)
    pinky = convert_mediapipe_coords(pinky)

    try:
        # Step 1: 构造当前帧的局部坐标系
        # Y轴：从wrist指向middle
        v_middle = middle - wrist
        if np.linalg.norm(v_middle) == 0:
            return [np.nan, np.nan, np.nan, np.nan]
        y_axis = normalize_vector(v_middle)

        # 从wrist到pinky的向量
        v_pinky = pinky - wrist
        if np.linalg.norm(v_pinky) == 0:
            return [np.nan, np.nan, np.nan, np.nan]

        # Z轴：掌心法向量（使用叉积）
        # 为了保证左右手镜像对称时得到相同角度，我们需要统一坐标系构建方式
        if is_right:
            # 右手：y_axis × v_pinky，掌心向外
            z_axis = np.cross(y_axis, v_pinky)
        else:
            # 左手：为了保证镜像对称，我们先将坐标系在x轴上镜像
            # 镜像后的pinky向量
            v_pinky_mirrored = np.array([-v_pinky[0], v_pinky[1], v_pinky[2]])
            # 使用镜像后的向量计算z轴
            z_axis = np.cross(y_axis, v_pinky_mirrored)
            # 再将z轴在x方向镜像回来
            z_axis = np.array([-z_axis[0], z_axis[1], z_axis[2]])

        if np.linalg.norm(z_axis) == 0:
            return [np.nan, np.nan, np.nan, np.nan]
        z_axis = normalize_vector(z_axis)

        # X轴：通过y轴和z轴的叉积得到，确保右手坐标系
        x_axis = normalize_vector(np.cross(y_axis, z_axis))

        # 构建当前帧的旋转矩阵
        R_hand = np.column_stack([x_axis, y_axis, z_axis])

        # Step 2: 定义参考坐标系（Stop手势）
        # 左右手使用相同的参考坐标系
        x_ref = np.array([1, 0, 0])  # X轴向右
        y_ref = np.array([0, 1, 0])  # Y轴向上
        z_ref = np.array([0, 0, 1])  # Z轴向前（掌心向外）

        R_ref = np.column_stack([x_ref, y_ref, z_ref])

        # Step 3: 计算从参考坐标系到当前坐标系的相对旋转
        R_delta = R_ref.T @ R_hand

        # Step 4: 使用scipy提取YXZ顺序的Euler角度
        # 抑制万向锁警告
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Gimbal lock detected")
            euler_rad = R.from_matrix(R_delta).as_euler('yxz', degrees=False)

        y_angle = euler_rad[0]  # Y轴旋转
        x_angle = euler_rad[1]  # X轴旋转
        z_angle = euler_rad[2]  # Z轴旋转

        # 角度标准化到[-180, 180]度范围
        def normalize_angle(angle_rad):
            angle_deg = np.degrees(angle_rad)
            # 标准化到[-180, 180]
            while angle_deg > 180:
                angle_deg -= 360
            while angle_deg < -180:
                angle_deg += 360
            return angle_deg

        y_angle_deg = normalize_angle(y_angle)
        x_angle_deg = normalize_angle(x_angle)
        z_angle_deg = normalize_angle(z_angle)

        # 四舍五入到15度的倍数
        def round15_deg(angle_deg):
            return int(np.round(angle_deg / 15.0)) * 15

        y_angle_deg = round15_deg(y_angle_deg)
        x_angle_deg = round15_deg(x_angle_deg)
        z_angle_deg = round15_deg(z_angle_deg)

        return [z_axis, x_angle_deg, y_angle_deg, z_angle_deg]

    except Exception as e:
        print(f"计算手部旋转时出错: {e}")
        return [np.nan, np.nan, np.nan, np.nan]


def calculate_hand_orientations(wrist_seq, middle_seq, pinky_seq, is_right=True):
    """
    计算一序列帧的手部朝向

    参数:
    wrist_seq: 手腕坐标序列
    middle_seq: 中指MCP关节坐标序列
    pinky_seq: 小指MCP关节坐标序列
    is_right: 是否为右手

    返回:
    手部朝向结果列表
    """
    num_frames = len(wrist_seq)
    hand_orientation = []

    for i in range(num_frames):
        wrist = wrist_seq[i]
        middle = middle_seq[i]
        pinky = pinky_seq[i]
        result = compute_hand_rotation(wrist, middle, pinky, is_right=is_right)
        hand_orientation.append(result)

    return hand_orientation


def verify_rotation(wrist, middle, pinky, is_right=True):
    """
    验证旋转计算的正确性
    通过重构坐标系并与原始坐标系比较
    """
    result = compute_hand_rotation(wrist, middle, pinky, is_right)

    if any(np.isnan(x) if not isinstance(x, np.ndarray) else np.any(np.isnan(x)) for x in result):
        print("计算失败，返回NaN值")
        return

    z_axis, x_angle, y_angle, z_angle = result

    print(f"原始坐标 (MediaPipe坐标系):")
    print(f"  Wrist: {wrist}")
    print(f"  Middle: {middle}")
    print(f"  Pinky: {pinky}")
    print(f"  是否右手: {is_right}")

    print(f"\n计算得到的旋转角度:")
    print(f"  Y轴旋转: {y_angle:.2f}°")
    print(f"  X轴旋转: {x_angle:.2f}°")
    print(f"  Z轴旋转: {z_angle:.2f}°")

    print(f"\n计算得到的Z轴向量 (标准坐标系):")
    print(f"  Z轴: {z_axis}")


# 使用示例和测试
def example_usage():
    """使用示例 - 使用MediaPipe坐标系"""

    print("=== 手部旋转计算示例 (MediaPipe坐标系) ===")

    # 测试对称性：左右手执行相同旋转应该得到相同角度
    print("\n=== 对称性测试 ===")

    # Stop手势（参考位置）- MediaPipe坐标系
    print("1. Stop手势测试:")
    # 右手：wrist在中心，middle向上（y为负），pinky向左上
    wrist_r = [0.5, 0.5, 0.0]  # 中心位置
    middle_r = [0.5, 0.3, 0.0]  # 向上（MediaPipe中y向下为正，所以向上是负值）
    pinky_r = [0.3, 0.4, 0.0]  # 向左上

    # 左手：wrist在中心，middle向上，pinky向右上（镜像位置）
    wrist_l = [0.5, 0.5, 0.0]  # 中心位置
    middle_l = [0.5, 0.3, 0.0]  # 向上
    pinky_l = [0.7, 0.4, 0.0]  # 向右上（镜像）

    result_r = compute_hand_rotation(wrist_r, middle_r, pinky_r, is_right=True)
    result_l = compute_hand_rotation(wrist_l, middle_l, pinky_l, is_right=False)

    print(f"  右手: Y={result_r[2]:.1f}°, X={result_r[1]:.1f}°, Z={result_r[3]:.1f}°")
    print(f"  左手: Y={result_l[2]:.1f}°, X={result_l[1]:.1f}°, Z={result_l[3]:.1f}°")

    # 掌心相对（沿Y轴内旋）
    print("\n2. 掌心相对测试（预期Y轴旋转约-90°）:")
    # 右手向左转：pinky相对wrist向近摄像头方向（z为负）
    wrist_r = [0.5, 0.5, 0.0]
    middle_r = [0.5, 0.3, 0.0]  # 向上
    pinky_r = [0.5, 0.4, -0.2]  # 向近摄像头（右手内旋）

    # 左手向右转：pinky相对wrist向近摄像头方向（镜像动作）
    wrist_l = [0.5, 0.5, 0.0]
    middle_l = [0.5, 0.3, 0.0]  # 向上
    pinky_l = [0.5, 0.4, -0.2]  # 向近摄像头（左手内旋）

    result_r = compute_hand_rotation(wrist_r, middle_r, pinky_r, is_right=True)
    result_l = compute_hand_rotation(wrist_l, middle_l, pinky_l, is_right=False)

    print(f"  右手: Y={result_r[2]:.1f}°, X={result_r[1]:.1f}°, Z={result_r[3]:.1f}°")
    print(f"  左手: Y={result_l[2]:.1f}°, X={result_l[1]:.1f}°, Z={result_l[3]:.1f}°")

    # 掌心向下（沿X轴旋转）
    print("\n3. 掌心向下测试（预期X轴旋转约+90°）:")
    # 右手：middle向远摄像头，pinky在middle左侧向远摄像头
    wrist_r = [0.5, 0.5, 0.0]
    middle_r = [0.5, 0.5, 0.2]  # 向远摄像头
    pinky_r = [0.3, 0.5, 0.2]  # 左远方

    # 左手：middle向远摄像头，pinky在middle右侧向远摄像头（镜像）
    wrist_l = [0.5, 0.5, 0.0]
    middle_l = [0.5, 0.5, 0.2]  # 向远摄像头
    pinky_l = [0.7, 0.5, 0.2]  # 右远方（镜像）

    result_r = compute_hand_rotation(wrist_r, middle_r, pinky_r, is_right=True)
    result_l = compute_hand_rotation(wrist_l, middle_l, pinky_l, is_right=False)

    print(f"  右手: Y={result_r[2]:.1f}°, X={result_r[1]:.1f}°, Z={result_r[3]:.1f}°")
    print(f"  左手: Y={result_l[2]:.1f}°, X={result_l[1]:.1f}°, Z={result_l[3]:.1f}°")

    # 额外的测试：掌心向上
    print("\n4. 掌心向上测试（预期X轴旋转约-90°）:")
    # 右手：middle向近摄像头，pinky在middle左侧向近摄像头
    wrist_r = [0.5, 0.5, 0.0]
    middle_r = [0.5, 0.5, -0.2]  # 向近摄像头
    pinky_r = [0.3, 0.5, -0.2]  # 左近方

    # 左手：middle向近摄像头，pinky在middle右侧向近摄像头（镜像）
    wrist_l = [0.5, 0.5, 0.0]
    middle_l = [0.5, 0.5, -0.2]  # 向近摄像头
    pinky_l = [0.7, 0.5, -0.2]  # 右近方（镜像）

    result_r = compute_hand_rotation(wrist_r, middle_r, pinky_r, is_right=True)
    result_l = compute_hand_rotation(wrist_l, middle_l, pinky_l, is_right=False)

    print(f"  右手: Y={result_r[2]:.1f}°, X={result_r[1]:.1f}°, Z={result_r[3]:.1f}°")
    print(f"  左手: Y={result_l[2]:.1f}°, X={result_l[1]:.1f}°, Z={result_l[3]:.1f}°")


if __name__ == "__main__":
    example_usage()