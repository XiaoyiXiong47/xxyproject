"""
Created by: Xiaoyi Xiong
Date: 11/04/2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d
import cv2
import mediapipe as mp
from mpl_toolkits.mplot3d import Axes3D
import extract_coordinates
from typing import List, Tuple
from scipy.spatial.distance import cosine

def is_same_shape(angle1: np.ndarray, angle2: np.ndarray, threshold: float = 0.95) -> bool:
    """比较两个角度向量是否表示相似的手型"""
    similarity = 1 - cosine(angle1, angle2)
    return similarity > threshold


def segment_signs_from_velocity_and_shape(
        angles: List[np.ndarray],  # 每帧的手指角度向量
        velocity: np.ndarray  # 速度序列（已平滑）
    ) -> List[Tuple[int, int]]:
    """
    基于速度极小值分段，并合并角度相似的段
    :param angles: 每帧的手型角度向量（shape=(n_frames, n_dims)）
    :param velocity: 手腕的平滑速度曲线
    :return: List of (start_frame, end_frame)
    """

    # 第一步：用速度极小值作为 keyframes
    def find_velocity_minima(v):
        from scipy.signal import argrelextrema
        minima = argrelextrema(v, np.less, order=3)[0]
        return minima.tolist()

    def merge_close_keyframes(keyframes, min_gap=5):
        if not keyframes:
            return []
        merged = [keyframes[0]]
        for kf in keyframes[1:]:
            if kf - merged[-1] >= min_gap:
                merged.append(kf)
        return merged

    keyframes = find_velocity_minima(velocity)
    keyframes = merge_close_keyframes(sorted(set(keyframes)), min_gap=5)

    # 第二步：生成 segments
    segments = [(keyframes[i], keyframes[i + 1]) for i in range(len(keyframes) - 1)]
    print("segments:", segments)

    # 第三步：按手型相似性合并段
    merged_segments = []
    if not segments:
        return merged_segments

    prev_start, prev_end = segments[0]
    prev_shape = np.mean(angles[prev_start:prev_end], axis=0)

    for start, end in segments[1:]:
        current_shape = np.mean(angles[start:end], axis=0)
        if is_same_shape(prev_shape, current_shape):
            prev_end = end  # 合并
        else:
            merged_segments.append((prev_start, prev_end))
            prev_start, prev_end = start, end
            prev_shape = current_shape

    merged_segments.append((prev_start, prev_end))
    return merged_segments

def detect_pause_then_motion(hand_landmarks, pause_length=5, pause_thresh=0.001, motion_thresh=0.002):
    """
    检测“短暂停顿 + 紧接运动”模式。
    :param positions: 手腕或掌心坐标序列 [ [x, y, z], ... ]
    :param pause_length: 静止的最短帧数（例如帧率为30时，5帧=0.17s）
    :param pause_thresh: 静止速度阈值
    :param motion_thresh: 开始移动的速度阈值
    :return: 所有检测到的动作开始帧索引（list）
    """
    velocity = compute_velocity(hand_landmarks)
    n = len(velocity)
    motion_starts = []

    i = 0
    while i < n - pause_length - 1:
        # 检测是否是静止段
        if np.all(velocity[i:i+pause_length] < pause_thresh):
            # 检查静止段之后是否立即开始移动
            for j in range(i + pause_length, min(i + pause_length + 5, n)):
                if velocity[j] > motion_thresh:
                    motion_starts.append(j)
                    i = j + 1  # 跳过这段，避免重复检测
                    break
            else:
                i += 1
        else:
            i += 1
    return motion_starts


def load_video(file_path):
    """
    Use OpenCV to load given sign video.
    :param file_path: The path of sign video
    :return:
    """

    video_id = file_path.split('\\')[-1]
    cap = cv2.VideoCapture(file_path)
    return cap, video_id


# -------------------------------
# Step 1. 数据输入（假设 Nx21x3 的 numpy array）
# -------------------------------
# input: hand_landmarks[frame_idx, joint_idx, coord]，比如 (300, 21, 3)

def get_wrist_trajectory(hand_landmarks):
    return hand_landmarks[:, 0, :]  # wrist 是 index 0


def interpolate_nan_rows(data):
    """
    Linear interpolate nan rows
    :param data: (N x 3) array contains nan
    :return:
    """
    for dim in range(data.shape[1]):
        valid = ~np.isnan(data[:, dim])
        if np.sum(valid) == 0:
            # 如果整列都是 NaN，跳过插值
            continue
        data[:, dim] = np.interp(
            x=np.arange(len(data)),
            xp=np.flatnonzero(valid),
            fp=data[valid, dim]
        )
    return data




# -------------------------------
# Step 2. 提取手腕轨迹 + 平滑
# -------------------------------

def compute_velocity(trajectory):
    # trajectory: Nx3
    diffs = np.diff(trajectory, axis=0)
    velocity = np.linalg.norm(diffs, axis=1)
    return velocity


def smooth_signal(signal, sigma=2):
    return gaussian_filter1d(signal, sigma=sigma)


# -------------------------------
# Step 3. 识别运动段（去掉静止）
# -------------------------------

def get_movement_segment(velocity, threshold=0.01):
    if velocity is None or len(velocity) == 0:
        return 0, 0  # 返回空段
    is_moving = velocity > threshold
    if not np.any(is_moving):
        return 0, 0
    start = np.argmax(is_moving)
    end = len(is_moving) - np.argmax(is_moving[::-1])
    return start, end


# -------------------------------
# Step 4. 关键帧检测
# -------------------------------

def find_velocity_minima(velocity, order=10):
    return argrelextrema(velocity, np.less, order=order)[0]


def find_direction_changes(trajectory, threshold_deg=45):
    v1 = trajectory[1:-1] - trajectory[:-2]
    v2 = trajectory[2:] - trajectory[1:-1]
    dot = np.sum(v1 * v2, axis=1)
    norm = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    angle_rad = np.arccos(np.clip(dot / (norm + 1e-8), -1, 1))
    angle_deg = np.degrees(angle_rad)
    return np.where(angle_deg > threshold_deg)[0] + 1


def find_position_extremes(trajectory):
    if trajectory is None or len(trajectory) == 0:
        return []  # 或 return None，具体取决于后续处理方式
    x_ext = [np.argmax(trajectory[:, 0]), np.argmin(trajectory[:, 0])]
    y_ext = [np.argmax(trajectory[:, 1]), np.argmin(trajectory[:, 1])]
    z_ext = [np.argmax(trajectory[:, 2]), np.argmin(trajectory[:, 2])]
    return list(set(x_ext + y_ext + z_ext))



def detect_turning_points_2d(x, y, smooth_sigma=2, threshold_k=1.0):
    """

    :param trajectory:
    :param smooth_sigma:
    :return:
    """
    # x = trajectory[:, 0]
    # y = trajectory[:, 1]

    # 1. 计算一阶导数（速度）
    dx = np.gradient(x)
    dy = np.gradient(y)

    # 2. 计算二阶导数（加速度）
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # 3. 计算曲率
    numerator = dx * ddy - dy * ddx
    denominator = (dx ** 2 + dy ** 2) ** 1.5
    curvature = numerator / (denominator + 1e-8)  # 防止除0

    # 4. 平滑曲率
    curvature_smooth = gaussian_filter1d(curvature, sigma=smooth_sigma)

    # 5. 检测曲率局部极大值
    turning_idx = argrelextrema(curvature_smooth, np.greater)[0]

    if len(turning_idx) == 0:
        return np.array([]), curvature_smooth

        # 6. 自适应阈值筛选
    curvature_at_turning = np.abs(curvature_smooth[turning_idx])
    mean_c = np.mean(curvature_at_turning)
    std_c = np.std(curvature_at_turning)
    threshold = mean_c + threshold_k * std_c

    # 7. 保留曲率超过 threshold 的点
    selected_idx = turning_idx[curvature_at_turning > threshold]
    return turning_idx, curvature_smooth



def detect_adaptive_significant_turning_points_3d(x, y, z, smooth_sigma=2, threshold_k=0.5):
    # 1. 一阶导数（速度）
    dx = np.gradient(x)
    dy = np.gradient(y)
    dz = np.gradient(z)

    # 2. 二阶导数（加速度）
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    ddz = np.gradient(dz)

    # 3. 计算叉乘
    cross_prod = np.cross(np.vstack((dx, dy, dz)).T, np.vstack((ddx, ddy, ddz)).T)
    cross_norm = np.linalg.norm(cross_prod, axis=1)

    # 4. 计算速度模
    velocity_norm = np.sqrt(dx**2 + dy**2 + dz**2)

    # 5. 曲率计算
    curvature = cross_norm / (velocity_norm**3 + 1e-8)  # 防止除0

    # 6. 平滑曲率
    curvature_smooth = gaussian_filter1d(curvature, sigma=smooth_sigma)

    # 7. 曲率局部极大值
    turning_idx = argrelextrema(curvature_smooth, np.greater)[0]

    if len(turning_idx) == 0:
        return np.array([]), curvature_smooth

    # 8. 自适应阈值筛选
    curvature_at_turning = np.abs(curvature_smooth[turning_idx])
    mean_c = np.mean(curvature_at_turning)
    std_c = np.std(curvature_at_turning)
    threshold = mean_c + threshold_k * std_c

    selected_idx = turning_idx[curvature_at_turning > threshold]

    return selected_idx, curvature_smooth


def remove_close_points_by_max_angle(indices, angles_all, min_gap=10):
    if len(indices) == 0:
        return np.array([])

    indices = np.array(sorted(indices))
    filtered = []
    group = [indices[0]]

    for idx in indices[1:]:
        if idx - group[-1] < min_gap:
            group.append(idx)
        else:
            # ⚠️ 先把 group 映射到 angles_all 的合法范围内
            valid_group = [i for i in group if i < len(angles_all)]
            if not valid_group:
                group = [idx]
                continue
            best_idx = valid_group[np.argmax(angles_all[np.array(valid_group)])]
            filtered.append(best_idx)
            group = [idx]

    if group:
        valid_group = [i for i in group if i < len(angles_all)]
        if valid_group:
            best_idx = valid_group[np.argmax(angles_all[np.array(valid_group)])]
            filtered.append(best_idx)

    return np.array(filtered)


def detect_turning_points_auto_with_best(x, y, z, step_ratio=0.03, min_step=5, k=1.0, min_gap=10):
    """
    自动选择step和阈值，并且相近点只保留角度最大的
    """
    points = np.vstack((x, y, z)).T
    n_points = len(points)

    # automatically determine step
    step = max(int(n_points * step_ratio), min_step)
    # if step >= n_points // 2:
    #     step = n_points // 4
    step = 5

    vectors = points[step:] - points[:-step]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    vectors_norm = vectors / norms

    cos_angles = np.sum(vectors_norm[:-1] * vectors_norm[1:], axis=1)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles = np.arccos(cos_angles) * 180 / np.pi

    mean_angle = np.mean(angles)
    std_angle = np.std(angles)
    angle_threshold_deg = mean_angle + k * std_angle

    # 初步检测拐点
    turning_points_idx = np.where(angles > angle_threshold_deg)[0] + step

    # 去重：优先保留角度最大的
    turning_points_idx_filtered = remove_close_points_by_max_angle(turning_points_idx, angles, min_gap=min_gap)

    return turning_points_idx_filtered, angles, step, angle_threshold_deg


# -------------------------------
# Step 5. 整合关键帧
# -------------------------------


def merge_close_keyframes(keyframes, min_gap=5):
    merged = []
    last = -min_gap
    for k in sorted(keyframes):
        if k - last >= min_gap:
            merged.append(k)
            last = k
    return merged


def detect_keyframes(trajectory, velocity):
    velocity_minima = find_velocity_minima(velocity)
    direction_changes = find_direction_changes(trajectory)
    extremes = find_position_extremes(trajectory)
    # combine all key frames candidates
    all_candidates = sorted(set(velocity_minima) | set(direction_changes) | set(extremes))
    # merge key frames that are too close
    filtered_keyframes = merge_close_keyframes(all_candidates, min_gap=5)

    return filtered_keyframes

def detect_keyframes_by_velocity(velocity):
    velocity_minima = find_velocity_minima(velocity)
    # merge key frames that are too close
    filtered_keyframes = merge_close_keyframes(sorted(set(velocity_minima)), min_gap=5)

    return filtered_keyframes

def detect_keyframes_by_trajectory(trajectory):
    direction_changes = find_direction_changes(trajectory)
    extremes = find_position_extremes(trajectory)
    candidates = sorted(set(direction_changes) | set(extremes))
    filtered_keyframes = merge_close_keyframes(candidates, min_gap=5)
    return filtered_keyframes


# -------------------------------
# Step 6. 可视化（2D）
# -------------------------------

def plot_keyframes(leftorright, trajectory, keyframes):
    plt.figure(figsize=(8, 6))
    plt.plot(trajectory[:, 0], trajectory[:, 1], label="Trajectory")
    plt.scatter(trajectory[keyframes, 0], trajectory[keyframes, 1], color='red', label="Keyframes")
    plt.legend()
    plt.title(f'{leftorright} Wrist Trajectory + Keyframes')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.show()


# -------------------------------
# 主函数封装
# -------------------------------

def detect_keyframes_using_multi_indicators(left_wrist, right_wrist, left_angles, right_angles, left_normal, right_normal):
    """

    :param left_wrist:
    :param right_wrist:
    :param left_angles:
    :param right_angles:
    :param left_normal:
    :param right_normal:
    :return:
    """

    left_seg = []
    right_seg = []
    left_mid = []
    right_mid = []
    return left_seg, right_seg, left_mid, right_mid


def process_hand_landmarks(left_wrist, right_wrist, left_angles, right_angles):
    """
    Given a sequence of left and right wrist, also the joint angles at each frame.
    Return two lists containing tuples (start, end) for each hand.
    :param left_wrist: List of coordinates of left wrist
    :param right_wrist:List of coordinates of right wrist
    :param left_angles: 每一帧的手指角度特征
    :param right_angles: 每一帧的手指角度特征
    :return:
    """
    left_seg = []
    right_seg = []

    # Process left hand if data is valid
    if isinstance(left_wrist, list) and len(left_wrist) > 0:
        left_wrist_array = np.array(left_wrist)
        if left_wrist_array.ndim == 2 and left_wrist_array.shape[1] == 3:
            left_wrist_trajectory = interpolate_nan_rows(left_wrist_array)
            left_velocity = compute_velocity(left_wrist_trajectory)
            left_velocity_smooth = smooth_signal(left_velocity)
            # print("left_velocity_smooth:", left_velocity_smooth)
            left_seg = segment_signs_from_velocity_and_shape(left_angles, left_velocity_smooth)
            print("left_seg:", left_seg)
        else:
            print(f"[Warning] Invalid left_wrist shape: {left_wrist_array.shape}, skipping left hand processing.")
    else:
        print("[Info] No left hand data provided.")

    # Process right hand if data is valid
    if isinstance(right_wrist, list) and len(right_wrist) > 0:
        right_wrist_array = np.array(right_wrist)
        if right_wrist_array.ndim == 2 and right_wrist_array.shape[1] == 3:
            right_wrist_trajectory = interpolate_nan_rows(right_wrist_array)
            right_velocity = compute_velocity(right_wrist_trajectory)
            right_velocity_smooth = smooth_signal(right_velocity)
            # print("right_velocity_smooth:", right_velocity_smooth)
            right_seg = segment_signs_from_velocity_and_shape(right_angles, right_velocity_smooth)
            # starts = detect_pause_then_motion(right_wrist)
            # print("Detected keyframes after short pauses:", starts)
            print("right_seg:", right_seg)
        else:
            print(f"[Warning] Invalid right_wrist shape: {right_wrist_array.shape}, skipping right hand processing.")
    else:
        print("[Info] No right hand data provided.")

    return left_seg, right_seg

    # -------------------------------

    # left_start, left_end = get_movement_segment(left_velocity_smooth)
    # right_start, right_end = get_movement_segment(right_velocity_smooth)
    #
    # # start, end = get_movement_segment(velocity_smooth)
    # motion_traj_left = left_wrist_trajectory[left_start:left_end]
    # motion_traj_right = right_wrist_trajectory[right_start:right_end]
    #
    # motion_vel_left = left_velocity_smooth[left_start:left_end - 1]  # 注意 diff 少1帧
    # motion_vel_right = left_velocity_smooth[right_start:right_end - 1]  # 注意 diff 少1帧
    #
    # keyframes_left = detect_keyframes(motion_traj_left, motion_vel_left)
    # keyframes_right = detect_keyframes(motion_traj_right, motion_vel_right)

    # # keyframes_left = merge_close_keyframes(keyframes_left, min_gap=5)
    # # keyframes_right = merge_close_keyframes(keyframes_right, min_gap=5)
    #
    # keyframes_left = [k + left_start for k in keyframes_left]  # 恢复原始帧索引
    # keyframes_right = [k + right_start for k in keyframes_right]  # 恢复原始帧索引
    #
    # # plot_keyframes('Left', left_wrist_trajectory, keyframes_left)
    # # plot_keyframes('Right', right_wrist_trajectory, keyframes_right)
    #
    #
    # # return keyframes_left, keyframes_right
    # # 从运动轨迹中检测关键帧之间的中间 turning points
    # midpoints_left = []
    # for i in range(len(keyframes_left) - 1):
    #     seg_start = keyframes_left[i] - left_start
    #     seg_end = keyframes_left[i + 1] - left_start
    #     if seg_start < 0 or seg_end > len(motion_traj_left):
    #         midpoints_left.append([])
    #         continue
    #     seg = motion_traj_left[seg_start:seg_end]
    #     if len(seg) >= 10:
    #         mid_idx, _, _, _ = detect_turning_points_auto_with_best(
    #             seg[:, 0], seg[:, 1], seg[:, 2],
    #             step_ratio=0.03, min_step=5, k=1.0, min_gap=10
    #         )
    #         midpoints_left.append([keyframes_left[i] + j for j in mid_idx])
    #     else:
    #         midpoints_left.append([])
    #
    # midpoints_right = []
    # for i in range(len(keyframes_right) - 1):
    #     seg_start = keyframes_right[i] - right_start
    #     seg_end = keyframes_right[i + 1] - right_start
    #     if seg_start < 0 or seg_end > len(motion_traj_right):
    #         midpoints_right.append([])
    #         continue
    #     seg = motion_traj_right[seg_start:seg_end]
    #     if len(seg) >= 10:
    #         mid_idx, _, _, _ = detect_turning_points_auto_with_best(
    #             seg[:, 0], seg[:, 1], seg[:, 2],
    #             step_ratio=0.03, min_step=5, k=1.0, min_gap=10
    #         )
    #         midpoints_right.append([keyframes_right[i] + j for j in mid_idx])
    #     else:
    #         midpoints_right.append([])
    #
    # return keyframes_left, keyframes_right, midpoints_left, midpoints_right

def main():
    file_path = r'D:\project_codes\WLASL\start_kit\raw_videos\04593.mp4'
    cap, video_id = load_video(file_path)

    # Get video frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    pose, hands, face_mesh = extract_coordinates.init_mediapipe()

    left_hand = []
    right_hand = []

    left_wrist = []
    right_wrist = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # hand detection
        hand_results = hands.process(rgb_frame)
        multi_hands = hand_results.multi_hand_landmarks
        # frame append
        frames.append(rgb_frame)
        if multi_hands:  # if hand key points are detected
            if len(multi_hands) == 2:
                for hand_landmarks, handedness in zip(multi_hands, hand_results.multi_handedness):
                    landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                    wrist = landmarks[0]
                    if handedness.classification[0].label == "Left":
                        left_wrist.append(wrist)
                        left_hand.append(landmarks)
                        # current_left_right["left"] = landmarks
                    elif handedness.classification[0].label == "Right":
                        right_wrist.append(wrist)
                        right_hand.append(landmarks)
                        # current_left_right["right"] = landmarks

            elif len(multi_hands) == 1:
                hand_landmarks = multi_hands[0]
                handedness = hand_results.multi_handedness[0]
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                wrist = landmarks[0]
                if handedness.classification[0].label == "Left":
                    left_wrist.append(wrist)
                    left_hand.append(landmarks)
                    right_wrist.append([np.nan, np.nan, np.nan])
                    right_hand.append(np.full(21, np.nan))
                elif handedness.classification[0].label == "Right":
                    right_wrist.append(wrist)
                    right_hand.append(landmarks)
                    left_wrist.append([np.nan, np.nan, np.nan])
                    left_hand.append(np.full(21, np.nan))

        else:
        #     hand not detected at current frame, insert with [np.nan, np.nan, np.nan]
            left_wrist.append([np.nan, np.nan, np.nan])
            left_hand.append(np.full(21, np.nan))
            right_wrist.append([np.nan, np.nan, np.nan])
            right_hand.append(np.full(21, np.nan))
    k_l, k_r, midpoints_left, midpoints_right = process_hand_landmarks(left_wrist, right_wrist)
    print("k_l:", k_l)
    print("k_r:", k_r)
    print("midpoints_left:", midpoints_left)
    print("midpoints_right:", midpoints_right)


if __name__ == '__main__':
    main()





#
# # t = np.linspace(0, 2*np.pi, 300)
# # x = np.cos(t) + 0.1*np.sin(3*t)
# # y = np.sin(t) + 0.1*np.sin(2*t)
# # z = 0.5*np.sin(2*t)
#
# t = np.linspace(0, 4*np.pi, 500)
# x = np.cos(t) + 0.2*np.sin(5*t)
# y = np.sin(t) + 0.2*np.cos(3*t)
# z = t/np.pi + 0.1*np.sin(7*t)
#
#
# # detecting
# turning_points_idx_filtered, angles, auto_step, auto_angle_threshold = detect_turning_points_auto_with_best(
#     x, y, z,
#     step_ratio=0.03,
#     min_step=5,
#     k=1.0,
#     min_gap=10
# )
#
# print(f"Automatic chosen step: {auto_step}")
# print(f"Automatic chosen angle_threshold_deg: {auto_angle_threshold:.2f}°")
# print(f"The number of turning points being kept: {len(turning_points_idx_filtered)}")
#
# # visualization
# fig = plt.figure(figsize=(10,8))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x, y, z, 'k-', label='Trajectory')
# ax.scatter(x[turning_points_idx_filtered], y[turning_points_idx_filtered], z[turning_points_idx_filtered], color='red', marker='x', s=100, label='Turning Points')
# ax.set_title('Turning Points with Best Angle Selected in Dense Areas')
# ax.legend()
# plt.show()
#


