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

def load_video(file_path):
    """
    Use OpenCV to load given sign video.
    :param file_path: The path of sign video
    :return:
    """

    video_id = file_path.split('\\')[-1]
    cap = cv2.VideoCapture(file_path)
    return cap, video_id


def init_mediapipe():
    # Initialise MediaPipe Hands and Face Detection
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_detection
    hands = mp_hands.Hands()
    face_detection = mp_face.FaceDetection()
    return mp_hands, mp_face, hands, face_detection


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
    is_moving = velocity > threshold
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

def process_hand_landmarks():

    keyframes = {'left': [], 'right': []}

    # file_path = r'D:\project_codes\WLASL\start_kit\raw_videos\04797.mp4'
    # file_path = r'D:\project_codes\WLASL\start_kit\raw_videos\00629.mp4'
    file_path = '/Users/xiongxiaoyi/Downloads/demo/04851.mp4'
    cap, video_id = load_video(file_path)
    mp_hands, mp_face, hands, face_detection = init_mediapipe()
    frame_index = 0
    frames = []
    left_wrist = []
    right_wrist = []
    # left = []
    # right = []
    # all_hand_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # hand detection
        hand_results = hands.process(rgb_frame)

        # current_left_right = {}
        # current_left_right["left"] = np.full((21, 3), np.nan)
        # current_left_right["right"] = np.full((21, 3), np.nan)
        # frame append
        frames.append(rgb_frame)
        if hand_results.multi_hand_landmarks:  # if hand key points are detected
            for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                wrist = landmarks[0]
                if handedness.classification[0].label == "Left":
                    left_wrist.append(wrist)
                    # current_left_right["left"] = landmarks
                elif handedness.classification[0].label == "Right":
                    right_wrist.append(wrist)
                    # current_left_right["right"] = landmarks
        # else:
        #     hand not detected at current frame, insert with [np.nan, np.nan, np.nan]
            left_wrist.append([np.nan, np.nan, np.nan])
            right_wrist.append([np.nan, np.nan, np.nan])
        #     # left.append(np.full((21, 3), np.nan))
        #     # right.append(np.full((21, 3), np.nan))
        #     current_left_right["left"] = np.full((21, 3), np.nan)
        #     current_left_right["right"] = np.full((21, 3), np.nan)

            # show result
        cv2.imshow(video_id, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # all_hand_frames.append(current_left_right)

    cap.release()
    cv2.destroyAllWindows()


    left_wrist_trajectory = interpolate_nan_rows(np.array(left_wrist))
    right_wrist_trajectory = interpolate_nan_rows(np.array(right_wrist))

    # wrist_traj = get_wrist_trajectory(hand_landmarks)
    # velocity = compute_velocity(wrist_traj)

    left_velocity = compute_velocity(left_wrist_trajectory)
    right_velocity = compute_velocity(right_wrist_trajectory)

    left_velocity_smooth = smooth_signal(left_velocity)
    right_velocity_smooth = smooth_signal(right_velocity)

    left_start, left_end = get_movement_segment(left_velocity_smooth)
    right_start, right_end = get_movement_segment(right_velocity_smooth)

    # start, end = get_movement_segment(velocity_smooth)
    motion_traj_left = left_wrist_trajectory[left_start:left_end]
    motion_traj_right = right_wrist_trajectory[right_start:right_end]

    motion_vel_left = left_velocity_smooth[left_start:left_end - 1]  # 注意 diff 少1帧
    motion_vel_right = left_velocity_smooth[right_start:right_end - 1]  # 注意 diff 少1帧

    keyframes_left = detect_keyframes(motion_traj_left, motion_vel_left)
    keyframes_right = detect_keyframes(motion_traj_right, motion_vel_right)

    # keyframes_left = merge_close_keyframes(keyframes_left, min_gap=5)
    # keyframes_right = merge_close_keyframes(keyframes_right, min_gap=5)

    keyframes_left = [k + left_start for k in keyframes_left]  # 恢复原始帧索引
    keyframes_right = [k + right_start for k in keyframes_right]  # 恢复原始帧索引

    # plot_keyframes('Left', left_wrist_trajectory, keyframes_left)
    # plot_keyframes('Right', right_wrist_trajectory, keyframes_right)


    # return keyframes_left, keyframes_right
    # 从运动轨迹中检测关键帧之间的中间 turning points
    midpoints_left = []
    for i in range(len(keyframes_left) - 1):
        seg_start = keyframes_left[i] - left_start
        seg_end = keyframes_left[i + 1] - left_start
        if seg_start < 0 or seg_end > len(motion_traj_left):
            midpoints_left.append([])
            continue
        seg = motion_traj_left[seg_start:seg_end]
        if len(seg) >= 10:
            mid_idx, _, _, _ = detect_turning_points_auto_with_best(
                seg[:, 0], seg[:, 1], seg[:, 2],
                step_ratio=0.03, min_step=5, k=1.0, min_gap=10
            )
            midpoints_left.append([keyframes_left[i] + j for j in mid_idx])
        else:
            midpoints_left.append([])

    midpoints_right = []
    for i in range(len(keyframes_right) - 1):
        seg_start = keyframes_right[i] - right_start
        seg_end = keyframes_right[i + 1] - right_start
        if seg_start < 0 or seg_end > len(motion_traj_right):
            midpoints_right.append([])
            continue
        seg = motion_traj_right[seg_start:seg_end]
        if len(seg) >= 10:
            mid_idx, _, _, _ = detect_turning_points_auto_with_best(
                seg[:, 0], seg[:, 1], seg[:, 2],
                step_ratio=0.03, min_step=5, k=1.0, min_gap=10
            )
            midpoints_right.append([keyframes_right[i] + j for j in mid_idx])
        else:
            midpoints_right.append([])

    return keyframes_left, keyframes_right, midpoints_left, midpoints_right

k_l, k_r, midpoints_left, midpoints_right = process_hand_landmarks()
print("k_l:", k_l)
print("k_r:", k_r)
print("midpoints_left:", midpoints_left)
print("midpoints_right:", midpoints_right)



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


