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

def find_velocity_minima(velocity, order=5):
    return argrelextrema(velocity, np.less, order=order)[0]


def find_direction_changes(trajectory, threshold_deg=30):
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


# -------------------------------
# Step 5. 整合关键帧
# -------------------------------

def detect_keyframes(trajectory, velocity):
    velocity_minima = find_velocity_minima(velocity)
    direction_changes = find_direction_changes(trajectory)
    extremes = find_position_extremes(trajectory)
    keyframes = sorted(set(velocity_minima) | set(direction_changes) | set(extremes))
    return keyframes


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

    file_path = r'D:\project_codes\WLASL\start_kit\raw_videos\01992.mp4'
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

    keyframes_left = [k + left_start for k in keyframes_left]  # 恢复原始帧索引
    keyframes_right = [k + right_start for k in keyframes_right]  # 恢复原始帧索引

    plot_keyframes('Left', left_wrist_trajectory, keyframes_left)
    plot_keyframes('Right', right_wrist_trajectory, keyframes_right)


    return keyframes_left, keyframes_right


k_l, k_r = process_hand_landmarks()