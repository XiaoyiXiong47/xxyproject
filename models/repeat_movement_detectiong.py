"""
Created by: Xiaoyi Xiong
Date: 01/05/2025
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
# from . import keyframes_detection_and_notation
from utils import preprocess

def init_mediapipe():
    # Initialise MediaPipe Hands and Face Detection
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh
    hands = mp_hands.Hands()
    return mp_hands, mp_face, hands


def load_video(file_path):
    """
    Use OpenCV to load given sign video.
    :param file_path: The path of sign video
    :return:
    """

    video_id = file_path.split('\\')[-1]
    cap = cv2.VideoCapture(file_path)
    return cap, video_id




def extract_features(hand_keypoints):
    """
    Flatten 42 keypoints into 1D array
    hand_keypoints: np.ndarray of shape (n_frames, 42, 3)
    """
    return hand_keypoints.reshape(hand_keypoints.shape[0], -1)  # shape: (n_frames, 126)


def compute_similarity_matrix(features):
    sim_matrix = cosine_similarity(features)  # shape: (n_frames, n_frames)
    return sim_matrix

def plot_similarity_matrix(sim_matrix):
    plt.figure(figsize=(6, 6))
    plt.imshow(sim_matrix, cmap='hot', interpolation='nearest')
    plt.title("Self-Similarity Matrix")
    plt.colorbar()
    plt.show()

def detect_repeats(sim_matrix, similarity_threshold=0.95, min_frame_gap=10):
    """
    查找所有与自己相似的时间段（不太靠近自己）
    返回：[(start1, end1), (start2, end2)] 表示重复动作对
    """
    n = sim_matrix.shape[0]
    repeats = []

    for i in range(n):
        for j in range(i + min_frame_gap, n):
            if sim_matrix[i, j] > similarity_threshold:
                repeats.append((i, j))

    return repeats

def convert_to_array_2d(hand_keypoints):
    """
    将 object-array 或 list 转为标准二维 array: (n_frames, 126)
    """
    if isinstance(hand_keypoints, np.ndarray) and hand_keypoints.ndim == 1:
        return np.stack(hand_keypoints)
    elif isinstance(hand_keypoints, list):
        return np.stack(hand_keypoints)
    return hand_keypoints


def estimate_motion_length(hand_keypoints, fps=30):
    hand_keypoints_array = np.stack(hand_keypoints, axis=0)
    wrist = hand_keypoints_array[:, 0, :]  # 以左手 wrist 为例
    velocity = np.linalg.norm(np.diff(wrist, axis=0), axis=1)
    motion_frames = velocity > np.percentile(velocity, 70)  # 活跃帧
    motion_regions = np.diff(np.where(np.concatenate(([motion_frames[0]],
                                                      motion_frames[:-1] != motion_frames[1:],
                                                      [True])))[0])[::2]
    if len(motion_regions) == 0:
        return 20  # fallback
    return int(np.mean(motion_regions))


def estimate_repeat_count(repeat_pairs):
    """
    基于重复段对的数量估算重复次数（例如 (10, 60), (60, 110) → 重复了3次）
    """
    if not repeat_pairs:
        return 1
    frames = sorted(set([x for pair in repeat_pairs for x in pair]))
    return len(frames) // 2 + 1


file_path = r'D:\project_codes\WLASL\start_kit\raw_videos\04797.mp4'      # three times
cap, video_id = load_video(file_path)
mp_hands, mp_face, hands = init_mediapipe()
frames = []
left = []
right = []
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
            if handedness.classification[0].label == "Left":
                left.append(landmarks)
                # current_left_right["left"] = landmarks
            elif handedness.classification[0].label == "Right":
                right.append(landmarks)
                # current_left_right["right"] = landmarks
            else:
                # hand not detected at current frame, insert with [np.nan, np.nan, np.nan]
                left.append(np.full((21, 3), np.nan))
                right.append(np.full((21, 3), np.nan))

    # show result
    cv2.imshow(video_id, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

left = np.array(left)
right = np.array(right)
hand_keypoints = preprocess.process_frame(left, right)


left_geatures = extract_features(left)


features = extract_features(convert_to_array_2d(hand_keypoints))
sim_matrix = compute_similarity_matrix(features)
plot_similarity_matrix(sim_matrix)

estimated_len = estimate_motion_length(hand_keypoints)
min_frame_gap = int(estimated_len * 0.8)

repeats = detect_repeats(sim_matrix, similarity_threshold=0.7, min_frame_gap=min_frame_gap)
print("Detected repeats:", repeats)

repeat_count = estimate_repeat_count(repeats)
print("Estimated repeat_count:", repeat_count)


print("Features shape:", features.shape)
print("Feature ranges：", np.min(features), "to", np.max(features))
print("Exists NaN?:", np.isnan(features).any())