"""
Created by: Xiaoyi Xiong
Date: 04/03/2025
"""

import argparse
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import extract_coordinates
import keyframes_detection_and_notation
import hand_rotation
import hand_location
import construct_xml
import os
import subprocess
import json

import cv2
import mediapipe as mp
import pandas as pd
import os
from tqdm import tqdm
import argparse

#
# # Initialize MediaPipe models
# mp_holistic = mp.solutions.holistic
# mp_hands = mp.solutions.hands
# mp_pose = mp.solutions.pose
#
# # Define CSV columns (complete WLASL format)
# CSV_COLUMNS = [
#     # Left hand index finger
#     "indexDIP_left_X", "indexDIP_left_Y",
#     "indexDIP_right_X", "indexDIP_right_Y",
#     "indexMCP_left_X", "indexMCP_left_Y",
#     "indexMCP_right_X", "indexMCP_right_Y",
#     "indexPIP_left_X", "indexPIP_left_Y",
#     "indexPIP_right_X", "indexPIP_right_Y",
#     "indexTip_left_X", "indexTip_left_Y",
#     "indexTip_right_X", "indexTip_right_Y",
#
#     # Label
#     "labels",
#
#     # Body landmarks (left)
#     "leftEar_X", "leftEar_Y",
#     "leftElbow_X", "leftElbow_Y",
#     "leftEye_X", "leftEye_Y",
#     "leftShoulder_X", "leftShoulder_Y",
#     "leftWrist_X", "leftWrist_Y",
#
#     # Left hand little finger
#     "littleDIP_left_X", "littleDIP_left_Y",
#     "littleDIP_right_X", "littleDIP_right_Y",
#     "littleMCP_left_X", "littleMCP_left_Y",
#     "littleMCP_right_X", "littleMCP_right_Y",
#     "littlePIP_left_X", "littlePIP_left_Y",
#     "littlePIP_right_X", "littlePIP_right_Y",
#     "littleTip_left_X", "littleTip_left_Y",
#     "littleTip_right_X", "littleTip_right_Y",
#
#     # Left hand middle finger
#     "middleDIP_left_X", "middleDIP_left_Y",
#     "middleDIP_right_X", "middleDIP_right_Y",
#     "middleMCP_left_X", "middleMCP_left_Y",
#     "middleMCP_right_X", "middleMCP_right_Y",
#     "middlePIP_left_X", "middlePIP_left_Y",
#     "middlePIP_right_X", "middlePIP_right_Y",
#     "middleTip_left_X", "middleTip_left_Y",
#     "middleTip_right_X", "middleTip_right_Y",
#
#     # Neck and nose
#     "neck_X", "neck_Y",
#     "nose_X", "nose_Y",
#
#     # Body landmarks (right)
#     "rightEar_X", "rightEar_Y",
#     "rightElbow_X", "rightElbow_Y",
#     "rightEye_X", "rightEye_Y",
#     "rightShoulder_X", "rightShoulder_Y",
#     "rightWrist_X", "rightWrist_Y",
#
#     # Left hand ring finger
#     "ringDIP_left_X", "ringDIP_left_Y",
#     "ringDIP_right_X", "ringDIP_right_Y",
#     "ringMCP_left_X", "ringMCP_left_Y",
#     "ringMCP_right_X", "ringMCP_right_Y",
#     "ringPIP_left_X", "ringPIP_left_Y",
#     "ringPIP_right_X", "ringPIP_right_Y",
#     "ringTip_left_X", "ringTip_left_Y",
#     "ringTip_right_X", "ringTip_right_Y",
#
#     # Root position
#     "root_X", "root_Y",
#
#     # Left hand thumb
#     "thumbCMC_left_X", "thumbCMC_left_Y",
#     "thumbCMC_right_X", "thumbCMC_right_Y",
#     "thumbIP_left_X", "thumbIP_left_Y",
#     "thumbIP_right_X", "thumbIP_right_Y",
#     "thumbMP_left_X", "thumbMP_left_Y",
#     "thumbMP_right_X", "thumbMP_right_Y",
#     "thumbTip_left_X", "thumbTip_left_Y",
#     "thumbTip_right_X", "thumbTip_right_Y",
#
#     # Video info
#     "video_fps", "video_size_height", "video_size_width",
#
#     # Wrists
#     "wrist_left_X", "wrist_left_Y",
#     "wrist_right_X", "wrist_right_Y"
# ]
#
# def extract_video_keypoints(video_path, label, target_length=204, fill_value=0):
#     """Extract and aggregate keypoints for an entire video into one row"""
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#     # Initialize storage for all frames' keypoints
#     frame_data = {col: [] for col in CSV_COLUMNS if col not in ['labels', 'video_fps', 'video_size_height', 'video_size_width']}
#
#     with mp_holistic.Holistic(
#         static_image_mode=False,
#         model_complexity=2,
#         refine_face_landmarks=True
#     ) as holistic:
#
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = holistic.process(frame_rgb)
#
#             # Initialize current frame keypoints with zeros
#             keypoints = {k: 0.0 for k in frame_data.keys()}
#
#             # Process left hand landmarks
#             if results.left_hand_landmarks:
#                 left_hand = results.left_hand_landmarks.landmark
#
#                 # Index finger
#                 keypoints["indexDIP_left_X"] = left_hand[mp_hands.HandLandmark.INDEX_FINGER_DIP].x
#                 keypoints["indexDIP_left_Y"] = left_hand[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
#                 keypoints["indexMCP_left_X"] = left_hand[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
#                 keypoints["indexMCP_left_Y"] = left_hand[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
#                 keypoints["indexPIP_left_X"] = left_hand[mp_hands.HandLandmark.INDEX_FINGER_PIP].x
#                 keypoints["indexPIP_left_Y"] = left_hand[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
#                 keypoints["indexTip_left_X"] = left_hand[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
#                 keypoints["indexTip_left_Y"] = left_hand[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
#
#                 # Little finger
#                 keypoints["littleDIP_left_X"] = left_hand[mp_hands.HandLandmark.PINKY_DIP].x
#                 keypoints["littleDIP_left_Y"] = left_hand[mp_hands.HandLandmark.PINKY_DIP].y
#                 keypoints["littleMCP_left_X"] = left_hand[mp_hands.HandLandmark.PINKY_MCP].x
#                 keypoints["littleMCP_left_Y"] = left_hand[mp_hands.HandLandmark.PINKY_MCP].y
#                 keypoints["littlePIP_left_X"] = left_hand[mp_hands.HandLandmark.PINKY_PIP].x
#                 keypoints["littlePIP_left_Y"] = left_hand[mp_hands.HandLandmark.PINKY_PIP].y
#                 keypoints["littleTip_left_X"] = left_hand[mp_hands.HandLandmark.PINKY_TIP].x
#                 keypoints["littleTip_left_Y"] = left_hand[mp_hands.HandLandmark.PINKY_TIP].y
#
#                 # Middle finger
#                 keypoints["middleDIP_left_X"] = left_hand[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x
#                 keypoints["middleDIP_left_Y"] = left_hand[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y
#                 keypoints["middleMCP_left_X"] = left_hand[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
#                 keypoints["middleMCP_left_Y"] = left_hand[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
#                 keypoints["middlePIP_left_X"] = left_hand[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x
#                 keypoints["middlePIP_left_Y"] = left_hand[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
#                 keypoints["middleTip_left_X"] = left_hand[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
#                 keypoints["middleTip_left_Y"] = left_hand[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
#
#                 # Ring finger
#                 keypoints["ringDIP_left_X"] = left_hand[mp_hands.HandLandmark.RING_FINGER_DIP].x
#                 keypoints["ringDIP_left_Y"] = left_hand[mp_hands.HandLandmark.RING_FINGER_DIP].y
#                 keypoints["ringMCP_left_X"] = left_hand[mp_hands.HandLandmark.RING_FINGER_MCP].x
#                 keypoints["ringMCP_left_Y"] = left_hand[mp_hands.HandLandmark.RING_FINGER_MCP].y
#                 keypoints["ringPIP_left_X"] = left_hand[mp_hands.HandLandmark.RING_FINGER_PIP].x
#                 keypoints["ringPIP_left_Y"] = left_hand[mp_hands.HandLandmark.RING_FINGER_PIP].y
#                 keypoints["ringTip_left_X"] = left_hand[mp_hands.HandLandmark.RING_FINGER_TIP].x
#                 keypoints["ringTip_left_Y"] = left_hand[mp_hands.HandLandmark.RING_FINGER_TIP].y
#
#                 # Thumb
#                 keypoints["thumbCMC_left_X"] = left_hand[mp_hands.HandLandmark.THUMB_CMC].x
#                 keypoints["thumbCMC_left_Y"] = left_hand[mp_hands.HandLandmark.THUMB_CMC].y
#                 keypoints["thumbIP_left_X"] = left_hand[mp_hands.HandLandmark.THUMB_IP].x
#                 keypoints["thumbIP_left_Y"] = left_hand[mp_hands.HandLandmark.THUMB_IP].y
#                 keypoints["thumbMP_left_X"] = left_hand[mp_hands.HandLandmark.THUMB_MCP].x
#                 keypoints["thumbMP_left_Y"] = left_hand[mp_hands.HandLandmark.THUMB_MCP].y
#                 keypoints["thumbTip_left_X"] = left_hand[mp_hands.HandLandmark.THUMB_TIP].x
#                 keypoints["thumbTip_left_Y"] = left_hand[mp_hands.HandLandmark.THUMB_TIP].y
#
#                 # Left wrist
#                 keypoints["wrist_left_X"] = left_hand[mp_hands.HandLandmark.WRIST].x
#                 keypoints["wrist_left_Y"] = left_hand[mp_hands.HandLandmark.WRIST].y
#
#             # Process right hand landmarks
#             if results.right_hand_landmarks:
#                 right_hand = results.right_hand_landmarks.landmark
#
#                 # Index finger
#                 keypoints["indexDIP_right_X"] = right_hand[mp_hands.HandLandmark.INDEX_FINGER_DIP].x
#                 keypoints["indexDIP_right_Y"] = right_hand[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
#                 keypoints["indexMCP_right_X"] = right_hand[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
#                 keypoints["indexMCP_right_Y"] = right_hand[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
#                 keypoints["indexPIP_right_X"] = right_hand[mp_hands.HandLandmark.INDEX_FINGER_PIP].x
#                 keypoints["indexPIP_right_Y"] = right_hand[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
#                 keypoints["indexTip_right_X"] = right_hand[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
#                 keypoints["indexTip_right_Y"] = right_hand[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
#
#                 # Little finger
#                 keypoints["littleDIP_right_X"] = right_hand[mp_hands.HandLandmark.PINKY_DIP].x
#                 keypoints["littleDIP_right_Y"] = right_hand[mp_hands.HandLandmark.PINKY_DIP].y
#                 keypoints["littleMCP_right_X"] = right_hand[mp_hands.HandLandmark.PINKY_MCP].x
#                 keypoints["littleMCP_right_Y"] = right_hand[mp_hands.HandLandmark.PINKY_MCP].y
#                 keypoints["littlePIP_right_X"] = right_hand[mp_hands.HandLandmark.PINKY_PIP].x
#                 keypoints["littlePIP_right_Y"] = right_hand[mp_hands.HandLandmark.PINKY_PIP].y
#                 keypoints["littleTip_right_X"] = right_hand[mp_hands.HandLandmark.PINKY_TIP].x
#                 keypoints["littleTip_right_Y"] = right_hand[mp_hands.HandLandmark.PINKY_TIP].y
#
#                 # Middle finger
#                 keypoints["middleDIP_right_X"] = right_hand[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x
#                 keypoints["middleDIP_right_Y"] = right_hand[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y
#                 keypoints["middleMCP_right_X"] = right_hand[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
#                 keypoints["middleMCP_right_Y"] = right_hand[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
#                 keypoints["middlePIP_right_X"] = right_hand[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x
#                 keypoints["middlePIP_right_Y"] = right_hand[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
#                 keypoints["middleTip_right_X"] = right_hand[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
#                 keypoints["middleTip_right_Y"] = right_hand[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
#
#                 # Ring finger
#                 keypoints["ringDIP_right_X"] = right_hand[mp_hands.HandLandmark.RING_FINGER_DIP].x
#                 keypoints["ringDIP_right_Y"] = right_hand[mp_hands.HandLandmark.RING_FINGER_DIP].y
#                 keypoints["ringMCP_right_X"] = right_hand[mp_hands.HandLandmark.RING_FINGER_MCP].x
#                 keypoints["ringMCP_right_Y"] = right_hand[mp_hands.HandLandmark.RING_FINGER_MCP].y
#                 keypoints["ringPIP_right_X"] = right_hand[mp_hands.HandLandmark.RING_FINGER_PIP].x
#                 keypoints["ringPIP_right_Y"] = right_hand[mp_hands.HandLandmark.RING_FINGER_PIP].y
#                 keypoints["ringTip_right_X"] = right_hand[mp_hands.HandLandmark.RING_FINGER_TIP].x
#                 keypoints["ringTip_right_Y"] = right_hand[mp_hands.HandLandmark.RING_FINGER_TIP].y
#
#                 # Thumb
#                 keypoints["thumbCMC_right_X"] = right_hand[mp_hands.HandLandmark.THUMB_CMC].x
#                 keypoints["thumbCMC_right_Y"] = right_hand[mp_hands.HandLandmark.THUMB_CMC].y
#                 keypoints["thumbIP_right_X"] = right_hand[mp_hands.HandLandmark.THUMB_IP].x
#                 keypoints["thumbIP_right_Y"] = right_hand[mp_hands.HandLandmark.THUMB_IP].y
#                 keypoints["thumbMP_right_X"] = right_hand[mp_hands.HandLandmark.THUMB_MCP].x
#                 keypoints["thumbMP_right_Y"] = right_hand[mp_hands.HandLandmark.THUMB_MCP].y
#                 keypoints["thumbTip_right_X"] = right_hand[mp_hands.HandLandmark.THUMB_TIP].x
#                 keypoints["thumbTip_right_Y"] = right_hand[mp_hands.HandLandmark.THUMB_TIP].y
#
#                 # Right wrist
#                 keypoints["wrist_right_X"] = right_hand[mp_hands.HandLandmark.WRIST].x
#                 keypoints["wrist_right_Y"] = right_hand[mp_hands.HandLandmark.WRIST].y
#
#             # Process pose landmarks
#             if results.pose_landmarks:
#                 pose = results.pose_landmarks.landmark
#
#                 # Left body parts
#                 keypoints["leftEar_X"] = pose[mp_pose.PoseLandmark.LEFT_EAR].x
#                 keypoints["leftEar_Y"] = pose[mp_pose.PoseLandmark.LEFT_EAR].y
#                 keypoints["leftElbow_X"] = pose[mp_pose.PoseLandmark.LEFT_ELBOW].x
#                 keypoints["leftElbow_Y"] = pose[mp_pose.PoseLandmark.LEFT_ELBOW].y
#                 keypoints["leftEye_X"] = pose[mp_pose.PoseLandmark.LEFT_EYE].x
#                 keypoints["leftEye_Y"] = pose[mp_pose.PoseLandmark.LEFT_EYE].y
#                 keypoints["leftShoulder_X"] = pose[mp_pose.PoseLandmark.LEFT_SHOULDER].x
#                 keypoints["leftShoulder_Y"] = pose[mp_pose.PoseLandmark.LEFT_SHOULDER].y
#                 keypoints["leftWrist_X"] = pose[mp_pose.PoseLandmark.LEFT_WRIST].x
#                 keypoints["leftWrist_Y"] = pose[mp_pose.PoseLandmark.LEFT_WRIST].y
#
#                 # Right body parts
#                 keypoints["rightEar_X"] = pose[mp_pose.PoseLandmark.RIGHT_EAR].x
#                 keypoints["rightEar_Y"] = pose[mp_pose.PoseLandmark.RIGHT_EAR].y
#                 keypoints["rightElbow_X"] = pose[mp_pose.PoseLandmark.RIGHT_ELBOW].x
#                 keypoints["rightElbow_Y"] = pose[mp_pose.PoseLandmark.RIGHT_ELBOW].y
#                 keypoints["rightEye_X"] = pose[mp_pose.PoseLandmark.RIGHT_EYE].x
#                 keypoints["rightEye_Y"] = pose[mp_pose.PoseLandmark.RIGHT_EYE].y
#                 keypoints["rightShoulder_X"] = pose[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
#                 keypoints["rightShoulder_Y"] = pose[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
#                 keypoints["rightWrist_X"] = pose[mp_pose.PoseLandmark.RIGHT_WRIST].x
#                 keypoints["rightWrist_Y"] = pose[mp_pose.PoseLandmark.RIGHT_WRIST].y
#
#                 # Center body parts
#                 keypoints["neck_X"] = pose[mp_pose.PoseLandmark.NOSE].x  # Using nose as neck proxy
#                 keypoints["neck_Y"] = pose[mp_pose.PoseLandmark.NOSE].y
#                 keypoints["nose_X"] = pose[mp_pose.PoseLandmark.NOSE].x
#                 keypoints["nose_Y"] = pose[mp_pose.PoseLandmark.NOSE].y
#                 keypoints["root_X"] = (pose[mp_pose.PoseLandmark.LEFT_HIP].x + pose[mp_pose.PoseLandmark.RIGHT_HIP].x) / 2
#                 keypoints["root_Y"] = (pose[mp_pose.PoseLandmark.LEFT_HIP].y + pose[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
#
#             # Append all keypoints for this frame
#             for k, v in keypoints.items():
#                 frame_data[k].append(v)
#
#     cap.release()
#
#     video_row = {
#         'labels': label,
#         'video_fps': fps,
#         'video_size_height': height,
#         'video_size_width': width
#     }
#     for i in frame_data:
#         if len(frame_data[i]) < target_length:
#             frame_data[i] +=  [fill_value] * (target_length - len(frame_data[i]))
#     frame_data.update(video_row)
#     return frame_data
#
#
# def load_gloss_to_label_map(label_txt_path):
#     """从 WLASL100labels.txt 加载 gloss -> label_id（int）"""
#     gloss2label = {}
#     with open(label_txt_path, "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) == 2:
#                 label_id, gloss = parts
#                 gloss2label[gloss] = int(label_id)
#     return gloss2label
#
#
# def load_video_to_gloss_map(json_path):
#     """从 WLASL_v0.3.json 加载 video_id -> gloss 映射"""
#     import json
#     with open(json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#
#     video2gloss = {}
#     for entry in data:
#         gloss = entry["gloss"]
#         for inst in entry["instances"]:
#             video_id = inst["video_id"]
#             video2gloss[video_id] = gloss
#     return video2gloss
#
# def process_single_video(video_dir, video_id, output_dir):
#     gloss2label = load_gloss_to_label_map("../WLASL100labels.txt")
#     video2gloss = load_video_to_gloss_map("../WLASL_v0.3.json")
#
#     if video_id not in video2gloss:
#         print(f"❌ Cannot find video_id '{video_id}' in JSON mapping.")
#         return None
#
#     gloss = video2gloss[video_id]
#     label = gloss2label.get(gloss)
#     if label is None:
#         print(f"❌ Cannot find label for gloss '{gloss}'")
#         return None
#
#     video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
#     video_path = None
#     for ext in video_extensions:
#         candidate = os.path.join(video_dir, video_id + ext)
#         if os.path.exists(candidate):
#             video_path = candidate
#             break
#
#     if video_path is None:
#         print(f"❌ Video file for ID '{video_id}' not found in directory {video_dir}")
#         return None
#
#     print(f"🔍 Processing video: {video_path}")
#     row_data = extract_video_keypoints(video_path, video_id)
#
#     df = pd.DataFrame([row_data], columns=CSV_COLUMNS)
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, f"{video_id}.csv")
#     df.to_csv(output_path, index=False)
#     print(f"✅ Saved keypoints to: {output_path}")
#
def parse_args():
    parser = argparse.ArgumentParser(description="参数解析函数示例")
    parser.add_argument('--video_dir', type=str, required=True,
                        help='The path to the dataset that is being annoted')
    parser.add_argument('--video_id', type=str, required=True,
                        help='The video name, without extension')
    parser.add_argument('--dataset', type=str, required=True, help='The name of the dataset.')
    parser.add_argument('--gloss', type=str, required=False, help='The gloss the video.')
    # parser.add_argument('--gloss', type=str, required=False, help='The gloss of the dataset, will be recognised later')
    return parser.parse_args()


# Calculate angle between two vectors
def calculate_angle(a, b, c, normal):
    """ Angle between keypoints a, b, and c (in degrees) """

    a = np.array(a)  # node 1
    b = np.array(b)  # node 2 - where the angle is calculated
    c = np.array(c)  # node 3

    ba = a - b  # Vector BA
    bc = c - b  # Vector BC

    # calculate angle
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)  # in degrees

def convert_angles(x):
    """convert angle between point a, b, and c to 0-90 [natation]"""
    y = 180 - x
    if 0 <= y and y < 10:
        converted = 0
    elif 10 <= y and y < 25:
        converted = 15
    elif 25 <= y and y < 40:
        converted = 30
    elif 40 <= y and y < 55:
        converted = 45
    elif 55 <= y and y < 70:
        converted = 60
    elif 70 <= y and y < 85:
        converted = 75
    else:
        converted = 90
    return converted    # in notation form

def init_mediapipe():
    # Initialise MediaPipe Hands and Face Detection
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_detection
    hands = mp_hands.Hands()
    face_detection = mp_face.FaceDetection()
    return mp_hands, mp_face, hands, face_detection


def load_video(file_path):
    """
    Use OpenCV to load given sign video.
    :param file_path: The path of sign video
    :return:
    """

    video_id = file_path.split('\\')[-1]
    cap = cv2.VideoCapture(file_path)
    return cap, video_id

def extract_finger_angles_all_frames(hand_landmarks_3d, joint_triplets, normal_vector=None):
    """
    提取每一帧的手指角度特征
    :param hand_landmarks_3d: shape=(n_frames, 21, 3)
    :param joint_triplets: List of (i, j, k)
    :param normal_vector: 参考法向量（可选）
    :return: angles_array, shape=(n_frames, len(joint_triplets))
    """
    n_frames = len(hand_landmarks_3d)
    n_triplets = len(joint_triplets)
    angles_array = np.zeros((n_frames, n_triplets))

    for frame_idx in range(n_frames):
        landmarks = hand_landmarks_3d[frame_idx]
        frame_angles = []
        for i, (a_idx, b_idx, c_idx) in enumerate(joint_triplets):
            a, b, c = landmarks[a_idx], landmarks[b_idx], landmarks[c_idx]
            angle = calculate_angle(a, b, c, normal_vector)
            angle_value = convert_angles(angle)
            frame_angles.append(angle_value)
        angles_array[frame_idx] = frame_angles

    return angles_array


def print_keyframe_angles(hand_name, keyframes, all_angles):
    print(f"\n{hand_name} hand keyframe angles:")
    for idx in keyframes:
        if idx < len(all_angles):
            angles = all_angles[idx]
            if all(np.isnan(angles)):  # 全是 nan，表示该帧未检测到该手
                print(f"[Frame {idx}] No hand detected.")
                continue
            print(f"[Frame {idx}]")
            for f in range(5):
                j1 = angles[f * 3 + 0]
                j2 = angles[f * 3 + 1]
                j3 = angles[f * 3 + 2]
                j1_str = f"{j1:.1f}" if not np.isnan(j1) else "NaN"
                j2_str = f"{j2:.1f}" if not np.isnan(j2) else "NaN"
                j3_str = f"{j3:.1f}" if not np.isnan(j3) else "NaN"
                print(f"  f{f}: j1={j1_str}, j2={j2_str}, j3={j3_str}")


def main():
    """
    Process the input video for the first time. Identify the type of the movement and determine key frames for
    furture notation.
    :return:
    """

    joint_triplets = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4),  # f0 - j1, j2, j3
        (0, 5, 6), (5, 6, 7), (6, 7, 8),  # f1 - j1, j2, j3
        (0, 9, 10), (9, 10, 11), (10, 11, 12),  # f2 - j1, j2, j3
        (0, 13, 14), (13, 14, 15), (14, 15, 16),  # f3 - j1, j2, j3
        (0, 17, 18), (17, 18, 19), (18, 19, 20),  # f4 - j1, j2, j3
    ]
    joint_names = [
        "f0_j1", "f0_j2", "f0_j3",
        "f1_j1", "f1_j2", "f1_j3",
        "f2_j1", "f2_j2", "f2_j3",
        "f3_j1", "f3_j2", "f3_j3",
        "f4_j1", "f4_j2", "f4_j3"
    ]
    args = parse_args()
    video_dir = args.video_dir
    video_id = args.video_id
    dataset = args.dataset
    gloss = args.gloss
    video_path = os.path.join(video_dir, video_id+".mp4")
    print("Now processing:", video_path)
    # output_dir = "..\data\predicted_label"


    # # gloss recognition using Siformer
    # # first pre-preocess the video and extract skeleton data into csv file
    #
    # process_single_video(video_dir, video_id, output_dir)
    # subprocess.run([
    #     "python", "../utils/data_preprocess.py",
    #     "--video_dir", video_dir,
    #     "--video_id", video_id,
    #     "--output_dir", "../slr-model/Siformer/datasets/temp"
    # ]) # this will generate a csv file containing one row of skeleton sequence data of input video into temp.csv file
    #
    # # 上面的代码没问题了
    # # secondly make prediction
    # subprocess.run([
    #     "python",   "../slr-model/Siformer/predict.py",
    #     "--model_path", "../slr-model/Siformer/out-checkpoints/WLASL100v3/checkpoint_t_10.pth",
    #     "--csv_path", "../slr-model/Siformer/datasets/temp/"+video_id+".csv"
    # ], capture_output=True, text=True)
    # predicted_path = os.path.join("../data/predicted_label/", video_id+'.csv')
    # data = pd.read_csv(predicted_path)
    #
    # for item in data:
    #     print(item)
    #     # print(f"Video ID: {item[0]}, Predicted Label: {item[1]}")

    video_output_path = os.path.join(r'D:\project_codes\xxyproject\data\processed\video_with_frame_id', f'frame'+video_id+'.mp4')

    right_hand, left_hand, right_wrist, left_wrist, pose_landmarks = extract_coordinates.get_coordinates(video_path)
    print("Coordinates successfully detected!")


    left_hand_angles = extract_finger_angles_all_frames(left_hand, joint_triplets)
    right_hand_angles = extract_finger_angles_all_frames(right_hand, joint_triplets)


    left_wrist_seq, left_middle_seq, left_pinky_seq = extract_coordinates.extract_keypoints_from_hand_seq(left_hand)
    right_wrist_seq, right_middle_seq, right_pinky_seq = extract_coordinates.extract_keypoints_from_hand_seq(right_hand)


    left_seg, right_seg, left_mid, right_mid = keyframes_detection_and_notation.process_hand_landmarks(left_wrist, right_wrist,left_hand_angles, right_hand_angles)

    left_location = hand_location.calculate_hand_locations(
        pose_landmarks, left_hand, is_left_hand=False
    )
    right_location = hand_location.calculate_hand_locations(
        pose_landmarks, right_hand, is_left_hand=True
    )

    left_orientation = hand_rotation.calculate_hand_orientations(left_wrist_seq, left_middle_seq, left_pinky_seq, is_right=False)
    right_orientation = hand_rotation.calculate_hand_orientations(right_wrist_seq, right_middle_seq, right_pinky_seq, is_right=True)

    cap, _ = load_video(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 设置编码器并创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或使用 'XVID'
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print("Failed to open VideoWriter!")
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)
        #
        # cv2.imshow(video_id, frame)
        # if cv2.waitKey(200) & 0xFF == ord('q'):
        #     break
        frame_idx+=1
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if not gloss:
        gloss = "N/A"

    xml_name = fr"D:\project_codes\xxyproject\data\annotations\out-{video_id}.xml"
    construct_xml.generate_xml(left_hand_angles, right_hand_angles, left_orientation, right_orientation, left_location,right_location, left_seg, right_seg, left_mid, right_mid, dataset, gloss, output_path=xml_name)



def test():
    args = parse_args()
    video_path = args.video_path
    dataset = args.dataset
    # Windows path
    # file_path = r'D:\project_codes\WLASL\start_kit\raw_videos\04593.mp4'      # only left hand detected
    # file_path = r'D:\project_codes\WLASL\start_kit\raw_videos\04797.mp4'      # three times
    # file_path = r'D:\project_codes\WLASL\start_kit\raw_videos\00625.mp4'      # stable hand shape
    # file_path = r'D:\project_codes\WLASL\start_kit\raw_videos\00632.mp4'

    # MacOS path
    # file_path = '/Users/xiongxiaoyi/Downloads/demo/04854.mp4'
    cap, video_id = load_video(video_path)

    # Get video frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    time_per_frame = 1 / fps

    # Initialize variables to store previous wrist positions
    prev_left_wrist = None
    prev_right_wrist = None

    # store every frames
    frames = []
    all_orientations = []
    yaw, pitch, roll = 0, 0, 0

    # Lists to store speed data
    left_speeds = []
    right_speeds = []
    frame_numbers = []

    left_wrist = []
    right_wirst = []
    frame_count = 0

    mp_hands, mp_face, hands, face_detection = init_mediapipe()

    joint_triplets = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4),  # f0 - j1, j2, j3
        (0, 5, 6), (5, 6, 7), (6, 7, 8),  # f1 - j1, j2, j3
        (0, 9, 10), (9, 10, 11), (10, 11, 12),  # f2 - j1, j2, j3
        (0, 13, 14), (13, 14, 15), (14, 15, 16),  # f3 - j1, j2, j3
        (0, 17, 18), (17, 18, 19), (18, 19, 20),  # f4 - j1, j2, j3
    ]
    joint_names = [
        "f0_j1", "f0_j2", "f0_j3",
        "f1_j1", "f1_j2", "f1_j3",
        "f2_j1", "f2_j2", "f2_j3",
        "f3_j1", "f3_j2", "f3_j3",
        "f4_j1", "f4_j2", "f4_j3"
    ]
    all_angles = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # hand detection
        hand_results = hands.process(rgb_frame)

        # frame append
        frames.append(rgb_frame)

        frame_angles = []
        if hand_results.multi_hand_landmarks:   # if hand key points are detected
            print("+++++++++++++++++++++++++++++++++++++")
            print("len(hand_results.multi_hand_landmarks)", len(hand_results.multi_hand_landmarks))
            for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                # # in 2D-coordinate
                # landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in hand_landmarks.landmark]
                # in 3D-coordinate
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                # print("landmarks", np.array(landmarks).flatten())
                # wrist landmark
                wrist = landmarks[0]
                print("wrist_coord", wrist)

                # Determine if it's left or right hand
                if handedness.classification[0].label == "Left":
                    left_w = landmarks[0]
                    print("Left hand")
                    if prev_left_wrist is not None:
                        # Calculate Euclidean distance between current and previous wrist position
                        distance = np.linalg.norm(np.array(left_w) - np.array(prev_left_wrist))
                        speed = distance / time_per_frame
                        left_speeds.append(speed)
                        frame_numbers.append(frame_count)  # Append frame number only when speed is calculated
                    prev_left_wrist = left_w
                else: # right hand
                    right_w = landmarks[0]
                    print("Right hand")
                    if prev_right_wrist is not None:
                        # Calculate Euclidean distance between current and previous wrist position
                        distance = np.linalg.norm(np.array(right_w) - np.array(prev_right_wrist))
                        speed = distance / time_per_frame
                        right_speeds.append(speed)
                        frame_numbers.append(frame_count)
                    prev_right_wrist = right_w

                frame_angles = []

                normal_vector, yaw, pitch, roll = compute_hand_rotation(landmarks[0], landmarks[5], landmarks[17])
                print(f"Yaw: {yaw}, Pitch: {pitch}, Roll: {roll}")
                all_angles.append(frame_angles)


                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


                for triplet in joint_triplets:
                    a, b, c = [landmarks[i] for i in triplet]
                    angle = calculate_angle(a, b, c, normal_vector)
                    frame_angles.append(convert_angles(angle))

                print("frame_count:", frame_count)

                num_finger = 0
                while num_finger <= 4:
                    print(f"f{num_finger}_j1: {frame_angles[num_finger*3 + 0]:.2f}, j2: {frame_angles[num_finger*3 + 1]:.2f}, j3: {frame_angles[num_finger*3 + 2]:.2f}")
                    num_finger += 1


                # draw hand landmarks
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            left_speeds.append(0)
            frame_numbers.append(frame_count)
            frame_angles = [np.nan] * len(joint_triplets)
            all_angles.append(frame_angles)


        # face detection
        face_results = face_detection.process(rgb_frame)
        if face_results.detections:
            for detection in face_results.detections:
                # bounding box
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(frame, bbox, (0, 255, 0), 2)

        frame_count += 1


        # show result
        cv2.imshow(video_id, frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

    keyframes_left, keyframes_right, midpoints_left, midpoints_right = keyframes_detection_and_notation.process_hand_landmarks(
        left_wrist, right_wrist
    )

    print("\n=== Detected Keyframes ===")
    print("Left hand keyframes:", keyframes_left)
    print("Right hand keyframes:", keyframes_right)
    print("Left hand midpoints between keyframes:", midpoints_left)
    print("Right hand midpoints between keyframes:", midpoints_right)

    xml_tree = construct_xml.build_xml_frames_with_frame_index(all_angles)
    construct_xml.save_pretty_xml(xml_tree, "hand_angles.xml")

    print(frame_numbers)

    # Plot the speed curves
    plt.figure(figsize=(10, 5))
    plt.plot(frame_numbers[:len(left_speeds)], left_speeds, label='Left Hand Speed')
    plt.plot(frame_numbers[:len(right_speeds)], right_speeds, label='Right Hand Speed')
    plt.xlabel('Frame Number')
    plt.ylabel('Speed (pixels/second)')
    plt.title('Wrist Speed Over Time')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
    # try_new_method()