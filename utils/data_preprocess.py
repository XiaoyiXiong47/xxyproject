#!/usr/bin/python3
import cv2
import mediapipe as mp
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

# Initialize MediaPipe models
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# Define CSV columns (complete WLASL format)
CSV_COLUMNS = [
    # Left hand index finger
    "indexDIP_left_X", "indexDIP_left_Y",
    "indexDIP_right_X", "indexDIP_right_Y",
    "indexMCP_left_X", "indexMCP_left_Y",
    "indexMCP_right_X", "indexMCP_right_Y",
    "indexPIP_left_X", "indexPIP_left_Y",
    "indexPIP_right_X", "indexPIP_right_Y",
    "indexTip_left_X", "indexTip_left_Y",
    "indexTip_right_X", "indexTip_right_Y",

    # Label
    "labels",

    # Body landmarks (left)
    "leftEar_X", "leftEar_Y",
    "leftElbow_X", "leftElbow_Y",
    "leftEye_X", "leftEye_Y",
    "leftShoulder_X", "leftShoulder_Y",
    "leftWrist_X", "leftWrist_Y",

    # Left hand little finger
    "littleDIP_left_X", "littleDIP_left_Y",
    "littleDIP_right_X", "littleDIP_right_Y",
    "littleMCP_left_X", "littleMCP_left_Y",
    "littleMCP_right_X", "littleMCP_right_Y",
    "littlePIP_left_X", "littlePIP_left_Y",
    "littlePIP_right_X", "littlePIP_right_Y",
    "littleTip_left_X", "littleTip_left_Y",
    "littleTip_right_X", "littleTip_right_Y",

    # Left hand middle finger
    "middleDIP_left_X", "middleDIP_left_Y",
    "middleDIP_right_X", "middleDIP_right_Y",
    "middleMCP_left_X", "middleMCP_left_Y",
    "middleMCP_right_X", "middleMCP_right_Y",
    "middlePIP_left_X", "middlePIP_left_Y",
    "middlePIP_right_X", "middlePIP_right_Y",
    "middleTip_left_X", "middleTip_left_Y",
    "middleTip_right_X", "middleTip_right_Y",

    # Neck and nose
    "neck_X", "neck_Y",
    "nose_X", "nose_Y",

    # Body landmarks (right)
    "rightEar_X", "rightEar_Y",
    "rightElbow_X", "rightElbow_Y",
    "rightEye_X", "rightEye_Y",
    "rightShoulder_X", "rightShoulder_Y",
    "rightWrist_X", "rightWrist_Y",

    # Left hand ring finger
    "ringDIP_left_X", "ringDIP_left_Y",
    "ringDIP_right_X", "ringDIP_right_Y",
    "ringMCP_left_X", "ringMCP_left_Y",
    "ringMCP_right_X", "ringMCP_right_Y",
    "ringPIP_left_X", "ringPIP_left_Y",
    "ringPIP_right_X", "ringPIP_right_Y",
    "ringTip_left_X", "ringTip_left_Y",
    "ringTip_right_X", "ringTip_right_Y",

    # Root position
    "root_X", "root_Y",

    # Left hand thumb
    "thumbCMC_left_X", "thumbCMC_left_Y",
    "thumbCMC_right_X", "thumbCMC_right_Y",
    "thumbIP_left_X", "thumbIP_left_Y",
    "thumbIP_right_X", "thumbIP_right_Y",
    "thumbMP_left_X", "thumbMP_left_Y",
    "thumbMP_right_X", "thumbMP_right_Y",
    "thumbTip_left_X", "thumbTip_left_Y",
    "thumbTip_right_X", "thumbTip_right_Y",

    # Video info
    "video_fps", "video_size_height", "video_size_width",

    # Wrists
    "wrist_left_X", "wrist_left_Y",
    "wrist_right_X", "wrist_right_Y"
]

def extract_video_keypoints(video_path, label, target_length=204, fill_value=0):
    """Extract and aggregate keypoints for an entire video into one row"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize storage for all frames' keypoints
    frame_data = {col: [] for col in CSV_COLUMNS if col not in ['labels', 'video_fps', 'video_size_height', 'video_size_width']}

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        refine_face_landmarks=True
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            # Initialize current frame keypoints with zeros
            keypoints = {k: 0.0 for k in frame_data.keys()}

            # Process left hand landmarks
            if results.left_hand_landmarks:
                left_hand = results.left_hand_landmarks.landmark

                # Index finger
                keypoints["indexDIP_left_X"] = left_hand[mp_hands.HandLandmark.INDEX_FINGER_DIP].x
                keypoints["indexDIP_left_Y"] = left_hand[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
                keypoints["indexMCP_left_X"] = left_hand[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
                keypoints["indexMCP_left_Y"] = left_hand[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
                keypoints["indexPIP_left_X"] = left_hand[mp_hands.HandLandmark.INDEX_FINGER_PIP].x
                keypoints["indexPIP_left_Y"] = left_hand[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
                keypoints["indexTip_left_X"] = left_hand[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
                keypoints["indexTip_left_Y"] = left_hand[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

                # Little finger
                keypoints["littleDIP_left_X"] = left_hand[mp_hands.HandLandmark.PINKY_DIP].x
                keypoints["littleDIP_left_Y"] = left_hand[mp_hands.HandLandmark.PINKY_DIP].y
                keypoints["littleMCP_left_X"] = left_hand[mp_hands.HandLandmark.PINKY_MCP].x
                keypoints["littleMCP_left_Y"] = left_hand[mp_hands.HandLandmark.PINKY_MCP].y
                keypoints["littlePIP_left_X"] = left_hand[mp_hands.HandLandmark.PINKY_PIP].x
                keypoints["littlePIP_left_Y"] = left_hand[mp_hands.HandLandmark.PINKY_PIP].y
                keypoints["littleTip_left_X"] = left_hand[mp_hands.HandLandmark.PINKY_TIP].x
                keypoints["littleTip_left_Y"] = left_hand[mp_hands.HandLandmark.PINKY_TIP].y

                # Middle finger
                keypoints["middleDIP_left_X"] = left_hand[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x
                keypoints["middleDIP_left_Y"] = left_hand[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y
                keypoints["middleMCP_left_X"] = left_hand[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
                keypoints["middleMCP_left_Y"] = left_hand[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
                keypoints["middlePIP_left_X"] = left_hand[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x
                keypoints["middlePIP_left_Y"] = left_hand[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
                keypoints["middleTip_left_X"] = left_hand[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
                keypoints["middleTip_left_Y"] = left_hand[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y

                # Ring finger
                keypoints["ringDIP_left_X"] = left_hand[mp_hands.HandLandmark.RING_FINGER_DIP].x
                keypoints["ringDIP_left_Y"] = left_hand[mp_hands.HandLandmark.RING_FINGER_DIP].y
                keypoints["ringMCP_left_X"] = left_hand[mp_hands.HandLandmark.RING_FINGER_MCP].x
                keypoints["ringMCP_left_Y"] = left_hand[mp_hands.HandLandmark.RING_FINGER_MCP].y
                keypoints["ringPIP_left_X"] = left_hand[mp_hands.HandLandmark.RING_FINGER_PIP].x
                keypoints["ringPIP_left_Y"] = left_hand[mp_hands.HandLandmark.RING_FINGER_PIP].y
                keypoints["ringTip_left_X"] = left_hand[mp_hands.HandLandmark.RING_FINGER_TIP].x
                keypoints["ringTip_left_Y"] = left_hand[mp_hands.HandLandmark.RING_FINGER_TIP].y

                # Thumb
                keypoints["thumbCMC_left_X"] = left_hand[mp_hands.HandLandmark.THUMB_CMC].x
                keypoints["thumbCMC_left_Y"] = left_hand[mp_hands.HandLandmark.THUMB_CMC].y
                keypoints["thumbIP_left_X"] = left_hand[mp_hands.HandLandmark.THUMB_IP].x
                keypoints["thumbIP_left_Y"] = left_hand[mp_hands.HandLandmark.THUMB_IP].y
                keypoints["thumbMP_left_X"] = left_hand[mp_hands.HandLandmark.THUMB_MCP].x
                keypoints["thumbMP_left_Y"] = left_hand[mp_hands.HandLandmark.THUMB_MCP].y
                keypoints["thumbTip_left_X"] = left_hand[mp_hands.HandLandmark.THUMB_TIP].x
                keypoints["thumbTip_left_Y"] = left_hand[mp_hands.HandLandmark.THUMB_TIP].y

                # Left wrist
                keypoints["wrist_left_X"] = left_hand[mp_hands.HandLandmark.WRIST].x
                keypoints["wrist_left_Y"] = left_hand[mp_hands.HandLandmark.WRIST].y

            # Process right hand landmarks
            if results.right_hand_landmarks:
                right_hand = results.right_hand_landmarks.landmark

                # Index finger
                keypoints["indexDIP_right_X"] = right_hand[mp_hands.HandLandmark.INDEX_FINGER_DIP].x
                keypoints["indexDIP_right_Y"] = right_hand[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
                keypoints["indexMCP_right_X"] = right_hand[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
                keypoints["indexMCP_right_Y"] = right_hand[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
                keypoints["indexPIP_right_X"] = right_hand[mp_hands.HandLandmark.INDEX_FINGER_PIP].x
                keypoints["indexPIP_right_Y"] = right_hand[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
                keypoints["indexTip_right_X"] = right_hand[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
                keypoints["indexTip_right_Y"] = right_hand[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

                # Little finger
                keypoints["littleDIP_right_X"] = right_hand[mp_hands.HandLandmark.PINKY_DIP].x
                keypoints["littleDIP_right_Y"] = right_hand[mp_hands.HandLandmark.PINKY_DIP].y
                keypoints["littleMCP_right_X"] = right_hand[mp_hands.HandLandmark.PINKY_MCP].x
                keypoints["littleMCP_right_Y"] = right_hand[mp_hands.HandLandmark.PINKY_MCP].y
                keypoints["littlePIP_right_X"] = right_hand[mp_hands.HandLandmark.PINKY_PIP].x
                keypoints["littlePIP_right_Y"] = right_hand[mp_hands.HandLandmark.PINKY_PIP].y
                keypoints["littleTip_right_X"] = right_hand[mp_hands.HandLandmark.PINKY_TIP].x
                keypoints["littleTip_right_Y"] = right_hand[mp_hands.HandLandmark.PINKY_TIP].y

                # Middle finger
                keypoints["middleDIP_right_X"] = right_hand[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x
                keypoints["middleDIP_right_Y"] = right_hand[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y
                keypoints["middleMCP_right_X"] = right_hand[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
                keypoints["middleMCP_right_Y"] = right_hand[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
                keypoints["middlePIP_right_X"] = right_hand[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x
                keypoints["middlePIP_right_Y"] = right_hand[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
                keypoints["middleTip_right_X"] = right_hand[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
                keypoints["middleTip_right_Y"] = right_hand[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y

                # Ring finger
                keypoints["ringDIP_right_X"] = right_hand[mp_hands.HandLandmark.RING_FINGER_DIP].x
                keypoints["ringDIP_right_Y"] = right_hand[mp_hands.HandLandmark.RING_FINGER_DIP].y
                keypoints["ringMCP_right_X"] = right_hand[mp_hands.HandLandmark.RING_FINGER_MCP].x
                keypoints["ringMCP_right_Y"] = right_hand[mp_hands.HandLandmark.RING_FINGER_MCP].y
                keypoints["ringPIP_right_X"] = right_hand[mp_hands.HandLandmark.RING_FINGER_PIP].x
                keypoints["ringPIP_right_Y"] = right_hand[mp_hands.HandLandmark.RING_FINGER_PIP].y
                keypoints["ringTip_right_X"] = right_hand[mp_hands.HandLandmark.RING_FINGER_TIP].x
                keypoints["ringTip_right_Y"] = right_hand[mp_hands.HandLandmark.RING_FINGER_TIP].y

                # Thumb
                keypoints["thumbCMC_right_X"] = right_hand[mp_hands.HandLandmark.THUMB_CMC].x
                keypoints["thumbCMC_right_Y"] = right_hand[mp_hands.HandLandmark.THUMB_CMC].y
                keypoints["thumbIP_right_X"] = right_hand[mp_hands.HandLandmark.THUMB_IP].x
                keypoints["thumbIP_right_Y"] = right_hand[mp_hands.HandLandmark.THUMB_IP].y
                keypoints["thumbMP_right_X"] = right_hand[mp_hands.HandLandmark.THUMB_MCP].x
                keypoints["thumbMP_right_Y"] = right_hand[mp_hands.HandLandmark.THUMB_MCP].y
                keypoints["thumbTip_right_X"] = right_hand[mp_hands.HandLandmark.THUMB_TIP].x
                keypoints["thumbTip_right_Y"] = right_hand[mp_hands.HandLandmark.THUMB_TIP].y

                # Right wrist
                keypoints["wrist_right_X"] = right_hand[mp_hands.HandLandmark.WRIST].x
                keypoints["wrist_right_Y"] = right_hand[mp_hands.HandLandmark.WRIST].y

            # Process pose landmarks
            if results.pose_landmarks:
                pose = results.pose_landmarks.landmark

                # Left body parts
                keypoints["leftEar_X"] = pose[mp_pose.PoseLandmark.LEFT_EAR].x
                keypoints["leftEar_Y"] = pose[mp_pose.PoseLandmark.LEFT_EAR].y
                keypoints["leftElbow_X"] = pose[mp_pose.PoseLandmark.LEFT_ELBOW].x
                keypoints["leftElbow_Y"] = pose[mp_pose.PoseLandmark.LEFT_ELBOW].y
                keypoints["leftEye_X"] = pose[mp_pose.PoseLandmark.LEFT_EYE].x
                keypoints["leftEye_Y"] = pose[mp_pose.PoseLandmark.LEFT_EYE].y
                keypoints["leftShoulder_X"] = pose[mp_pose.PoseLandmark.LEFT_SHOULDER].x
                keypoints["leftShoulder_Y"] = pose[mp_pose.PoseLandmark.LEFT_SHOULDER].y
                keypoints["leftWrist_X"] = pose[mp_pose.PoseLandmark.LEFT_WRIST].x
                keypoints["leftWrist_Y"] = pose[mp_pose.PoseLandmark.LEFT_WRIST].y

                # Right body parts
                keypoints["rightEar_X"] = pose[mp_pose.PoseLandmark.RIGHT_EAR].x
                keypoints["rightEar_Y"] = pose[mp_pose.PoseLandmark.RIGHT_EAR].y
                keypoints["rightElbow_X"] = pose[mp_pose.PoseLandmark.RIGHT_ELBOW].x
                keypoints["rightElbow_Y"] = pose[mp_pose.PoseLandmark.RIGHT_ELBOW].y
                keypoints["rightEye_X"] = pose[mp_pose.PoseLandmark.RIGHT_EYE].x
                keypoints["rightEye_Y"] = pose[mp_pose.PoseLandmark.RIGHT_EYE].y
                keypoints["rightShoulder_X"] = pose[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
                keypoints["rightShoulder_Y"] = pose[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
                keypoints["rightWrist_X"] = pose[mp_pose.PoseLandmark.RIGHT_WRIST].x
                keypoints["rightWrist_Y"] = pose[mp_pose.PoseLandmark.RIGHT_WRIST].y

                # Center body parts
                keypoints["neck_X"] = pose[mp_pose.PoseLandmark.NOSE].x  # Using nose as neck proxy
                keypoints["neck_Y"] = pose[mp_pose.PoseLandmark.NOSE].y
                keypoints["nose_X"] = pose[mp_pose.PoseLandmark.NOSE].x
                keypoints["nose_Y"] = pose[mp_pose.PoseLandmark.NOSE].y
                keypoints["root_X"] = (pose[mp_pose.PoseLandmark.LEFT_HIP].x + pose[mp_pose.PoseLandmark.RIGHT_HIP].x) / 2
                keypoints["root_Y"] = (pose[mp_pose.PoseLandmark.LEFT_HIP].y + pose[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2

            # Append all keypoints for this frame
            for k, v in keypoints.items():
                frame_data[k].append(v)

    cap.release()

    video_row = {
        'labels': label,
        'video_fps': fps,
        'video_size_height': height,
        'video_size_width': width
    }
    for i in frame_data:
        if len(frame_data[i]) < target_length:
            frame_data[i] +=  [fill_value] * (target_length - len(frame_data[i]))
    frame_data.update(video_row)
    return frame_data


def load_gloss_to_label_map(label_txt_path):
    """从 WLASL100labels.txt 加载 gloss -> label_id（int）"""
    gloss2label = {}
    with open(label_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                label_id, gloss = parts
                gloss2label[gloss] = int(label_id)
    return gloss2label


def load_video_to_gloss_map(json_path, valid_gloss_set=None):
    """从 WLASL_v0.3.json 加载 video_id -> gloss 映射"""
    import json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    video2gloss = {}
    for entry in data:
        gloss = entry["gloss"]
        if valid_gloss_set and gloss not in valid_gloss_set:
            continue
        for inst in entry["instances"]:
            video_id = inst["video_id"]
            video2gloss[video_id] = gloss
    return video2gloss


# def generate_wlasl_csv(video_dir, output_csv="wlasl_output.csv"):
#     """处理所有视频并生成 CSV，每个视频一行，包含 keypoints + label"""
#     all_data = []
#
#     # 读取映射
#     gloss2label = load_gloss_to_label_map("..\WLASL100labels.txt")
#     video2gloss = load_video_to_gloss_map("..\WLASL_v0.3.json", valid_gloss_set=gloss2label.keys())
#
#     # 收集视频文件
#     video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
#     video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(video_extensions)]
#
#     for video_file in tqdm(video_files, desc="Extracting keypoints"):
#         video_path = os.path.join(video_dir, video_file)
#         video_id = os.path.splitext(video_file)[0]
#
#         gloss = video2gloss.get(video_id)
#         if gloss is None:
#             print(f"⚠️ 未在 JSON 中找到 gloss: {video_id}")
#             continue
#
#         label_id = gloss2label.get(gloss)
#         if label_id is None:
#             print(f"⚠️ 未在 labels.txt 中找到 label: {gloss}")
#             continue
#
#         try:
#             video_row = extract_video_keypoints(video_path, label_id)
#             all_data.append(video_row)
#         except Exception as e:
#             print(f"❌ Error processing {video_file}: {str(e)}")
#             continue
#
#     # 写入 CSV
#     final_df = pd.DataFrame(all_data, columns=CSV_COLUMNS)
#     final_df.to_csv(output_csv, index=False)
#     print(f"\n✅ 成功处理 {len(all_data)} 个视频，已保存到 {output_csv}")


def generate_wlasl_csv_multi(split_dirs, output_csv="wlasl_output.csv"):
    """处理多个 split 下的视频并合并成一个大 CSV"""
    all_data = []

    # 加载映射
    gloss2label = load_gloss_to_label_map("..\WLASL100labels.txt")
    video2gloss = load_video_to_gloss_map("..\WLASL_v0.3.json", valid_gloss_set=gloss2label.keys())

    for split_dir in split_dirs:
        print(f"\n📂 正在处理目录: {split_dir}")
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        video_files = [f for f in os.listdir(split_dir) if f.lower().endswith(video_extensions)]

        for video_file in tqdm(video_files, desc=f"Processing {os.path.basename(split_dir)}"):
            video_path = os.path.join(split_dir, video_file)
            video_id = os.path.splitext(video_file)[0]

            gloss = video2gloss.get(video_id)
            if gloss is None:
                print(f"⚠️ 未在 JSON 中找到 gloss: {video_id}")
                continue

            label_id = gloss2label.get(gloss)
            if label_id is None:
                print(f"⚠️ 未在 labels.txt 中找到 label: {gloss}")
                continue

            try:
                video_row = extract_video_keypoints(video_path, label_id)
                all_data.append(video_row)
            except Exception as e:
                print(f"❌ Error processing {video_file}: {str(e)}")
                continue

    # 写入合并的 CSV
    final_df = pd.DataFrame(all_data, columns=CSV_COLUMNS)
    final_df.to_csv(output_csv, index=False)
    print(f"\n✅ 成功处理 {len(all_data)} 个视频，已保存为: {output_csv}")

# Example usage
if __name__ == "__main__":
    # video_directory = r"..\data\raw\WLASL100\train"  # Change to your video directory
    # output_filename = "..\slr-model\Siformer\datasets\wlasl100_train_v2.csv"
    # generate_wlasl_csv(video_directory, output_filename)
    #
    base_dir = r"..\data\raw\WLASL100"
    splits = ["train", "val", "test"]
    split_dirs = [os.path.join(base_dir, s) for s in splits]

    output_csv = "..\slr-model\Siformer\datasets\wlasl100_full.csv"
    generate_wlasl_csv_multi(split_dirs, output_csv)