"""
Created by: Xiaoyi Xiong
Date: 04/03/2025
"""

import argparse
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
import extract_coordinates
import keyframes_detection_and_notation
import hand_rotation
import hand_location
import construct_xml

from sklearn.preprocessing import StandardScaler
#import pandas as pd
import xml.dom.minidom as minidom

#from utils import preprocess

def parse_args():
    parser = argparse.ArgumentParser(description="参数解析函数示例")
    parser.add_argument('--data_path', type=str, required=True,
                        help='The path to the dataset that is being annoted')
    parser.add_argument('--dataset', type=str, required=True, help='The name of the dataset.')
    return parser.parse_args()

def diff_in_z(v1, v2):
    z_diff = abs(v1[2] - v2[2])
    z_thresh = 0.1
    if z_diff > z_thresh:
        return True
    return False




def compute_hand_rotation(wrist, index, picky):
    """Given coordinates of wrist, index, and picky, compute normal vector of the palm"""
    v1 = np.array(picky) - np.array(wrist)
    v2 = np.array(index) - np.array(wrist)

    normal = np.cross(v1, v2)

    # normalisation
    normal = normal / np.linalg.norm(normal)

    # ratation matrix
    ref_vector = np.array([0, 0, 1])  # reference direction（Z-axis）
    rot_matrix = np.linalg.inv(R.align_vectors([normal], [ref_vector])[0].as_matrix())

    # convert to Euler angle (yaw, pitch, roll)
    yaw, pitch, roll = R.from_matrix(rot_matrix).as_euler('zyx', degrees=True)

    return normal, yaw, pitch, roll

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



def review_keyframes(video_path, keyframes):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_idx = 0
    keyframe_set = set(keyframes)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in keyframe_set:
            # 显示帧号
            cv2.putText(frame, f"Keyframe: {frame_idx}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

            # 显示帧画面
            cv2.imshow("Keyframe Viewer", frame)

            # 等待按键，按任意键继续，按'q'退出
            key = cv2.waitKey(0)
            if key == ord('q'):
                print("User quit.")
                break
        else:
            # 非关键帧跳过显示
            pass

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

# --------------------------------------
# Construct and write XML file

def build_element(tag, content):
    # 创建元素和属性
    elem = ET.Element(tag)
    for key, value in content.items():
        if key.startswith("@"):
            attr_name = key[1:]
            elem.set(attr_name, value)
        elif isinstance(value, dict):
            child = build_element(key, value)
            elem.append(child)
        elif isinstance(value, list):
            for item in value:
                child = build_element(key, item)
                elem.append(child)
        else:
            elem.text = str(value)
    return elem

def build_xml_from_dict(root_tag, data_dict):
    root = ET.Element(root_tag)
    for tag, content in data_dict.items():
        root.append(build_element(tag, content))
    return ET.ElementTree(root)


def build_xml_frames_with_frame_index(all_angles):
    root = ET.Element("sequence")

    for idx, angle_row in enumerate(all_angles):
        frame = ET.SubElement(root, "frame", index=str(idx))  # keep frame number

        hand = ET.SubElement(frame, "hand", side="A")  # one hand

        # j1 to j3 for f0 to f4
        for i in range(5):
            j1 = angle_row[i * 3] if i * 3 < len(angle_row) else ""
            j2 = angle_row[i * 3 + 1] if i * 3 + 1 < len(angle_row) else ""
            j3 = angle_row[i * 3 + 2] if i * 3 + 2 < len(angle_row) else ""
            finger = ET.SubElement(hand, f"f{i}")
            finger.set("j1", f"{j1:.1f}" if j1 != "" and not np.isnan(j1) else "")
            finger.set("j2", f"{j2:.1f}" if j2 != "" and not np.isnan(j2) else "")
            finger.set("j3", f"{j3:.1f}" if j3 != "" and not np.isnan(j3) else "")

        # keep structure
        # ET.SubElement(hand, "orientation", xAngle="", yAngle="", zAngle="")
        # 添加 orientation
        orientation = ET.SubElement(hand, "orientation")
        orientation.set("xAngle", f"{yaw:.1f}" if not np.isnan(yaw) else "")
        orientation.set("yAngle", f"{pitch:.1f}" if not np.isnan(pitch) else "")
        orientation.set("zAngle", f"{roll:.1f}" if not np.isnan(roll) else "")
        location = ET.SubElement(hand, "location")
        ET.SubElement(location, "loc", x="", y="", z="")
        ET.SubElement(hand, "movement")

    return ET.ElementTree(root)

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


def first_time():
    """
    Process the input video for the first time. Identify the type of the movement and determine key frames for
    furture notation.
    :return:
    """
    args = parse_args()
    file_path = args.data_path
    dataset = args.dataset

    left_hand, right_hand, left_wrist, right_wrist = extract_coordinates.get_coordinates(file_path)
    print("Coordinates successfully detected!")
    # print("left_hand:", left_hand)
    # print("right_hand:", right_hand)
    # print("left_wrist:", left_wrist)
    # print("right_wrist:", right_wrist)


    keyframes_left, keyframes_right, midpoints_left, midpoints_right = keyframes_detection_and_notation.process_hand_landmarks(
        left_wrist, right_wrist)

    print("keyframes_left:", keyframes_left)
    print("keyframes_right:", keyframes_right)
    print("midpoints_left:", midpoints_left)
    print("midpoints_right:", midpoints_right)

    review_keyframes(file_path, keyframes_left)






def main():
    args = parse_args()
    file_path = args.data_path
    dataset = args.dataset
    # Windows path
    # file_path = r'D:\project_codes\WLASL\start_kit\raw_videos\04593.mp4'      # only left hand detected
    # file_path = r'D:\project_codes\WLASL\start_kit\raw_videos\04797.mp4'      # three times
    # file_path = r'D:\project_codes\WLASL\start_kit\raw_videos\00625.mp4'      # stable hand shape
    # file_path = r'D:\project_codes\WLASL\start_kit\raw_videos\00632.mp4'

    # MacOS path
    # file_path = '/Users/xiongxiaoyi/Downloads/demo/04854.mp4'
    cap, video_id = load_video(file_path)

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

    xml_tree = build_xml_frames_with_frame_index(all_angles)
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
    first_time()
