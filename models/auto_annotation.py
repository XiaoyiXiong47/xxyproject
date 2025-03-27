"""
Created by: Xiaoyi Xiong
Date: 04/03/2025
"""

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R

# Calculate angle between two vectors
def calculate_angle(a, b, c):
    """ Angle between keypoints a, b, and c (in degrees) """
    a = np.array(a)  # node 1
    b = np.array(b)  # node 2 - where the angle is calculated
    c = np.array(c)  # node 3

    ba = a - b  # Vector BA
    bc = c - b  # Vector BC

    # calculate angle
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # 防止浮点数误差
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


def load_video():
    # load video

    # Windows path
    # file_path = r'D:\project_codes\WLASL\start_kit\raw_videos\04593.mp4'      # only left hand detected
    # file_path = r'D:\project_codes\WLASL\start_kit\raw_videos\04797.mp4'      # three times
    file_path = r'D:\project_codes\WLASL\start_kit\raw_videos\00625.mp4'      # stable hand shape

    # MacOS path
    video_id = file_path.split('\\')[-1]
    cap = cv2.VideoCapture(file_path)
    return cap, video_id


def write_xml(file, element):
    """Initialise the XML file"""
    # create root element
    root = ET.Element("doc")

    # create sem
    sem = ET.SubElement(root, "sem")
    sem.attrib["type"] = element["type"]
    # ...

    # Create the tree and write to a file
    tree = ET.ElementTree(root)
    tree.write(file)


def append_element_to_xml(file, element):
    """ Append element into XML file"""
    # Load the existing XML file
    tree = ET.parse(file)
    root = tree.getroot()

    # Append new data
    new_sem = ET.SubElement(root, "sem")
    new_sem.attrib["type"] = element["type"]
    # ...

    # Save the changes back to the file
    tree.write(file)


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

    return yaw, pitch, roll


def main():
    cap, video_id = load_video()

    # Get video frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    time_per_frame = 1 / fps

    # Initialize variables to store previous wrist positions
    prev_left_wrist = None
    prev_right_wrist = None

    # Lists to store speed data
    left_speeds = []
    right_speeds = []
    frame_numbers = []

    frame_count = 0

    mp_hands, mp_face, hands, face_detection = init_mediapipe()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # hand detection
        hand_results = hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            print("+++++++++++++++++++++++++++++++++++++")
            print("len(hand_results.multi_hand_landmarks)", len(hand_results.multi_hand_landmarks))
            for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in hand_landmarks.landmark]

                # wrist landmark
                wrist = landmarks[0]
                # Determine if it's left or right hand
                if handedness.classification[0].label == "Left":
                    left_wrist = landmarks[0]
                    print("Left hand")
                    if prev_left_wrist is not None:
                        # Calculate Euclidean distance between current and previous wrist position
                        distance = np.linalg.norm(np.array(left_wrist) - np.array(prev_left_wrist))
                        speed = distance / time_per_frame
                        left_speeds.append(speed)
                        frame_numbers.append(frame_count)  # Append frame number only when speed is calculated
                    prev_left_wrist = left_wrist
                else: # right hand
                    right_wrist = landmarks[0]
                    print("Right hand")
                    if prev_right_wrist is not None:
                        # Calculate Euclidean distance between current and previous wrist position
                        distance = np.linalg.norm(np.array(right_wrist) - np.array(prev_right_wrist))
                        speed = distance / time_per_frame
                        right_speeds.append(speed)
                        frame_numbers.append(frame_count)
                    prev_right_wrist = right_wrist

                # calculate bending angle for each finger
                thumb_angles = []
                index_angles = []
                middle_angles = []
                ring_angles = []
                pinky_angles = []
                thumb_angles.append(convert_angles(calculate_angle(landmarks[0], landmarks[1], landmarks[2])))
                thumb_angles.append(convert_angles(calculate_angle(landmarks[1], landmarks[2], landmarks[3])))
                thumb_angles.append(convert_angles(calculate_angle(landmarks[2], landmarks[3], landmarks[4])))

                index_angles.append(convert_angles(calculate_angle(landmarks[0], landmarks[5], landmarks[6])))
                index_angles.append(convert_angles(calculate_angle(landmarks[5], landmarks[6], landmarks[7])))
                index_angles.append(convert_angles(calculate_angle(landmarks[6], landmarks[7], landmarks[8])))

                middle_angles.append(convert_angles(calculate_angle(landmarks[0], landmarks[9], landmarks[10])))
                middle_angles.append(convert_angles(calculate_angle(landmarks[9], landmarks[10], landmarks[11])))
                middle_angles.append(convert_angles(calculate_angle(landmarks[10], landmarks[11], landmarks[12])))

                ring_angles.append(convert_angles(calculate_angle(landmarks[0], landmarks[13], landmarks[14])))
                ring_angles.append(convert_angles(calculate_angle(landmarks[13], landmarks[14], landmarks[15])))
                ring_angles.append(convert_angles(calculate_angle(landmarks[14], landmarks[15], landmarks[16])))

                pinky_angles.append(convert_angles(calculate_angle(landmarks[0], landmarks[17], landmarks[18])))
                pinky_angles.append(convert_angles(calculate_angle(landmarks[17], landmarks[18], landmarks[19])))
                pinky_angles.append(convert_angles(calculate_angle(landmarks[18], landmarks[19], landmarks[20])))

                print(f"f0_j1: {thumb_angles[0]:.2f}, j2: {thumb_angles[1]:.2f}, 3: {thumb_angles[2]:.2f}")
                print(f"f1_j1: {index_angles[0]:.2f}, j2: {index_angles[1]:.2f}, j3: {index_angles[2]:.2f}")
                print(f"f2_j1: {middle_angles[0]:.2f}, j2: {middle_angles[1]:.2f}, j3: {middle_angles[2]:.2f}")
                print(f"f3_j1: {ring_angles[0]:.2f}, j2: {ring_angles[1]:.2f}, j3: {ring_angles[2]:.2f}")
                print(f"f4_j1: {pinky_angles[0]:.2f}, j2: {pinky_angles[1]:.2f}, j3: {pinky_angles[2]:.2f}")


                # draw hand landmarks
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            left_speeds.append(0)
            frame_numbers.append(frame_count)

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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

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

main()