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
from sklearn.preprocessing import StandardScaler
import pandas as pd
import xml.dom.minidom as minidom

from utils import preprocess

def diff_in_z(v1, v2):
    z_diff = abs(v1[2] - v2[2])
    z_thresh = 0.1
    if z_diff > z_thresh:
        return True
    return False

def save_pretty_xml(xml_tree, filename="hand_angles.xml"):
    xml_str = ET.tostring(xml_tree.getroot(), encoding="utf-8")

    dom = minidom.parseString(xml_str)
    pretty_xml_as_string = dom.toprettyxml(indent="  ")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(pretty_xml_as_string)

    print(f"{filename} saved!")


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

    return normal, yaw, pitch, roll

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
        ET.SubElement(hand, "orientation", xAngle="", yAngle="", zAngle="")
        location = ET.SubElement(hand, "location")
        ET.SubElement(location, "loc", x="", y="", z="")
        ET.SubElement(hand, "movement")

    return ET.ElementTree(root)



def main():
    # Windows path
    # file_path = r'D:\project_codes\WLASL\start_kit\raw_videos\04593.mp4'      # only left hand detected
    file_path = r'D:\project_codes\WLASL\start_kit\raw_videos\04797.mp4'      # three times
    # file_path = r'D:\project_codes\WLASL\start_kit\raw_videos\00625.mp4'      # stable hand shape
    # file_path = r'D:\project_codes\WLASL\start_kit\raw_videos\00626.mp4'
    cap, video_id = load_video(file_path)

    # Get video frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    time_per_frame = 1 / fps

    # Initialize variables to store previous wrist positions
    prev_left_wrist = None
    prev_right_wrist = None

    # store every frames
    frames = []

    # Lists to store speed data
    left_speeds = []
    right_speeds = []
    frame_numbers = []

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

                frame_angles = []

                normal_vector, yaw, pitch, roll = compute_hand_rotation(landmarks[0], landmarks[5], landmarks[17])
                print(f"Yaw: {yaw}, Pitch: {pitch}, Roll: {roll}")

                for triplet in joint_triplets:
                    a, b, c = [landmarks[i] for i in triplet]
                    angle = calculate_angle(a, b, c, normal_vector)
                    frame_angles.append(convert_angles(angle))

                print("frame_count:", frame_count)

                num_finger = 0
                while num_finger <= 4:
                    print(f"f{num_finger}_j1: {frame_angles[num_finger*3 + 0]:.2f}, j2: {frame_angles[num_finger*3 + 1]:.2f}, j3: {frame_angles[num_finger*3 + 2]:.2f}")
                    num_finger += 1

                all_angles.append(frame_angles)


                # # calculate bending angle for each finger
                # thumb_angles = []
                # index_angles = []
                # middle_angles = []
                # ring_angles = []
                # pinky_angles = []
                # thumb_angles.append(convert_angles(calculate_angle(landmarks[0], landmarks[1], landmarks[2])))
                # thumb_angles.append(convert_angles(calculate_angle(landmarks[1], landmarks[2], landmarks[3])))
                # thumb_angles.append(convert_angles(calculate_angle(landmarks[2], landmarks[3], landmarks[4])))
                #
                # index_angles.append(convert_angles(calculate_angle(landmarks[0], landmarks[5], landmarks[6])))
                # index_angles.append(convert_angles(calculate_angle(landmarks[5], landmarks[6], landmarks[7])))
                # index_angles.append(convert_angles(calculate_angle(landmarks[6], landmarks[7], landmarks[8])))
                #
                # middle_angles.append(convert_angles(calculate_angle(landmarks[0], landmarks[9], landmarks[10])))
                # middle_angles.append(convert_angles(calculate_angle(landmarks[9], landmarks[10], landmarks[11])))
                # middle_angles.append(convert_angles(calculate_angle(landmarks[10], landmarks[11], landmarks[12])))
                #
                # ring_angles.append(convert_angles(calculate_angle(landmarks[0], landmarks[13], landmarks[14])))
                # ring_angles.append(convert_angles(calculate_angle(landmarks[13], landmarks[14], landmarks[15])))
                # ring_angles.append(convert_angles(calculate_angle(landmarks[14], landmarks[15], landmarks[16])))
                #
                # pinky_angles.append(convert_angles(calculate_angle(landmarks[0], landmarks[17], landmarks[18])))
                # pinky_angles.append(convert_angles(calculate_angle(landmarks[17], landmarks[18], landmarks[19])))
                # pinky_angles.append(convert_angles(calculate_angle(landmarks[18], landmarks[19], landmarks[20])))
                #
                # print(f"f0_j1: {thumb_angles[0]:.2f}, j2: {thumb_angles[1]:.2f}, 3: {thumb_angles[2]:.2f}")
                # print(f"f1_j1: {index_angles[0]:.2f}, j2: {index_angles[1]:.2f}, j3: {index_angles[2]:.2f}")
                # print(f"f2_j1: {middle_angles[0]:.2f}, j2: {middle_angles[1]:.2f}, j3: {middle_angles[2]:.2f}")
                # print(f"f3_j1: {ring_angles[0]:.2f}, j2: {ring_angles[1]:.2f}, j3: {ring_angles[2]:.2f}")
                # print(f"f4_j1: {pinky_angles[0]:.2f}, j2: {pinky_angles[1]:.2f}, j3: {pinky_angles[2]:.2f}")


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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


    #
    # df = pd.DataFrame(all_angles, columns=joint_names)
    # df.to_csv("hand_joint_angles.csv", index=False)
    # print("Hand joint angles saved at hand_joint_angles.csv")

    xml_tree = build_xml_frames_with_frame_index(all_angles)
    save_pretty_xml(xml_tree, "hand_angles.xml")

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