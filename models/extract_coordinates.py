"""
Created by: Xiaoyi Xiong
Date: 06/05/2025
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import StandardScaler
import pandas as pd
import xml.dom.minidom as minidom
import keyframes_detection_and_notation
import hand_location

def init_mediapipe():
    """
    Initialise MediaPipe Hand, Pose and Face Detection
    :return: hands:

    """
    #
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh
    pose = mp_pose.Pose(static_image_mode=False)
    hands = mp_hands.Hands(static_image_mode=False)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    return pose, hands, face_mesh


def load_video(file_path):
    """
    Use OpenCV to load given sign video.
    :param file_path: The path of sign video
    :return: cap:
            video_id: the id of the loaded video
    """

    video_id = file_path.split('\\')[-1]
    cap = cv2.VideoCapture(file_path)
    return cap, video_id


def get_coordinates(file_path):
    """"""

    cap, video_id = load_video(file_path)

    # Get video frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    # time_per_frame = 1 / fps

    frame_count = 0
    frames = []
    pose, hands, face_mesh = init_mediapipe()

    left_hand = []
    right_hand = []

    left_wrist = []
    right_wrist = []

    left_hand_location_by_frame = []
    right_hand_location_by_frame = []

    pose_landmarks_seq = []

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
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    # landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                    wrist = landmarks[0]
                    if handedness.classification[0].label == "Left":
                        left_wrist.append(wrist)
                        left_hand.append(landmarks)
                        # current_left_right["left"] = landmarks
                    elif handedness.classification[0].label == "Right":
                        right_wrist.append(wrist)
                        right_hand.append(landmarks)
                        # current_left_right["right"] = landmarks
                    # draw hand landmarks
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks)
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
                # # draw hand landmarks
                # mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, hands.HAND_CONNECTIONS)
        else:
        #     hand not detected at current frame, insert with [np.nan, np.nan, np.nan]
            left_wrist.append([np.nan, np.nan, np.nan])
            left_hand.append(np.full(21, np.nan))
            right_wrist.append([np.nan, np.nan, np.nan])
            right_hand.append(np.full(21, np.nan))

        # pose detection
        pose_results = pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            pose_landmarks = pose_results.pose_landmarks.landmark
        else:
            pose_landmarks = None
        pose_landmarks_seq.append(pose_landmarks)  # 不管有没有都 append，占位


        frame_count += 1

        # # show result
        # cv2.imshow(video_id, frame)
        # if cv2.waitKey(100) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()

    return left_hand, right_hand, left_wrist, right_wrist, pose_landmarks_seq
    # return left_hand, right_hand, left_wrist, right_wrist, left_hand_location_by_frame, right_hand_location_by_frame, pose_landmarks_seq

def extract_keypoints_from_hand_seq(hand_seq):
    wrist_seq = []
    thumb_seq = []
    index_seq = []

    for landmarks in hand_seq:
        if isinstance(landmarks, np.ndarray) and landmarks.shape == (21, 3):
            wrist_seq.append(landmarks[0])   # wrist
            thumb_seq.append(landmarks[4])   # thumb tip
            index_seq.append(landmarks[8])   # index tip
        else:
            # 补nan（无效帧）
            wrist_seq.append([np.nan, np.nan, np.nan])
            thumb_seq.append([np.nan, np.nan, np.nan])
            index_seq.append([np.nan, np.nan, np.nan])

    return wrist_seq, thumb_seq, index_seq


def main():
    file_path = r'D:\project_codes\WLASL\start_kit\raw_videos\04797.mp4'
    left_hand, right_hand, left_wrist, right_wrist, pose_landmarks_seq = get_coordinates(file_path)
    return left_hand, right_hand, left_wrist, right_wrist, pose_landmarks_seq

if __name__ == '__main__':
    main()
