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
    pose = mp_pose.Pose(static_image_mode=False)
    hands = mp_hands.Hands(static_image_mode=False)
    mp_face_mesh = mp.solutions.face_mesh
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
    time_per_frame = 1 / fps

    frame_count = 0
    frames = []
    pose, hands, face_mesh = init_mediapipe()

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

        # keyframes_left, keyframes_right, midpoints_left, midpoints_right = keyframes_detection_and_notation.process_hand_landmarks(left_wrist, right_wrist)



        # face detection
        face_result = face_mesh.process(rgb_frame)
        if face_result.multi_face_landmarks:
            for face_landmarks in face_result.multi_face_landmarks:
                lm = face_landmarks.landmark




        # pose detection

        left_hand_location_by_frame = []
        right_hand_location_by_frame = []
        pose_results = pose.process(rgb_frame)
        pose_landmarks = pose_results.pose_landmarks.landmark if pose_results.pose_landmarks else None

        if pose_landmarks and multi_hands:
            for idx, hand_landmarks in enumerate(multi_hands):
                hand_pos = hand_location.get_hand_position(15 if idx == 0 else 16, pose_landmarks, hand_landmarks.landmark)
                label = "Left" if idx == 0 else "Right"
                if hand_pos is not None:
                    if label == "Left":
                        left_hand_location_by_frame.append(hand_pos)
                    else:
                        right_hand_location_by_frame.append(hand_pos)
        else:
            left_hand_location_by_frame.append(-1)
            right_hand_location_by_frame.append(-1)


        frame_count += 1

        # # show result
        # cv2.imshow(video_id, frame)
        # if cv2.waitKey(100) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()

    return left_hand, right_hand, left_wrist, right_wrist


def main():
    file_path = r'D:\project_codes\WLASL\start_kit\raw_videos\04797.mp4'
    left_hand, right_hand, keyframes_left, keyframes_right, midpoints_left, midpoints_right = get_coordinates(file_path)
    return left_hand, right_hand, keyframes_left, keyframes_right, midpoints_left, midpoints_right

if __name__ == '__main__':
    main()
