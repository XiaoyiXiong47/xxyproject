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
    file_path = r'D:\project_codes\WLASL\start_kit\raw_videos\00632.mp4'
    cap, video_id = load_video(file_path)

    # Get video frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    time_per_frame = 1 / fps

    frame_count = 0
    frames = []
    pose, hands, face_mesh = init_mediapipe()

    left_hand = []
    right_hand = []

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

        # hand landmarks detection
        if hand_results.multi_hand_landmarks:  # if hand key points are detected
            if len(hand_results.multi_hand_landmarks) == 2:
                # both hands are detected
                for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                    landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                    if handedness.classification[0].label == "Left":
                        left_hand.append(landmarks)
                    else:  # right hand
                        right_hand = landmarks
            elif len(hand_results.multi_hand_landmarks) == 1:
                for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                    landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                    if handedness.classification[0].label == "Left":
                        left_hand.append(landmarks)
                        right_hand.append(np.full(21, np.nan))
                    else:  # right hand
                        right_hand = landmarks
                        left_hand.append(np.full(21, np.nan))

                # draw hand landmarks
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, hands.HAND_CONNECTIONS)
        else:
            # no hand detected
            left_hand.append(np.full(21, np.nan))
            right_hand.append(np.full(21, np.nan))

        # face detection
        face_results = face_mesh.process(rgb_frame)
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

    return left_hand, right_hand, shoulders, facial



