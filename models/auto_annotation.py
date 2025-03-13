"""
Created by: Xiaoyi Xiong
Date: 04/03/2025
"""

import cv2
import mediapipe as mp
import numpy as np

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

# Initialise MediaPipe Hands and Face Detection
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
hands = mp_hands.Hands()
face_detection = mp_face.FaceDetection()

# load video
file_path = r'D:\project_codes\WLASL\start_kit\raw_videos\00414.mp4'
video_id = file_path.split('\\')[-1]
cap = cv2.VideoCapture(file_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # hand detection
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in hand_landmarks.landmark]

            # 计算每根手指的弯曲角度
            thumb_angle = calculate_angle(landmarks[1], landmarks[2], landmarks[3])
            index_angle = calculate_angle(landmarks[5], landmarks[6], landmarks[7])
            middle_angle = calculate_angle(landmarks[9], landmarks[10], landmarks[11])
            ring_angle = calculate_angle(landmarks[13], landmarks[14], landmarks[15])
            pinky_angle = calculate_angle(landmarks[17], landmarks[18], landmarks[19])

            print(f"Thumb: {thumb_angle:.2f}, Index: {index_angle:.2f}, Middle: {middle_angle:.2f}, Ring: {ring_angle:.2f}, Pinky: {pinky_angle:.2f}")

            # draw hand landmarks
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    ## face detection
    # face_results = face_detection.process(rgb_frame)
    # if face_results.detections:
    #     for detection in face_results.detections:
    #         # bounding box
    #         bboxC = detection.location_data.relative_bounding_box
    #         ih, iw, _ = frame.shape
    #         bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
    #                int(bboxC.width * iw), int(bboxC.height * ih)
    #         cv2.rectangle(frame, bbox, (0, 255, 0), 2)

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode



    model_path = r"D:\anaconda3\Lib\site-packages\mediapipe"
    # Create a face landmarker instance with the video mode:
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO)

    with FaceLandmarker.create_from_options(options) as landmarker:
        mp.solutions.drawing_utils.draw_landmarks(frame, landmarker, FaceLandmarker.HAND_CONNECTIONS)
    # show result
    cv2.imshow(video_id, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()