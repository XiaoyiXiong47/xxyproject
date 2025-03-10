"""
Created by: Xiaoyi Xiong
Date: 04/03/2025
"""

import cv2
import mediapipe as mp

# 初始化MediaPipe Hands和Face Detection
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
hands = mp_hands.Hands()
face_detection = mp_face.FaceDetection()

# 读取视频
cap = cv2.VideoCapture('D:\project_codes\WLASL\start_kit\raw_videos\00414.mp4')

while cap.isOpened():
    # print("1")
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 手部检测
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # 绘制手部关键点
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 面部检测
    face_results = face_detection.process(rgb_frame)
    if face_results.detections:
        for detection in face_results.detections:
            # 绘制面部边界框
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, bbox, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('Sign Language Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()