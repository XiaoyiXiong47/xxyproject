"""
Created by: Xiaoyi Xiong
Date: 01/05/2025
"""

# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# from mediapipe import solutions
# from mediapipe.framework.formats import landmark_pb2
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
#
# def draw_landmarks_on_image(rgb_image, detection_result):
#     face_landmarks_list = detection_result.face_landmarks
#     annotated_image = np.copy(rgb_image)
#
#     # Loop through the detected faces to visualize.
#     for idx in range(len(face_landmarks_list)):
#         face_landmarks = face_landmarks_list[idx]
#
#         # Draw the face landmarks.
#         face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#         face_landmarks_proto.landmark.extend([
#             landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
#         ])
#
#         solutions.drawing_utils.draw_landmarks(
#                 image=annotated_image,
#                 landmark_list=face_landmarks_proto,
#                 connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
#                 landmark_drawing_spec=None,
#                 connection_drawing_spec=mp.solutions.drawing_styles
#                 .get_default_face_mesh_tesselation_style())
#         solutions.drawing_utils.draw_landmarks(
#                 image=annotated_image,
#                 landmark_list=face_landmarks_proto,
#                 connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
#                 landmark_drawing_spec=None,
#                 connection_drawing_spec=mp.solutions.drawing_styles
#                 .get_default_face_mesh_contours_style())
#         solutions.drawing_utils.draw_landmarks(
#                 image=annotated_image,
#                 landmark_list=face_landmarks_proto,
#                 connections=mp.solutions.face_mesh.FACEMESH_IRISES,
#                   landmark_drawing_spec=None,
#                   connection_drawing_spec=mp.solutions.drawing_styles
#                   .get_default_face_mesh_iris_connections_style())
#
#     return annotated_image
#
# def plot_face_blendshapes_bar_graph(face_blendshapes):
#     # Extract the face blendshapes category names and scores.
#     face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
#     face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
#     # The blendshapes are ordered in decreasing score value.
#     face_blendshapes_ranks = range(len(face_blendshapes_names))
#
#     fig, ax = plt.subplots(figsize=(12, 12))
#     bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
#     ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
#     ax.invert_yaxis()
#
#     # Label each bar with values
#     for score, patch in zip(face_blendshapes_scores, bar.patches):
#         plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")
#
#     ax.set_xlabel('Score')
#     ax.set_title("Face Blendshapes")
#     plt.tight_layout()
#     plt.show()




# # STEP 2: Create an FaceLandmarker object.
# base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
# options = vision.FaceLandmarkerOptions(base_options=base_options,
#                                        output_face_blendshapes=True,
#                                        output_facial_transformation_matrixes=True,
#                                        num_faces=1)
# detector = vision.FaceLandmarker.create_from_options(options)
#
# # STEP 3: Load the input image.
# image = mp.Image.create_from_file(r"C:\Users\GGPC\OneDrive\Desktop\business-person.png")
#
# # STEP 4: Detect face landmarks from the input image.
# detection_result = detector.detect(image)
#
# # STEP 5: Process the detection result. In this case, visualize it.
# annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
# # cv2.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
# cv2.imshow("Annotated Image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))  # ✅ 正确
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0])
# print(detection_result.facial_transformation_matrixes)

# =================================================================================
#
# import cv2
# import mediapipe as mp
#
#
# def init_mediapipe():
#     # Initialise MediaPipe Hands and Face Detection
#     mp_hands = mp.solutions.hands
#     mp_face_mesh = mp.solutions.face_mesh
#     hands = mp_hands.Hands()
#     return mp_hands, mp_face_mesh, hands
#
#
# def load_video(file_path):
#     """
#     Use OpenCV to load given sign video.
#     :param file_path: The path of sign video
#     :return:
#     """
#
#     video_id = file_path.split('\\')[-1]
#     cap = cv2.VideoCapture(file_path)
#     return cap, video_id
#
#
# file_path = r'D:\project_codes\WLASL\start_kit\raw_videos\00625.mp4'
# cap, video_id = load_video(file_path)
# mp_hands, mp_face_mesh, hands = init_mediapipe()
# mp_drawing = mp.solutions.drawing_utils
# frames = []
# left = []
# right = []
#
# with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             break
#
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(frame_rgb)
#
#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 # visualization
#                 mp_drawing.draw_landmarks(
#                     image=frame,
#                     landmark_list=face_landmarks,
#                     connections=mp_face_mesh.FACEMESH_TESSELATION,
#                     landmark_drawing_spec=None,
#                     connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))
#
#                 # 获取眉毛关键点
#                 landmarks = face_landmarks.landmark
#
#                 # 左眉毛（以用户角度）常用点：
#                 left_brow = {
#                     "眉头Eyebrow head": landmarks[55],
#                     "眉峰Eyebrow arch": landmarks[65],
#                     "眉尾Eyebrow tail": landmarks[52]
#                 }
#
#                 # 右眉毛
#                 right_brow = {
#                     "眉头Eyebrow head": landmarks[285],
#                     "眉峰Eyebrow arch": landmarks[295],
#                     "眉尾Eyebrow tail": landmarks[282]
#                 }
#
#                 # 可以将其绘制在图像上查看
#                 h, w, _ = frame.shape
#                 for name, point in left_brow.items():
#                     x, y = int(point.x * w), int(point.y * h)
#                     cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
#                     cv2.putText(frame, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
#
#                 for name, point in right_brow.items():
#                     x, y = int(point.x * w), int(point.y * h)
#                     cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
#                     cv2.putText(frame, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
#
#         cv2.imshow("Face Mesh", frame)
#         if cv2.waitKey(5) & 0xFF == 27:
#             break
#
# cap.release()
# cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import numpy as np



# 距离函数
def euclidean(p1, p2):
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

# 判断眼睛状态
def eye_status(upper, lower, inner, outer):
    openness = euclidean(upper, lower) / euclidean(inner, outer)
    if openness < 0.18:
        return "Closed"
    elif openness < 0.25:
        return "Normal"
    else:
        return "Wide"

# 判断眉毛是否抬起
def eyebrow_status(brow, upper_eye, lower_eye):
    ratio = euclidean(brow, upper_eye) / euclidean(upper_eye, lower_eye)
    return "Raised" if ratio > 1.8 else "Neutral"





def main():
    mp_face_mesh = mp.solutions.face_mesh
    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
    # video_path = r'D:\project_codes\WLASL\start_kit\raw_videos\00625.mp4'
    video_path = r'D:\project_codes\WLASL\start_kit\raw_videos\00632.mp4'
    cap = cv2.VideoCapture(video_path)

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:
                    lm = face_landmarks.landmark

                    # left eye（from user's view）
                    l_inner, l_outer = lm[133], lm[33]
                    l_upper, l_lower = lm[159], lm[145]
                    l_eyebrow = lm[65]

                    # right eye（from user's view）
                    r_inner, r_outer = lm[362], lm[263]
                    r_upper, r_lower = lm[386], lm[374]
                    r_eyebrow = lm[295]

                    # status
                    left_eye_state = eye_status(l_upper, l_lower, l_inner, l_outer)
                    right_eye_state = eye_status(r_upper, r_lower, r_inner, r_outer)

                    left_brow_state = eyebrow_status(l_eyebrow, l_upper, l_lower)
                    right_brow_state = eyebrow_status(r_eyebrow, r_upper, r_lower)

                    # text annotation
                    cv2.putText(frame, f"L Eye: {left_eye_state}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                                2)
                    cv2.putText(frame, f"R Eye: {right_eye_state}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 2)
                    cv2.putText(frame, f"L Brow: {left_brow_state}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 0, 0), 2)
                    cv2.putText(frame, f"R Brow: {right_brow_state}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 0, 0), 2)

                    # visualization
                    h, w, _ = frame.shape
                    for idx in [65, 295, 159, 145, 386, 374]:  # 眉毛、上/下眼睑点
                        x, y = int(lm[idx].x * w), int(lm[idx].y * h)
                        cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

            cv2.imshow("Face Expression Detection", frame)
            if cv2.waitKey(100) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
