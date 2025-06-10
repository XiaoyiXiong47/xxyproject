"""
Created by: Xiaoyi Xiong
Date: 06/05/2025
"""
import cv2
import numpy as np
import mediapipe as mp


# 坐标轴转换函数
def convert_axes(hand_coord, ref_coord, flip_x=False):
    z = hand_coord[2] - ref_coord[2]  # 向前为正（z反转）
    x = -(hand_coord[0] - ref_coord[0])  if flip_x else (hand_coord[0] - ref_coord[0]) # 向左为正 (x反转)
    y = hand_coord[1] - ref_coord[1] # 向下为正
    return np.array([x, y, z])


def get_hand_position(pose_landmarks, hand_landmarks, is_left_hand):
    """
    Calculate the hand position relative to the center of the shoulders,
    scaled by palm length (half of shoulder width), and rounded to the nearest 50 units.
    :param pose_landmarks: extracted post landmarks at specific frame
    :param hand_landmarks: Sequence of one hand landmarks at each frame
    :return: (rel_x, rel_y, rel_z): relative distances to center of the shoulders in palm_length units
    """
    if pose_landmarks is None or hand_landmarks is None or len(hand_landmarks) == 0:
        return None, None, None

    shoulder_left = np.array([
        pose_landmarks[11].x,
        pose_landmarks[11].y,
        pose_landmarks[11].z
    ])
    shoulder_right = np.array([
        pose_landmarks[12].x,
        pose_landmarks[12].y,
        pose_landmarks[12].z
    ])
    shoulder_mid = (shoulder_left + shoulder_right) / 2 # middle point of two shoulders

    index_mcp = hand_landmarks[5]
    hand_pos = np.array([
        index_mcp[0],
        index_mcp[1],
        index_mcp[2]
    ])

    # 坐标轴转换（默认以摄像头为正面）
    relative_pos = convert_axes(hand_pos, shoulder_mid, flip_x=is_left_hand)

    # 使用肩宽的一半作为单位手掌长度
    shoulder_distance = np.linalg.norm(shoulder_left - shoulder_right)
    if shoulder_distance == 0:
        return [np.nan, np.nan, np.nan]
    palm_length = shoulder_distance / 2

    scale = 100 / palm_length
    scaled = relative_pos * scale
    # scaled = hand_pos * scale
    rounded = np.round(scaled / 50) * 50
    return rounded.astype(int)    # 坐标轴转换（默认以摄像头为正面）
    # relative_pos = convert_axes(hand_pos, shoulder_mid)
    #
    # # 使用肩宽的一半作为单位手掌长度
    # shoulder_distance = np.linalg.norm(shoulder_left - shoulder_right)
    # if shoulder_distance == 0:
    #     return [np.nan, np.nan, np.nan]
    # palm_length = shoulder_distance / 2
    #
    # scale = 100 / palm_length
    # scaled = relative_pos * scale
    # rounded = np.round(scaled / 50) * 50
    # return rounded.astype(int)



def calculate_hand_locations(
    pose_landmarks_seq, hand_landmarks_seq,
    keyframes, midpoints, is_left_hand
):
    """
    根据给定帧序列（pose + hand landmarks）只在关键帧和中点帧上计算手的位置。
    其他帧填充 [np.nan]。

    :param hand_index: 要获取的位置关键点索引（通常是15或16）
    :param pose_landmarks_seq: pose landmarks 每帧列表
    :param hand_landmarks_seq: hand landmarks 每帧列表（每帧是21个关键点）
    :param keyframes: set 或 list，表示起止帧
    :param midpoints: set 或 list，表示中间帧
    :return: hand_location_by_frame: list，每帧为[rel_x, rel_y, rel_z]或[np.nan]
    """
    # valid_frames = set(keyframes).union(midpoints)
    valid_frames = set(midpoints) | {f for pair in keyframes for f in pair}
    # print("Valid frames:", valid_frames)

    hand_location_by_frame = []
    for i in range(len(hand_landmarks_seq)):
        pose_lms = pose_landmarks_seq[i]
        hand_lms = hand_landmarks_seq[i]
        # print(
        #     f"Frame {i}: pose_lms is None? {pose_lms is None}, hand_lms length: {len(hand_lms) if isinstance(hand_lms, list) else 'N/A'}")
        if pose_lms and isinstance(hand_lms, np.ndarray) and len(hand_lms) == 21:
            try:
                loc = get_hand_position(pose_lms, hand_lms, is_left_hand)
                hand_location_by_frame.append(loc)
            except Exception as e:
                print(f"Error at frame {i}: {e}")
                hand_location_by_frame.append([np.nan, np.nan, np.nan])
        else:
            hand_location_by_frame.append([np.nan, np.nan, np.nan])

    return hand_location_by_frame



# 主函数处理视频
def main():
    # 初始化 MediaPipe
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    pose = mp_pose.Pose(static_image_mode=False)
    hands = mp_hands.Hands(static_image_mode=False)
    file_path = r'D:\project_codes\WLASL\start_kit\raw_videos\49374.mp4'
    video_path = '/Users/xiongxiaoyi/Downloads/demo/00624.mp4'

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 检测 Pose
        pose_results = pose.process(image_rgb)
        pose_landmarks = pose_results.pose_landmarks.landmark if pose_results.pose_landmarks else None

        # 检测 Hands
        hands_results = hands.process(image_rgb)
        multi_hands = hands_results.multi_hand_landmarks

        if pose_landmarks and multi_hands:
            for idx, hand_landmarks in enumerate(multi_hands):
                hand_pos = get_hand_position(15 if idx == 0 else 16, pose_landmarks, hand_landmarks.landmark)
                label = "Left" if idx == 0 else "Right"
                if hand_pos is not None:
                    print(f"Frame {frame_count}: {label} Hand Position: {hand_pos}")

        #你可以取消注释以下两行实时查看结果：
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

