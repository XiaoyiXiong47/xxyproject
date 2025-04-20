"""
Created by: Xiaoyi Xiong
Date: 04/04/2025
"""
import cv2
import numpy as np
from utils import preprocess
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import os
from models import auto_annotation
from utils import preprocess
import pickle

os.environ["OMP_NUM_THREADS"] = "2"

def extract_sequence_from_frame(video_path, mediapipe_hand):

    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
    sequences = []
    cap, _ = auto_annotation.load_video(video_path)
    frame_features = []
    frame_index = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = mediapipe_hand.process(rgb_frame)

        # create left and right for hand landmark
        left = None
        right = None
        # if hand key points are detected
        if hand_results.multi_hand_landmarks:
            # print(f"Detected {len(hand_results.multi_hand_landmarks)} hand(s)")
            for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                raw_lms = hand_landmarks.landmark
                if not raw_lms or len(raw_lms) != 21:
                    # Invalid landmark data
                    landmarks = np.zeros((21, 3), dtype=np.float32)
                    # landmarks = np.zeros((1, 3), dtype=np.float32)
                else:
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in raw_lms], dtype=np.float32)
                    # landmarks = landmarks[0]    # wrist point
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in raw_lms], dtype=np.float32)

                # print(
                # f"✅ landmarks shape: {landmarks.shape}, dtype: {landmarks.dtype}, label: {handedness.classification[0].label}")
                # landmarks = np.array([lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark)  # shape (3 x 21, )
                # print(handedness.classification[0])
                if handedness.classification[0].label == "Left":
                    left = landmarks
                elif handedness.classification[0].label == "Right":
                    right = landmarks
                else:
                    continue
        else:
            continue
            # print("No hands detected in this frame.")

        both_hands = preprocess.process_frame(left, right)
        # print(f"Frame {frame_index}: both_hands shape = {both_hands.shape}")
        # frame_features shape = (126, n_frames)
        frame_features.append(both_hands)  # Add coordinates for both hand at every frame to frame_features.
        frame_index += 1

    cap.release()
    if len(frame_features) < 2:
        print(f"Not enough valid frames in video {video_path}")
    frame_features_shapes = [i.shape for i in frame_features]
    # print(frame_features_shapes)
    sequence = np.stack(frame_features)
    sequence = preprocess.interpolate_sequence(sequence)

    # standardise sequence
    scaler = StandardScaler()
    sequence = scaler.fit_transform(sequence)  # shape remains (num_frames, 126)

    # ✅ 可选：增强特征，加上速度和加速度（更利于动作识别）
    velocity = np.diff(sequence, axis=0, prepend=sequence[0:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
    sequence = np.concatenate([sequence, velocity, acceleration], axis=1)  # shape: (num_frames, 126 * 3)

    sequences.append(sequence)

    print(f"Video {video_path}.mp4 has been processed and trained.")
    return sequences


def get_hmm_train_sequences(label_to_ids_dict, video_base_path, mediapipe_hand):
    """
    Extract frame sequence for all videos, return data suitable for HMM model training.
    :param label_to_ids_dict:
    :param video_base_path:
    :param mediapipe_hand:
    :return:
    """
    all_sequences = {}

    for label, video_ids in label_to_ids_dict.items():

        print(f"Processing label: {label}")

        for vid in video_ids:
            if not vid:
                continue

            video_path = os.path.join(video_base_path, f"{vid}.mp4")
            sequences = extract_sequence_from_frame(video_path, mediapipe_hand)

        all_sequences[label] = sequences

    return all_sequences


def hmm_train(all_sequences):
    """
    Train a HMM model for each motion.

    :param all_sequences:
    :return:
    """
    # 对每类运动轨迹训练一个 HMM
    models = {}
    for label, sequences in all_sequences.items():
        # X（每个视频的关键点序列）和 y（标签）

        sequences = [sequences[i] for i in range(len(sequences))]
        lengths = [len(seq) for seq in sequences]
        concat_data = np.concatenate(sequences)

        model = hmm.GaussianHMM(n_components=3, n_iter=300)
        model.fit(concat_data, lengths)
        models[label] = model

    return models

def hmm_predict(sequence, models):
    """
    Given sequence, use hmm model to predict the motion within sequence

    :param sequence:
    :param models: dict[str, GaussianHMM], trained HMMs
    :return: best_label: Predicted label
    :return: scores: dict[label, score]
    """
    scores = {}
    for label, model in models.items():
        try:
            score = model.score(sequence)
        except:
            score = -np.inf
        scores[label] = score
    best_label = max(scores, key=scores.get)

    return best_label, scores


motion_labels = ['linear_motion', 'circular_motion', 'curve_motion']
label_to_ids_dict = {
    "linear_motion": ["00624", "00626", "00629", "00634", "01991", "03119", "03120", "03121", "03122", "03124", "03125",
                 "03437", "03439", "03441", "04593"],

    "circular_motion": ["00414", "00415", "00416", "00421", "01385", "01386", "01391", "01986", "01987", "01988",
                   "01992", "02228", "02229", "02230", "02231", "02233"],

    "curve_motion": ["01383", "01384", "01387", "01460", "01461", "01462", "01463", "01464", "01466", ""]
}


#"compond_movements": ["00623", "00631", "00632"]
# "wavy_repeated_motion": ["02583", "02584", "02585", "02586", "02587", "02589", "02999", "03000",
#                          "03001", "03002", "00303", "00305", "03118", "03267", "03268", "03270", "03272", "03273",
#                          "03274",
#                         "03277", "03278", "03435", "04505", "04507", "04508", "04511"]
# mark到了03438.mp4



# HMM implementation



# for given video, construct time sequence
mp_hands, mp_face, hands, face_detection = auto_annotation.init_mediapipe()

video_path = r'D:\project_codes\WLASL\start_kit\raw_videos'
hmm_train_data = get_hmm_train_sequences(label_to_ids_dict, video_path, hands)
trained_hmm_models = hmm_train(hmm_train_data)

with open('all_hmm_models.pkl', 'wb') as f:
    pickle.dump(trained_hmm_models, f)


# sequence = np.stack([preprocess.process_frame(f['left'], f['right']) for f in frames])
#
# # 对所有帧的坐标进行标准化，让模型更专注于动作变化：
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(sequence.reshape(-1, sequence.shape[-1]))
# sequence = X_scaled.reshape(sequence.shape)
#
# # 特征增强：速度 / 加速度
# # 为每一帧增加一阶差分（速度）和二阶差分（加速度）
# velocity = np.diff(sequence, axis=0, prepend=sequence[0:1])
# acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
# sequence_augmented = np.concatenate([sequence, velocity, acceleration], axis=1)

