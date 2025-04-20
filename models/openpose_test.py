"""
Created by: Xiaoyi Xiong
Date: 28/03/2025
"""
import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json

# === 设置 OpenPose 的路径 ===
# dir_path = '/Users/xiongxiaoyi/Documents/ProjectCode/openpose'  # 修改为你自己的路径
dir_path = r'D:\project_codes'
sys.path.append(os.path.join(dir_path, 'build/python'))
from openpose import pyopenpose as op

# === OpenPose 参数 ===
params = {
    "model_folder": os.path.join(dir_path, "models"),
    "hand": True,
    "hand_detector": 2,
    "number_people_max": 1,
    "render_pose": 0,
    "disable_multi_thread": True,
}

# === 初始化 OpenPose ===
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# === 视频读取 ===
video_path = r'D:\project_codes\WLASL\start_kit\raw_videos\00625.mp4'   # 修改为你的视频
cap = cv2.VideoCapture(video_path)
frame_num = 0

right_index_finger_traj = []  # 保存右手 index 指尖位置

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])

    # 右手关键点
    right_hand = datum.handKeypoints[1]  # shape: (1, 21, 3)
    if right_hand is not None and len(right_hand) > 0:
        # 第8号点是 index finger tip (指尖)
        x, y, conf = right_hand[0][8]  # 0 是第一个人
        if conf > 0.2:  # 可选阈值，过滤低置信度点
            right_index_finger_traj.append((x, y))

    frame_num += 1
    print(f"Processed frame {frame_num}", end='\r')

cap.release()

# === 绘制轨迹图 ===
x_vals = [pt[0] for pt in right_index_finger_traj]
y_vals = [pt[1] for pt in right_index_finger_traj]

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, marker='o', linewidth=1)
plt.title("Trajectory of Right Hand Index Finger (Tip)")
plt.xlabel("X")
plt.ylabel("Y")
plt.gca().invert_yaxis()  # 图像坐标系是左上为(0,0)，所以Y轴反向
plt.grid(True)
plt.savefig("right_hand_index_trajectory.png")
plt.show()

print("✔️ 完成：轨迹图已保存为 right_hand_index_trajectory.png")
