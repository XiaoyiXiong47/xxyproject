"""
Created by: Xiaoyi Xiong
Date: 29/05/2025
"""
import os
import pandas as pd

# 设置路径
label_csv = "../WLASL100_labels.csv"
video_dir = "D:/project_codes/WLASL/start_kit/sample_videos_100"

# 读取 CSV 文件
df = pd.read_csv(label_csv)

# 读取文件夹中的视频文件名（不含扩展名）
video_files = [f.split('.')[0] for f in os.listdir(video_dir) if f.endswith('.mp4')]

# 创建 video_id 到 label 的映射字典
video_to_label = dict(zip(df['video_id'].astype(str), df['label_id']))

# 输出每个视频文件的 label
for vid in video_files:
    label = video_to_label.get(vid)
    if label is not None:
        print(f"{vid}.mp4 -> label: {label}")
    else:
        print(f"{vid}.mp4 -> label: Not found in CSV")
