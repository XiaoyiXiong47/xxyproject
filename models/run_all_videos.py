import os
import random
import subprocess
import json
import pandas as pd

# video_dir = "/Users/xiongxiaoyi/Downloads/demo"
video_dir = r'D:\project_codes\WLASL\start_kit\sample_videos_100'
video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]


# 读取预测结果和标签信息
predicted_df = pd.read_csv(r"D:\project_codes\xxyproject\data\predicted_label\predicted_gloss.csv")  # 包含 video_id, label
labels_df = pd.read_csv("..\WLASL100_labels.csv")     # 包含 label_id, gloss

# 创建 label -> gloss 映射
gloss_map = dict(zip(labels_df["label_id"], labels_df["gloss"]))

# 创建 video_id -> label 映射（video_id 为 int 类型）
video_label_map = dict(zip(predicted_df["video_id"], predicted_df["label"]))
# 打乱顺序
# random.shuffle(video_files)

# video_files = random.sample(video_files, 10)

with open("run_order.log", "w") as f:
    for video in video_files:
        f.write(video + "\n")

for video in video_files:
    full_path = os.path.join(video_dir, video)
    try:
        video_id = int(os.path.splitext(video)[0].lstrip("0"))  # "00623.mp4" -> 623
        label = video_label_map.get(video_id)
        gloss = gloss_map.get(label, "N/A") if label is not None else "N/A"
    except Exception as e:
        gloss = "N/A"
    cmd = [
        "python", "./auto_annotation.py",
        "--data_path", full_path,
        "--dataset", "WLASL",
        "--gloss", gloss
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd)



