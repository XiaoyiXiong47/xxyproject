import os
import random
import subprocess
import json
import pandas as pd
# make sure the current working directory is under /xxyproject/models
print("当前工作目录：", os.getcwd())
# video_dir = "/Users/xiongxiaoyi/Downloads/demo"
video_dir = r"D:\project_codes\xxyproject\data\raw\WLASL100\train"
video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

#
# # 读取预测结果和标签信息
predicted_df = pd.read_csv(r"D:\project_codes\xxyproject\data\predicted_label\predicted_gloss.csv")  # 包含 video_id, label
labels_txt_path  = r"D:\project_codes\xxyproject\WLASL100labels.txt"     # 包含 label_id, gloss

# 读取 label -> gloss 映射（按行读取）
# 提取每行最后一个词（真正的 gloss）
with open(labels_txt_path, "r", encoding="utf-8") as f:
    gloss_list = [line.strip().split(maxsplit=1)[-1] for line in f.readlines()]
gloss_map = {i: gloss for i, gloss in enumerate(gloss_list, start=0)}
print(gloss_map)

# # 创建 video_id -> gloss 映射
# video_gloss_map = {
#     str(row["video_id"] + 1).zfill(5): gloss_map.get(row["label"], "N/A")
#     for _, row in predicted_df.iterrows()
# }
# print(video_gloss_map)
#

#
# # 创建 label -> gloss 映射
# gloss_map = dict(zip(labels_df["label_id"], labels_df["gloss"]))
#
# # 创建 video_id -> label 映射（video_id 为 int 类型）
# video_label_map = dict(zip(predicted_df["video_id"], predicted_df["label"]))
# 打乱顺序
# random.shuffle(video_files)

# video_files = random.sample(video_files, 10)
# video_files = ["00629", "05234", "05732", "09950","11309","13636","14673","15038","15043","22953","28125","28201",
#                "51063","53269","57633","57940","58502","63669","65300","65403","69368"]

with open("run_order.log", "w") as f:
    for video in video_files:
        f.write(video + "\n")

# for video in video_files:
for i in range(len(video_files)):
    video = video_files[i].split(".")[0]
    # gloss = video_gloss_map.get(video, "N/A")
    gloss = gloss_map.get(predicted_df.iloc[i]["label"], "N/A")
    # full_path = os.path.join(video_dir, video + ".mp4")

    cmd = [
        "python", "./auto_annotation.py",
        "--video_dir", video_dir,
        "--video_id", video,
        "--dataset", "WLASL100",
        "--gloss", gloss
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd)



