"""
Created by: Xiaoyi Xiong
Date: 27/05/2025
"""
# import json
# import pandas as pd
#
# # 读取 JSON 数据
# with open("../WLASL/start_kit/WLASL_v0.3.json") as f:
#     data = json.load(f)
#
# # 只保留前100个 gloss（即WLASL100）
# wlasl100_data = data[:100]
#
# # 提取 gloss、video_id 和 label_id
# records = []
# for label_id, entry in enumerate(wlasl100_data):
#     gloss = entry['gloss']
#     for instance in entry['instances']:
#         records.append({
#             'video_id': instance['video_id'],
#             'gloss': gloss,
#             'label_id': label_id,
#             'split': instance.get('split', 'unknown')
#         })
#
# df = pd.DataFrame(records)
# df.to_csv('WLASL100_labels.csv', index=False)



import os
import json
import shutil
from tqdm import tqdm

# 配置路径
project_root = "D:/project_codes/xxyproject"
json_path = os.path.join(project_root, "WLASL_v0.3.json")
gloss_txt_path = os.path.join(project_root, "WLASL100labels.txt")
videos_dir = os.path.join(project_root, "data", "raw", "videos")
output_root = os.path.join(project_root, "data", "raw", "WLASL100")

# 创建输出文件夹结构
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_root, split), exist_ok=True)

# Step 1: 读取 WLASL100 的 gloss 名称
wlasl100_glosses = set()
with open(gloss_txt_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            wlasl100_glosses.add(parts[1])

# Step 2: 加载 JSON 数据
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Step 3: 映射 video_id -> (gloss, split)
video_to_split = {}
for entry in data:
    gloss = entry["gloss"]
    if gloss in wlasl100_glosses:
        for inst in entry["instances"]:
            video_to_split[inst["video_id"]] = inst["split"]

# Step 4: 扫描原始视频文件夹并复制（带进度条）
video_files = [f for f in os.listdir(videos_dir) if f.endswith(".mp4")]

for file_name in tqdm(video_files, desc="Copying videos"):
    video_id = os.path.splitext(file_name)[0]
    if video_id in video_to_split:
        split = video_to_split[video_id]
        src = os.path.join(videos_dir, file_name)
        dst = os.path.join(output_root, split, file_name)
        shutil.copy2(src, dst)

print("✅ 所有 WLALS100 视频已根据 split 分类复制完成。")
