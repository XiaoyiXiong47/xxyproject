"""
Created by: Xiaoyi Xiong
Date: 29/05/2025
"""
import os
import json
import shutil

# 路径设置
json_path = "D:/project_codes/WLASL/start_kit/WLASL100.json"
raw_videos_dir = "D:/project_codes/WLASL/start_kit/raw_videos"
target_dir = "D:/project_codes/WLASL/start_kit/wlasl100videos"

# 创建目标文件夹（如果不存在）
os.makedirs(target_dir, exist_ok=True)

# 读取 JSON 文件
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取所有 video_id 并构造成 mp4 文件名集合
video_ids = set()
for gloss_entry in data:
    for instance in gloss_entry.get("instances", []):
        video_ids.add(instance["video_id"].zfill(5) + ".mp4")

# 开始复制
copied_count = 0
for video_file in os.listdir(raw_videos_dir):
    if video_file in video_ids:
        src = os.path.join(raw_videos_dir, video_file)
        dst = os.path.join(target_dir, video_file)
        shutil.copyfile(src, dst)
        copied_count += 1

print(f"完成：共复制了 {copied_count} 个视频文件。")
