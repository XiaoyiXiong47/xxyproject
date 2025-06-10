"""
Created by: Xiaoyi Xiong
Date: 08/06/2025
"""
import json

def print_top_n_glosses(json_path, n=10):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    glosses = [entry['gloss'] for entry in data[:n]]
    for i, gloss in enumerate(glosses, 1):
        print(f"{i}. {gloss}")




# 使用下面这段 Python 脚本来统计：你的视频文件（以 video_id.mp4 命名）一共对应了 WLASL_v0.3.json 中多少个不同的 gloss。
import os
import json

def count_matched_glosses(json_path, videos_folder):
    # 读取 WLASL json
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取视频文件名集合（不含扩展名）
    video_ids_in_folder = {
        os.path.splitext(f)[0] for f in os.listdir(videos_folder) if f.endswith('.mp4')
    }

    # 遍历所有 gloss，查找其中是否有 instance 的 video_id 出现在视频文件夹中
    matched_glosses = set()

    for entry in data:
        for instance in entry.get("instances", []):
            if instance["video_id"] in video_ids_in_folder:
                matched_glosses.add(entry["gloss"])
                break  # 这个 gloss 已经匹配上，继续下一个 gloss

    print(f"Total matched glosses: {len(matched_glosses)}")
    return matched_glosses



import json

def extract_gloss_video_label(json_path, output_txt_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_txt_path, 'w', encoding='utf-8') as out_file:
        for label, entry in enumerate(data):
            gloss = entry['gloss']
            for instance in entry.get('instances', []):
                video_id = instance['video_id']
                out_file.write(f"{gloss} {video_id} {label}\n")

    print(f"Saved gloss, video_id, and label to: {output_txt_path}")


import os
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed

def count_frames(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return (video_path, -1)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return (video_path, frame_count)
    except Exception as e:
        return (video_path, -1)

def get_max_frame_video_multiprocess(videos_folder, num_workers=8):
    video_files = [os.path.join(videos_folder, f) for f in os.listdir(videos_folder)
                   if f.endswith('.mp4') or f.endswith('.avi')]

    max_video = ""
    max_frames = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(count_frames, video_path) for video_path in video_files]

        for future in as_completed(futures):
            video_path, frame_count = future.result()
            if frame_count > max_frames:
                max_frames = frame_count
                max_video = os.path.basename(video_path)

    return max_video, max_frames

# 示例调用
if __name__ == "__main__":
    json_path = "../WLASL_v0.3.json"
    videos_folder = "../data/raw/WLASL100/val"  # 替换成你的视频文件夹路径
    # videos_folder = "../data/raw/videos"  # 替换成你的视频文件夹路径

    # print_top_n_glosses("WLASL_v0.3.json", n=10)

    # matched_glosses = count_matched_glosses(json_path, videos_folder)

    ## 提取gloss, video_id, label
    # extract_gloss_video_label(json_path, "../gloss_video_label.txt")

    max_video, max_frames = get_max_frame_video_multiprocess(videos_folder)
    print(f"Video with the most number of frames: {max_video}, the number of frames is: {max_frames}")


