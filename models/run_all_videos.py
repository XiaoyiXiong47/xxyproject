import os
import random
import subprocess

video_dir = "/Users/xiongxiaoyi/Downloads/demo"
video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

# 打乱顺序
random.shuffle(video_files)

video_files = random.sample(video_files, 10)

with open("run_order.log", "w") as f:
    for video in video_files:
        f.write(video + "\n")

for video in video_files:
    full_path = os.path.join(video_dir, video)
    cmd = [
        "python", "./auto_annotation.py",
        "--data_path", full_path,
        "--dataset", "WLASL",
        "--gloss", "HAHA"
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd)



