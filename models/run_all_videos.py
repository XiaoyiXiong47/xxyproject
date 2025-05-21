import os
import subprocess

video_dir = "/Users/xiongxiaoyi/Downloads/demo"
video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

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
