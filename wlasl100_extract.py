"""
Created by: Xiaoyi Xiong
Date: 27/05/2025
"""
import json
import pandas as pd

# 读取 JSON 数据
with open("../WLASL/start_kit/WLASL_v0.3.json") as f:
    data = json.load(f)

# 只保留前100个 gloss（即WLASL100）
wlasl100_data = data[:100]

# 提取 gloss、video_id 和 label_id
records = []
for label_id, entry in enumerate(wlasl100_data):
    gloss = entry['gloss']
    for instance in entry['instances']:
        records.append({
            'video_id': instance['video_id'],
            'gloss': gloss,
            'label_id': label_id,
            'split': instance.get('split', 'unknown')
        })

df = pd.DataFrame(records)
df.to_csv('WLASL100_labels.csv', index=False)
