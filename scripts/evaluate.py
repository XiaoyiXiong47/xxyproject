"""
Created by: Xiaoyi Xiong
Date: 04/03/2025
"""
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_metrics(actual, predicted, tolerance=0):
    matched_pred = set()
    matched_gt = set()

    for i, a in enumerate(actual):
        for j, p in enumerate(predicted):
            if abs(a - p) <= tolerance and j not in matched_pred:
                matched_gt.add(i)
                matched_pred.add(j)
                break

    true_positives = len(matched_gt)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(actual) if actual else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0

    return round(precision, 3), round(recall, 3), round(f1, 3)



if __name__ == "__main__":
    # # keyframe evaluation
    # # 加载原始 JSON 文件
    # with open("result-keyframes.json", "r") as f:
    #     results = json.load(f)
    # # Step 2: Compute metrics for each video
    # precisions, recalls, f1s = [], [], []
    #
    # for video_id, data in results.items():
    #     actual = data.get("actual", [])
    #     predicted = data.get("predicted", [])
    #     precision, recall, f1 = compute_metrics(actual, predicted, tolerance=2)
    #
    #     data["precision"] = precision
    #     data["recall"] = recall
    #     data["f1"] = f1
    #
    #     precisions.append(precision)
    #     recalls.append(recall)
    #     f1s.append(f1)
    #
    # # Step 3: Write updated JSON
    # with open("keyframe_eval_with_metrics.json", "w") as f:
    #     json.dump(results, f, indent=4)
    #
    # # Step 4: Print averages
    # avg_precision = round(sum(precisions) / len(precisions), 3)
    # avg_recall = round(sum(recalls) / len(recalls), 3)
    # avg_f1 = round(sum(f1s) / len(f1s), 3)
    # #
    # print("Keyframes detection result:")
    # print("Average Precision:", avg_precision)
    # print("Average Recall:", avg_recall)
    # print("Average F1 Score:", avg_f1)
    # print("\n\n")

    # Keyframe evaluation
    print("========================")
    print("Keyframes evaluation")
    with open("result-keyframes.json", "r") as f:
        data = json.load(f)
    total_actual = 0
    total_predicted = 0
    total_correct = 0
    num_videos = len(data)

    for video_id, frames in data.items():
        actual = set(frames['actual'])
        predicted = set(frames['predicted'])
        correct = actual.intersection(predicted)

        total_actual += len(actual)
        total_predicted += len(predicted)
        total_correct += len(correct)

    avg_actual = total_actual / num_videos
    avg_predicted = total_predicted / num_videos
    avg_correct = total_correct / num_videos

    print(f"Average number of actual keyframes: {avg_actual:.2f}")
    print(f"Average number of predicted keyframes: {avg_predicted:.2f}")
    print(f"Average number of correctly predicted keyframes: {avg_correct:.2f}")
    print("\n\n")

    # joint angles evaluation
    print("========================")
    print("Joint angles evaluation:")
    with open("result_joint_angles.json", "r") as f:
        data = json.load(f)

    mae_list = []
    cos_sim_list = []

    print("Per-sample evaluation:")

    # 遍历每个视频/样本
    for key in data["actual"]:
        actual = np.array(data["actual"][key])
        predicted = np.array(data["predicted"][key])

        actual_vec = actual.flatten()
        pred_vec = predicted.flatten()

        # 计算 MAE
        mae = np.mean(np.abs(actual_vec - pred_vec))
        mae_list.append(mae)

        # 计算 Cosine Similarity
        cos_sim = cosine_similarity(actual_vec.reshape(1, -1), pred_vec.reshape(1, -1))[0][0]
        cos_sim_list.append(cos_sim)

        # print(f"Video {key}: MAE = {mae:.2f}, Cosine Similarity = {cos_sim:.3f}")

    # 总体平均结果
    average_mae = np.mean(mae_list)
    average_cos_sim = np.mean(cos_sim_list)

    print("Overall Evaluation:")
    print(f"Average MAE: {average_mae:.2f}")
    print(f"Average Cosine Similarity: {average_cos_sim:.3f}")
    print("\n\n")


    # orientation evaluation
    print("========================")
    print("Hand orientation result evaluation:")
    with open("result_orientation.json", "r") as f:
        data = json.load(f)


    def angular_difference(a1, a2):
        """计算两个角度之间的最小环绕差值"""
        diff = np.abs(np.array(a1) - np.array(a2))
        return np.minimum(diff, 360 - diff)


    errors_y, errors_x, errors_z = [], [], []

    for key in data["actual"]:
        actual = data["actual"][key]
        predicted = data["predicted"][key]

        diff = angular_difference(actual, predicted)
        errors_y.append(diff[0])
        errors_x.append(diff[1])
        errors_z.append(diff[2])

    mae_y = np.mean(errors_y)
    mae_x = np.mean(errors_x)
    mae_z = np.mean(errors_z)

    # 总体平均 MAE
    maae = np.mean([mae_y, mae_x, mae_z])

    print(f"MAAE (mean angular error): {maae:.2f}°")
    print(f"Y-axis (yAngle) error: {mae_y:.2f}°")
    print(f"X-axis (xAngle) error: {mae_x:.2f}°")
    print(f"Z-axis (zAngle) error: {mae_z:.2f}°")
    print("\n\n")


    # hand location evaluation
    print("========================")
    print("Hand location result evaluation:")
    with open("result-location.json", "r") as f:
        data = json.load(f)
    mae_results = {}
    all_errors = []

    for key in data["actual"]:
        actual = np.array(data["a ctual"][key])
        predicted = np.array(data["predicted"][key])
        mae = np.mean(np.abs(actual - predicted))
        mae_results[key] = mae
        all_errors.extend(np.abs(actual - predicted))  # 汇总误差用于整体 MAE

    # # 打印每个样本的 MAE
    # for k, v in mae_results.items():
    #     print(f"Sample {k}: MAE = {v:.2f}")

    # 总体 MAE
    overall_mae = np.mean(all_errors)
    print(f"Overall MAE: {overall_mae:.2f}")