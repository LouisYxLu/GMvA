# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 11:40:27 2025

@author: Lenovo
"""

import argparse
import torch
from utils import load_and_preprocess_data, check_data, extract_trajectories, prepare_trajectory_features, create_target_matrix
from model import TrajectoryMatchingNet
from train import train_model
from evaluate import evaluate_model, load_model

def estimate_margin(model, data_A, data_B, times, window_size, sample_windows=20):
    """
    估计合适的 TripletLoss margin
    
    Args:
        model (nn.Module): 训练模型
        data_A, data_B (pd.DataFrame): 两个数据源
        times (list): 时间序列（train_times 或 test_times）
        window_size (int): 窗口大小
        sample_windows (int): 采样多少个窗口来估计 (默认20)
    
    Returns:
        float: 推荐的 margin
    """
    model.eval()
    pos_diffs = []
    neg_diffs = []
    
    with torch.no_grad():
        for i in range(min(len(times) - window_size + 1, sample_windows)):
            time_window = times[i:i+window_size]
            trajectories_A, ids_A = extract_trajectories(data_A, time_window, "ID")
            trajectories_B, ids_B = extract_trajectories(data_B, time_window, "ID")

            if not trajectories_A or not trajectories_B:
                continue

            features_A = prepare_trajectory_features(trajectories_A)
            features_B = prepare_trajectory_features(trajectories_B)

            if features_A.size(0) == 0 or features_B.size(0) == 0:
                continue

            # 模型输出相似度
            output = model(features_A, features_B, return_uncertainty=True)
            similarity_matrix = output["similarity"]

            # 构建目标矩阵
            target_matrix, pairs = create_target_matrix(ids_A, ids_B, data_A, data_B)
            positive_pairs = [(i, j) for i, j in pairs]
            negative_pairs = [
                (i, j) for i in range(similarity_matrix.size(0))
                for j in range(similarity_matrix.size(1))
                if (i, j) not in pairs
            ]

            if not positive_pairs or not negative_pairs:
                continue

            pos_scores = torch.tensor([similarity_matrix[i, j].item() for i, j in positive_pairs])
            neg_scores = torch.tensor([similarity_matrix[i, j].item() for i, j in negative_pairs])

            pos_mean = pos_scores.mean().item()
            neg_mean = neg_scores.mean().item()
            diff = neg_mean - pos_mean  # 正负样本平均差

            pos_diffs.append(pos_mean)
            neg_diffs.append(neg_mean)

    if not pos_diffs or not neg_diffs:
        print("⚠️ 没有统计到有效的正负样本，无法估计 margin")
        return 0.2  # 默认值
    
    avg_pos = sum(pos_diffs) / len(pos_diffs)
    avg_neg = sum(neg_diffs) / len(neg_diffs)
    avg_diff = avg_neg - avg_pos

    # 推荐 margin = 差值的 0.5 ~ 0.8 倍
    recommended = max(0.05, round(avg_diff * 0.6, 2))

    print(f"📊 正样本平均相似度: {avg_pos:.4f}")
    print(f"📊 负样本平均相似度: {avg_neg:.4f}")
    print(f"📊 平均差值: {avg_diff:.4f}")
    print(f"✅ 推荐 margin ≈ {recommended}")

    return recommended

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], 
                        help='Run mode: train or test')
    parser.add_argument('--weights_path', type=str, default='checkpoint/GMvA3_model_weights.pth',
                        help='Path to the model weights file')
    parser.add_argument('--AIS_data_train_file', type=str, default='AIS_Train_Train.xlsx',
                        help='Path to the Excel file for training data A')
    parser.add_argument('--Video_data_train_file', type=str, default='Video_Train_Train.xlsx',
                        help='Path to the Excel file for training data B')
    parser.add_argument('--AIS_data_test_file', type=str, default='AIS_Test.xlsx',
                        help='Path to the Excel file for test data A')
    parser.add_argument('--CCTV_data_test_file', type=str, default='Video_Test.xlsx',
                        help='Path to the Excel file for test data B')
    parser.add_argument('--num_epochs', type=str, default=100,
                        help='train epochs numbers')
    parser.add_argument('--window_size', type=str, default=6,
                        help='windows size')
    args = parser.parse_args()

    print("Starting data loading and preprocessing...")
    data_A, data_B, train_times, test_times = load_and_preprocess_data(
        args.AIS_data_train_file, args.Video_data_train_file, 
        args.AIS_data_test_file, args.CCTV_data_test_file
    )
    check_data(data_A, data_B)
    
    print("Extracting sample trajectories to determine feature dimension...")
    sample_window = train_times[:args.window_size]
    sample_trajectories_A, _ = extract_trajectories(data_A, sample_window, 'ID')
    
    if not sample_trajectories_A:
        print("Unable to extract trajectories from sample window. Please check the data.")
        return
    
    sample_features_A = prepare_trajectory_features(sample_trajectories_A)
    input_dim = sample_features_A.size(2)
    
    print(f"Feature dimension: {input_dim}")
    
    print("Initializing model...")
    model = TrajectoryMatchingNet(input_dim=input_dim, hidden_dim=128)
    
    margin = estimate_margin(model, data_A, data_B, train_times, args.window_size)
    print(f"Using margin={margin} for TripletLoss")
    
    if args.mode == 'train':
        print("Starting training mode...")
        train_model(model, data_A, data_B, train_times, args.window_size, args.num_epochs, args.weights_path)
        print("Evaluating trained model...")
        avg_accuracy, avg_precision, avg_recall = evaluate_model(
            model, data_A, data_B, test_times, args.window_size)
    else:
        print("Starting test mode...")
        if load_model(model, args.weights_path):
            model.eval()
            avg_accuracy, avg_precision, avg_recall = evaluate_model(
                model, data_A, data_B, test_times, args.window_size)
        else:
            print("Pre-trained model not found. Please train the model or check the weights file path.")
            return
    
    print("\nOverall performance:")
    print(f"Average Accuracy: {avg_accuracy:.2f}%")
    print(f"Average Precision: {avg_precision:.2f}")
    print(f"Average Recall: {avg_recall:.2f}")

    # 保存结果到txt文档
    with open("results_log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"Average Accuracy: {avg_accuracy:.2f}%\n")
    
    print("Processing completed successfully!")
        

if __name__ == "__main__":
    for i in range(100):
        main()