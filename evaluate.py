# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 11:40:27 2025

@author: Lenovo
"""

import os
import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from utils import extract_trajectories, prepare_trajectory_features, create_target_matrix

def evaluate_model(model, data_A, data_B, test_times, window_size):
    """
    evaluate model performance
    """
    model.eval()
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    test_count = 0
    with torch.no_grad():
        for i in range(len(test_times) - window_size + 1):
            time_window = test_times[i:i+window_size]
            end_time = time_window[-1]
            data_A_window = data_A[data_A['DateTime'].isin(time_window)]
            data_B_window = data_B[data_B['DateTime'].isin(time_window)]
            trajectories_A, ids_A = extract_trajectories(data_A, time_window, 'ID')
            trajectories_B, ids_B = extract_trajectories(data_B, time_window, 'ID')
            if not trajectories_A or not trajectories_B:
                continue
            features_A = prepare_trajectory_features(trajectories_A)
            features_B = prepare_trajectory_features(trajectories_B)
            if features_A.size(0) == 0 or features_B.size(0) == 0:
                continue
            similarity_matrix = model(features_A, features_B)
            similarity_np = similarity_matrix.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(-similarity_np)
            predicted_matches = list(zip(row_ind, col_ind))
            target_matrix, true_matches = create_target_matrix(ids_A, ids_B, data_A, data_B)
            pred_set = set(predicted_matches)
            true_set = set(true_matches)
            correct_matches = len(pred_set & true_set)
            total_pred = max(len(pred_set), 1)
            total_true = max(len(true_set), 1)
            accuracy = (correct_matches / total_true * 100) if true_set else 0
            precision = correct_matches / total_pred if pred_set else 0
            recall = correct_matches / total_true if true_set else 0
            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            test_count += 1
            visualize_trajectory_matching(
                data_A_window, data_B_window, 
                ids_A, ids_B, 
                predicted_matches, true_matches, 
                time_window, (accuracy, precision, recall)
            )
            print(f"Window ending at {end_time}")
            print(f"Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}")
            print("-" * 50)
    if test_count > 0:
        avg_accuracy = total_accuracy / test_count
        avg_precision = total_precision / test_count
        avg_recall = total_recall / test_count
        return avg_accuracy, avg_precision, avg_recall
    else:
        return 0, 0, 0

def visualize_trajectory_matching(data_A, data_B, ids_A, ids_B, pred_matches, true_matches, time_window, metrics):
    """
    visualize trajectory matching result
    """
    print(pred_matches, true_matches)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    end_time = time_window[-1]
    for id_a in ids_A:
        traj_A = data_A[data_A['ID'] == id_a].sort_values('DateTime')
        if len(traj_A) >= 2:
            ax1.plot(traj_A['X'], traj_A['Y'], 'b-', alpha=0.5)
            ax1.scatter(traj_A['X'].iloc[-1], traj_A['Y'].iloc[-1], c='blue', s=30)
            ax2.plot(traj_A['X'], traj_A['Y'], 'b-', alpha=0.5)
            ax2.scatter(traj_A['X'].iloc[-1], traj_A['Y'].iloc[-1], c='blue', s=30)
    for id_b in ids_B:
        traj_B = data_B[data_B['ID'] == id_b].sort_values('DateTime')
        if len(traj_B) >= 2:
            ax1.plot(traj_B['X'], traj_B['Y'], 'r-', alpha=0.5)
            ax1.scatter(traj_B['X'].iloc[-1], traj_B['Y'].iloc[-1], c='red', s=30)
            ax2.plot(traj_B['X'], traj_B['Y'], 'r-', alpha=0.5)
            ax2.scatter(traj_B['X'].iloc[-1], traj_B['Y'].iloc[-1], c='red', s=30)
    for i, j in pred_matches:
        if i < len(ids_A) and j < len(ids_B):
            id_a, id_b = ids_A[i], ids_B[j]
            a_last = data_A[(data_A['ID'] == id_a)]
            b_last = data_B[(data_B['ID'] == id_b)]
            if not a_last.empty and not b_last.empty:
                ax1.plot([a_last['X'].iloc[0], b_last['X'].iloc[0]],
                         [a_last['Y'].iloc[0], b_last['Y'].iloc[0]],
                         'g--', linewidth=1.5, alpha=0.7)
    ax1.set_title('Predicted Trajectory Matching')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    for i, j in true_matches:
        if i < len(ids_A) and j < len(ids_B):
            id_a, id_b = ids_A[i], ids_B[j]
            a_last = data_A[(data_A['ID'] == id_a)]
            b_last = data_B[(data_B['ID'] == id_b)]
            if not a_last.empty and not b_last.empty:
                ax2.plot([a_last['X'].iloc[0], b_last['X'].iloc[0]],
                         [a_last['Y'].iloc[0], b_last['Y'].iloc[0]],
                         'g--', linewidth=1.5, alpha=0.7)
    ax2.set_title('Ground Truth Matching')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    accuracy, precision, recall = metrics
    plt.suptitle(f'Time Window: {time_window[0]} to {end_time}\n'
                 f'Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}')
    plt.show()
    plt.close()

def load_model(model, weights_path='output/model_weights.pth'):
    """
    load trained model
    """
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model weights from {weights_path}")
        return True
    else:
        print(f"No saved weights found at {weights_path}")
        return False