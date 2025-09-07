# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 11:40:27 2025

@author: Lenovo
"""

import torch
import numpy as np
import pandas as pd
import os
from torch_geometric.utils import dense_to_sparse

def check_data(data_A, data_B):
    """Check data validity"""
    print("Checking AIS data:")
    print(f"Number of rows: {len(data_A)}")
    print(f"Columns: {data_A.columns.tolist()}")
    print(f"DateTime format example: {data_A['DateTime'].iloc[0]}")
    print("\nChecking CCTV data:")
    print(f"Number of rows: {len(data_B)}")
    print(f"Columns: {data_B.columns.tolist()}")
    print(f"DateTime format example: {data_B['DateTime'].iloc[0]}")
    
    # Check for missing values
    print("\nMissing values in data A:")
    print(data_A.isnull().sum())
    print("\nMissing values in data B:")
    print(data_B.isnull().sum())

def load_and_preprocess_data(data_A_train_file, data_B_train_file, data_A_test_file, data_B_test_file):
    """Load and preprocess Excel data"""
    data_A_Train = pd.read_excel(data_A_train_file)
    data_B_Train = pd.read_excel(data_B_train_file)
    data_A_Test = pd.read_excel(data_A_test_file)
    data_B_Test = pd.read_excel(data_B_test_file)
    
    # Ensure DateTime column format is correct
    data_A_Train['DateTime'] = pd.to_datetime(data_A_Train['DateTime'])
    data_B_Train['DateTime'] = pd.to_datetime(data_B_Train['DateTime'])
    data_A_Test['DateTime'] = pd.to_datetime(data_A_Test['DateTime'])
    data_B_Test['DateTime'] = pd.to_datetime(data_B_Test['DateTime'])

    data_A_Train = data_A_Train.sort_values(['DateTime', 'ID'])
    data_A_Test = data_A_Test.sort_values(['DateTime', 'ID'])
    data_B_Train = data_B_Train.sort_values(['DateTime', 'ID'])
    data_B_Test = data_B_Test.sort_values(['DateTime', 'ID'])

    # Combine training and test data
    data_A = pd.concat([data_A_Train, data_A_Test])
    data_B = pd.concat([data_B_Train, data_B_Test])
    
    # Get unique timestamps for training and test sets
    train_times = sorted(data_A_Train['DateTime'].unique())
    test_times = sorted(data_A_Test['DateTime'].unique())
    
    return data_A, data_B, train_times, test_times

def build_temporal_edge_index(num_nodes, seq_len):
    """
    Build temporal edge index
    """
    adj = torch.zeros((seq_len, seq_len))
    for t in range(seq_len - 1):
        adj[t, t + 1] = 1
    edge_index_single, _ = dense_to_sparse(adj)
    edge_indices = []
    for i in range(num_nodes):
        offset = i * seq_len
        edge_index_offset = edge_index_single + offset
        edge_indices.append(edge_index_offset)
    edge_index = torch.cat(edge_indices, dim=1)
    return edge_index

def interpolate_missing_points(traj):
    """
    Interpolate missing trajectory points using linear interpolation
    
    Args:
        traj (np.array): Trajectory array with shape [k, 2], may contain NaN values
    
    Returns:
        np.array: Trajectory array with interpolated values
    """
    df = pd.DataFrame(traj, columns=['X', 'Y'])
    
    # Check if all values are NaN
    if df.isna().all().all():
        return np.zeros_like(traj)
    
    # Linear interpolation for missing values
    df_interpolated = df.interpolate(method='linear', limit_direction='both')
    
    # For edge cases where interpolation might still leave NaN values,
    # use forward fill then backward fill
    df_interpolated = df_interpolated.fillna(method='ffill').fillna(method='bfill')
    
    # If still NaN (edge case: only one point or all NaN), fill with zeros
    df_interpolated = df_interpolated.fillna(0.0)
    
    return df_interpolated.values

def extract_trajectories(data, time_window, id_col='ID'):
    """
    Extract trajectories from data with interpolation for missing points
    """
    trajectories = {}
    data_in_window = data[data['DateTime'].isin(time_window)]
    for id_val, group in data_in_window.groupby(id_col):
        group = group.sort_values('DateTime')
        traj_features = []
        for t in time_window:
            t_data = group[group['DateTime'] == t]
            if len(t_data) > 0:
                row = t_data.iloc[0]
                traj_features.append([float(row['X']), float(row['Y'])])
            else:
                traj_features.append([np.nan, np.nan])
        
        # Count valid (non-NaN) points before interpolation
        traj_array = np.array(traj_features)
        valid_points = np.sum(~np.isnan(traj_array).any(axis=1))
        
        # Only keep trajectories with at least 3 valid points
        if valid_points >= 3:
            # Apply interpolation to fill missing points
            traj_interpolated = interpolate_missing_points(traj_array)
            trajectories[id_val] = traj_interpolated
    
    ids = list(trajectories.keys())
    return trajectories, ids

def prepare_trajectory_features(traj_dict):
    """
    Prepare trajectory features for model input
    Note: Now receives interpolated trajectories, so no need for nan_to_num
    """
    features = []
    for traj_id, traj in traj_dict.items():
        # Since trajectories are already interpolated, we can use them directly
        features.append(traj)
    if features:
        return torch.FloatTensor(features)
    else:
        return torch.FloatTensor(0, 0, 0)

def create_target_matrix(ids_A, ids_B, data_A, data_B):
    """
    Create target matching matrix based on GTID
    """
    N_A = len(ids_A)
    N_B = len(ids_B)
    target_matrix = torch.zeros((N_A, N_B))
    true_matches = []
    id_to_gtid_B = {}
    for _, row in data_B.drop_duplicates('ID').iterrows():
        id_to_gtid_B[row['ID']] = row['GTID']
    for i, id_a in enumerate(ids_A):
        for j, id_b in enumerate(ids_B):
            gtid_b = id_to_gtid_B.get(id_b)
            if gtid_b is not None and id_a == gtid_b:
                target_matrix[i, j] = 1.0
                true_matches.append((i, j))
                break
    return target_matrix, true_matches

# Create output directory
if not os.path.exists('output'):
    os.makedirs('output')