# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 11:40:27 2025

@author: Lenovo
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from utils import build_temporal_edge_index
        
class TemporalAttention(nn.Module):
    """Implements temporal attention to aggregate features across time steps."""
    def __init__(self, hidden_dim, num_heads=4):
        """
        Initialize the TemporalAttention module.

        Args:
            hidden_dim (int): Dimension of the input features.
            num_heads (int): Number of attention heads for multi-head attention (default: 4).
        """
        super().__init__()
        # Multi-head attention layer to capture temporal dependencies
        self.temporal_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        """
        Forward pass for temporal attention.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_dim].

        Returns:
            torch.Tensor: Aggregated features of shape [batch_size, hidden_dim], averaged over time steps.
        """
        # Apply multi-head attention (query, key, value are the same input tensor)
        attn_out, _ = self.temporal_attn(x, x, x)
        # Add residual connection to preserve input information
        x = x + attn_out
        # Average across time dimension (seq_len) to get per-trajectory features
        return x.mean(dim=1)

class SpatialAttention(nn.Module):
    """Implements spatial attention to model interactions between trajectories."""
    def __init__(self, hidden_dim, num_heads=4):
        """
        Initialize the SpatialAttention module.

        Args:
            hidden_dim (int): Dimension of the input features.
            num_heads (int): Number of attention heads for multi-head attention (default: 4).
        """
        super().__init__()
        # Multi-head attention layer to capture spatial relationships between trajectories
        self.spatial_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

    def forward(self, feat_A, feat_B):
        """
        Forward pass for spatial attention.

        Args:
            feat_A (torch.Tensor): Features of dataset A, shape [N_A, hidden_dim].
            feat_B (torch.Tensor): Features of dataset B, shape [N_B, hidden_dim].

        Returns:
            torch.Tensor: Updated features for dataset A, shape [N_A, hidden_dim].
        """
        # Prepare query (Q) by adding a sequence dimension
        Q = feat_A.unsqueeze(1)  # Shape: [N_A, 1, hidden_dim]
        # Prepare key (K) and value (V) by expanding feat_B to match Q's batch size
        K = feat_B.unsqueeze(0).expand(feat_A.size(0), feat_B.size(0), feat_B.size(1))  # Shape: [N_A, N_B, hidden_dim]
        V = K  # Use same tensor for key and value
        # Apply multi-head attention to compute spatial interactions
        attn_out, _ =  self.spatial_attn(Q, K, V)  # Shape: [N_A, 1, hidden_dim]
        # Remove sequence dimension
        attn_out = attn_out.squeeze(1) + feat_A  # Shape: [N_A, hidden_dim]
        # Add residual connection to preserve input features
        return attn_out

class FeedForwardNetwork(nn.Module):
    """Implements a feed-forward network for feature transformation."""
    def __init__(self, input_dim, hidden_dim, ffn_dim=None):
        """
        Initialize the FeedForwardNetwork module.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Output feature dimension.
            ffn_dim (int, optional): Hidden layer dimension (default: hidden_dim * 2).
        """
        super().__init__()
        ffn_dim = ffn_dim or hidden_dim * 2
        # Define feed-forward network with two linear layers, ReLU, and normalization
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x):
        """
        Forward pass for the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape [..., input_dim].

        Returns:
            torch.Tensor: Transformed tensor of shape [..., hidden_dim].
        """
        return self.ffn(x)


class UncertaintySimilarity(nn.Module):
    """Implements uncertainty-aware similarity computation with confidence adjustment."""
    
    def __init__(self, hidden_dim, epsilon=1e-6, temperature=1.0):
        """
        Initialize the UncertaintySimilarity module.

        Args:
            hidden_dim (int): Hidden dimension for internal representations.
            epsilon (float): Small value to ensure numerical stability (default: 1e-6).
            temperature (float): Temperature parameter for confidence scaling (default: 1.0).
            use_sigmoid (bool): Whether to apply Sigmoid to the final similarity score (default: True).
        """
        super().__init__()
        
        # Shared feature transformation layer
        self.feature_transform = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )

        # Head for predicting mean similarity
        self.mean_head = nn.Linear(hidden_dim, 1)

        # Head for predicting log variance (to ensure positivity)
        self.var_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensures variance is positive
        )

        self.epsilon = epsilon
        self.temperature = temperature

    def forward(self, paired_features):
        """
        Forward pass for uncertainty-aware similarity computation.

        Args:
            paired_features (torch.Tensor): Input tensor of shape [..., hidden_dim * 2].

        Returns:
            dict: Dictionary containing:
                - 'similarity': Final similarity scores of shape [..., 1].
                - 'mean': Predicted mean similarity scores of shape [..., 1].
                - 'variance': Predicted variance of shape [..., 1].
                - 'uncertainty': Predicted uncertainty (standard deviation) of shape [..., 1].
                - 'confidence': Confidence scores based on uncertainty of shape [..., 1].
        """
        # Transform input features
        features = self.feature_transform(paired_features)

        # Predict mean (mu) and log-variance
        mean = self.mean_head(features)  # Mean similarity
        log_variance = self.var_head(features)  # Log variance
        
        variance = torch.exp(log_variance) + self.epsilon # Convert log-variance to variance and stabilize

        # Compute standard deviation (uncertainty)
        uncertainty = torch.sqrt(variance)
        
        confidence = torch.exp(-uncertainty * self.temperature) # Confidence factor

        # Compute uncertainty-weighted similarity  
        similarity = mean * confidence#+ uncertainty #* epsilon1

        return {
            'similarity': similarity,       # Final similarity score
            'mean': mean,                   # Predicted mean
            'variance': variance,           # Predicted variance
            'uncertainty': uncertainty,     # Predicted uncertainty (std dev)
            'confidence': confidence        # Confidence based on uncertainty
        }
    

class TrajectoryMatchingNet(nn.Module):
    """Main network for matching trajectories between two datasets using temporal and spatial attention."""
    def __init__(self, input_dim, hidden_dim=128, gat_heads=4, uncertainty_temperature=1.25):
        """
        Initialize the TrajectoryMatchingNet.

        Args:
            input_dim (int): Dimension of input trajectory features (e.g., 2 for X, Y coordinates).
            hidden_dim (int): Hidden dimension for internal representations (default: 128).
            gat_heads (int): Number of attention heads for GAT layers (default: 4).
            uncertainty_temperature (float): Temperature for uncertainty scaling (default: 1.0).
            · When temperature = 1.0 (default): uncertainty affects confidence at its original scale
            · When temperature > 1.0: the effect of uncertainty is amplified, making the model more sensitive to uncertainty
            · When temperature < 1.0: the effect of uncertainty is reduced, making the model less sensitive to uncertainty
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Feature extraction layers for datasets A and V
        self.feature_input_A = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.feature_input_V = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())

        # Graph Attention Network (GAT) layers for temporal graph processing
        self.gat_A = GATConv(hidden_dim, hidden_dim, heads=gat_heads, concat=False)
        self.gat_V = GATConv(hidden_dim, hidden_dim, heads=gat_heads, concat=False)
        
        # Temporal attention layers to aggregate features across time
        self.temporal_attention_A = TemporalAttention(hidden_dim)
        self.temporal_attention_V = TemporalAttention(hidden_dim)

        # Spatial attention layers to model interactions between trajectories
        self.spatial_attention_A = SpatialAttention(hidden_dim=hidden_dim)
        self.spatial_attention_V = SpatialAttention(hidden_dim=hidden_dim)
        
        # Feed-forward networks to fuse temporal and spatial features
        self.ffn_A = FeedForwardNetwork(hidden_dim * 2, hidden_dim)
        self.ffn_V = FeedForwardNetwork(hidden_dim * 2, hidden_dim)

        # Uncertainty-aware similarity computation
        self.similarity = UncertaintySimilarity(hidden_dim, temperature=uncertainty_temperature)

    def forward(self, traj_A, traj_B, return_uncertainty=False):
        """
        Forward pass for trajectory matching.

        Args:
            traj_A (torch.Tensor): Trajectories from dataset A, shape [N_A, seq_len, input_dim].
            traj_B (torch.Tensor): Trajectories from dataset B, shape [N_B, seq_len, input_dim].
            return_uncertainty (bool): Whether to return uncertainty information (default: True).

        Returns:
            dict or torch.Tensor: 
                If return_uncertainty is True, returns a dictionary containing:
                    - 'similarity': Similarity matrix of shape [N_A, N_B]
                    - 'uncertainty': Uncertainty matrix of shape [N_A, N_B]
                    - 'confidence': Confidence matrix of shape [N_A, N_B]
                If return_uncertainty is False, returns only the similarity matrix.
        """

        N_A, seq_len, _ = traj_A.shape
        N_B, _, _ = traj_B.shape

        # Extract initial features using linear layers
        feat_A = self.feature_input_A(traj_A)  # Shape: [N_A, seq_len, hidden_dim]
        feat_B = self.feature_input_V(traj_B)  # Shape: [N_B, seq_len, hidden_dim]
    
        # Build temporal edge indices for graph processing
        edge_index_A = build_temporal_edge_index(N_A, seq_len).to(traj_A.device)
        edge_index_B = build_temporal_edge_index(N_B, seq_len).to(traj_B.device)
    
        # Apply GAT to capture temporal relationships within trajectories
        feat_A_gnn = self.gat_A(feat_A.reshape(N_A*seq_len, self.hidden_dim), edge_index_A).reshape(N_A, seq_len, self.hidden_dim)
        feat_B_gnn = self.gat_V(feat_B.reshape(N_B*seq_len, self.hidden_dim), edge_index_B).reshape(N_B, seq_len, self.hidden_dim)
    
        # Apply temporal attention to aggregate features across time
        feat_A_temporal = self.temporal_attention_A(feat_A_gnn + feat_A)  # Shape: [N_A, hidden_dim]
        feat_B_temporal = self.temporal_attention_V(feat_B_gnn + feat_B)  # Shape: [N_B, hidden_dim]
    
        # Apply spatial attention to model interactions between trajectories
        feat_A_spatial = self.spatial_attention_A(feat_A_temporal, feat_A_temporal)  # Shape: [N_A, hidden_dim]
        feat_B_spatial = self.spatial_attention_V(feat_B_temporal, feat_B_temporal)  # Shape: [N_B, hidden_dim]
 
        # Concatenate temporal and spatial features
        feat_A_concat = torch.cat([feat_A_temporal, feat_A_spatial], dim=-1)  # Shape: [N_A, hidden_dim*2]
        feat_B_concat = torch.cat([feat_B_temporal, feat_B_spatial], dim=-1)  # Shape: [N_B, hidden_dim*2]
    
        # Fuse features using feed-forward networks with residual connections
        feat_A_final = self.ffn_A(feat_A_concat) + feat_A_temporal + feat_A_spatial  # Shape: [N_A, hidden_dim]
        feat_B_final = self.ffn_V(feat_B_concat) + feat_B_temporal + feat_B_spatial  # Shape: [N_B, hidden_dim]

        # Compute similarity scores with uncertainty for all pairs of trajectories
        feat_A_expanded = feat_A_final.unsqueeze(1).expand(N_A, N_B, -1)  # Shape: [N_A, N_B, hidden_dim]
        feat_B_expanded = feat_B_final.unsqueeze(0).expand(N_A, N_B, -1)  # Shape: [N_A, N_B, hidden_dim]
        paired_features = torch.cat([feat_A_expanded, feat_B_expanded], dim=2)  # Shape: [N_A, N_B, hidden_dim*2]
        
        # Get uncertainty-aware similarity scores
        similarity_output = self.similarity(paired_features.view(-1, self.hidden_dim * 2))
        
        # Reshape outputs to matrix form
        similarity_matrix = similarity_output['similarity'].view(N_A, N_B)
        
        if return_uncertainty:
            uncertainty_matrix = similarity_output['uncertainty'].view(N_A, N_B)
            confidence_matrix = similarity_output['confidence'].view(N_A, N_B)
            mean_matrix = similarity_output['mean'].view(N_A, N_B)
            
            return {
                'similarity': similarity_matrix,
                'mean': mean_matrix,
                'uncertainty': uncertainty_matrix,
                'confidence': confidence_matrix
            }
        else:
            return similarity_matrix