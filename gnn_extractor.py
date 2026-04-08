import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

class GNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom PyTorch Graph Convolutional Network (GCN) Extractor built strictly
    for MaskablePPO over Variable DAG Scheduling Environments.
    
    Processes the exact 'node_features' and 'adj_matrix' cleanly to pool topological
    structures avoiding 'flattening' rigid limits.
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        # We process all spatial variables and return them concatenated up to 'features_dim'.
        super().__init__(observation_space, features_dim)
        
        # 1. Identify dimensional metrics from observation_space dictionary
        # node_features shape: (max_nodes, 5)
        node_feat_dim = observation_space.spaces["node_features"].shape[1]
        
        # global_state shape: (3,)
        self.global_dim = observation_space.spaces["global_state"].shape[0]
        
        # Let's target graph extraction pooling embedding side to equal (features_dim - global_dim)
        # So when concatenated we equal exactly features_dim to match SB3 Output!
        self.graph_embedding_dim = features_dim - self.global_dim
        
        # --- Graph Node Embedding Layers ---
        self.fc_in = nn.Linear(node_feat_dim, 32)
        
        # Linear projections for Message Passing GCN layer processing
        self.gcn1 = nn.Linear(32, 64)
        self.gcn2 = nn.Linear(64, self.graph_embedding_dim)
        
    def forward(self, observations: dict) -> torch.Tensor:
        """
        Forward pass executing Graph Convolution across the nodes natively 
        using dense Batched Matrix Multiplication across Padded Nodes!
        """
        # Node attributes: (batch_size, max_nodes, 5)
        X = observations["node_features"]
        
        # Adjacency matrix: (batch_size, max_nodes, max_nodes)
        A = observations["adj_matrix"]
        
        batch_size = A.size(0)
        max_nodes = A.size(1)
        
        # --- The A + I Self-Loop Trick ---
        # Add Self-Loops so nodes keep their own features during Graph Message Passing!
        identity = torch.eye(max_nodes, device=A.device).unsqueeze(0).expand(batch_size, -1, -1)
        A_hat = A + identity
        
        # --- Pre-computation ---
        X = F.relu(self.fc_in(X))
        
        # --- GCN Layer 1: H' = relu( A_hat * (X * W1) ) ---
        H1 = self.gcn1(X)                   # (Batch, max_nodes, 64)
        H1 = torch.bmm(A_hat, H1)           # Message passing! (Batch, max_nodes, 64)
        H1 = F.relu(H1)
        
        # --- GCN Layer 2: H'' = relu( A_hat * (H1 * W2) ) ---
        H2 = self.gcn2(H1)                              # (Batch, max_nodes, graph_embedding_dim)
        H2 = torch.bmm(A_hat, H2)                       # (Batch, max_nodes, graph_embedding_dim)
        node_embeddings = F.relu(H2)
        
        # --- Global Pooling (Readout) ---
        # We use Max Pooling across the dimension of max_nodes (dim=1) to smash 
        # the graph down to a single powerful embedding vector irrespective of actual filled structure!
        graph_global_repr = torch.max(node_embeddings, dim=1)[0]   # (Batch, graph_embedding_dim)
        
        # --- Merge with Global Hardware State ---
        g_state = observations["global_state"] # (Batch, 3)
        
        # The final Extractor Tensor matching SB3 strict boundary logic
        final_features = torch.cat([graph_global_repr, g_state], dim=1)  # (Batch, features_dim)
        
        return final_features
