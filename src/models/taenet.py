"""
Task-Enhanced Network (TAENet) モジュール
論文の4.1.1節に対応する実装
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TaskEnhancedNetwork(nn.Module):
    """
    Task-Enhanced Network (TAENet)
    論文の4.1.1節に対応
    """
    def __init__(self, embedding_dim, num_heads=2, gamma=1.0):
        """
        初期化関数
        
        Args:
            embedding_dim (int): 埋め込み表現の次元数
            num_heads (int): マルチヘッドアテンションのヘッド数
            gamma (float): 調整係数
        """
        super(TaskEnhancedNetwork, self).__init__()
        
        # マルチヘッドアテンション
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Task-Enhanced Gate (TAEGate)
        self.taegate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
        self.gamma = gamma
        
    def forward(self, task_embedding, treatment_enhanced_embedding):
        """
        順伝播関数
        
        Args:
            task_embedding (torch.Tensor): タスク埋め込み [batch_size, embedding_dim]
            treatment_enhanced_embedding (torch.Tensor): トリートメント強化埋め込み [batch_size, feature_dim+1, embedding_dim]
            
        Returns:
            torch.Tensor: タスク調整係数
        """
        batch_size = task_embedding.size(0)
        
        # タスク埋め込みをクエリとして使用
        query = task_embedding.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        
        # トリートメント強化埋め込みをキーと値として使用
        key = treatment_enhanced_embedding.detach()  # 勾配を切断
        value = treatment_enhanced_embedding.detach()  # 勾配を切断
        
        # マルチヘッドアテンション - 論文の式(5)に対応
        attn_output, _ = self.multihead_attention(query, key, value)  # [batch_size, 1, embedding_dim]
        E_pri = attn_output.squeeze(1)  # [batch_size, embedding_dim]
        
        # Task-Enhanced Gate - 論文の式(6)に対応
        delta_ta = self.gamma * self.taegate(E_pri)  # [batch_size, 1]
        
        return delta_ta
