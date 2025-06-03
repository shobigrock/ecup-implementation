"""
初期埋め込み表現モジュール
特徴量とトリートメントの初期埋め込み表現を生成します
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class InitialEmbedding(nn.Module):
    """
    特徴量とトリートメントの初期埋め込み表現を生成するモジュール
    論文の2.2節に対応
    """
    def __init__(self, feature_dim, embedding_dim, treatment_count):
        """
        初期化関数
        
        Args:
            feature_dim (int): 特徴量の次元数
            embedding_dim (int): 埋め込み表現の次元数
            treatment_count (int): トリートメントの種類数
        """
        super(InitialEmbedding, self).__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.treatment_count = treatment_count
        
        # 特徴量の変換層
        self.feature_transform = nn.Linear(feature_dim, feature_dim * embedding_dim)
        
        # トリートメントの埋め込み層
        self.treatment_embedding = nn.Embedding(treatment_count, embedding_dim)
        
        # タスク埋め込み層（クリックとコンバージョンの2タスク）
        self.task_embedding = nn.Embedding(2, embedding_dim)
        
    def forward(self, features, treatment):
        """
        順伝播関数
        
        Args:
            features (torch.Tensor): 特徴量テンソル [batch_size, feature_dim]
            treatment (torch.Tensor): トリートメントテンソル [batch_size]
            
        Returns:
            tuple: 特徴量埋め込み、トリートメント埋め込み、タスク埋め込み（クリック、コンバージョン）
        """
        batch_size = features.size(0)
        
        # 特徴量の埋め込み表現
        # 論文の式(3)と(4)に対応
        feature_embedding = self.feature_transform(features)
        feature_embedding = feature_embedding.view(batch_size, self.feature_dim, self.embedding_dim)
        
        # トリートメントの埋め込み表現
        treatment_embedding = self.treatment_embedding(treatment).unsqueeze(1)
        
        # タスク埋め込み表現
        click_task_embedding = self.task_embedding(torch.zeros(batch_size, dtype=torch.long, device=features.device))
        conversion_task_embedding = self.task_embedding(torch.ones(batch_size, dtype=torch.long, device=features.device))
        
        return feature_embedding, treatment_embedding, click_task_embedding, conversion_task_embedding
