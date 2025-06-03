"""
ECUP (Entire Chain Uplift Modeling with Context-Enhanced Learning) モデル
論文の全体構造を実装
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding import InitialEmbedding
from .tenet import TENet
from .ecenet import ECENet

class ECUP(nn.Module):
    """
    ECUP (Entire Chain Uplift Modeling with Context-Enhanced Learning) モデル
    論文の全体構造を実装
    """
    def __init__(self, feature_dim=99, embedding_dim=16, hidden_dims=[128, 64, 32], 
                 attention_dim=8, treatment_count=5, num_heads=2, gamma=1.0):
        """
        初期化関数
        
        Args:
            feature_dim (int): 特徴量の次元数
            embedding_dim (int): 埋め込み表現の次元数
            hidden_dims (list): 隠れ層の次元数リスト
            attention_dim (int): アテンション機構の次元数
            treatment_count (int): トリートメントの種類数
            num_heads (int): マルチヘッドアテンションのヘッド数
            gamma (float): 調整係数
        """
        super(ECUP, self).__init__()
        
        # 初期埋め込み表現
        self.initial_embedding = InitialEmbedding(feature_dim, embedding_dim, treatment_count)
        
        # Treatment-Enhanced Network
        self.tenet = TENet(feature_dim, embedding_dim, attention_dim)
        
        # Entire Chain-Enhanced Network
        self.ecenet = ECENet(feature_dim, embedding_dim, hidden_dims, treatment_count, num_heads, gamma)
        
    def forward(self, features, treatment):
        """
        順伝播関数
        
        Args:
            features (torch.Tensor): 特徴量テンソル [batch_size, feature_dim]
            treatment (torch.Tensor): トリートメントテンソル [batch_size]
            
        Returns:
            tuple: CTRとCTCVRの予測値（制御群と処置群）
        """
        # 初期埋め込み表現の生成
        feature_embedding, treatment_embedding, click_task_embedding, conversion_task_embedding = \
            self.initial_embedding(features, treatment)
        
        # Treatment-Enhanced Network
        treatment_enhanced_embedding = self.tenet(feature_embedding, treatment_embedding)
        
        # Entire Chain-Enhanced Network
        ctr_control, ctr_treatment, ctcvr_control, ctcvr_treatment = \
            self.ecenet(treatment_enhanced_embedding, click_task_embedding, conversion_task_embedding, treatment)
        
        return ctr_control, ctr_treatment, ctcvr_control, ctcvr_treatment
    
    def predict_uplift(self, features, treatment_idx):
        """
        特定のトリートメントに対するアップリフトスコアを予測
        
        Args:
            features (torch.Tensor): 特徴量テンソル [batch_size, feature_dim]
            treatment_idx (int): 予測するトリートメントのインデックス（1-4）
            
        Returns:
            tuple: CTRとCTCVRのアップリフトスコア
        """
        # 制御群（treatment=0）の予測
        treatment_zeros = torch.zeros(features.size(0), dtype=torch.long, device=features.device)
        ctr_control, _, ctcvr_control, _ = self.forward(features, treatment_zeros)
        
        # 指定されたトリートメント群の予測
        treatment_t = torch.full((features.size(0),), treatment_idx, dtype=torch.long, device=features.device)
        ctr_t_control, ctr_t_treatment, ctcvr_t_control, ctcvr_t_treatment = self.forward(features, treatment_t)
        
        # treatment_idxに対応する処置群の予測値を取得
        ctr_t = ctr_t_treatment[:, treatment_idx-1].unsqueeze(1)
        ctcvr_t = ctcvr_t_treatment[:, treatment_idx-1].unsqueeze(1)
        
        # アップリフトスコアの計算（処置群 - 制御群）
        ctr_uplift = ctr_t - ctr_control
        ctcvr_uplift = ctcvr_t - ctcvr_control
        
        return ctr_uplift, ctcvr_uplift
