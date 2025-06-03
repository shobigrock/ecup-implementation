"""
Treatment-Enhanced Network (TENet) モジュール
論文の4.2節に対応する実装
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TreatmentAwareUnit(nn.Module):
    """
    Treatment-Aware Unit (TAU)
    論文の4.2.1節に対応
    """
    def __init__(self, feature_dim, embedding_dim, attention_dim):
        """
        初期化関数
        
        Args:
            feature_dim (int): 特徴量フィールド数
            embedding_dim (int): 埋め込み表現の次元数
            attention_dim (int): アテンション機構の次元数
        """
        super(TreatmentAwareUnit, self).__init__()
        
        # Self-attention用の変換行列
        self.W_Q = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.W_K = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.W_V = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.W_P = nn.Linear(attention_dim, embedding_dim, bias=False)
        
        # Treatment情報抽出器
        self.treatment_extractor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU()
        )
        
        # Treatment-aware特徴生成
        self.tau_transform = nn.Linear(embedding_dim * 2, embedding_dim)
        
        # Bit-level重み生成
        self.bit_weight_transform = nn.Linear(embedding_dim * 2, embedding_dim)
        
        self.attention_dim = attention_dim
        
    def forward(self, feature_embedding, treatment_embedding):
        """
        順伝播関数
        
        Args:
            feature_embedding (torch.Tensor): 特徴量埋め込み [batch_size, feature_dim, embedding_dim]
            treatment_embedding (torch.Tensor): トリートメント埋め込み [batch_size, 1, embedding_dim]
            
        Returns:
            tuple: Treatment-aware特徴表現とBit-level重み
        """
        batch_size, feature_dim, embedding_dim = feature_embedding.size()
        
        # Self-attention network - 論文の式(9)-(11)に対応
        Q = self.W_Q(feature_embedding)  # [batch_size, feature_dim, attention_dim]
        K = self.W_K(feature_embedding)  # [batch_size, feature_dim, attention_dim]
        V = self.W_V(feature_embedding)  # [batch_size, feature_dim, attention_dim]
        
        # 式(10)のアテンション計算
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.attention_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        E_V = torch.matmul(attention_weights, V)  # [batch_size, feature_dim, attention_dim]
        
        # 式(11)の次元復元
        E_att = self.W_P(E_V)  # [batch_size, feature_dim, embedding_dim]
        
        # Treatment情報抽出 - 論文の式(12)に対応
        treatment_info = self.treatment_extractor(treatment_embedding)  # [batch_size, 1, embedding_dim]
        
        # Treatment-aware特徴生成 - 論文の式(13)に対応
        treatment_info_expanded = treatment_info.expand(-1, feature_dim, -1)
        concat_embedding = torch.cat([E_att, treatment_info_expanded], dim=-1)  # [batch_size, feature_dim, embedding_dim*2]
        E_TAU = torch.tanh(self.tau_transform(concat_embedding))  # [batch_size, feature_dim, embedding_dim]
        
        # Bit-level重み生成 - 論文の式(14)に対応
        W_b = self.bit_weight_transform(concat_embedding)  # [batch_size, feature_dim, embedding_dim]
        
        return E_TAU, W_b

class TreatmentEnhancedGate(nn.Module):
    """
    Treatment-Enhanced Gate (TEGate)
    論文の4.2.2節に対応
    """
    def __init__(self):
        """
        初期化関数
        """
        super(TreatmentEnhancedGate, self).__init__()
        
    def forward(self, feature_embedding, E_TAU, W_b, treatment_embedding):
        """
        順伝播関数
        
        Args:
            feature_embedding (torch.Tensor): 初期特徴量埋め込み [batch_size, feature_dim, embedding_dim]
            E_TAU (torch.Tensor): Treatment-aware特徴表現 [batch_size, feature_dim, embedding_dim]
            W_b (torch.Tensor): Bit-level重み [batch_size, feature_dim, embedding_dim]
            treatment_embedding (torch.Tensor): トリートメント埋め込み [batch_size, 1, embedding_dim]
            
        Returns:
            torch.Tensor: Treatment-enhanced特徴表現
        """
        # 式(15)のゲーティング機構
        gate = torch.sigmoid(W_b)
        E_r = feature_embedding * gate + E_TAU * (1 - gate)  # [batch_size, feature_dim, embedding_dim]
        
        # 式(16)のトリートメント特徴の結合
        E_r_with_treatment = torch.cat([E_r, treatment_embedding], dim=1)  # [batch_size, feature_dim+1, embedding_dim]
        
        return E_r_with_treatment

class TENet(nn.Module):
    """
    Treatment-Enhanced Network (TENet)
    論文の4.2節に対応
    """
    def __init__(self, feature_dim, embedding_dim, attention_dim):
        """
        初期化関数
        
        Args:
            feature_dim (int): 特徴量フィールド数
            embedding_dim (int): 埋め込み表現の次元数
            attention_dim (int): アテンション機構の次元数
        """
        super(TENet, self).__init__()
        
        self.tau = TreatmentAwareUnit(feature_dim, embedding_dim, attention_dim)
        self.tegate = TreatmentEnhancedGate()
        
    def forward(self, feature_embedding, treatment_embedding):
        """
        順伝播関数
        
        Args:
            feature_embedding (torch.Tensor): 特徴量埋め込み [batch_size, feature_dim, embedding_dim]
            treatment_embedding (torch.Tensor): トリートメント埋め込み [batch_size, 1, embedding_dim]
            
        Returns:
            torch.Tensor: Treatment-enhanced特徴表現
        """
        E_TAU, W_b = self.tau(feature_embedding, treatment_embedding)
        E_r_with_treatment = self.tegate(feature_embedding, E_TAU, W_b, treatment_embedding)
        
        return E_r_with_treatment
