"""
Entire Chain-Enhanced Network (ECENet) モジュール
論文の4.1節に対応する実装
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .taenet import TaskEnhancedNetwork

class DNN(nn.Module):
    """
    Deep Neural Network (DNN) モジュール
    """
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.1):
        """
        初期化関数
        
        Args:
            input_dim (int): 入力次元数
            hidden_dims (list): 隠れ層の次元数リスト
            dropout_rate (float): ドロップアウト率
        """
        super(DNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
            
        self.dnn = nn.ModuleList(layers)
        
    def forward(self, x, task_scale=None):
        """
        順伝播関数
        
        Args:
            x (torch.Tensor): 入力テンソル
            task_scale (torch.Tensor, optional): タスク調整係数
            
        Returns:
            torch.Tensor: 出力テンソル
        """
        for i, layer in enumerate(self.dnn):
            if isinstance(layer, nn.Linear) and task_scale is not None:
                # 論文の式(7)に対応 - タスク調整係数による重み付け
                x = layer(x)
                x = x * task_scale
            else:
                x = layer(x)
                
        return x

class ECENet(nn.Module):
    """
    Entire Chain-Enhanced Network (ECENet)
    論文の4.1節に対応
    """
    def __init__(self, feature_dim, embedding_dim, hidden_dims, treatment_count=5, num_heads=2, gamma=1.0):
        """
        初期化関数
        
        Args:
            feature_dim (int): 特徴量フィールド数
            embedding_dim (int): 埋め込み表現の次元数
            hidden_dims (list): 隠れ層の次元数リスト
            treatment_count (int): トリートメントの種類数
            num_heads (int): マルチヘッドアテンションのヘッド数
            gamma (float): 調整係数
        """
        super(ECENet, self).__init__()
        
        # Task-Enhanced Network
        self.taenet = TaskEnhancedNetwork(embedding_dim, num_heads, gamma)
        
        # 入力次元数計算（特徴量+トリートメント）
        input_dim = (feature_dim + 1) * embedding_dim
        
        # CTRタスク用DNN
        self.ctr_dnn = DNN(input_dim, hidden_dims)
        self.ctr_output = nn.Linear(hidden_dims[-1], 1)
        
        # CTCVRタスク用DNN
        self.ctcvr_dnn = DNN(input_dim, hidden_dims)
        self.ctcvr_output = nn.Linear(hidden_dims[-1], 1)
        
        # 制御グループ（treatment=0）用の出力層
        self.ctr_control = nn.Linear(hidden_dims[-1], 1)
        self.ctcvr_control = nn.Linear(hidden_dims[-1], 1)
        
        # トリートメント別出力層（treatment=1,2,3,4）
        self.ctr_treatment = nn.ModuleList([nn.Linear(hidden_dims[-1], 1) for _ in range(treatment_count-1)])
        self.ctcvr_treatment = nn.ModuleList([nn.Linear(hidden_dims[-1], 1) for _ in range(treatment_count-1)])
        
    def forward(self, treatment_enhanced_embedding, click_task_embedding, conversion_task_embedding, treatment):
        """
        順伝播関数
        
        Args:
            treatment_enhanced_embedding (torch.Tensor): トリートメント強化埋め込み [batch_size, feature_dim+1, embedding_dim]
            click_task_embedding (torch.Tensor): クリックタスク埋め込み [batch_size, embedding_dim]
            conversion_task_embedding (torch.Tensor): コンバージョンタスク埋め込み [batch_size, embedding_dim]
            treatment (torch.Tensor): トリートメントテンソル [batch_size]
            
        Returns:
            tuple: CTRとCTCVRの予測値（制御群と処置群）
        """
        batch_size = treatment_enhanced_embedding.size(0)
        
        # 埋め込みをフラット化
        flat_embedding = treatment_enhanced_embedding.view(batch_size, -1)  # [batch_size, (feature_dim+1)*embedding_dim]
        
        # タスク調整係数の計算
        ctr_scale = self.taenet(click_task_embedding, treatment_enhanced_embedding)
        ctcvr_scale = self.taenet(conversion_task_embedding, treatment_enhanced_embedding)
        
        # CTRタスクの予測
        ctr_hidden = self.ctr_dnn(flat_embedding, ctr_scale)
        ctr_control_output = torch.sigmoid(self.ctr_control(ctr_hidden))
        
        # CTCVRタスクの予測
        ctcvr_hidden = self.ctcvr_dnn(flat_embedding, ctcvr_scale)
        ctcvr_control_output = torch.sigmoid(self.ctcvr_control(ctcvr_hidden))
        
        # トリートメント別の予測値を格納するリスト
        ctr_treatment_outputs = []
        ctcvr_treatment_outputs = []
        
        # 各トリートメントの予測値を計算
        for i in range(len(self.ctr_treatment)):
            ctr_treatment_output = torch.sigmoid(self.ctr_treatment[i](ctr_hidden))
            ctcvr_treatment_output = torch.sigmoid(self.ctcvr_treatment[i](ctcvr_hidden))
            
            ctr_treatment_outputs.append(ctr_treatment_output)
            ctcvr_treatment_outputs.append(ctcvr_treatment_output)
        
        # トリートメント別の予測値をスタック
        ctr_treatment_stack = torch.cat([o for o in ctr_treatment_outputs], dim=1)  # [batch_size, treatment_count-1]
        ctcvr_treatment_stack = torch.cat([o for o in ctcvr_treatment_outputs], dim=1)  # [batch_size, treatment_count-1]
        
        return ctr_control_output, ctr_treatment_stack, ctcvr_control_output, ctcvr_treatment_stack
