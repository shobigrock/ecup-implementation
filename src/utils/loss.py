"""
損失関数と学習ユーティリティモジュール
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score

class UpliftLoss(nn.Module):
    """
    アップリフトモデリング用の損失関数
    """
    def __init__(self):
        """
        初期化関数
        """
        super(UpliftLoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='none')
        
    def forward(self, ctr_control, ctr_treatment, ctcvr_control, ctcvr_treatment, 
                click, conversion, treatment):
        """
        順伝播関数
        
        Args:
            ctr_control (torch.Tensor): 制御群のCTR予測 [batch_size, 1]
            ctr_treatment (torch.Tensor): 処置群のCTR予測 [batch_size, treatment_count-1]
            ctcvr_control (torch.Tensor): 制御群のCTCVR予測 [batch_size, 1]
            ctcvr_treatment (torch.Tensor): 処置群のCTCVR予測 [batch_size, treatment_count-1]
            click (torch.Tensor): クリックラベル [batch_size]
            conversion (torch.Tensor): コンバージョンラベル [batch_size]
            treatment (torch.Tensor): トリートメントラベル [batch_size]
            
        Returns:
            torch.Tensor: 損失値
        """
        batch_size = click.size(0)
        treatment_count = ctr_treatment.size(1) + 1
        
        # クリックとコンバージョンラベルの整形
        click = click.view(-1, 1)
        conversion = conversion.view(-1, 1)
        
        # 制御群と処置群のマスク
        control_mask = (treatment == 0).float().view(-1, 1)
        treatment_masks = [(treatment == i).float().view(-1, 1) for i in range(1, treatment_count)]
        
        # CTRタスクの損失
        ctr_control_loss = self.bce_loss(ctr_control, click) * control_mask
        ctr_treatment_losses = [self.bce_loss(ctr_treatment[:, i-1:i], click) * treatment_masks[i-1] 
                               for i in range(1, treatment_count)]
        
        # CTCVRタスクの損失
        # コンバージョンはクリックが発生した場合のみ考慮
        click_mask = (click == 1).float()
        ctcvr_control_loss = self.bce_loss(ctcvr_control, conversion) * control_mask * click_mask
        ctcvr_treatment_losses = [self.bce_loss(ctcvr_treatment[:, i-1:i], conversion) * treatment_masks[i-1] * click_mask 
                                 for i in range(1, treatment_count)]
        
        # 全ての損失を合計
        total_loss = ctr_control_loss.sum()
        for loss in ctr_treatment_losses:
            total_loss += loss.sum()
        total_loss += ctcvr_control_loss.sum()
        for loss in ctcvr_treatment_losses:
            total_loss += loss.sum()
        
        # バッチサイズで正規化
        total_loss = total_loss / batch_size
        
        return total_loss

def evaluate_uplift(model, data_loader, device, treatment_idx=1):
    """
    アップリフトモデルの評価
    
    Args:
        model (nn.Module): 評価するモデル
        data_loader (DataLoader): 評価用データローダー
        device (torch.device): 計算デバイス
        treatment_idx (int): 評価するトリートメントのインデックス
        
    Returns:
        tuple: CTRとCTCVRのAUUC（Area Under the Uplift Curve）
    """
    model.eval()
    
    all_features = []
    all_treatments = []
    all_clicks = []
    all_conversions = []
    all_ctr_uplifts = []
    all_ctcvr_uplifts = []
    
    with torch.no_grad():
        for batch in data_loader:
            features = batch['features'].to(device)
            treatments = batch['treatment'].to(device)
            clicks = batch['click'].to(device)
            conversions = batch['conversion'].to(device)
            
            # アップリフトスコアの予測
            ctr_uplift, ctcvr_uplift = model.predict_uplift(features, treatment_idx)
            
            all_features.append(features.cpu().numpy())
            all_treatments.append(treatments.cpu().numpy())
            all_clicks.append(clicks.cpu().numpy())
            all_conversions.append(conversions.cpu().numpy())
            all_ctr_uplifts.append(ctr_uplift.cpu().numpy())
            all_ctcvr_uplifts.append(ctcvr_uplift.cpu().numpy())
    
    # 配列の結合
    all_features = np.concatenate(all_features, axis=0)
    all_treatments = np.concatenate(all_treatments, axis=0)
    all_clicks = np.concatenate(all_clicks, axis=0)
    all_conversions = np.concatenate(all_conversions, axis=0)
    all_ctr_uplifts = np.concatenate(all_ctr_uplifts, axis=0).flatten()
    all_ctcvr_uplifts = np.concatenate(all_ctcvr_uplifts, axis=0).flatten()
    
    # 制御群と処置群のマスク
    control_mask = (all_treatments == 0)
    treatment_mask = (all_treatments == treatment_idx)
    
    # 実際のアップリフト計算用のデータ抽出
    control_clicks = all_clicks[control_mask]
    treatment_clicks = all_clicks[treatment_mask]
    control_conversions = all_conversions[control_mask]
    treatment_conversions = all_conversions[treatment_mask]
    
    # 実際のアップリフト計算
    actual_ctr_uplift = treatment_clicks.mean() - control_clicks.mean()
    actual_ctcvr_uplift = treatment_conversions.mean() - control_conversions.mean()
    
    # AUUCの計算（簡易版 - 実際にはCausalMLなどのライブラリを使用）
    # ここでは予測アップリフトスコアと実際のラベルの相関をAUCで代用
    ctr_auuc = roc_auc_score(all_clicks, all_ctr_uplifts) if len(np.unique(all_clicks)) > 1 else 0.5
    ctcvr_auuc = roc_auc_score(all_conversions, all_ctcvr_uplifts) if len(np.unique(all_conversions)) > 1 else 0.5
    
    return ctr_auuc, ctcvr_auuc, actual_ctr_uplift, actual_ctcvr_uplift
