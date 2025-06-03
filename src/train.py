"""
学習ループと実行スクリプト
"""
import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_processor import DataProcessor
from models.ecup import ECUP
from utils.loss import UpliftLoss, evaluate_uplift

def train(config):
    """
    モデルの学習を実行
    
    Args:
        config (dict): 設定パラメータ
    """
    # ロギングの設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config['output_dir'], 'training.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # 出力ディレクトリの作成
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # デバイスの設定
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # データの読み込み
    logger.info("Loading data...")
    data_processor = DataProcessor(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        test_size=config['test_size'],
        val_size=config['val_size'],
        random_state=config['random_seed']
    )
    train_loader, val_loader, test_loader = data_processor.load_data()
    logger.info(f"Data loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # モデルの初期化
    logger.info("Initializing model...")
    model = ECUP(
        feature_dim=data_processor.get_feature_dim(),
        embedding_dim=config['embedding_dim'],
        hidden_dims=config['hidden_dims'],
        attention_dim=config['attention_dim'],
        treatment_count=data_processor.get_treatment_count(),
        num_heads=config['num_heads'],
        gamma=config['gamma']
    ).to(device)
    
    # 損失関数とオプティマイザの設定
    criterion = UpliftLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # 学習ループ
    logger.info("Starting training...")
    best_val_ctcvr_auuc = 0.0
    
    for epoch in range(config['num_epochs']):
        # 訓練モード
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
            # データをデバイスに転送
            features = batch['features'].to(device)
            treatment = batch['treatment'].to(device)
            click = batch['click'].to(device)
            conversion = batch['conversion'].to(device)
            
            # 勾配のリセット
            optimizer.zero_grad()
            
            # 順伝播
            ctr_control, ctr_treatment, ctcvr_control, ctcvr_treatment = model(features, treatment)
            
            # 損失の計算
            loss = criterion(ctr_control, ctr_treatment, ctcvr_control, ctcvr_treatment, 
                            click, conversion, treatment)
            
            # 逆伝播と最適化
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # エポックごとの平均損失
        train_loss /= len(train_loader)
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']}, Train Loss: {train_loss:.4f}")
        
        # 検証
        logger.info("Evaluating on validation set...")
        val_ctr_auuc, val_ctcvr_auuc, val_ctr_uplift, val_ctcvr_uplift = evaluate_uplift(
            model, val_loader, device, treatment_idx=config['eval_treatment_idx']
        )
        logger.info(f"Validation - CTR AUUC: {val_ctr_auuc:.4f}, CTCVR AUUC: {val_ctcvr_auuc:.4f}")
        logger.info(f"Validation - Actual CTR Uplift: {val_ctr_uplift:.4f}, Actual CTCVR Uplift: {val_ctcvr_uplift:.4f}")
        
        # 最良モデルの保存
        if val_ctcvr_auuc > best_val_ctcvr_auuc:
            best_val_ctcvr_auuc = val_ctcvr_auuc
            model_path = os.path.join(config['output_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ctr_auuc': val_ctr_auuc,
                'val_ctcvr_auuc': val_ctcvr_auuc,
            }, model_path)
            logger.info(f"New best model saved to {model_path}")
    
    # 最終モデルの保存
    final_model_path = os.path.join(config['output_dir'], 'final_model.pth')
    torch.save({
        'epoch': config['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # テストセットでの評価
    logger.info("Evaluating on test set...")
    # 最良モデルの読み込み
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_ctr_auuc, test_ctcvr_auuc, test_ctr_uplift, test_ctcvr_uplift = evaluate_uplift(
        model, test_loader, device, treatment_idx=config['eval_treatment_idx']
    )
    logger.info(f"Test - CTR AUUC: {test_ctr_auuc:.4f}, CTCVR AUUC: {test_ctcvr_auuc:.4f}")
    logger.info(f"Test - Actual CTR Uplift: {test_ctr_uplift:.4f}, Actual CTCVR Uplift: {test_ctcvr_uplift:.4f}")
    
    # 結果の保存
    results = {
        'test_ctr_auuc': float(test_ctr_auuc),
        'test_ctcvr_auuc': float(test_ctcvr_auuc),
        'test_ctr_uplift': float(test_ctr_uplift),
        'test_ctcvr_uplift': float(test_ctcvr_uplift),
    }
    
    with open(os.path.join(config['output_dir'], 'results.yaml'), 'w') as f:
        yaml.dump(results, f)
    
    logger.info("Training completed!")

def main():
    parser = argparse.ArgumentParser(description='Train ECUP model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # 設定ファイルの読み込み
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 学習の実行
    train(config)

if __name__ == '__main__':
    main()
