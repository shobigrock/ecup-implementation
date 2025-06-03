"""
評価スクリプト
"""
import os
import argparse
import yaml
import torch
import numpy as np
import logging
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_processor import DataProcessor
from models.ecup import ECUP
from utils.loss import evaluate_uplift

def evaluate(args):
    """
    モデルの評価を実行
    
    Args:
        args: コマンドライン引数
    """
    # ロギングの設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(os.path.dirname(args.model_path), 'evaluation.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # 設定ファイルの読み込み
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
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
    _, _, test_loader = data_processor.load_data()
    logger.info(f"Data loaded. Test batches: {len(test_loader)}")
    
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
    
    # モデルの読み込み
    logger.info(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 評価
    logger.info("Evaluating model...")
    results = {}
    
    # 各トリートメントに対して評価
    for treatment_idx in range(1, data_processor.get_treatment_count()):
        logger.info(f"Evaluating for treatment {treatment_idx}...")
        ctr_auuc, ctcvr_auuc, ctr_uplift, ctcvr_uplift = evaluate_uplift(
            model, test_loader, device, treatment_idx=treatment_idx
        )
        
        results[f'treatment_{treatment_idx}'] = {
            'ctr_auuc': float(ctr_auuc),
            'ctcvr_auuc': float(ctcvr_auuc),
            'ctr_uplift': float(ctr_uplift),
            'ctcvr_uplift': float(ctcvr_uplift)
        }
        
        logger.info(f"Treatment {treatment_idx} - CTR AUUC: {ctr_auuc:.4f}, CTCVR AUUC: {ctcvr_auuc:.4f}")
        logger.info(f"Treatment {treatment_idx} - Actual CTR Uplift: {ctr_uplift:.4f}, Actual CTCVR Uplift: {ctcvr_uplift:.4f}")
    
    # 結果の保存
    output_dir = os.path.dirname(args.model_path)
    with open(os.path.join(output_dir, 'evaluation_results.yaml'), 'w') as f:
        yaml.dump(results, f)
    
    logger.info(f"Evaluation results saved to {os.path.join(output_dir, 'evaluation_results.yaml')}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate ECUP model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # 評価の実行
    evaluate(args)

if __name__ == '__main__':
    main()
