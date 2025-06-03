"""
データセット処理モジュール
MT-LIFTデータセットの読み込みと前処理を行います
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader

class MTLIFTDataset(Dataset):
    """
    MT-LIFTデータセット用のPyTorchデータセットクラス
    """
    def __init__(self, features, treatments, click_labels, conversion_labels):
        """
        初期化関数
        
        Args:
            features (np.ndarray): 特徴量配列
            treatments (np.ndarray): トリートメントラベル配列
            click_labels (np.ndarray): クリックラベル配列
            conversion_labels (np.ndarray): コンバージョンラベル配列
        """
        self.features = torch.FloatTensor(features)
        self.treatments = torch.LongTensor(treatments)
        self.click_labels = torch.FloatTensor(click_labels)
        self.conversion_labels = torch.FloatTensor(conversion_labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'treatment': self.treatments[idx],
            'click': self.click_labels[idx],
            'conversion': self.conversion_labels[idx]
        }

class DataProcessor:
    """
    MT-LIFTデータセットの処理クラス
    """
    def __init__(self, data_dir, batch_size=2048, test_size=0.2, val_size=0.1, random_state=42):
        """
        初期化関数
        
        Args:
            data_dir (str): データセットのディレクトリパス
            batch_size (int): バッチサイズ
            test_size (float): テストデータの割合
            val_size (float): 検証データの割合
            random_state (int): 乱数シード
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.feature_scaler = StandardScaler()
        
    def load_data(self):
        """
        データの読み込みと前処理を行います
        
        Returns:
            tuple: 訓練用、検証用、テスト用のDataLoaderオブジェクト
        """
        # 訓練データとテストデータの読み込み
        train_path = os.path.join(self.data_dir, 'train.csv')
        test_path = os.path.join(self.data_dir, 'test.csv')
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # 特徴量とラベルの分離
        feature_cols = [f'f{i}' for i in range(99)]  # f0-f98
        
        X_train = train_df[feature_cols].values
        treatment_train = train_df['treatment'].values
        click_train = train_df['click'].values
        conversion_train = train_df['conversion'].values
        
        X_test = test_df[feature_cols].values
        treatment_test = test_df['treatment'].values
        click_test = test_df['click'].values
        conversion_test = test_df['conversion'].values
        
        # 特徴量の標準化
        X_train = self.feature_scaler.fit_transform(X_train)
        X_test = self.feature_scaler.transform(X_test)
        
        # 訓練データから検証データを分割
        val_size_adjusted = self.val_size / (1 - self.test_size)
        np.random.seed(self.random_state)
        indices = np.random.permutation(len(X_train))
        val_count = int(len(X_train) * val_size_adjusted)
        val_indices = indices[:val_count]
        train_indices = indices[val_count:]
        
        X_val = X_train[val_indices]
        treatment_val = treatment_train[val_indices]
        click_val = click_train[val_indices]
        conversion_val = conversion_train[val_indices]
        
        X_train = X_train[train_indices]
        treatment_train = treatment_train[train_indices]
        click_train = click_train[train_indices]
        conversion_train = conversion_train[train_indices]
        
        # データセットの作成
        train_dataset = MTLIFTDataset(X_train, treatment_train, click_train, conversion_train)
        val_dataset = MTLIFTDataset(X_val, treatment_val, click_val, conversion_val)
        test_dataset = MTLIFTDataset(X_test, treatment_test, click_test, conversion_test)
        
        # DataLoaderの作成
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def get_feature_dim(self):
        """
        特徴量の次元数を返します
        
        Returns:
            int: 特徴量の次元数
        """
        return 99  # f0-f98の99次元
    
    def get_treatment_count(self):
        """
        トリートメントの種類数を返します
        
        Returns:
            int: トリートメントの種類数
        """
        return 5  # 0-4の5種類
