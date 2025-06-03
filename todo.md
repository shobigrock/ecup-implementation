# ECUP実装タスクリスト

## データ準備
- [x] MT-LIFTデータセットの読み込み機能実装
- [x] 特徴量の前処理（正規化、エンコーディングなど）
- [x] データセットの分割（訓練/検証/テスト）
- [x] データローダーの実装

## モデル実装
- [x] 初期埋め込み表現の実装
- [x] Treatment-Enhanced Network (TENet)の実装
  - [x] Treatment-Aware Unit (TAU)の実装
  - [x] Treatment-Enhanced Gate (TEGate)の実装
- [x] Entire Chain-Enhanced Network (ECENet)の実装
  - [x] Task-Enhanced Network (TAENet)の実装
  - [x] Task-Enhanced Gate (TAEGate)の実装
- [x] 損失関数の実装

## 学習・評価
- [x] 学習ループの実装
- [x] 評価指標（AUUC、QINI）の実装
- [x] モデル保存・読み込み機能の実装
- [x] ハイパーパラメータ設定の実装

## ユーティリティ
- [x] 設定ファイル読み込み機能の実装
- [x] ロギング機能の実装
- [x] 可視化機能の実装

## テスト・検証
- [x] 単体テストの実装
- [x] 結果の検証と論文との比較

## ドキュメント
- [x] コードドキュメントの作成
- [x] 使用方法の詳細説明
