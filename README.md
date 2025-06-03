# ECUP: Entire Chain Uplift Modeling with Context-Enhanced Learning

このリポジトリは、論文「Entire Chain Uplift Modeling with Context-Enhanced Learning for Intelligent Marketing」の実装です。

## 概要

ECUPは、マーケティングにおけるアップリフトモデリングの課題を解決するための手法です。主に以下の2つの問題に対処します：

1. **Chain-bias問題**: ユーザー行動の連鎖（インプレッション→クリック→コンバージョン）において、各段階でのトリートメントの影響が異なることによるバイアス
2. **Treatment-unadaptive問題**: 異なるトリートメントに対する特徴量の適応性の不足

## 主要コンポーネント

ECUPは以下の2つの主要コンポーネントで構成されています：

1. **Entire Chain-Enhanced Network (ECENet)**
   - ユーザー行動パターンを活用して、チェーン全体でのITE（Individual Treatment Effect）を推定
   - タスク情報を活用してコンテキスト認識を強化

2. **Treatment-Enhanced Network (TENet)**
   - Treatment-Aware Unit (TAU)を通じてトリートメント認識特徴を生成
   - Treatment-Enhanced Gate (TEGate)を通じてビットレベルの特徴調整を実現

## データセット

実装では[MT-LIFT](https://github.com/mtdjdsp/mt-lift)データセットを使用します。このデータセットは美団（Meituan）のフードデリバリープラットフォームから収集された大規模な非バイアスデータセットで、以下の特徴を持ちます：

- 5,541,842サンプル
- 99の特徴量（f0-f98）
- クリックとコンバージョンの2つのラベル
- 5種類のトリートメント（0-4の範囲）

## 使用方法

```bash
# 環境設定
pip install -r requirements.txt

# データの準備
python src/data/prepare_data.py --data_dir /path/to/mt-lift

# モデルの学習
python src/train.py --config configs/ecup_config.yaml

# 評価
python src/evaluate.py --model_path /path/to/saved/model --test_data /path/to/test/data
```

## 引用

```
@inproceedings{huang2024entire,
  title={Entire Chain Uplift Modeling with Context-Enhanced Learning for Intelligent Marketing},
  author={Huang, Yinqiu and Wang, Shuli and Gao, Min and Wei, Xue and Li, Changhao and Luo, Chuan and Zhu, Yinhua and Xiao, Xiong and Luo, Yi},
  booktitle={Companion Proceedings of the ACM on Web Conference 2024},
  pages={226--234},
  year={2024}
}
```
