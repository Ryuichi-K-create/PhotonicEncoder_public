# PhotonicEncoder: 光回路と機械学習の融合による次世代データ圧縮・分類

> **"Light meets AI"** — 光の物理現象を利用した高速・低消費電力なデータ処理と、最先端の機械学習モデルの融合。

## 🌟 はじめに
現代のAI技術は膨大な計算リソースを必要とします。本プロジェクトは、電子回路（GPU/CPU）の代わりに**光回路（Photonic Circuits）**を用いてデータの前処理（圧縮・特徴抽出）を行うことで、計算の高速化と大幅な省電力化を目指す研究プロジェクトです。

特に、光の「位相（Phase）」と「非線形性」を巧みに利用することで、従来のデジタル計算では高負荷だった処理を、光が回路を通過する一瞬のうちに完了させることを可能にします。

---

## 📖 研究概要
本研究では、入力データを光の位相に写像する**位相変調型光エンコーダ（Phase Modulation Encoder）**を用い、光学的な非線形変換とランダム干渉を組み合わせた高速・低電力なデータ圧縮を実現することを目指しています。

### 背景と課題
位相変調は光強度検出と組み合わせることで強い**非線形性**を生み出し、特徴抽出能力に優れます。しかし、従来の構成では光回路を一度通過するだけで非線形変換が浅く、高圧縮下で情報が失われやすいという課題がありました。

### 提案手法: Photonic-DEQ Encoder
そこで本研究では、この非線形光エンコーダを **Deep Equilibrium Model（DEQ）** の枠組みで反復的に利用し、固定点となる潜在表現を取得する **Photonic-DEQ Encoder** を提案しました。
これにより、物理的な光回路のループ構造を、無限層のニューラルネットワークとして数理的にモデル化し、学習させることが可能になります。

### 主な成果
- **非線形性の優位性**: 位相変調由来の非線形性が、特徴抽出において強度変調方式よりも有利であることを実証しました。
- **精度と効率の両立**: Fashion-MNISTをはじめとしたさまざまな画像・表データを用いた評価において、従来法と比較して高圧縮時の分類精度向上と、PC側（デジタル側）モデルの軽量化を同時に実現しました。

---

## 🚀 応用と未来
本技術が実用化されれば、以下のようなインパクトが期待されます。

1.  **エッジデバイスの高度化**: 自動運転車やドローンなど、電力制約の厳しい環境で大量のセンサーデータをリアルタイムに処理。
2.  **通信トラフィックの削減**: データセンター間などで、情報を光のまま圧縮して送信することで、通信帯域と消費電力を大幅に削減。
3.  **次世代光AIチップ**: ムーアの法則の限界を超える、新しい計算パラダイムの確立。

### 🔬 今後の研究展望
- **多様なデータセットでの検証**: 現在のMNIST/CIFARに加え、より高解像度な画像データや、時系列データなどへの適用を進め、汎用性を検証します。
- **タスクの拡張**: 分類（Classification）だけでなく、画像の復元（Reconstruction）や回帰（Regression）タスクにもモデルを適用し、光エンコーダの表現能力を多角的に評価します。

---

## 🛠️ 技術詳細とコード構成

### 💻 技術スタック
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy&logoColor=white)

本リポジトリのコードは、PyTorchを用いて光回路の挙動をシミュレートし、機械学習モデルと統合して学習・評価を行うためのものです。

> ⚠️ **Note**: 本リポジトリは公開用であり、特許・秘密保持の観点から、核心となる光回路シミュレーションの一部はマスクまたは簡略化されています。

### システム構成
```mermaid
graph LR
    Input[入力画像] -->|光変調| Encoder["光エンコーダ<br>(PM/IM)"]
    Encoder -->|光干渉・検出| Compressed[圧縮表現]
    Compressed -->|デジタル処理| DEQ["Deep Equilibrium Model<br>(復元・分類)"]
    DEQ --> Output[クラス分類/画像復元]
```

### 🧠 実装モデルとアルゴリズム解説

本プロジェクトでは、光回路の出力を処理するために以下の機械学習モデルを実装・比較しています。

#### 1. Deep Equilibrium Models (DEQ)
- **概要**: 層を無限に積み重ねる代わりに、重みを共有した層を反復的に適用し、出力が収束する「平衡点（Fixed Point）」を求めるモデルです。これにより、少ないパラメータ数で非常に深いネットワークと同等の表現力を得ることができます。本研究では、光回路の物理的なループ構造をDEQとしてモデル化しています。
- **関連コード**: 
  - `models/OtherModels.py`: DEQのソルバー（Anderson Accelerationなど）や不動点探索ロジックを実装。
  - `models/IntegrationModel.py`: 光エンコーダとDEQを統合した `PhotonicDEQ` モデルなどを定義。

#### 2. 多層パーセプトロン (MLP) & CNN
- **概要**: 圧縮された特徴量からの分類を行うベースラインおよび比較用モデルとして使用しています。
- **関連コード**: 
  - `models/Classifiers.py`: シンプルな全結合層 (MLP) や、畳み込みニューラルネットワーク (CNN) の定義。

### ディレクトリ構成
- **`main.py`**: プロジェクトのエントリーポイント。実験設定（データセット、モデル、パラメータ）を行い、学習・評価プロセスを実行します。
- **`models/`**: モデル定義の中核
    - `IntegrationModel.py`: 光エンコーダとデジタル分類器を統合したモデル。
    - `Classifiers.py`: MLPやCNNなどの分類器定義。
    - `OtherModels.py`: DEQ (Deep Equilibrium Models) 関連のモジュール。
- **`train/`**: 学習・評価用スクリプト
    - `training.py`: 学習ループの実装。
    - `evaluate.py`: モデルの評価と可視化。
- **`dataloader/`**: データセット読み込み（MNIST, Fashion-MNIST, CIFAR-10など）。

### シミュレーション手法
光の伝搬、干渉、変調といった物理現象を、PyTorchの微分可能な演算として実装しています。これにより、光回路のパラメータ（物理的な設計値）を、ニューラルネットワークの重みとして誤差逆伝播法で最適化することが可能です。

---

## 🏆 発表実績

本研究の成果は、以下の学会・会議で発表されています（または予定されています）。

- **NOLTA 2025** (International Symposium on Nonlinear Theory and its Applications)
    - 沖縄で開催された非線形理論の国際学会にて発表。（予稿集など未公開）
    - 2025年10月28日-31日
    - [http://nolta2025.org/](http://nolta2025.org/)
- **学術変革領域（A）第7回領域会議**
    - 2025年12月15-16日、学術総合センタービルにて発表予定。
- **第73回 応用物理学会 春季学術講演会**
    - 2026年3月、発表予定。

---

## 📚 参考文献

1.  **Deep Equilibrium Models**
    - Bai, S., Kolter, J. Z., & Koltun, V. (2019). *Deep Equilibrium Models*. NeurIPS.
    - [https://arxiv.org/abs/1909.01377](https://arxiv.org/abs/1909.01377)
2.  **Photonic Encoder**
    - X. Wang et al., “Integrated photonic encoder for low power and high-speed image processing,” Nature Communications, 15, 4510 (2024).
    - [https://www.nature.com/articles/s41467-024-48099-2](https://www.nature.com/articles/s41467-024-48099-2)

---

## 🏫 所属・関連リンク
- **金沢大学 砂田・新山研究室 (Sunada Lab)**
  - [https://sn-lab.w3.kanazawa-u.ac.jp/]

*Author: Ryuichi-K-create*
