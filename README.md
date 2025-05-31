# PhotonicEncoder: 光エンコーダを用いた圧縮・分類モデル

本研究では、**光回路上のエンコーダ**を用いてデータ圧縮を行い、その後PC上で機械学習による**分類**を行う手法を提案する。光回路を利用することで、高速かつ低消費電力な前処理（圧縮）を可能にする。

##  研究目的
- **位相変調（Phase Modulation, PM）型エンコーダ**を中心とした、光学的圧縮手法の有効性を評価。  
- 他手法（強度変調型, MZM型, LI型）との比較により、PM型の**非線形性が特徴抽出に有利である**ことを検証。  
- 復元精度と分類精度をタスクに応じて評価。

---

##  モデル構成

Input Image
↓
[光学エンコーダ (PM / IM / MZM / LI)]
↓
Compressed Representation
↓
[Classifier (MLP, CNN, DEQ)]
↓
Class Label


- **エンコーダ**：  
  - `PMEncoder`：位相変調  
  - `IMEncoder`：強度変調  
  - `MZMEncoder`：MZ変調  
  - `LIEncoder`：線形変調（比較用）  

- **後段**：  
  - 復元：AutoEncoder  
  - 分類：MLP, DEQ（Deep Equilibrium Model）  

---

## ディレクトリ構成
PhotonicEncoder/
├── main.py # 実行スクリプト（train および test）
├── evaluate.py # 評価・可視化用スクリプト
├── training.py # 学習ループ本体
├── IntegrationModel.py # 光学エンコーダ＋分類器モデル定義
├── Classifiers.py # MLPなど分類器定義
├── dataloader.py # MNIST/CIFAR 系のデータ読み込みモジュール
├── OtherModels.py # DEQFixedPoint, Cell, anderson などの補助モデル
├── test0527.ipynb # 実験用 Jupyter Notebook
