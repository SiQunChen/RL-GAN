# RL-GAN for HVAC

本專案提出了一種新穎的框架，利用強化學習 (Reinforcement Learning, RL) 來引導一個預訓練好的生成對抗網路 (Generative Adversarial Network, GAN)，以生成更具目標導向性與物理一致性的暖通空調 (HVAC) 系統時間序列數據。

傳統的 GAN 在生成數據時缺乏對特定效能目標（如節能、維持舒適溫度）的直接控制。本框架透過建立一個以 GAN 為核心的虛擬環境，讓 RL Agent 在此環境中學習如何透過微調 GAN 的輸入，來生成兼具「數據真實性」與「特定效能指標」的數據。

## 專案架構與核心檔案

```
.
├── data/
│   └── ChillerFinal500Ping_res.csv     # 原始真實 HVAC 數據
├── saved_models/
│   └── hvac_cgan_500ping.pth           # 預訓練好的基礎 GAN 模型
├── logs/
│   └── 20250922-223235_也還不錯/         # 訓練後 RL 模型的儲存位置 (範例)
│       ├── best_model.zip              # 訓練好的 RL Agent
│       └── hvac_rl_gan_generated.csv   # 透過 RL Agent 生成的數據
│       └── ... (其他日誌與評估報告)
├── gan_models.py                       # 定義 GAN 的生成器與評論家 (Critic) 架構
├── hvac_env.py                         # 核心檔案：將 GAN 包裝成一個 OpenAI Gym 的 RL 環境
├── train_rl.py                         # **主程式1：用於訓練 RL Agent**
├── generate.py                         # 負責載入訓練好的 RL Agent 來生成數據
├── 效能指標評估.py                      # 分析腳本：評估數據的物理效能 (舒適度、能耗)
├── 仿真度評估.py                        # 分析腳本：從統計與機器學習角度評估數據的真實性
└── run_evaluation.py                   # **主程式2：自動化執行數據生成與完整評估流程**
```

## 環境設定

在開始之前，請確保您已安裝所有必要的 Python 套件。建議在虛擬環境中進行安裝：

```bash
pip install torch gymnasium pandas numpy scikit-learn stable-baselines3 tqdm matplotlib seaborn statsmodels tabulate dtw-python
```

## 如何使用

本專案主要包含兩個核心流程：**模型訓練**與**評估與生成**。

### 1\. 訓練 RL Agent

如果您想要從頭開始訓練一個新的 RL Agent，請執行 `train_rl.py`。

```bash
python train_rl.py
```

**這個腳本會做什麼？**

  - 載入位於 `./saved_models/hvac_cgan_500ping.pth` 的預訓練 GAN 模型。
  - 載入位於 `./data/ChillerFinal500Ping_res.csv` 的真實數據作為環境參考。
  - 初始化 `GanHvacEnv` 環境，此環境的動態由 GAN 的生成器驅動。
  - 使用 Stable Baselines 3 的 SAC (Soft Actor-Critic) 演算法來訓練 Agent。
  - 訓練過程中，Agent 的目標是最大化一個綜合獎勵函數，該函數包含：
      - **HVAC 效能獎勵**: 鼓勵生成低能耗且溫度舒適的數據。
      - **真實性獎勵**: 透過 GAN 的判別器，鼓勵生成看起來更真實的數據。
      - **物理一致性獎勵**: 根據能量守恆定律，鼓勵生成符合物理定律的數據。
  - 訓練完成後，最佳的模型 (`best_model.zip`) 和 TensorBoard 日誌會被儲存在一個以時間戳命名的 `logs/` 資料夾中，例如 `logs/20250924-015030/`。

### 2\. 生成數據並進行完整評估

當您訓練好一個 RL Agent 後（或使用一個已有的模型），您可以使用 `run_evaluation.py` 來自動化地生成數據並產出完整的分析報告。

**使用方法非常簡單，只需在指令後方指定模型所在的日誌資料夾路徑即可。**

```bash
# 將 "your_model_log_directory_path" 替換成您實際的模型資料夾路徑
# 例如: python run_evaluation.py ./logs/20250922-223235_也還不錯
python run_evaluation.py [your_model_log_directory_path]
```

**這個腳本會做什麼？**

1.  **【數據生成】**:

      - 自動找到您指定的資料夾中的 `best_model.zip`。
      - 呼叫 `generate.py` 的核心功能，載入 RL Agent 來引導 GAN 生成與原始數據相同長度的仿真數據。
      - 生成的數據會被儲存為 `hvac_rl_gan_generated.csv`，同樣位於該模型資料夾中。

2.  **【仿真度綜合評估】**:

      - 呼叫 `仿真度評估.py`，對比 `hvac_rl_gan_generated.csv` 與原始真實數據。
      - 從多個維度進行深度分析，包含：
          - 描述性統計分析 (均值、標準差)。
          - 分佈保真度 (Wasserstein 距離)。
          - 時間序列特性 (ACF, DTW)。
          - 相關性結構分析。
          - **機器學習實用性 (TSTR)**: 在生成數據上訓練模型，在真實數據上測試，評估其 R² 分數。
          - **多變量結構評估 (PCA)**。
      - 產出一個名為 `comprehensive_analysis_results` 的子資料夾，裡面包含詳細的文字報告 (`comprehensive_report.txt`) 與所有分析圖表（如綜合評分雷達圖）。

3.  **【效能與保真度比較】**:

      - 呼叫 `效能指標評估.py`。
      - 同時載入**真實數據**、**傳統 GAN 生成的數據** (`synthetic_data_500ping.csv`) 以及**本方法 (RL+GAN) 生成的數據**。
      - 計算並印出三者的效能指標比較表，包含舒適度 (RMSE)、平均功耗，以及一個新的綜合指標 **CPI (Comfort-Power Index)**。CPI 值越低，代表模型的綜合性能越好。
      - 繪製多個比較圖表，如機率密度分佈圖、特徵相關性熱圖、自相關函數圖等，讓您可以直觀地比較三種數據的差異。
